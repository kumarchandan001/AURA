"""
realtime_vitals.py — Sliding-Window Real-Time BPM & HRV Extraction.

Provides a lightweight, thread-safe vital-signs processor that accepts
individual Green-channel samples, accumulates them in a circular buffer,
and periodically re-computes BPM and RMSSD over the most recent window.

This is the real-time counterpart of Phase 2's offline ``signal_processor.py``,
optimised for low-latency incremental processing suitable for the affective
feedback loop.

Classes:
    RealtimeVitals — Circular-buffer BPM/HRV extractor.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


class RealtimeVitals:
    """Sliding-window real-time BPM and RMSSD computation.

    Accumulates Green-channel spatial means and their timestamps in a
    fixed-duration circular buffer.  On each call to :meth:`compute`,
    the buffer contents are detrended, bandpass-filtered, and peak-detected
    to yield instantaneous BPM and RMSSD.

    Parameters
    ----------
    window_seconds : float
        Duration of the analysis window (default: 10 s).
    min_window_seconds : float
        Minimum data required before attempting computation (default: 4 s).

    Attributes
    ----------
    latest_bpm : float
        Most recently computed BPM (NaN if unavailable).
    latest_rmssd : float
        Most recently computed RMSSD in ms (NaN if unavailable).

    Examples
    --------
    >>> vitals = RealtimeVitals(window_seconds=10)
    >>> vitals.add_sample(green_value=142.3, timestamp=1001.234)
    >>> bpm, rmssd = vitals.compute()
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        min_window_seconds: float = 4.0,
    ) -> None:
        self._window_seconds = window_seconds
        self._min_window_seconds = min_window_seconds

        self._green_buffer: Deque[float] = deque()
        self._ts_buffer: Deque[float] = deque()
        self._lock = threading.Lock()

        self.latest_bpm: float = float("nan")
        self.latest_rmssd: float = float("nan")
        self.latest_peaks_count: int = 0

        logger.info(
            "RealtimeVitals initialised — window=%.1fs  min=%.1fs",
            window_seconds,
            min_window_seconds,
        )

    def add_sample(self, green_value: float, timestamp: float) -> None:
        """Add a single Green-channel sample to the buffer.

        Parameters
        ----------
        green_value : float
            Spatial mean of the Green channel from the ROI.
        timestamp : float
            Monotonic ``perf_counter`` timestamp.
        """
        if np.isnan(green_value):
            return

        with self._lock:
            self._green_buffer.append(green_value)
            self._ts_buffer.append(timestamp)

            # Evict samples outside the window.
            cutoff = timestamp - self._window_seconds
            while self._ts_buffer and self._ts_buffer[0] < cutoff:
                self._ts_buffer.popleft()
                self._green_buffer.popleft()

    def compute(self) -> Tuple[float, float]:
        """Run BPM and RMSSD extraction on the current buffer contents.

        Returns
        -------
        tuple[float, float]
            ``(bpm, rmssd)`` — both NaN if insufficient data.
        """
        with self._lock:
            if len(self._green_buffer) < 10:
                return float("nan"), float("nan")

            green = np.array(self._green_buffer, dtype=np.float64)
            timestamps = np.array(self._ts_buffer, dtype=np.float64)

        duration = timestamps[-1] - timestamps[0]
        if duration < self._min_window_seconds:
            return float("nan"), float("nan")

        n_samples = len(green)
        fs = (n_samples - 1) / duration

        if fs < 5.0:
            # Sampling rate too low for cardiac band.
            return float("nan"), float("nan")

        try:
            bpm, rmssd, n_peaks = self._process_window(green, timestamps, fs)
            self.latest_bpm = bpm
            self.latest_rmssd = rmssd
            self.latest_peaks_count = n_peaks
            return bpm, rmssd
        except Exception as e:
            logger.debug("Vitals computation failed: %s", e)
            return float("nan"), float("nan")

    def _process_window(
        self,
        green: np.ndarray,
        timestamps: np.ndarray,
        fs: float,
    ) -> Tuple[float, float, int]:
        """Internal DSP pipeline: detrend → filter → peaks → BPM/RMSSD.

        Parameters
        ----------
        green : numpy.ndarray
            Green channel values.
        timestamps : numpy.ndarray
            Corresponding timestamps.
        fs : float
            Computed sampling rate.

        Returns
        -------
        tuple[float, float, int]
            ``(bpm, rmssd_ms, n_peaks)``
        """
        # Detrend: linear + rolling mean.
        from scipy.signal import detrend as sp_detrend

        sig = sp_detrend(green, type="linear")
        win = max(int(fs * 1.5), 3)
        if win % 2 == 0:
            win += 1
        rolling = pd.Series(sig).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
        sig = sig - rolling

        # Bandpass filter: 0.7–3.0 Hz.
        nyq = fs / 2.0
        high = min(3.0, nyq * 0.95)
        low = 0.7
        if low >= high:
            return float("nan"), float("nan"), 0

        b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
        pad = min(3 * max(len(b), len(a)), len(sig) - 1)
        if pad < 1:
            return float("nan"), float("nan"), 0
        filtered = filtfilt(b, a, sig, padlen=pad)

        # Peak detection.
        min_dist = max(int(fs * (60.0 / 180.0)), 1)
        prominence = 0.3 * np.std(filtered)
        if prominence < 1e-6:
            return float("nan"), float("nan"), 0

        peaks, _ = find_peaks(filtered, distance=min_dist, prominence=prominence)

        if len(peaks) < 2:
            return float("nan"), float("nan"), len(peaks)

        # BPM from peak times.
        peak_times = timestamps[peaks]
        ibi_s = np.diff(peak_times)
        valid_ibi = ibi_s[(ibi_s > 0.33) & (ibi_s < 1.5)]  # 40–180 BPM range.

        if len(valid_ibi) < 1:
            return float("nan"), float("nan"), len(peaks)

        mean_bpm = float(60.0 / np.mean(valid_ibi))

        # RMSSD.
        if len(valid_ibi) < 2:
            return mean_bpm, float("nan"), len(peaks)

        ibi_ms = valid_ibi * 1000.0
        successive_diffs = np.diff(ibi_ms)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

        return mean_bpm, rmssd, len(peaks)

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        with self._lock:
            self._green_buffer.clear()
            self._ts_buffer.clear()
        self.latest_bpm = float("nan")
        self.latest_rmssd = float("nan")
        self.latest_peaks_count = 0
