"""
signal_processor.py — Phase 2: Signal Conditioning, Pulse Extraction & HRV.

This script performs offline digital signal processing on the raw optical data
captured during Phase 1 to extract clinically meaningful physiological metrics.

Processing Pipeline:
    1. Load ``phase1_raw_vitals.csv`` and compute exact sampling rate from
       the monotonic timestamp column.
    2. Detrend the Green channel to remove slow baseline wander caused by
       ambient illumination drift and involuntary subject movement.
    3. Apply a zero-phase Butterworth bandpass filter (0.7–3.0 Hz) to isolate
       the cardiac pulse band (42–180 BPM).
    4. Detect systolic peaks via ``scipy.signal.find_peaks`` with adaptive
       prominence and minimum inter-beat distance constraints.
    5. Derive Inter-Beat Intervals (IBI), instantaneous BPM, and compute
       RMSSD — the gold-standard time-domain HRV metric for autonomic
       nervous system assessment.
    6. Generate a publication-quality 3-panel research figure and export
       aggregated metrics to the terminal.

Usage:
    python signal_processor.py
    python signal_processor.py --input phase1_raw_vitals.csv --output phase2_signal_analysis.png

Author : Project SENTIO Research Team
Version: 0.2.0 (Phase 2 — Signal Processing)
"""

from __future__ import annotations
from scipy.signal import detrend
from scipy import signal as sp_signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings

import argparse
import sys
from pathlib import Path

# ── Dynamic Data Paths ────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "manuscript"


# ======================================================================
# Constants
# ======================================================================

# Cardiac passband: 42 BPM → 180 BPM.
LOW_CUTOFF_HZ: float = 0.7
HIGH_CUTOFF_HZ: float = 3.0

# Butterworth filter order.
FILTER_ORDER: int = 4

# Minimum recording duration (seconds) for reliable HRV.
MIN_DURATION_S: float = 5.0

# Minimum number of peaks needed for HRV calculation.
MIN_PEAKS_HRV: int = 4


# ======================================================================
# DSP Functions
# ======================================================================

def compute_sampling_rate(timestamps: np.ndarray) -> float:
    """Compute the exact average sampling rate from monotonic timestamps.

    Instead of assuming a fixed FPS, this function calculates the true
    mean inter-frame interval from the recorded ``perf_counter`` values.
    This is critical because webcam drivers rarely deliver frames at their
    nominal rate, and filter cutoff accuracy depends on correct Fs.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1-D array of monotonic ``perf_counter`` values (seconds).

    Returns
    -------
    float
        Average sampling frequency in Hz.

    Raises
    ------
    ValueError
        If fewer than 2 timestamps are provided.
    """
    if len(timestamps) < 2:
        raise ValueError(
            f"Need at least 2 timestamps to compute Fs; got {len(timestamps)}."
        )

    dt = np.diff(timestamps)
    mean_dt = np.mean(dt)

    if mean_dt <= 0:
        raise ValueError(
            f"Non-positive mean inter-frame interval ({mean_dt:.6f} s). "
            "Timestamps may be corrupted."
        )

    fs = 1.0 / mean_dt
    return float(fs)


def detrend_signal(
    raw_signal: np.ndarray,
    fs: float,
    window_seconds: float = 1.5,
) -> np.ndarray:
    """Remove slow baseline wander from the raw optical signal.

    Applies a two-stage detrend:
        1. **Linear detrend** via ``scipy.signal.detrend`` to remove any
           global linear trend (e.g., from camera auto-exposure ramping).
        2. **Rolling-mean subtraction** with a window of ``window_seconds``
           to eliminate slower oscillatory drift (respiration, lighting
           changes) without distorting the cardiac pulse.

    Parameters
    ----------
    raw_signal : numpy.ndarray
        1-D raw Green channel signal.
    fs : float
        Sampling frequency in Hz.
    window_seconds : float, optional
        Duration (seconds) of the rolling-mean smoothing window.
        Default is 1.5 s (large enough to preserve the ~1 s cardiac cycle).

    Returns
    -------
    numpy.ndarray
        Detrended signal with zero mean.
    """
    # Stage 1: Remove linear trend.
    linear_detrended = detrend(raw_signal, type="linear")

    # Stage 2: Subtract rolling mean.
    window_size = max(int(fs * window_seconds), 3)
    # Ensure odd window for symmetric kernel.
    if window_size % 2 == 0:
        window_size += 1

    rolling_mean = pd.Series(linear_detrended).rolling(
        window=window_size, center=True, min_periods=1
    ).mean().to_numpy()

    detrended = linear_detrended - rolling_mean
    return detrended


def design_bandpass_filter(
    fs: float,
    low_hz: float = LOW_CUTOFF_HZ,
    high_hz: float = HIGH_CUTOFF_HZ,
    order: int = FILTER_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter for the cardiac pulse band.

    The Nyquist-normalised cutoffs are computed from the dynamically
    measured sampling rate ``fs`` to ensure accurate filter placement
    regardless of the camera's actual frame delivery rate.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    low_hz : float
        Lower cutoff frequency in Hz (default: 0.7 Hz = 42 BPM).
    high_hz : float
        Upper cutoff frequency in Hz (default: 3.0 Hz = 180 BPM).
    order : int
        Butterworth filter order (default: 4).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(b, a)`` — Numerator and denominator polynomial coefficients.

    Raises
    ------
    ValueError
        If the cutoff frequencies violate the Nyquist criterion.
    """
    nyquist = fs / 2.0

    if high_hz >= nyquist:
        warnings.warn(
            f"Upper cutoff ({high_hz} Hz) ≥ Nyquist ({nyquist:.1f} Hz). "
            f"Clamping to {nyquist * 0.95:.2f} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )
        high_hz = nyquist * 0.95

    if low_hz <= 0:
        raise ValueError(f"Lower cutoff must be > 0; got {low_hz}.")
    if low_hz >= high_hz:
        raise ValueError(
            f"Lower cutoff ({low_hz} Hz) must be < upper cutoff ({high_hz} Hz)."
        )

    wn = [low_hz / nyquist, high_hz / nyquist]
    b, a = sp_signal.butter(order, wn, btype="bandpass")
    return b, a


def apply_bandpass_filter(
    sig: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Uses ``scipy.signal.filtfilt`` for forward-backward filtering, which
    eliminates phase distortion — essential for accurate peak timing in
    rPPG.  The signal is padded using the ``gust`` method to minimise
    edge transients.

    Parameters
    ----------
    sig : numpy.ndarray
        1-D detrended signal.
    b, a : numpy.ndarray
        Filter coefficients from :func:`design_bandpass_filter`.

    Returns
    -------
    numpy.ndarray
        Filtered signal (same length as input).
    """
    # Require sufficient length for filtfilt padding.
    pad_len = 3 * max(len(b), len(a))
    if len(sig) <= pad_len:
        warnings.warn(
            f"Signal length ({len(sig)}) is very short relative to filter "
            f"order (padlen={pad_len}). Results may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
        pad_len = len(sig) - 1

    filtered = sp_signal.filtfilt(b, a, sig, padlen=pad_len)
    return filtered


def detect_peaks(
    filtered_signal: np.ndarray,
    fs: float,
    min_bpm: float = 42.0,
    max_bpm: float = 180.0,
) -> np.ndarray:
    """Detect cardiac peaks in the bandpass-filtered Green signal.

    Uses ``scipy.signal.find_peaks`` with physiologically motivated
    constraints:
        - **Minimum distance** between peaks derived from ``max_bpm``.
        - **Adaptive prominence** set to 30% of the signal's standard
          deviation to reject noise spikes without missing weak pulses.

    Parameters
    ----------
    filtered_signal : numpy.ndarray
        1-D bandpass-filtered signal.
    fs : float
        Sampling frequency in Hz.
    min_bpm : float
        Lower physiological bound (minimum HR), used to set max peak
        distance (default: 42 BPM).
    max_bpm : float
        Upper physiological bound (maximum HR), used to set min peak
        distance (default: 180 BPM).

    Returns
    -------
    numpy.ndarray
        Array of sample indices where peaks were detected.
    """
    # Minimum distance: at max_bpm, peaks are separated by (60/max_bpm) s.
    min_distance_samples = int(fs * (60.0 / max_bpm))
    min_distance_samples = max(min_distance_samples, 1)

    # Adaptive prominence threshold.
    prominence = 0.3 * np.std(filtered_signal)

    peaks, properties = sp_signal.find_peaks(
        filtered_signal,
        distance=min_distance_samples,
        prominence=prominence,
    )

    return peaks


def compute_ibi_and_bpm(
    peaks: np.ndarray,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Inter-Beat Intervals (IBI), instantaneous BPM, and mean BPM.

    Parameters
    ----------
    peaks : numpy.ndarray
        Sample indices of detected peaks.
    timestamps : numpy.ndarray
        Monotonic timestamp array corresponding to each sample.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        ``(ibi_ms, inst_bpm, mean_bpm)``
        - ``ibi_ms`` — Inter-beat intervals in milliseconds.
        - ``inst_bpm`` — Instantaneous BPM for each IBI.
        - ``mean_bpm`` — Average BPM over the recording.
    """
    if len(peaks) < 2:
        return np.array([]), np.array([]), float("nan")

    peak_times = timestamps[peaks]
    ibi_s = np.diff(peak_times)
    ibi_ms = ibi_s * 1000.0  # Convert to milliseconds.

    # Guard against zero-division.
    with np.errstate(divide="ignore", invalid="ignore"):
        inst_bpm = 60.0 / ibi_s

    # Reject physiologically impossible values.
    valid_mask = (inst_bpm >= 30) & (inst_bpm <= 220)
    if np.sum(valid_mask) == 0:
        return ibi_ms, inst_bpm, float("nan")

    mean_bpm = float(np.mean(inst_bpm[valid_mask]))
    return ibi_ms, inst_bpm, mean_bpm


def compute_rmssd(ibi_ms: np.ndarray) -> float:
    """Compute the Root Mean Square of Successive Differences (RMSSD).

    RMSSD is the gold-standard time-domain metric for parasympathetic
    (vagal) cardiac autonomic modulation.  Higher RMSSD indicates
    greater heart rate variability and better stress resilience.

    Interpretation guide (resting; finger-tip PPG reference ranges):
        - RMSSD < 20 ms  → Low HRV (high sympathetic drive / stress)
        - RMSSD 20–50 ms → Normal range
        - RMSSD > 50 ms  → High HRV (strong vagal tone)

    Parameters
    ----------
    ibi_ms : numpy.ndarray
        Inter-Beat Intervals in milliseconds.

    Returns
    -------
    float
        RMSSD in milliseconds, or ``NaN`` if insufficient data.
    """
    if len(ibi_ms) < 2:
        return float("nan")

    successive_diffs = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    return rmssd


# ======================================================================
# Visualization
# ======================================================================

def generate_research_plot(
    timestamps: np.ndarray,
    raw_green: np.ndarray,
    detrended_filtered: np.ndarray,
    peaks: np.ndarray,
    inst_bpm: np.ndarray,
    mean_bpm: float,
    rmssd: float,
    fs: float,
    output_path: Path,
) -> None:
    """Generate a publication-quality 3-panel signal analysis figure.

    Panel layout:
        1. Raw Green channel signal (unprocessed optical trace).
        2. Detrended + bandpass-filtered signal with detected peaks.
        3. Instantaneous BPM timeline.

    Parameters
    ----------
    timestamps : numpy.ndarray
        Monotonic timestamps (seconds).
    raw_green : numpy.ndarray
        Original Green channel spatial means.
    detrended_filtered : numpy.ndarray
        Signal after detrending and Butterworth filtering.
    peaks : numpy.ndarray
        Detected peak sample indices.
    inst_bpm : numpy.ndarray
        Instantaneous BPM values (one per IBI).
    mean_bpm : float
        Average BPM over the recording.
    rmssd : float
        RMSSD in milliseconds.
    fs : float
        Computed sampling frequency.
    output_path : Path
        File path for the saved figure.
    """
    # Normalise time axis to start at 0.
    t = timestamps - timestamps[0]

    # ── Figure setup ─────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=150)
    fig.suptitle(
        "Project SENTIO — Phase 2: Signal Analysis",
        fontsize=16,
        fontweight="bold",
        color="#00FFD0",
        y=0.98,
    )

    # Colour palette.
    raw_colour = "#4FC3F7"      # Light blue.
    filtered_colour = "#00E676"  # Bright green.
    peak_colour = "#FF1744"     # Red accent.
    bpm_colour = "#FFD740"      # Amber.
    grid_colour = "#333333"
    mean_line_colour = "#FF6D00"

    # ── Panel 1: Raw Green Signal ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(t, raw_green, color=raw_colour, linewidth=0.6, alpha=0.9)
    ax1.set_title("Raw Green Channel (Unprocessed Optical Trace)", fontsize=11, color="#BBBBBB")
    ax1.set_ylabel("Pixel Intensity", fontsize=10)
    ax1.grid(True, alpha=0.3, color=grid_colour)
    ax1.tick_params(colors="#999999")

    # ── Panel 2: Filtered Signal + Peaks ─────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, detrended_filtered, color=filtered_colour, linewidth=0.8, alpha=0.9, label="Filtered")
    if len(peaks) > 0:
        ax2.scatter(
            t[peaks], detrended_filtered[peaks],
            color=peak_colour, s=40, zorder=5, edgecolors="white",
            linewidths=0.5, label=f"Peaks (n={len(peaks)})",
        )
    ax2.set_title(
        f"Bandpass Filtered [{LOW_CUTOFF_HZ}–{HIGH_CUTOFF_HZ} Hz] + Peak Detection",
        fontsize=11, color="#BBBBBB",
    )
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.6)
    ax2.grid(True, alpha=0.3, color=grid_colour)
    ax2.tick_params(colors="#999999")

    # ── Panel 3: Instantaneous BPM ───────────────────────────────────
    ax3 = axes[2]
    if len(inst_bpm) > 0 and len(peaks) >= 2:
        # Each BPM value corresponds to the midpoint between two peaks.
        bpm_times = (t[peaks[:-1]] + t[peaks[1:]]) / 2.0
        ax3.plot(bpm_times, inst_bpm, color=bpm_colour, linewidth=1.2, marker="o",
                 markersize=3, alpha=0.9, label="Instantaneous BPM")

        if not np.isnan(mean_bpm):
            ax3.axhline(y=mean_bpm, color=mean_line_colour, linestyle="--",
                        linewidth=1.0, alpha=0.8, label=f"Mean: {mean_bpm:.1f} BPM")

        ax3.legend(loc="upper right", fontsize=9, framealpha=0.6)
    else:
        ax3.text(0.5, 0.5, "Insufficient peaks for BPM calculation",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=12, color="#FF5252")

    ax3.set_title("Instantaneous Heart Rate", fontsize=11, color="#BBBBBB")
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("BPM", fontsize=10)
    ax3.grid(True, alpha=0.3, color=grid_colour)
    ax3.tick_params(colors="#999999")

    # ── Metrics annotation box ───────────────────────────────────────
    metrics_text = (
        f"Fs = {fs:.1f} Hz   |   "
        f"Avg BPM = {mean_bpm:.1f}   |   "
        f"RMSSD = {rmssd:.1f} ms   |   "
        f"Peaks = {len(peaks)}"
    )
    fig.text(
        0.5, 0.015, metrics_text, ha="center", va="bottom",
        fontsize=10, color="#00FFD0",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A1A2E", edgecolor="#00FFD0", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {output_path.resolve()}")


# ======================================================================
# Main Pipeline
# ======================================================================

def run_phase2(input_csv: Path, output_png: Path) -> None:
    """Execute the complete Phase 2 signal processing pipeline.

    Parameters
    ----------
    input_csv : Path
        Path to the Phase 1 raw vitals CSV.
    output_png : Path
        Output path for the research figure.
    """
    separator = "=" * 62

    print(f"\n{separator}")
    print("  PROJECT SENTIO — Phase 2: Signal Processing")
    print(f"{separator}\n")

    # ── 1. Data ingestion ────────────────────────────────────────────
    if not input_csv.exists():
        print(f"  ✗ Input file not found: {input_csv.resolve()}")
        print("    Run Phase 1 first to generate the CSV.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"  ✓ Loaded {len(df)} frames from {input_csv.name}")

    # Validate required columns.
    required_cols = {"timestamp", "mean_g"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        sys.exit(1)

    # Drop rows where Green channel is NaN (face lost).
    valid_mask = df["mean_g"].notna() & df["timestamp"].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  ⚠ Dropped {n_dropped} NaN rows (face lost during capture)")
    df_valid = df[valid_mask].reset_index(drop=True)

    if len(df_valid) < 10:
        print("  ✗ Insufficient valid data points (< 10). Cannot process.")
        sys.exit(1)

    timestamps = df_valid["timestamp"].to_numpy(dtype=np.float64)
    raw_green = df_valid["mean_g"].to_numpy(dtype=np.float64)

    # ── 2. Compute exact sampling rate ───────────────────────────────
    fs = compute_sampling_rate(timestamps)
    duration = timestamps[-1] - timestamps[0]
    print(f"  ✓ Sampling rate: {fs:.2f} Hz (computed from timestamps)")
    print(f"  ✓ Recording duration: {duration:.2f} s")

    if duration < MIN_DURATION_S:
        print(
            f"\n  ⚠ WARNING: Recording duration ({duration:.1f} s) is below the\n"
            f"    minimum threshold ({MIN_DURATION_S} s) for reliable HRV.\n"
            f"    Results may not be physiologically meaningful.\n"
        )

    # ── 3. Detrend ───────────────────────────────────────────────────
    detrended = detrend_signal(raw_green, fs)
    print("  ✓ Detrended (linear + rolling mean, window=1.5 s)")

    # ── 4. Bandpass filter ───────────────────────────────────────────
    b, a = design_bandpass_filter(fs, LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ, FILTER_ORDER)
    filtered = apply_bandpass_filter(detrended, b, a)
    print(f"  ✓ Butterworth bandpass applied [{LOW_CUTOFF_HZ}–{HIGH_CUTOFF_HZ} Hz], order={FILTER_ORDER}")

    # ── 5. Peak detection ────────────────────────────────────────────
    peaks = detect_peaks(filtered, fs)
    print(f"  ✓ Peaks detected: {len(peaks)}")

    if len(peaks) < 2:
        print(
            "\n  ✗ Fewer than 2 peaks detected. Cannot compute BPM or HRV.\n"
            "    Possible causes:\n"
            "      - Very short recording\n"
            "      - Face not visible during capture\n"
            "      - Excessive motion artifact\n"
        )
        # Still generate the plot with what we have.
        ibi_ms = np.array([])
        inst_bpm = np.array([])
        mean_bpm = float("nan")
        rmssd = float("nan")
    else:
        # ── 6. BPM & HRV ────────────────────────────────────────────
        ibi_ms, inst_bpm, mean_bpm = compute_ibi_and_bpm(peaks, timestamps)
        rmssd = compute_rmssd(ibi_ms)

    # ── 7. Report ────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  RESULTS")
    print(f"{separator}")
    print(f"  Average Heart Rate : {mean_bpm:.1f} BPM")
    print(f"  RMSSD (HRV)        : {rmssd:.1f} ms")
    print(f"  Peaks detected     : {len(peaks)}")
    print(f"  Valid frames       : {len(df_valid)} / {len(df)}")
    print(f"  Effective Fs       : {fs:.2f} Hz")
    print(f"{separator}\n")

    # Interpret RMSSD.
    if not np.isnan(rmssd):
        if rmssd < 20:
            interpretation = "Low HRV — elevated sympathetic drive / stress"
        elif rmssd <= 50:
            interpretation = "Normal HRV range"
        else:
            interpretation = "High HRV — strong vagal tone"
        print(f"  HRV Interpretation : {interpretation}")
        print()

    # ── 8. Visualization ─────────────────────────────────────────────
    print("  Generating research plot...")
    generate_research_plot(
        timestamps=timestamps,
        raw_green=raw_green,
        detrended_filtered=filtered,
        peaks=peaks,
        inst_bpm=inst_bpm,
        mean_bpm=mean_bpm,
        rmssd=rmssd,
        fs=fs,
        output_path=output_png,
    )

    print("\n  ✓ Phase 2 complete.\n")


# ======================================================================
# Entry point
# ======================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with input/output paths.
    """
    parser = argparse.ArgumentParser(
        prog="SENTIO Phase 2",
        description="rPPG signal conditioning, pulse extraction, and HRV calculation.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "phase1_raw_vitals.csv"),
        help="Path to the Phase 1 CSV (default: phase1_raw_vitals.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MANUSCRIPT_DIR / "phase2_signal_analysis.png"),
        help="Output path for the research plot (default: phase2_signal_analysis.png).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phase2(
        input_csv=Path(args.input),
        output_png=Path(args.output),
    )
