"""
affective_state.py — Physiological State Manager for Affective Computing.

Monitors a rolling window of Heart Rate (BPM) and Heart Rate Variability
(RMSSD) to classify the user's cognitive/stress state in real-time.

State Machine:
    CALM     → RMSSD ≥ 30 ms AND BPM ≤ 85  (parasympathetic dominance)
    STRESSED → RMSSD < 30 ms OR  BPM > 85   (sympathetic activation)
    UNKNOWN  → Insufficient data or face lost

The thresholds are configurable and use hysteresis (separate thresholds for
entering vs. leaving a state) to prevent rapid oscillation at boundary values.

Classes:
    PhysiologicalState — Enum of possible affective states.
    AffectiveState     — Rolling-window state manager with hysteresis.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)


class PhysiologicalState(enum.Enum):
    """Discrete affective states derived from physiological signals."""

    UNKNOWN = "Unknown"
    CALM = "Calm"
    STRESSED = "Stressed"


@dataclass
class VitalSnapshot:
    """A single timestamped vital-signs reading.

    Attributes
    ----------
    timestamp : float
        ``perf_counter`` value when the reading was taken.
    bpm : float
        Instantaneous heart rate in beats per minute.
    rmssd : float
        RMSSD (ms) computed over the most recent inter-beat intervals.
    """

    timestamp: float
    bpm: float
    rmssd: float


class AffectiveState:
    """Real-time physiological state classifier with rolling-window smoothing.

    Maintains a fixed-duration sliding window of vital-signs snapshots and
    uses time-averaged BPM and RMSSD to classify the user's affective state.

    Hysteresis prevents rapid toggling:
        - To *enter* STRESSED: BPM must exceed ``bpm_stress_enter`` OR
          RMSSD must drop below ``rmssd_stress_enter``.
        - To *leave* STRESSED: BPM must fall below ``bpm_stress_leave`` AND
          RMSSD must rise above ``rmssd_stress_leave``.

    Parameters
    ----------
    window_seconds : float
        Duration of the rolling analysis window (default: 30 s).
    bpm_stress_enter : float
        BPM threshold to trigger STRESSED state (default: 85).
    bpm_stress_leave : float
        BPM threshold to exit STRESSED state (default: 78).
    rmssd_stress_enter : float
        RMSSD threshold (ms) to trigger STRESSED (default: 30).
    rmssd_stress_leave : float
        RMSSD threshold (ms) to exit STRESSED (default: 38).

    Attributes
    ----------
    current_state : PhysiologicalState
        The current classified affective state.
    avg_bpm : float
        Rolling-window average BPM.
    avg_rmssd : float
        Rolling-window average RMSSD.

    Examples
    --------
    >>> state_mgr = AffectiveState(window_seconds=30)
    >>> state_mgr.update(bpm=92.0, rmssd=22.0)
    >>> state_mgr.current_state
    PhysiologicalState.STRESSED
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        bpm_stress_enter: float = 85.0,
        bpm_stress_leave: float = 78.0,
        rmssd_stress_enter: float = 30.0,
        rmssd_stress_leave: float = 38.0,
    ) -> None:
        self._window_seconds = window_seconds
        self._bpm_enter = bpm_stress_enter
        self._bpm_leave = bpm_stress_leave
        self._rmssd_enter = rmssd_stress_enter
        self._rmssd_leave = rmssd_stress_leave

        self._buffer: Deque[VitalSnapshot] = deque()
        self._lock = threading.Lock()

        self.current_state: PhysiologicalState = PhysiologicalState.UNKNOWN
        self.avg_bpm: float = float("nan")
        self.avg_rmssd: float = float("nan")
        self._state_since: float = time.perf_counter()

        logger.info(
            "AffectiveState initialised — window=%.0fs  "
            "bpm_thresholds=(enter=%.0f, leave=%.0f)  "
            "rmssd_thresholds=(enter=%.0f, leave=%.0f)",
            window_seconds,
            bpm_stress_enter,
            bpm_stress_leave,
            rmssd_stress_enter,
            rmssd_stress_leave,
        )

    def update(self, bpm: float, rmssd: float) -> PhysiologicalState:
        """Ingest a new vital-signs reading and re-classify the state.

        Parameters
        ----------
        bpm : float
            Latest instantaneous BPM (may be NaN if unavailable).
        rmssd : float
            Latest RMSSD in ms (may be NaN if unavailable).

        Returns
        -------
        PhysiologicalState
            The (possibly updated) affective state.
        """
        now = time.perf_counter()

        with self._lock:
            # Append new reading.
            if not (np.isnan(bpm) or np.isnan(rmssd)):
                self._buffer.append(VitalSnapshot(timestamp=now, bpm=bpm, rmssd=rmssd))

            # Evict stale readings outside the window.
            cutoff = now - self._window_seconds
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()

            # Need at least 3 readings for stability.
            if len(self._buffer) < 3:
                self.current_state = PhysiologicalState.UNKNOWN
                self.avg_bpm = float("nan")
                self.avg_rmssd = float("nan")
                return self.current_state

            # Compute rolling averages.
            bpms = np.array([s.bpm for s in self._buffer])
            rmssds = np.array([s.rmssd for s in self._buffer])
            self.avg_bpm = float(np.mean(bpms))
            self.avg_rmssd = float(np.mean(rmssds))

            # Classify with hysteresis.
            prev_state = self.current_state

            if prev_state != PhysiologicalState.STRESSED:
                # Check entry conditions.
                if self.avg_bpm > self._bpm_enter or self.avg_rmssd < self._rmssd_enter:
                    self.current_state = PhysiologicalState.STRESSED
            else:
                # Check exit conditions (both must be met to leave STRESSED).
                if self.avg_bpm < self._bpm_leave and self.avg_rmssd > self._rmssd_leave:
                    self.current_state = PhysiologicalState.CALM

            if prev_state == PhysiologicalState.UNKNOWN and self.current_state == PhysiologicalState.UNKNOWN:
                # First real classification.
                if self.avg_bpm <= self._bpm_enter and self.avg_rmssd >= self._rmssd_enter:
                    self.current_state = PhysiologicalState.CALM

            if self.current_state != prev_state:
                self._state_since = now
                logger.info(
                    "State transition: %s → %s  (avg_bpm=%.1f, avg_rmssd=%.1f)",
                    prev_state.value,
                    self.current_state.value,
                    self.avg_bpm,
                    self.avg_rmssd,
                )

        return self.current_state

    def get_state_duration(self) -> float:
        """Return how long the current state has been active (seconds)."""
        return time.perf_counter() - self._state_since

    def get_summary(self) -> dict:
        """Return a snapshot of the current affective state for the UI.

        Returns
        -------
        dict
            Keys: ``state``, ``avg_bpm``, ``avg_rmssd``, ``buffer_size``,
            ``state_duration_s``.
        """
        with self._lock:
            return {
                "state": self.current_state.value,
                "avg_bpm": round(self.avg_bpm, 1) if not np.isnan(self.avg_bpm) else None,
                "avg_rmssd": round(self.avg_rmssd, 1) if not np.isnan(self.avg_rmssd) else None,
                "buffer_size": len(self._buffer),
                "state_duration_s": round(self.get_state_duration(), 1),
            }
