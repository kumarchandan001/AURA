"""
telemetry_manager.py — Live HUD, In-Memory Buffering & CSV Persistence.

Responsibilities:

    1. **Live HUD** — Render an OpenCV window showing the annotated camera
       feed with real-time overlays for FPS, frame counter, motion status,
       head pose, and RGB signal strength.
    2. **In-memory buffer** — Accumulate per-frame telemetry rows in a
       pre-allocated list (converted to a Pandas DataFrame on export).
    3. **CSV export** — On graceful shutdown, flush the buffer to
       ``phase1_raw_vitals.csv`` with the canonical column schema.

Column schema (immutable contract for Phase 2):
    ``timestamp, frame_id, actual_fps, mean_r, mean_g, mean_b,
    head_pitch, head_yaw, head_roll, motion_flag``

Classes:
    TelemetryManager — Orchestrates HUD rendering, buffering, and export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# HUD styling constants.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1
_TEXT_COLOUR = (0, 255, 200)       # Bright cyan-green.
_WARN_COLOUR = (0, 0, 255)        # Red for warnings.
_BG_COLOUR = (20, 20, 20)         # Near-black overlay strip.
_HUD_WINDOW = "SENTIO — Phase 1 Acquisition"


class TelemetryManager:
    """Live telemetry HUD, data buffering, and CSV persistence.

    Parameters
    ----------
    output_path : str or Path
        Destination for the CSV export (default: ``phase1_raw_vitals.csv``).
    fps_window : int
        Number of recent frames used to compute a rolling FPS estimate.

    Attributes
    ----------
    frame_id : int
        Monotonically increasing frame counter.

    Examples
    --------
    >>> tm = TelemetryManager()
    >>> tm.update(frame, timestamp, mean_rgb, pose, motion_flag)
    >>> tm.render()
    >>> tm.export()
    """

    # Canonical column order — this is the Phase 2 contract.
    COLUMNS: list[str] = [
        "timestamp",
        "frame_id",
        "actual_fps",
        "mean_r",
        "mean_g",
        "mean_b",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "motion_flag",
    ]

    def __init__(
        self,
        output_path: str | Path = str(Path(__file__).resolve().parent.parent.parent.parent /
                                      "research" / "data" / "phase1_raw_vitals.csv"),
        fps_window: int = 30,
    ) -> None:
        self._output_path: Path = Path(output_path)
        self._fps_window: int = fps_window

        # Data buffer — list-of-dicts is fastest for row-wise appends.
        self._buffer: List[Dict] = []

        # FPS estimation state.
        self._ts_history: List[float] = []
        self._current_fps: float = 0.0

        # Frame counter.
        self.frame_id: int = 0

        # Latest display frame (for render).
        self._display_frame: Optional[np.ndarray] = None
        self._latest_row: Optional[Dict] = None

        logger.info(
            "TelemetryManager initialised — output=%s  fps_window=%d",
            self._output_path,
            fps_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        annotated_frame: Optional[np.ndarray],
        timestamp: float,
        mean_r: float,
        mean_g: float,
        mean_b: float,
        pitch: float,
        yaw: float,
        roll: float,
        motion_flag: bool,
        face_detected: bool,
    ) -> None:
        """Ingest a single frame's telemetry, buffer it, and prepare the HUD.

        Parameters
        ----------
        annotated_frame : numpy.ndarray or None
            Frame with ROI overlays drawn (from FaceTracker).
        timestamp : float
            Monotonic ``perf_counter`` timestamp from the CameraSensor.
        mean_r, mean_g, mean_b : float
            Spatial-mean colour channels from the SignalExtractor.
        pitch, yaw, roll : float
            Head-pose Euler angles in degrees.
        motion_flag : bool
            Whether inter-frame motion exceeds the artefact threshold.
        face_detected : bool
            Whether a face was found in this frame.
        """
        self.frame_id += 1

        # --- Rolling FPS ------------------------------------------------
        self._ts_history.append(timestamp)
        if len(self._ts_history) > self._fps_window:
            self._ts_history = self._ts_history[-self._fps_window:]
        if len(self._ts_history) >= 2:
            dt = self._ts_history[-1] - self._ts_history[0]
            if dt > 0:
                self._current_fps = (len(self._ts_history) - 1) / dt

        # --- Buffer row --------------------------------------------------
        row = {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "actual_fps": round(self._current_fps, 2),
            "mean_r": round(mean_r, 4) if not np.isnan(mean_r) else float("nan"),
            "mean_g": round(mean_g, 4) if not np.isnan(mean_g) else float("nan"),
            "mean_b": round(mean_b, 4) if not np.isnan(mean_b) else float("nan"),
            "head_pitch": round(pitch, 2) if not np.isnan(pitch) else float("nan"),
            "head_yaw": round(yaw, 2) if not np.isnan(yaw) else float("nan"),
            "head_roll": round(roll, 2) if not np.isnan(roll) else float("nan"),
            "motion_flag": motion_flag,
        }
        self._buffer.append(row)
        self._latest_row = row

        # --- Prepare display frame ---------------------------------------
        if annotated_frame is not None:
            self._display_frame = annotated_frame
        else:
            # If no annotated frame, we may still have a raw frame from sensor
            self._display_frame = None

    def render(self, raw_frame: Optional[np.ndarray] = None) -> bool:
        """Draw the HUD overlay and display the window.

        Parameters
        ----------
        raw_frame : numpy.ndarray or None
            Fallback frame to display when no annotated frame is available.

        Returns
        -------
        bool
            ``False`` if the user pressed 'q' (quit signal); ``True`` otherwise.
        """
        frame = self._display_frame if self._display_frame is not None else raw_frame
        if frame is None:
            return True  # Nothing to show yet; keep running.

        display = frame.copy()
        h, w = display.shape[:2]

        # --- HUD background strip ----------------------------------------
        cv2.rectangle(display, (0, 0), (w, 110), _BG_COLOUR, -1)
        cv2.addWeighted(display, 0.7, frame, 0.3, 0, display)
        # Redraw the strip solidly at reduced opacity for readability.
        overlay_strip = display.copy()
        cv2.rectangle(overlay_strip, (0, 0), (w, 110), _BG_COLOUR, -1)
        cv2.addWeighted(overlay_strip, 0.6, display, 0.4, 0, display)

        row = self._latest_row or {}

        # --- Line 1: Project banner + FPS --------------------------------
        fps_text = f"SENTIO v0.1  |  FPS: {self._current_fps:.1f}  |  Frame: {self.frame_id}"
        cv2.putText(display, fps_text, (10, 22), _FONT, _FONT_SCALE, _TEXT_COLOUR, _FONT_THICK, cv2.LINE_AA)

        # --- Line 2: RGB signal ------------------------------------------
        r_val = row.get("mean_r", float("nan"))
        g_val = row.get("mean_g", float("nan"))
        b_val = row.get("mean_b", float("nan"))
        rgb_text = f"R: {r_val:.2f}  G: {g_val:.2f}  B: {b_val:.2f}"
        cv2.putText(display, rgb_text, (10, 46), _FONT, _FONT_SCALE, (180, 200, 255), _FONT_THICK, cv2.LINE_AA)

        # --- Line 3: Head pose -------------------------------------------
        p = row.get("head_pitch", float("nan"))
        y = row.get("head_yaw", float("nan"))
        r = row.get("head_roll", float("nan"))
        pose_text = f"Pitch: {p:.1f}  Yaw: {y:.1f}  Roll: {r:.1f}"
        cv2.putText(display, pose_text, (10, 70), _FONT, _FONT_SCALE, _TEXT_COLOUR, _FONT_THICK, cv2.LINE_AA)

        # --- Line 4: Motion status ---------------------------------------
        motion = row.get("motion_flag", False)
        if motion:
            status_text = "MOTION ARTIFACT DETECTED"
            colour = _WARN_COLOUR
        else:
            status_text = "Signal: STABLE"
            colour = (0, 255, 0)
        cv2.putText(display, status_text, (10, 94), _FONT, _FONT_SCALE, colour, _FONT_THICK, cv2.LINE_AA)

        # --- Instructions at bottom --------------------------------------
        cv2.putText(
            display,
            "Press 'q' to stop acquisition and export CSV",
            (10, h - 12),
            _FONT,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(_HUD_WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("Quit signal received from user.")
            return False
        return True

    def export(self) -> Path:
        """Flush the in-memory buffer to a CSV file.

        Returns
        -------
        Path
            Absolute path to the exported CSV.

        Raises
        ------
        ValueError
            If the buffer is empty (no frames were captured).
        """
        if not self._buffer:
            logger.warning("Buffer is empty — nothing to export.")
            raise ValueError("No telemetry data to export.")

        df = pd.DataFrame(self._buffer, columns=self.COLUMNS)
        df.to_csv(self._output_path, index=False)

        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        logger.info(
            "Exported %d frames (%.1f s) → %s",
            len(df),
            duration,
            self._output_path.resolve(),
        )
        print(f"\n{'='*60}")
        print("  SENTIO Phase 1 — Export Complete")
        print(f"  Frames captured : {len(df)}")
        print(f"  Duration        : {duration:.2f} s")
        print(f"  Mean FPS        : {df['actual_fps'].mean():.1f}")
        print(f"  Motion flags    : {df['motion_flag'].sum()}")
        print(f"  Output file     : {self._output_path.resolve()}")
        print(f"{'='*60}\n")

        return self._output_path.resolve()

    def cleanup(self) -> None:
        """Destroy any OpenCV windows."""
        cv2.destroyAllWindows()
        logger.info("TelemetryManager windows destroyed.")
