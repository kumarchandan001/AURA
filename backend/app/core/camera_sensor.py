"""
camera_sensor.py — Threaded, Zero-Drift Video Capture for rPPG Pipelines.

This module decouples frame acquisition from the main processing loop by
running ``cv2.VideoCapture.read()`` on a dedicated daemon thread.  The design
guarantees that:

    1. The grab-rate matches the camera's native FPS (no main-loop stalls).
    2. The consumer always receives the *most recent* frame (no stale queue).
    3. High-resolution ``time.perf_counter()`` timestamps are paired with every
       frame at capture-time, not consumption-time — critical for rPPG phase
       coherence.

Thread safety is ensured via a ``threading.Lock`` around the shared
(frame, timestamp) pair.

Classes:
    CameraSensor — Encapsulates threaded webcam capture with monotonic timing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraSensor:
    """High-fidelity, threaded webcam capture optimised for rPPG signal integrity.

    The sensor runs ``cv2.VideoCapture.read()`` in a tight background loop,
    storing only the latest frame and its monotonic timestamp.  Consumers call
    :meth:`read` from any thread to atomically retrieve the pair without
    blocking the capture cadence.

    Parameters
    ----------
    device_index : int, optional
        OpenCV device ordinal (default ``0`` — primary webcam).
    target_fps : int, optional
        Requested capture FPS.  The camera driver may clamp this to its own
        hardware ceiling; the *actual* FPS is measured and exposed for
        telemetry.
    resolution : tuple[int, int], optional
        Requested (width, height).  Defaults to 640×480.

    Attributes
    ----------
    is_running : bool
        ``True`` while the capture thread is alive.

    Examples
    --------
    >>> sensor = CameraSensor(device_index=0, target_fps=30)
    >>> sensor.start()
    >>> frame, ts = sensor.read()
    >>> sensor.stop()
    """

    def __init__(
        self,
        device_index: int = 0,
        target_fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
    ) -> None:
        self._device_index: int = device_index
        self._target_fps: int = target_fps
        self._resolution: Tuple[int, int] = resolution

        # Shared state — guarded by ``_lock``.
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock: threading.Lock = threading.Lock()

        # Internal bookkeeping.
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self.is_running: bool = False

        logger.info(
            "CameraSensor initialised — device=%d  target_fps=%d  resolution=%s",
            device_index,
            target_fps,
            resolution,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and launch the background capture thread.

        Raises
        ------
        RuntimeError
            If the camera device cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {self._device_index}. "
                "Ensure the webcam is connected and not in use by another process."
            )

        # Request capture parameters — the driver may override.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        # Minimise internal driver buffering to reduce latency.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened — actual resolution=%dx%d  reported_fps=%.1f",
            actual_w,
            actual_h,
            actual_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="CameraSensor-Thread", daemon=True
        )
        self._thread.start()
        self.is_running = True

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return the most recent ``(frame, timestamp)`` pair.

        Returns
        -------
        frame : numpy.ndarray or None
            The BGR frame, or ``None`` if no frame has been captured yet.
        timestamp : float
            Monotonic ``time.perf_counter()`` value recorded at capture-time.
        """
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._timestamp

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
        self.is_running = False
        logger.info("CameraSensor stopped and camera released.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Tight acquisition loop running on the daemon thread.

        Continuously grabs frames and pairs each with a monotonic
        ``perf_counter`` timestamp.  On read failure the loop logs a
        warning and retries; it exits only when the stop event is set.
        """
        consecutive_failures: int = 0
        max_failures: int = 30  # ~1 s at 30 FPS before giving up.

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                logger.error("Camera unexpectedly closed — exiting capture loop.")
                break

            ret, frame = self._cap.read()
            ts = time.perf_counter()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error(
                        "Exceeded %d consecutive capture failures — aborting.",
                        max_failures,
                    )
                    break
                continue

            consecutive_failures = 0

            with self._lock:
                self._frame = frame
                self._timestamp = ts

        self.is_running = False
        logger.debug("Capture loop terminated.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraSensor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
