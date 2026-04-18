"""
face_tracker.py — MediaPipe Face Landmarker ROI Isolation & Head-Pose Estimation.

This module wraps the MediaPipe **Face Landmarker Task API** (the modern
replacement for the deprecated ``mp.solutions.face_mesh``) to provide:

    1. **ROI masks** for two anatomically-optimal rPPG regions:
       - *Forehead*  — high capillary-bed density, minimal hair occlusion.
       - *Upper cheeks* (left + right) — strong pulsatile signal with low
         motion artefact relative to the jaw.
    2. **6-DoF head pose** (pitch, yaw, roll) derived from a Perspective-n-Point
       solve on canonical face landmarks.
    3. **Motion artefact flag** — a boolean that fires when the face bounding
       box displacement between consecutive frames exceeds a configurable
       pixel threshold.

All outputs are returned via the :class:`TrackingResult` dataclass so that
downstream consumers get a clean, typed interface.

Classes:
    TrackingResult — Immutable container for per-frame tracking data.
    FaceTracker    — Stateful tracker wrapping MediaPipe Face Landmarker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model path — bundled alongside this module.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent / "face_landmarker.task"
)

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices for ROI polygons.
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/
#            modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Forehead — a wide band above the eyebrows.
_FOREHEAD_INDICES: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]
# Refine: use a tighter forehead strip between eyebrow ridge and hairline.
_FOREHEAD_STRIP_INDICES: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356,
    # lower bound — just above brow ridge
    70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300,
]

# Left cheek — zygomatic region below the left eye.
_LEFT_CHEEK_INDICES: list[int] = [
    116, 117, 118, 119, 120, 121, 126, 142,
    36, 205, 206, 207, 187, 123, 116,
]

# Right cheek — mirror of the left.
_RIGHT_CHEEK_INDICES: list[int] = [
    345, 346, 347, 348, 349, 350, 355, 371,
    266, 425, 426, 427, 411, 352, 345,
]

# Canonical 3-D model points for PnP head-pose estimation (in mm).
# Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth.
_MODEL_POINTS_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye left corner
        [225.0, 170.0, -135.0],   # Right eye right corner
        [-150.0, -150.0, -125.0],  # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ],
    dtype=np.float64,
)

# Corresponding Face Mesh landmark indices.
_POSE_LANDMARK_IDS: list[int] = [1, 152, 33, 263, 61, 291]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackingResult:
    """Immutable per-frame output from :class:`FaceTracker`.

    Attributes
    ----------
    face_detected : bool
        Whether a face was found in the frame.
    forehead_mask : numpy.ndarray or None
        Binary mask (uint8) isolating the forehead ROI.
    left_cheek_mask : numpy.ndarray or None
        Binary mask isolating the left upper-cheek ROI.
    right_cheek_mask : numpy.ndarray or None
        Binary mask isolating the right upper-cheek ROI.
    pitch : float
        Head pitch in degrees (positive = looking up).
    yaw : float
        Head yaw in degrees (positive = turning right).
    roll : float
        Head roll in degrees (positive = tilting right).
    motion_flag : bool
        ``True`` when inter-frame face-box displacement exceeds threshold.
    annotated_frame : numpy.ndarray or None
        Copy of input frame with ROI contours drawn (for telemetry HUD).
    """

    face_detected: bool = False
    forehead_mask: Optional[np.ndarray] = field(default=None, repr=False)
    left_cheek_mask: Optional[np.ndarray] = field(default=None, repr=False)
    right_cheek_mask: Optional[np.ndarray] = field(default=None, repr=False)
    pitch: float = float("nan")
    yaw: float = float("nan")
    roll: float = float("nan")
    motion_flag: bool = False
    annotated_frame: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """Stateful face tracker producing ROI masks, head pose, and motion flags.

    Uses the modern MediaPipe **Face Landmarker Task API** in VIDEO running
    mode, which replaces the deprecated ``mp.solutions.face_mesh``.

    Parameters
    ----------
    min_detection_confidence : float
        MediaPipe detection confidence threshold (0–1).
    min_tracking_confidence : float
        MediaPipe tracking confidence threshold (0–1).
    motion_threshold_px : float
        If the face bounding-box centre moves more than this many pixels
        between consecutive frames, ``motion_flag`` is set.
    model_path : str or None
        Path to the ``face_landmarker.task`` model file.  If ``None``,
        defaults to the model bundled alongside this module.

    Examples
    --------
    >>> tracker = FaceTracker()
    >>> result = tracker.process(frame)
    >>> if result.face_detected:
    ...     roi = cv2.bitwise_and(frame, frame, mask=result.forehead_mask)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        motion_threshold_px: float = 15.0,
        model_path: Optional[str] = None,
    ) -> None:
        _model = model_path or _DEFAULT_MODEL_PATH

        if not Path(_model).exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {_model}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=_model)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Monotonically increasing timestamp for VIDEO mode.
        self._frame_timestamp_ms: int = 0

        self._motion_threshold: float = motion_threshold_px
        self._prev_centre: Optional[Tuple[float, float]] = None

        logger.info(
            "FaceTracker initialised (Task API) — det_conf=%.2f  trk_conf=%.2f  "
            "motion_thresh=%.1f px  model=%s",
            min_detection_confidence,
            min_tracking_confidence,
            motion_threshold_px,
            Path(_model).name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> TrackingResult:
        """Run face detection, ROI isolation, pose estimation, and motion check.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image from the camera sensor.

        Returns
        -------
        TrackingResult
            Fully populated result; fields are NaN / None when no face found.
        """
        h, w, _ = frame.shape

        # Convert BGR → RGB for MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Advance the monotonic timestamp.
        self._frame_timestamp_ms += 33  # ~30 FPS
        results = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not results.face_landmarks:
            logger.debug("No face detected in frame.")
            self._prev_centre = None
            return TrackingResult(face_detected=False)

        # Extract the first face's landmarks as pixel coordinates.
        face_lms = results.face_landmarks[0]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in face_lms], dtype=np.float64
        )

        # --- ROI masks ---------------------------------------------------
        forehead_mask = self._polygon_mask(pts, _FOREHEAD_STRIP_INDICES, h, w)
        left_cheek_mask = self._polygon_mask(pts, _LEFT_CHEEK_INDICES, h, w)
        right_cheek_mask = self._polygon_mask(pts, _RIGHT_CHEEK_INDICES, h, w)

        # --- Head pose ---------------------------------------------------
        pitch, yaw, roll = self._estimate_head_pose(pts, h, w)

        # --- Motion flag -------------------------------------------------
        motion_flag = self._check_motion(pts, w, h)

        # --- Annotated frame for HUD ------------------------------------
        annotated = self._draw_rois(frame.copy(), pts)

        return TrackingResult(
            face_detected=True,
            forehead_mask=forehead_mask,
            left_cheek_mask=left_cheek_mask,
            right_cheek_mask=right_cheek_mask,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            motion_flag=motion_flag,
            annotated_frame=annotated,
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("FaceTracker resources released.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_mask(
        pts: np.ndarray, indices: list[int], h: int, w: int
    ) -> np.ndarray:
        """Create a filled binary mask from landmark indices.

        Parameters
        ----------
        pts : numpy.ndarray
            (N, 2) array of all face landmark pixel coordinates.
        indices : list[int]
            Landmark indices forming the ROI polygon.
        h, w : int
            Frame dimensions.

        Returns
        -------
        numpy.ndarray
            Single-channel uint8 mask with 255 inside the polygon.
        """
        polygon = pts[indices].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    @staticmethod
    def _estimate_head_pose(
        pts: np.ndarray, h: int, w: int
    ) -> Tuple[float, float, float]:
        """Solve head pose via Perspective-n-Point.

        Parameters
        ----------
        pts : numpy.ndarray
            (N, 2) face landmark pixel coordinates.
        h, w : int
            Frame dimensions.

        Returns
        -------
        tuple[float, float, float]
            (pitch, yaw, roll) in degrees.
        """
        image_points = pts[_POSE_LANDMARK_IDS].astype(np.float64)

        # Approximate camera intrinsics (no calibration available).
        focal_length = w
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            logger.warning("PnP solve failed; returning NaN pose.")
            return (float("nan"), float("nan"), float("nan"))

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix → Euler angles (XYZ convention).
        pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
        yaw = float(
            np.degrees(np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2)))
        )
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))

        return pitch, yaw, roll

    def _check_motion(self, pts: np.ndarray, w: int, h: int) -> bool:
        """Detect sudden face displacement between frames.

        Computes the Euclidean distance between the current and previous
        bounding-box centres.  If it exceeds ``_motion_threshold``, the
        frame is flagged as containing a motion artefact.

        Parameters
        ----------
        pts : numpy.ndarray
            (N, 2) face landmark pixel coordinates.
        w, h : int
            Frame dimensions (used for normalisation reference only).

        Returns
        -------
        bool
            ``True`` if displacement exceeds the threshold.
        """
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        motion_flag = False
        if self._prev_centre is not None:
            dx = cx - self._prev_centre[0]
            dy = cy - self._prev_centre[1]
            displacement = np.sqrt(dx ** 2 + dy ** 2)
            if displacement > self._motion_threshold:
                motion_flag = True
                logger.debug("Motion artefact flagged — displacement=%.1f px", displacement)

        self._prev_centre = (cx, cy)
        return motion_flag

    @staticmethod
    def _draw_rois(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Draw semi-transparent ROI overlays on the frame for the HUD.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR frame to annotate (will be modified in-place).
        pts : numpy.ndarray
            (N, 2) face landmark pixel coordinates.

        Returns
        -------
        numpy.ndarray
            Annotated frame.
        """
        overlay = frame.copy()
        alpha = 0.25

        # Forehead — cyan overlay.
        forehead_poly = pts[_FOREHEAD_STRIP_INDICES].astype(np.int32)
        cv2.fillPoly(overlay, [forehead_poly], (255, 255, 0))  # Cyan in BGR.

        # Left cheek — green overlay.
        left_poly = pts[_LEFT_CHEEK_INDICES].astype(np.int32)
        cv2.fillPoly(overlay, [left_poly], (0, 255, 128))

        # Right cheek — green overlay.
        right_poly = pts[_RIGHT_CHEEK_INDICES].astype(np.int32)
        cv2.fillPoly(overlay, [right_poly], (0, 255, 128))

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw contours for clarity.
        cv2.polylines(frame, [forehead_poly], True, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [left_poly], True, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.polylines(frame, [right_poly], True, (0, 255, 128), 1, cv2.LINE_AA)

        return frame
