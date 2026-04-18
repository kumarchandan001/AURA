"""
signal_extractor.py — Multi-Channel Spatial Signal Extraction.

Given a BGR frame and a set of binary ROI masks, this module computes the
spatial mean of each colour channel (Red, Green, Blue) *exclusively* within
the mask region.

Design rationale
----------------
rPPG algorithms recover the blood-volume pulse from subtle, sub-pixel colour
fluctuations.  Spatial averaging across an ROI dramatically improves the
signal-to-noise ratio by collapsing sensor read noise while preserving the
pulsatile component (which is spatially coherent across the capillary bed).

We intentionally **do not** apply any temporal filter here — the raw optical
signal must be preserved for downstream algorithmic development (Phase 2).

Classes:
    SignalExtractor — Stateless per-frame RGB mean calculator.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SignalExtractor:
    """Extract per-frame spatial-mean RGB values from masked ROIs.

    This class is stateless; each call to :meth:`extract` processes a single
    frame independently.  Temporal aggregation is the responsibility of
    :class:`~sentio.telemetry_manager.TelemetryManager`.

    Examples
    --------
    >>> extractor = SignalExtractor()
    >>> r, g, b = extractor.extract(frame, forehead_mask, l_cheek, r_cheek)
    """

    @staticmethod
    def extract(
        frame: np.ndarray,
        forehead_mask: Optional[np.ndarray] = None,
        left_cheek_mask: Optional[np.ndarray] = None,
        right_cheek_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Compute combined spatial-mean R, G, B across all provided ROI masks.

        The masks are OR-combined so that a single ``cv2.mean`` call yields
        the aggregate colour value over the entire skin region.  If all masks
        are ``None``, NaN is returned for every channel.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image.
        forehead_mask : numpy.ndarray or None
            Binary uint8 mask for the forehead ROI.
        left_cheek_mask : numpy.ndarray or None
            Binary uint8 mask for the left cheek ROI.
        right_cheek_mask : numpy.ndarray or None
            Binary uint8 mask for the right cheek ROI.

        Returns
        -------
        tuple[float, float, float]
            ``(mean_r, mean_g, mean_b)`` — spatial averages within the ROI.
            Returns ``(NaN, NaN, NaN)`` when no valid mask pixels exist.
        """
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for mask in (forehead_mask, left_cheek_mask, right_cheek_mask):
            if mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Guard: no ROI pixels → NaN.
        if cv2.countNonZero(combined_mask) == 0:
            logger.debug("No ROI pixels available — returning NaN.")
            return (float("nan"), float("nan"), float("nan"))

        # cv2.mean returns (B, G, R, A) for a BGR image.
        mean_bgr = cv2.mean(frame, mask=combined_mask)
        mean_b, mean_g, mean_r = mean_bgr[0], mean_bgr[1], mean_bgr[2]

        return (mean_r, mean_g, mean_b)
