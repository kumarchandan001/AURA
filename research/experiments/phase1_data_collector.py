"""
main.py — Project SENTIO Phase 1 Pipeline Orchestrator.

This script wires the four core modules into a real-time acquisition loop:

    CameraSensor  →  FaceTracker  →  SignalExtractor  →  TelemetryManager

Execution flow:
    1. Start the threaded camera sensor.
    2. Enter the main loop (runs on the main thread):
       a. Grab the latest frame + timestamp from the sensor.
       b. Run face detection, ROI isolation, and pose estimation.
       c. Extract spatial-mean RGB within the ROI masks.
       d. Feed everything into the telemetry manager (buffer + HUD).
    3. On quit ('q') or exception, gracefully export the CSV and release
       all resources.

Usage:
    python main.py
    python main.py --device 1 --fps 60 --width 1280 --height 720

Author : Project SENTIO Research Team
Version: 0.1.0 (Phase 1 — Raw Signal Acquisition)
"""

from __future__ import annotations
from app.core.telemetry_manager import TelemetryManager
from app.core.signal_extractor import SignalExtractor
from app.core.face_tracker import FaceTracker, TrackingResult
from app.core.camera_sensor import CameraSensor
import time

import argparse
import logging
import sys
from pathlib import Path

# ── Dynamic Data Paths ────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "manuscript"


def _configure_logging(verbose: bool = False) -> None:
    """Set up structured logging to both console and a rotating log file.

    Parameters
    ----------
    verbose : bool
        If ``True``, set the console handler to DEBUG; otherwise INFO.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — keeps a full debug trace for post-hoc analysis.
    file_handler = logging.FileHandler("sentio_debug.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the acquisition pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with camera and tracker configuration.
    """
    parser = argparse.ArgumentParser(
        prog="SENTIO Phase 1",
        description="High-fidelity rPPG raw signal acquisition pipeline.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Camera device index (default: 0)."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target capture FPS (default: 30)."
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Capture width (default: 640)."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Capture height (default: 480)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "phase1_raw_vitals.csv"),
        help="Output CSV path (default: phase1_raw_vitals.csv).",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=15.0,
        help="Pixel displacement to trigger motion flag (default: 15.0).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG-level console logs."
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the Phase 1 acquisition pipeline.

    This is the main control loop.  It coordinates all four subsystems and
    handles graceful shutdown on user quit or unexpected exceptions.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    logger = logging.getLogger("sentio.pipeline")

    # --- Instantiate subsystems ------------------------------------------
    sensor = CameraSensor(
        device_index=args.device,
        target_fps=args.fps,
        resolution=(args.width, args.height),
    )
    tracker = FaceTracker(motion_threshold_px=args.motion_threshold)
    extractor = SignalExtractor()
    telemetry = TelemetryManager(output_path=args.output)

    logger.info("=" * 60)
    logger.info("  PROJECT SENTIO — Phase 1 Acquisition Pipeline")
    logger.info("  Camera device : %d", args.device)
    logger.info("  Target FPS    : %d", args.fps)
    logger.info("  Resolution    : %d × %d", args.width, args.height)
    logger.info("  Output CSV    : %s", args.output)
    logger.info("=" * 60)

    try:
        sensor.start()
        logger.info("Camera sensor started — entering acquisition loop.")

        # Allow the camera to warm up and auto-expose.
        time.sleep(0.5)

        while sensor.is_running:
            # 1. Grab latest frame from the threaded sensor.
            frame, timestamp = sensor.read()
            if frame is None:
                # Sensor hasn't produced a frame yet; yield CPU.
                time.sleep(0.001)
                continue

            # 2. Face detection + ROI masks + head pose + motion flag.
            try:
                result: TrackingResult = tracker.process(frame)
            except Exception as exc:
                logger.warning("FaceTracker error (non-fatal): %s", exc)
                result = TrackingResult(face_detected=False)

            # 3. Extract spatial-mean RGB from the ROI masks.
            if result.face_detected:
                mean_r, mean_g, mean_b = extractor.extract(
                    frame,
                    result.forehead_mask,
                    result.left_cheek_mask,
                    result.right_cheek_mask,
                )
                annotated = result.annotated_frame
            else:
                mean_r = mean_g = mean_b = float("nan")
                annotated = None

            # 4. Buffer telemetry + update HUD state.
            telemetry.update(
                annotated_frame=annotated,
                timestamp=timestamp,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                pitch=result.pitch,
                yaw=result.yaw,
                roll=result.roll,
                motion_flag=result.motion_flag,
                face_detected=result.face_detected,
            )

            # 5. Render the live HUD window; returns False on 'q'.
            if not telemetry.render(raw_frame=frame):
                logger.info("User initiated graceful shutdown.")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    except Exception as exc:
        logger.exception("Unhandled exception in pipeline: %s", exc)
    finally:
        # --- Graceful teardown -------------------------------------------
        logger.info("Shutting down subsystems...")
        sensor.stop()
        tracker.release()

        try:
            export_path = telemetry.export()
            logger.info("CSV exported to: %s", export_path)
        except ValueError:
            logger.warning("No data captured — CSV not exported.")

        telemetry.cleanup()
        logger.info("All resources released. SENTIO Phase 1 complete.")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    cli_args = _parse_args()
    _configure_logging(verbose=cli_args.verbose)
    run_pipeline(cli_args)
