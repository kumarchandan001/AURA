# Project SENTIO — Phase 1: Raw Signal Acquisition

**S**ignal **E**xtraction for **N**on-contact **T**elemetric **I**ntelligent **O**bservation

A high-fidelity, contactless physiological monitoring framework built on advanced **Remote Photoplethysmography (rPPG)**. This repository implements **Phase 1**: real-time raw optical signal acquisition from facial ROIs with sub-millisecond timestamping.

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  CameraSensor   │────▶│  FaceTracker  │────▶│ SignalExtractor  │────▶│ TelemetryManager  │
│ (threaded grab) │     │ (Face Mesh)   │     │ (RGB spatial μ)  │     │ (HUD + CSV)       │
└─────────────────┘     └──────────────┘     └──────────────────┘     └───────────────────┘
     background              main                  main                     main
      thread                thread                 thread                   thread
```

| Module | Responsibility |
|--------|---------------|
| `sentio/camera_sensor.py` | Lock-guarded threaded capture with `perf_counter` timestamps |
| `sentio/face_tracker.py` | MediaPipe Face Mesh → forehead & cheek ROI masks, PnP head pose, motion flag |
| `sentio/signal_extractor.py` | Spatial-mean R/G/B within combined ROI mask |
| `sentio/telemetry_manager.py` | Live OpenCV HUD, in-memory buffer, Pandas CSV export |
| `main.py` | CLI orchestrator wiring all subsystems |

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--fps` | `30` | Target capture FPS |
| `--width` | `640` | Capture width |
| `--height` | `480` | Capture height |
| `--output` | `phase1_raw_vitals.csv` | Output CSV path |
| `--motion-threshold` | `15.0` | Pixel displacement for motion flag |
| `-v / --verbose` | off | DEBUG-level console logs |

### 3. Stop & Export

Press **`q`** in the live HUD window. The pipeline will:
1. Stop the camera sensor thread
2. Release MediaPipe resources
3. Export all buffered data to `phase1_raw_vitals.csv`

---

## Output Schema

The CSV file contains **one row per captured frame** with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `float64` | Monotonic `perf_counter` value at capture-time |
| `frame_id` | `int` | Sequential frame number |
| `actual_fps` | `float64` | Rolling FPS estimate |
| `mean_r` | `float64` | Spatial mean of Red channel within ROI |
| `mean_g` | `float64` | Spatial mean of Green channel within ROI |
| `mean_b` | `float64` | Spatial mean of Blue channel within ROI |
| `head_pitch` | `float64` | Head pitch in degrees |
| `head_yaw` | `float64` | Head yaw in degrees |
| `head_roll` | `float64` | Head roll in degrees |
| `motion_flag` | `bool` | `True` if motion artefact detected |

> **Note:** When no face is detected, RGB and pose columns contain `NaN`.

---

## Design Decisions

- **No temporal filtering** — Raw optical signals are preserved for Phase 2 algorithmic R&D.
- **Threaded capture** — Decouples the camera driver's blocking I/O from the main processing loop, preventing frame-drop-induced aliasing.
- **`perf_counter` timestamps** — Monotonic, high-resolution, immune to NTP/system-clock jumps.
- **NaN on face loss** — Maintains temporal alignment of the signal vector even during occlusion.
- **CAP_DSHOW backend** — Selected for Windows to minimise driver latency versus MSMF.

---

## License

Internal research use only — Project SENTIO, 2026.
