"""
Project SENTIO — Signal Extraction for Non-contact Telemetric Intelligent Observation.

A high-fidelity, contactless physiological monitoring framework built on
advanced Remote Photoplethysmography (rPPG).

Modules:
    camera_sensor       — Threaded, jitter-free video capture.
    face_tracker         — MediaPipe Face Mesh ROI isolation & head-pose estimation.
    signal_extractor     — Per-frame multi-channel (RGB) spatial averaging.
    telemetry_manager    — Live HUD overlay, in-memory buffering, and CSV export.
    affective_state      — Rolling-window physiological state classifier.
    realtime_vitals      — Sliding-window real-time BPM & HRV extraction.
    ollama_client        — Async local LLM client with dynamic system prompting.
"""

__version__ = "0.4.0"
__author__ = "Project SENTIO Research Team"
