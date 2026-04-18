"""
server.py — FastAPI Backend Bridge for Project SENTIO.

Exposes the SENTIO rPPG pipeline as low-latency HTTP/WebSocket endpoints
for consumption by the React frontend dashboard.

Endpoints:
    GET  /video_feed      — MJPEG multipart stream (live webcam + ROI mesh).
    WS   /ws/telemetry    — 10 Hz JSON telemetry (BPM, HRV, State, pulse data).
    POST /api/chat        — Async LLM chat with physiological context injection.
    GET  /api/health      — System health check.

Usage:
    uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

Author : Project SENTIO Research Team
Version: 1.0.0 (Web Migration)
"""

from __future__ import annotations
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.core.signal_extractor import SignalExtractor
from app.core.realtime_vitals import RealtimeVitals
from app.core.ollama_client import OllamaClient
from app.core.face_tracker import FaceTracker
from app.core.camera_sensor import CameraSensor
from app.core.affective_state import AffectiveState

import asyncio
import io
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Ensure SENTIO root is importable ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


logger = logging.getLogger(__name__)

# Force UTF-8 on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )


# ======================================================================
# Pydantic Models
# ======================================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: str


# ======================================================================
# SENTIO Engine (Singleton)
# ======================================================================

class SentioEngine:
    """Central processing engine that wraps all SENTIO Phase 1-4 modules.

    Runs face tracking, signal extraction, and vitals computation in a
    dedicated background thread.  Thread-safe accessors expose the latest
    frame, telemetry snapshot, and pulse waveform batch for the API layer.
    """

    def __init__(
        self,
        camera_device: int = 0,
        target_fps: int = 30,
        llm_model: str = "qwen2:0.5b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        # ── Core modules ─────────────────────────────────────────
        self._sensor = CameraSensor(
            device_index=camera_device, target_fps=target_fps
        )
        self._tracker = FaceTracker()
        self._extractor = SignalExtractor()
        self._vitals = RealtimeVitals(
            window_seconds=10, min_window_seconds=4
        )
        self._affect = AffectiveState(window_seconds=30)
        self._llm = OllamaClient(model=llm_model, base_url=ollama_url)

        # ── Thread control ───────────────────────────────────────
        self._stop_event = threading.Event()
        self._proc_thread: Optional[threading.Thread] = None

        # ── Shared display frame ─────────────────────────────────
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # ── FPS tracking ─────────────────────────────────────────
        self._fps_counter: int = 0
        self._fps_timer: float = time.perf_counter()
        self._current_fps: float = 0.0

        # ── Pulse waveform batch (for chart streaming) ───────────
        self._green_batch: List[Dict] = []
        self._green_batch_lock = threading.Lock()
        self._green_t0: Optional[float] = None

        # ── Status ───────────────────────────────────────────────
        self.camera_ok: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open camera, launch processing thread."""
        try:
            self._sensor.start()
            time.sleep(0.3)
            self.camera_ok = True
            logger.info("Camera started.")
        except RuntimeError as e:
            self.camera_ok = False
            logger.error("Camera failed to start: %s", e)
            return

        self._stop_event.clear()
        self._proc_thread = threading.Thread(
            target=self._processing_loop,
            name="SentioEngine-Proc",
            daemon=True,
        )
        self._proc_thread.start()
        logger.info("SentioEngine processing loop started.")

    def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        self._stop_event.set()
        if self._proc_thread is not None:
            self._proc_thread.join(timeout=3.0)
        self._sensor.stop()
        self._tracker.release()
        self._llm.shutdown()
        logger.info("SentioEngine stopped.")

    # ------------------------------------------------------------------
    # Processing Loop (background thread)
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """Continuous frame processing: track → extract → vitals."""
        last_vitals = 0.0

        while not self._stop_event.is_set():
            frame, timestamp = self._sensor.read()
            if frame is None:
                time.sleep(0.005)
                continue

            # Face tracking.
            try:
                result = self._tracker.process(frame)
            except Exception:
                result = None

            # Signal extraction.
            mean_g = float("nan")
            if result and result.face_detected:
                _, mean_g, _ = self._extractor.extract(
                    frame,
                    result.forehead_mask,
                    result.left_cheek_mask,
                    result.right_cheek_mask,
                )
                self._vitals.add_sample(mean_g, timestamp)
                display = (
                    result.annotated_frame
                    if result.annotated_frame is not None
                    else frame
                )
            else:
                display = frame

            # Store display frame.
            with self._frame_lock:
                self._frame = display

            # Accumulate green values for pulse chart.
            if not np.isnan(mean_g):
                if self._green_t0 is None:
                    self._green_t0 = timestamp
                with self._green_batch_lock:
                    self._green_batch.append(
                        {
                            "t": round(timestamp - self._green_t0, 4),
                            "v": round(float(mean_g), 2),
                        }
                    )
                    # Safety cap.
                    if len(self._green_batch) > 120:
                        self._green_batch = self._green_batch[-120:]

            # FPS tracking.
            self._fps_counter += 1
            now = time.perf_counter()
            dt = now - self._fps_timer
            if dt >= 1.0:
                self._current_fps = self._fps_counter / dt
                self._fps_counter = 0
                self._fps_timer = now

            # Periodically recompute vitals (every 500 ms).
            if now - last_vitals >= 0.5:
                last_vitals = now
                bpm, rmssd = self._vitals.compute()
                self._affect.update(bpm, rmssd)

            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Accessors (thread-safe)
    # ------------------------------------------------------------------

    def get_frame_jpeg(self) -> Optional[bytes]:
        """Return the latest display frame as JPEG bytes."""
        with self._frame_lock:
            if self._frame is None:
                return None
            success, buf = cv2.imencode(
                ".jpg",
                self._frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
        return buf.tobytes() if success else None

    def get_telemetry(self) -> Dict:
        """Drain the pulse batch and return a telemetry snapshot."""
        bpm = self._vitals.latest_bpm
        rmssd = self._vitals.latest_rmssd
        state = self._affect.current_state.value
        fps = self._current_fps

        # Drain the green-channel batch accumulated since last call.
        with self._green_batch_lock:
            pulse = list(self._green_batch)
            self._green_batch.clear()

        return {
            "bpm": round(bpm, 1) if not np.isnan(bpm) else None,
            "rmssd": round(rmssd, 1) if not np.isnan(rmssd) else None,
            "state": state,
            "fps": round(fps, 1),
            "pulse": pulse,
        }

    def chat_sync(self, message: str) -> Dict[str, str]:
        """Synchronous LLM chat (called via run_in_executor)."""
        state = self._affect.current_state.value
        holder = [None]
        done = threading.Event()

        def callback(text: str):
            holder[0] = text
            done.set()

        self._llm.chat_async(
            user_message=message, state=state, callback=callback
        )
        done.wait(timeout=120)

        return {
            "response": holder[0] or "No response received.",
            "state": state,
        }

    def get_health(self) -> Dict:
        """System health summary."""
        return {
            "status": "ok",
            "camera": self.camera_ok and self._sensor.is_running,
            "llm": self._llm.is_available,
            "fps": round(self._current_fps, 1),
        }


# ======================================================================
# Engine Singleton (lazy — created at startup, not at import time)
# ======================================================================

engine: Optional[SentioEngine] = None


def _get_engine() -> SentioEngine:
    """Return the active engine instance (raises if not started)."""
    if engine is None:
        raise RuntimeError("SentioEngine not initialised yet.")
    return engine


# ======================================================================
# FastAPI Application
# ======================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create and start the engine on startup; tear down on shutdown."""
    global engine
    engine = SentioEngine()
    engine.start()
    yield
    engine.stop()
    engine = None


app = FastAPI(
    title="SENTIO Backend",
    description="Real-time rPPG & Affective AI API bridge.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Endpoints
# ======================================================================

# ── MJPEG Video Stream ───────────────────────────────────────────────

@app.get("/video_feed")
async def video_feed():
    """Multipart MJPEG stream of the live webcam with ROI overlay."""

    async def _generate():
        e = _get_engine()
        while True:
            jpeg = e.get_frame_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30 FPS cap.

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket Telemetry ──────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    """Broadcast live BPM, HRV, State, and pulse waveform at ~10 Hz."""
    await websocket.accept()
    logger.info("Telemetry WebSocket client connected.")
    e = _get_engine()
    try:
        while True:
            data = e.get_telemetry()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 10 Hz.
    except WebSocketDisconnect:
        logger.info("Telemetry WebSocket client disconnected.")
    except Exception as ws_err:
        logger.warning("WebSocket error: %s", ws_err)


# ── AI Chat ──────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Context-aware LLM chat with physiological state injection."""
    e = _get_engine()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: e.chat_sync(request.message)
    )
    return ChatResponse(**result)


# ── Health Check ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """System health status."""
    return _get_engine().get_health()
