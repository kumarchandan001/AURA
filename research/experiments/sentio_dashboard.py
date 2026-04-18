"""
sentio_dashboard.py — Presentation-Grade Affective AI Dashboard.

A professional, dark-themed CustomTkinter desktop dashboard for live
demonstration of the Project SENTIO contactless rPPG framework.

Architecture (4 concurrent loops):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Thread 0 (Main)     : CustomTkinter GUI event loop             │
    │  Thread 1 (Daemon)   : CameraSensor — frame acquisition         │
    │  Thread 2 (Daemon)   : Processing — face tracking + vitals      │
    │  Thread 3 (Pool)     : OllamaClient — LLM inference             │
    └──────────────────────────────────────────────────────────────────┘

Layout (Grid System):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Header:  SENTIO — Contactless Affective Computing Framework    │
    ├──────────────────────────┬───────────────────────────────────────┤
    │                          │  Biometrics Panel                    │
    │  Live Webcam Feed        │  BPM | HRV | State + Live Graph     │
    │  (MediaPipe ROI Mesh)    ├───────────────────────────────────────┤
    │                          │  Affective AI Chat                   │
    │  FPS counter             │  Messenger-style scrollable chat     │
    │                          │  with adaptive LLM responses         │
    └──────────────────────────┴───────────────────────────────────────┘

Usage:
    python sentio_dashboard.py
    python sentio_dashboard.py --device 0 --model phi3

Author : Project SENTIO Research Team
Version: 1.0.0 (Presentation-Grade Dashboard)
"""

from __future__ import annotations
from matplotlib.backends.backend_agg import FigureCanvasAgg
from app.core.signal_extractor import SignalExtractor
from app.core.realtime_vitals import RealtimeVitals
from app.core.ollama_client import OllamaClient
from app.core.face_tracker import FaceTracker
from app.core.camera_sensor import CameraSensor
from app.core.affective_state import AffectiveState, PhysiologicalState
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import matplotlib
import cv2
import customtkinter as ctk
from typing import Deque, Dict, List, Optional
from datetime import datetime
from collections import deque
import time
import threading

import argparse
import csv
import io
import logging
import queue
import sys
from pathlib import Path

# ── Dynamic Data Paths ────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "manuscript"

# Force UTF-8 output on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Matplotlib for embedded live graph.

matplotlib.use("Agg")  # Non-interactive backend — render to buffer.
# ── Project SENTIO modules ──────────────────────────────────────────

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

_APP_TITLE = "SENTIO — Contactless Affective Computing Framework"
_UI_UPDATE_MS = 33  # ~30 FPS refresh.
_VITALS_INTERVAL = 0.5  # Compute vitals every 500 ms.
_GRAPH_UPDATE_INTERVAL = 3  # Update graph every N UI cycles.
_GRAPH_WINDOW_S = 8.0  # Seconds visible on the scrolling graph.

# ── Colour Palette (Medical-Tech Dark Theme) ─────────────────────────
_BG_PRIMARY = "#0B0F19"  # Deep navy-black.
_BG_PANEL = "#111827"  # Slightly lighter panel.
_BG_CARD = "#1E293B"  # Card surface.
_BG_INPUT = "#1E293B"
_BG_HEADER = "#0F172A"

_FG_TEXT = "#E2E8F0"  # Primary text — warm grey.
_FG_DIM = "#64748B"  # Muted text.
_FG_ACCENT = "#22D3EE"  # Cyan — primary accent.
_FG_ACCENT_ALT = "#06B6D4"  # Slightly darker cyan.
_FG_SUCCESS = "#34D399"  # Green.
_FG_DANGER = "#F87171"  # Red.
_FG_WARN = "#FBBF24"  # Amber.
_FG_USER = "#60A5FA"  # Blue — user messages.
_FG_AI = "#A5B4FC"  # Indigo — AI messages.
_FG_SYSTEM = "#94A3B8"  # System messages.

_BORDER = "#334155"
_HOVER = "#1E3A5F"

# ── Recording CSV columns ────────────────────────────────────────────
_CSV_COLUMNS = [
    "timestamp_iso",
    "perf_counter",
    "mean_g",
    "bpm",
    "rmssd_ms",
    "state",
    "fps",
]


# ======================================================================
# Live Pulse Graph Renderer
# ======================================================================

class PulseGraphRenderer:
    """Renders a scrolling rPPG pulse waveform into a PIL image.

    Uses Matplotlib with the Agg backend to draw a live, transparent
    signal trace that integrates visually into the dark UI.

    Parameters
    ----------
    width_px : int
        Output image width in pixels.
    height_px : int
        Output image height in pixels.
    window_seconds : float
        Time window visible on the graph.
    """

    def __init__(
        self,
        width_px: int = 460,
        height_px: int = 140,
        window_seconds: float = _GRAPH_WINDOW_S,
    ) -> None:
        self._width = width_px
        self._height = height_px
        self._window = window_seconds

        self._times: Deque[float] = deque(maxlen=600)
        self._values: Deque[float] = deque(maxlen=600)
        self._start_time: Optional[float] = None

        # Build the figure once — reuse for speed.
        dpi = 100
        self._fig, self._ax = plt.subplots(
            figsize=(width_px / dpi, height_px / dpi), dpi=dpi
        )
        self._fig.patch.set_facecolor("none")
        self._ax.set_facecolor("none")
        self._line, = self._ax.plot([], [], color=_FG_ACCENT, linewidth=1.5, alpha=0.9)

        # Style axes minimally.
        self._ax.spines["top"].set_visible(False)
        self._ax.spines["right"].set_visible(False)
        self._ax.spines["bottom"].set_color(_FG_DIM)
        self._ax.spines["left"].set_color(_FG_DIM)
        self._ax.tick_params(colors=_FG_DIM, labelsize=7)
        self._ax.set_ylabel("Amplitude", fontsize=7, color=_FG_DIM, labelpad=2)
        self._ax.set_xlabel("", fontsize=7)

        self._canvas = FigureCanvasAgg(self._fig)

    def add_sample(self, value: float, timestamp: float) -> None:
        """Add a new signal sample."""
        if self._start_time is None:
            self._start_time = timestamp
        self._times.append(timestamp - self._start_time)
        self._values.append(value)

    def render(self) -> Optional[Image.Image]:
        """Render the current graph state to a PIL RGBA image."""
        if len(self._times) < 4:
            return None

        times = np.array(self._times)
        values = np.array(self._values)

        # Only show the last _window seconds.
        t_max = times[-1]
        t_min = max(0, t_max - self._window)
        mask = times >= t_min
        t_vis = times[mask]
        v_vis = values[mask]

        if len(t_vis) < 2:
            return None

        # Normalize values for display.
        v_std = np.std(v_vis)
        if v_std > 1e-6:
            v_norm = (v_vis - np.mean(v_vis)) / v_std
        else:
            v_norm = v_vis - np.mean(v_vis)

        self._line.set_data(t_vis, v_norm)
        self._ax.set_xlim(t_min, t_min + self._window)

        y_range = max(np.abs(v_norm).max() * 1.3, 1.0)
        self._ax.set_ylim(-y_range, y_range)

        # Render to RGBA buffer.
        self._canvas.draw()
        buf = self._canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", self._canvas.get_width_height(), buf)

        # Composite onto dark background.
        bg = Image.new("RGBA", img.size, (17, 24, 39, 255))  # _BG_PANEL
        bg.paste(img, mask=img)
        return bg.convert("RGB")

    def reset(self) -> None:
        """Clear all graph data."""
        self._times.clear()
        self._values.clear()
        self._start_time = None


# ======================================================================
# Sentio Dashboard Application
# ======================================================================

class SentioDashboard:
    """Presentation-grade CustomTkinter dashboard for Project SENTIO.

    Integrates live webcam feed with MediaPipe ROI overlay, real-time
    biometric telemetry with an embedded pulse graph, and an adaptive
    LLM chat interface — all running across concurrent threads without
    blocking the GUI.

    Parameters
    ----------
    camera_device : int
        Webcam device index.
    target_fps : int
        Target camera FPS.
    llm_model : str
        Ollama model name.
    ollama_url : str
        Ollama server URL.
    """

    def __init__(
        self,
        camera_device: int = 0,
        target_fps: int = 30,
        llm_model: str = "phi3",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        # ── Core SENTIO modules (Phase 1–4) ────────────────────────
        self._sensor = CameraSensor(
            device_index=camera_device, target_fps=target_fps
        )
        self._tracker = FaceTracker()
        self._extractor = SignalExtractor()
        self._vitals = RealtimeVitals(window_seconds=10, min_window_seconds=4)
        self._affect = AffectiveState(window_seconds=30)
        self._llm = OllamaClient(model=llm_model, base_url=ollama_url)

        # ── Thread communication ───────────────────────────────────
        self._llm_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # ── Display frame (shared between threads) ──────────────────
        self._display_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # ── Live filtered signal for the graph ─────────────────────
        self._filtered_value: float = 0.0
        self._filtered_lock = threading.Lock()

        # ── FPS tracking ───────────────────────────────────────────
        self._fps_counter: int = 0
        self._fps_timer: float = time.perf_counter()
        self._current_fps: float = 0.0

        # ── Recording state ────────────────────────────────────────
        self._is_recording: bool = False
        self._recording_data: List[Dict] = []
        self._record_lock = threading.Lock()

        # ── UI state trackers ──────────────────────────────────────
        self._graph_cycle: int = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._graph_photo: Optional[ImageTk.PhotoImage] = None
        self._pulse_renderer = PulseGraphRenderer(
            width_px=460, height_px=140
        )

        # ── Build CustomTkinter UI ─────────────────────────────────
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self._root = ctk.CTk()
        self._root.title(_APP_TITLE)
        self._root.geometry("1280x720")
        self._root.minsize(1100, 650)
        self._root.configure(fg_color=_BG_PRIMARY)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Try to maximize.
        try:
            self._root.state("zoomed")
        except Exception:
            pass

        self._build_ui()
        logger.info("SentioDashboard initialized.")

    # ══════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        """Construct the three-panel professional layout."""

        # ── Header Bar ──────────────────────────────────────────────
        header = ctk.CTkFrame(
            self._root, height=56, corner_radius=0, fg_color=_BG_HEADER,
            border_width=0,
        )
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Logo / Title.
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            title_frame, text="◉ SENTIO",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=_FG_ACCENT,
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text="  Contactless Affective Computing Framework",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=_FG_DIM,
        ).pack(side="left", padx=(4, 0))

        # Right side: Record toggle + status.
        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=20, pady=10)

        self._record_btn = ctk.CTkButton(
            right_header,
            text="● Start Recording",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            width=180, height=34,
            corner_radius=17,
            fg_color="#1E3A5F",
            hover_color="#2D5A88",
            text_color=_FG_TEXT,
            border_width=1,
            border_color=_FG_ACCENT_ALT,
            command=self._toggle_recording,
        )
        self._record_btn.pack(side="left", padx=(0, 12))

        # LLM status badge.
        llm_ok = self._llm.is_available
        llm_text = "LLM Online" if llm_ok else "LLM Offline"
        llm_color = _FG_SUCCESS if llm_ok else _FG_DANGER
        self._llm_badge = ctk.CTkLabel(
            right_header, text=f"● {llm_text}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=llm_color,
        )
        self._llm_badge.pack(side="left")

        # ── Main Content Area ──────────────────────────────────────
        content = ctk.CTkFrame(self._root, fg_color=_BG_PRIMARY)
        content.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        content.grid_columnconfigure(0, weight=3, minsize=480)
        content.grid_columnconfigure(1, weight=2, minsize=400)
        content.grid_rowconfigure(0, weight=1)

        # ── LEFT PANEL: Live Camera Feed ───────────────────────────
        self._left_panel = ctk.CTkFrame(
            content, corner_radius=12, fg_color=_BG_PANEL,
            border_width=1, border_color=_BORDER,
        )
        self._left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Panel header.
        cam_header = ctk.CTkFrame(self._left_panel, fg_color="transparent")
        cam_header.pack(fill="x", padx=16, pady=(12, 0))

        ctk.CTkLabel(
            cam_header, text="📹  Live Sensor Feed — ROI Mesh",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=_FG_ACCENT,
        ).pack(side="left")

        self._fps_label = ctk.CTkLabel(
            cam_header, text="FPS: --",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=_FG_DIM,
        )
        self._fps_label.pack(side="right")

        # Video canvas.
        self._video_label = ctk.CTkLabel(
            self._left_panel, text="",
            fg_color="#000000", corner_radius=8,
        )
        self._video_label.pack(
            fill="both", expand=True, padx=12, pady=(8, 12)
        )

        # ── RIGHT COLUMN ───────────────────────────────────────────
        right_col = ctk.CTkFrame(content, fg_color=_BG_PRIMARY)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_col.grid_rowconfigure(0, weight=0)  # Biometrics — fixed.
        right_col.grid_rowconfigure(1, weight=1)  # Chat — expandable.
        right_col.grid_columnconfigure(0, weight=1)

        # ── TOP-RIGHT: Biometrics Panel ────────────────────────────
        bio_panel = ctk.CTkFrame(
            right_col, corner_radius=12, fg_color=_BG_PANEL,
            border_width=1, border_color=_BORDER,
        )
        bio_panel.grid(row=0, column=0, sticky="new", pady=(0, 6))

        ctk.CTkLabel(
            bio_panel, text="💓  Physiological Telemetry",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=_FG_ACCENT, anchor="w",
        ).pack(fill="x", padx=16, pady=(12, 4))

        # ── BPM / HRV / State cards ─────────────────────────────────
        cards_row = ctk.CTkFrame(bio_panel, fg_color="transparent")
        cards_row.pack(fill="x", padx=12, pady=(4, 4))
        cards_row.grid_columnconfigure(0, weight=1)
        cards_row.grid_columnconfigure(1, weight=1)
        cards_row.grid_columnconfigure(2, weight=1)

        # BPM card.
        bpm_card = ctk.CTkFrame(
            cards_row, corner_radius=10, fg_color=_BG_CARD,
            border_width=1, border_color=_BORDER,
        )
        bpm_card.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")

        ctk.CTkLabel(
            bpm_card, text="HEART RATE",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM,
        ).pack(pady=(10, 0))

        self._bpm_value = ctk.CTkLabel(
            bpm_card, text="--",
            font=ctk.CTkFont(family="Consolas", size=32, weight="bold"),
            text_color=_FG_ACCENT,
        )
        self._bpm_value.pack()

        ctk.CTkLabel(
            bpm_card, text="BPM",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM,
        ).pack(pady=(0, 10))

        # HRV card.
        hrv_card = ctk.CTkFrame(
            cards_row, corner_radius=10, fg_color=_BG_CARD,
            border_width=1, border_color=_BORDER,
        )
        hrv_card.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

        ctk.CTkLabel(
            hrv_card, text="HRV (RMSSD)",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM,
        ).pack(pady=(10, 0))

        self._hrv_value = ctk.CTkLabel(
            hrv_card, text="--",
            font=ctk.CTkFont(family="Consolas", size=32, weight="bold"),
            text_color=_FG_ACCENT,
        )
        self._hrv_value.pack()

        ctk.CTkLabel(
            hrv_card, text="ms",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM,
        ).pack(pady=(0, 10))

        # State card.
        state_card = ctk.CTkFrame(
            cards_row, corner_radius=10, fg_color=_BG_CARD,
            border_width=1, border_color=_BORDER,
        )
        state_card.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")

        ctk.CTkLabel(
            state_card, text="COGNITIVE STATE",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM,
        ).pack(pady=(10, 0))

        self._state_badge = ctk.CTkLabel(
            state_card, text="Calibrating",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=_FG_DIM,
            corner_radius=8,
        )
        self._state_badge.pack(pady=(4, 0))

        self._state_icon = ctk.CTkLabel(
            state_card, text="⏳",
            font=ctk.CTkFont(size=20),
            text_color=_FG_DIM,
        )
        self._state_icon.pack(pady=(0, 10))

        # ── Live pulse graph ─────────────────────────────────────────
        graph_frame = ctk.CTkFrame(bio_panel, fg_color="transparent")
        graph_frame.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            graph_frame, text="LIVE rPPG PULSE WAVEFORM",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=_FG_DIM, anchor="w",
        ).pack(fill="x", padx=4, pady=(0, 2))

        self._graph_label = ctk.CTkLabel(
            graph_frame, text="  Acquiring signal...",
            fg_color=_BG_CARD, corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=_FG_DIM, height=140,
        )
        self._graph_label.pack(fill="x")

        # ── BOTTOM-RIGHT: Affective AI Chat ────────────────────────
        chat_panel = ctk.CTkFrame(
            right_col, corner_radius=12, fg_color=_BG_PANEL,
            border_width=1, border_color=_BORDER,
        )
        chat_panel.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        chat_panel.grid_rowconfigure(1, weight=1)
        chat_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            chat_panel, text="🤖  SENTIO AI — Adaptive Assistant",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=_FG_ACCENT, anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))

        # Chat display (scrollable textbox).
        self._chat_box = ctk.CTkTextbox(
            chat_panel,
            fg_color=_BG_CARD,
            text_color=_FG_TEXT,
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8,
            border_width=0,
            wrap="word",
            state="disabled",
            activate_scrollbars=True,
        )
        self._chat_box.grid(
            row=1, column=0, sticky="nsew", padx=12, pady=(0, 4)
        )

        # Configure text tags.
        self._chat_box.tag_config("user", foreground=_FG_USER)
        self._chat_box.tag_config("ai", foreground=_FG_AI)
        self._chat_box.tag_config("system", foreground=_FG_SYSTEM)
        self._chat_box.tag_config("state_tag", foreground=_FG_ACCENT)
        self._chat_box.tag_config("loading", foreground=_FG_WARN)

        # Input row.
        input_row = ctk.CTkFrame(chat_panel, fg_color="transparent")
        input_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        input_row.grid_columnconfigure(0, weight=1)

        self._chat_input = ctk.CTkEntry(
            input_row,
            placeholder_text="Ask SENTIO AI anything...",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color=_BG_INPUT,
            text_color=_FG_TEXT,
            border_color=_BORDER,
            corner_radius=10,
            height=38,
        )
        self._chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._chat_input.bind("<Return>", self._on_send)

        self._send_btn = ctk.CTkButton(
            input_row, text="Send ▶",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            width=90, height=38, corner_radius=10,
            fg_color=_FG_ACCENT_ALT, hover_color="#0891B2",
            text_color="#0B0F19",
            command=self._on_send,
        )
        self._send_btn.grid(row=0, column=1)

        # Welcome message.
        self._append_chat(
            "SENTIO AI",
            "Welcome to Project SENTIO — Presentation Dashboard.\n\n"
            "I'm your adaptive AI assistant. My communication style "
            "adjusts in real-time based on your physiological state:\n\n"
            "  • CALM → Detailed, technical responses\n"
            "  • STRESSED → Concise, supportive guidance\n\n"
            "Type a message below to begin.",
            "system",
        )

    # ══════════════════════════════════════════════════════════════
    # CHAT METHODS
    # ══════════════════════════════════════════════════════════════

    def _append_chat(
        self, sender: str, message: str, tag: str = "ai"
    ) -> None:
        """Thread-safe chat message append."""
        self._chat_box.configure(state="normal")
        self._chat_box.insert("end", f"\n{sender}:\n", tag)
        self._chat_box.insert("end", f"{message}\n", tag)
        self._chat_box.configure(state="disabled")
        self._chat_box.see("end")

    def _on_send(self, event=None) -> None:
        """Handle user message submission."""
        message = self._chat_input.get().strip()
        if not message:
            return

        self._chat_input.delete(0, "end")
        self._append_chat("You", message, "user")

        # Show cognitive state.
        state = self._affect.current_state.value
        self._append_chat(
            "⚡ State", f"[Responding in {state} mode]", "state_tag"
        )

        # Show loading indicator.
        self._append_chat(
            "SENTIO AI", "✦ Analyzing your query...", "loading"
        )

        # Submit to LLM asynchronously.
        def on_response(text: str):
            self._llm_queue.put(text)

        self._llm.chat_async(
            user_message=message,
            state=state,
            callback=on_response,
        )

    def _poll_llm_queue(self) -> None:
        """Check for LLM responses and update chat."""
        try:
            while True:
                response = self._llm_queue.get_nowait()
                # Remove the loading message by overwriting with response.
                # (Simple approach: just append the real response.)
                self._append_chat("SENTIO AI", response, "ai")
        except queue.Empty:
            pass

    # ══════════════════════════════════════════════════════════════
    # RECORDING
    # ══════════════════════════════════════════════════════════════

    def _toggle_recording(self) -> None:
        """Toggle session recording on/off."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin recording telemetry to memory."""
        with self._record_lock:
            self._recording_data.clear()
            self._is_recording = True

        self._record_btn.configure(
            text="■ Stop Recording",
            fg_color="#7F1D1D",
            hover_color="#991B1B",
            border_color=_FG_DANGER,
        )
        self._append_chat(
            "System", "📊 Recording started. Telemetry is being captured.", "system"
        )

    def _stop_recording(self) -> None:
        """Stop recording and save to CSV."""
        self._is_recording = False

        # Save data.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentio_session_{timestamp}.csv"
        output_path = Path(filename)

        with self._record_lock:
            data = list(self._recording_data)
            self._recording_data.clear()

        if data:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(data)
            msg = f"📊 Recording saved → {output_path.name} ({len(data)} samples)"
        else:
            msg = "📊 Recording stopped — no data captured."

        self._record_btn.configure(
            text="● Start Recording",
            fg_color="#1E3A5F",
            hover_color="#2D5A88",
            border_color=_FG_ACCENT_ALT,
        )
        self._append_chat("System", msg, "system")

    def _record_sample(
        self, mean_g: float, bpm: float, rmssd: float,
        state: str, fps: float,
    ) -> None:
        """Record a single telemetry sample if recording is active."""
        if not self._is_recording:
            return

        sample = {
            "timestamp_iso": datetime.now().isoformat(),
            "perf_counter": round(time.perf_counter(), 4),
            "mean_g": round(mean_g, 4),
            "bpm": round(bpm, 2) if not np.isnan(bpm) else "",
            "rmssd_ms": round(rmssd, 2) if not np.isnan(rmssd) else "",
            "state": state,
            "fps": round(fps, 1),
        }
        with self._record_lock:
            self._recording_data.append(sample)

    # ══════════════════════════════════════════════════════════════
    # PROCESSING THREAD
    # ══════════════════════════════════════════════════════════════

    def _processing_loop(self) -> None:
        """Background thread: frame acquisition → tracking → vitals."""
        last_vitals_time = 0.0

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
                mean_r, mean_g, mean_b = self._extractor.extract(
                    frame, result.forehead_mask,
                    result.left_cheek_mask, result.right_cheek_mask,
                )
                self._vitals.add_sample(mean_g, timestamp)
                display = (
                    result.annotated_frame
                    if result.annotated_frame is not None
                    else frame
                )
            else:
                display = frame

            # Update display frame.
            with self._frame_lock:
                self._display_frame = display

            # FPS tracking.
            self._fps_counter += 1
            now = time.perf_counter()
            dt = now - self._fps_timer
            if dt >= 1.0:
                self._current_fps = self._fps_counter / dt
                self._fps_counter = 0
                self._fps_timer = now

            # Periodically compute vitals.
            if now - last_vitals_time >= _VITALS_INTERVAL:
                last_vitals_time = now
                bpm, rmssd = self._vitals.compute()
                self._affect.update(bpm, rmssd)

                # Feed the pulse graph.
                if not np.isnan(mean_g):
                    self._pulse_renderer.add_sample(mean_g, timestamp)

                # Record sample.
                self._record_sample(
                    mean_g=mean_g if not np.isnan(mean_g) else 0.0,
                    bpm=bpm, rmssd=rmssd,
                    state=self._affect.current_state.value,
                    fps=self._current_fps,
                )

            time.sleep(0.001)  # Yield CPU.

    # ══════════════════════════════════════════════════════════════
    # UI UPDATE LOOP
    # ══════════════════════════════════════════════════════════════

    def _update_ui(self) -> None:
        """Periodic UI refresh (~30 FPS) on the main thread."""
        if self._stop_event.is_set():
            return

        # ── Update video feed ────────────────────────────────────
        with self._frame_lock:
            frame = self._display_frame

        if frame is not None:
            try:
                # Get the label's actual dimensions.
                lw = self._video_label.winfo_width()
                lh = self._video_label.winfo_height()

                if lw > 10 and lh > 10:
                    h, w = frame.shape[:2]
                    scale = min(lw / w, lh / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (nw, nh))
                else:
                    resized = frame

                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._photo = ImageTk.PhotoImage(image=img)
                self._video_label.configure(image=self._photo, text="")
            except Exception:
                pass

        # ── Update FPS counter ────────────────────────────────────
        self._fps_label.configure(
            text=f"FPS: {self._current_fps:.0f}",
            text_color=_FG_SUCCESS if self._current_fps >= 25 else _FG_WARN,
        )

        # ── Update biometric telemetry ────────────────────────────
        bpm = self._vitals.latest_bpm
        rmssd = self._vitals.latest_rmssd
        state = self._affect.current_state

        # BPM.
        bpm_text = f"{bpm:.0f}" if not np.isnan(bpm) else "--"
        bpm_color = _FG_DANGER if (not np.isnan(bpm) and bpm > 85) else (
            _FG_SUCCESS if not np.isnan(bpm) else _FG_DIM
        )
        self._bpm_value.configure(text=bpm_text, text_color=bpm_color)

        # HRV.
        hrv_text = f"{rmssd:.0f}" if not np.isnan(rmssd) else "--"
        hrv_color = _FG_DANGER if (not np.isnan(rmssd) and rmssd < 30) else (
            _FG_SUCCESS if not np.isnan(rmssd) else _FG_DIM
        )
        self._hrv_value.configure(text=hrv_text, text_color=hrv_color)

        # State badge.
        state_map = {
            PhysiologicalState.CALM: ("CALM", _FG_SUCCESS, "✅"),
            PhysiologicalState.STRESSED: ("STRESSED", _FG_DANGER, "⚠️"),
            PhysiologicalState.UNKNOWN: ("Calibrating", _FG_DIM, "⏳"),
        }
        s_text, s_color, s_icon = state_map.get(
            state, ("--", _FG_DIM, "")
        )
        self._state_badge.configure(text=s_text, text_color=s_color)
        self._state_icon.configure(text=s_icon, text_color=s_color)

        # ── Update pulse graph (throttled) ────────────────────────
        self._graph_cycle += 1
        if self._graph_cycle >= _GRAPH_UPDATE_INTERVAL:
            self._graph_cycle = 0
            graph_img = self._pulse_renderer.render()
            if graph_img is not None:
                self._graph_photo = ImageTk.PhotoImage(image=graph_img)
                self._graph_label.configure(
                    image=self._graph_photo, text=""
                )

        # ── Poll LLM responses ────────────────────────────────────
        self._poll_llm_queue()

        # ── Reschedule ────────────────────────────────────────────
        self._root.after(_UI_UPDATE_MS, self._update_ui)

    # ══════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Start all subsystems and enter the main loop."""
        logger.info("Starting SENTIO Presentation Dashboard...")

        # Start the camera.
        try:
            self._sensor.start()
        except RuntimeError as e:
            logger.error("Camera failed: %s", e)
            print(f"\n  ✗ Camera error: {e}\n")
            sys.exit(1)

        time.sleep(0.3)

        # Start processing thread.
        self._proc_thread = threading.Thread(
            target=self._processing_loop,
            name="SentioDash-Processing",
            daemon=True,
        )
        self._proc_thread.start()

        # Start UI update loop.
        self._root.after(100, self._update_ui)

        # Enter main loop (blocks).
        self._root.mainloop()

    def _on_close(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down SENTIO Dashboard...")
        self._stop_event.set()

        # Save recording if active.
        if self._is_recording:
            self._stop_recording()

        self._sensor.stop()
        self._tracker.release()
        self._llm.shutdown()

        # Clean up matplotlib.
        plt.close("all")

        self._root.destroy()
        logger.info("All resources released. Goodbye.")


# ======================================================================
# Entry Point
# ======================================================================

def _configure_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    fh = logging.FileHandler(
        "sentio_dashboard.log", mode="w", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="SENTIO Dashboard",
        description="Presentation-grade affective computing dashboard.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Camera device index."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target capture FPS."
    )
    parser.add_argument(
        "--model", type=str, default="phi3", help="Ollama model name."
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(verbose=args.verbose)

    app = SentioDashboard(
        camera_device=args.device,
        target_fps=args.fps,
        llm_model=args.model,
        ollama_url=args.ollama_url,
    )
    app.run()
