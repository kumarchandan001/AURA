"""
affective_agent.py — Phase 4: Multi-Threaded Affective AI Dashboard.

This is the crown jewel of Project SENTIO: a real-time, multi-threaded
tkinter dashboard that fuses live webcam-based rPPG physiological monitoring
with a local LLM whose communication style dynamically adapts to the user's
biometric stress level.

Architecture (4 concurrent loops):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Thread 0 (Main)     : tkinter GUI event loop                   │
    │  Thread 1 (Daemon)   : CameraSensor — frame acquisition         │
    │  Thread 2 (Daemon)   : Processing — face tracking + vitals      │
    │  Thread 3 (Pool)     : OllamaClient — LLM inference             │
    └──────────────────────────────────────────────────────────────────┘

    Communication between threads uses ``queue.Queue`` for the LLM response
    channel and atomic attribute reads for vitals/state (protected by locks
    inside each component).

Usage:
    python affective_agent.py
    python affective_agent.py --device 0 --model phi3

Author : Project SENTIO Research Team
Version: 0.4.0 (Phase 4 — Affective AI Integration)
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext
from typing import Optional

import cv2
import numpy as np
from app.core.affective_state import AffectiveState, PhysiologicalState
from app.core.camera_sensor import CameraSensor
from app.core.face_tracker import FaceTracker
from app.core.ollama_client import OllamaClient
from app.core.realtime_vitals import RealtimeVitals
from app.core.signal_extractor import SignalExtractor
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

_APP_TITLE = "SENTIO — Affective AI Dashboard"
_UPDATE_INTERVAL_MS = 33  # ~30 FPS UI refresh.
_VITALS_COMPUTE_INTERVAL = 0.5  # Compute vitals every 500ms.

# Colour scheme.
_BG_DARK = "#0D1117"
_BG_PANEL = "#161B22"
_BG_INPUT = "#21262D"
_FG_TEXT = "#C9D1D9"
_FG_DIM = "#8B949E"
_FG_ACCENT = "#00FFD0"
_FG_WARN = "#FF6B6B"
_FG_CALM = "#00E676"
_FG_STRESSED = "#FF5252"
_FG_USER = "#58A6FF"
_FG_AI = "#A5D6FF"
_BORDER = "#30363D"


# ======================================================================
# GUI Application
# ======================================================================

class SentioGUI:
    """Multi-threaded tkinter dashboard for affective AI interaction.

    Integrates live webcam feed, real-time physiological telemetry, and
    an adaptive LLM chat interface into a single responsive window.

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
        llm_model: str = "qwen2:0.5b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        # ── Core components ─────────────────────────────────────────
        self._sensor = CameraSensor(
            device_index=camera_device, target_fps=target_fps
        )
        self._tracker = FaceTracker()
        self._extractor = SignalExtractor()
        self._vitals = RealtimeVitals(window_seconds=10, min_window_seconds=4)
        self._affect = AffectiveState(window_seconds=30)
        self._llm = OllamaClient(model=llm_model, base_url=ollama_url)

        # ── Thread communication ────────────────────────────────────
        self._llm_response_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # ── Latest frame for display (set by processing thread) ─────
        self._display_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # ── Build the UI ────────────────────────────────────────────
        self._root = tk.Tk()
        self._root.title(_APP_TITLE)
        self._root.configure(bg=_BG_DARK)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.minsize(1100, 650)

        # Try to maximise window.
        try:
            self._root.state("zoomed")
        except tk.TclError:
            self._root.geometry("1280x720")

        self._build_ui()
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        logger.info("SentioGUI initialised.")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the three-panel dashboard layout."""

        # Fonts.
        self._font_title = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self._font_label = tkfont.Font(family="Segoe UI", size=11)
        self._font_value = tkfont.Font(family="Consolas", size=22, weight="bold")
        self._font_state = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self._font_chat = tkfont.Font(family="Consolas", size=10)
        self._font_input = tkfont.Font(family="Segoe UI", size=11)
        self._font_small = tkfont.Font(family="Segoe UI", size=9)

        # ── Main container ──────────────────────────────────────────
        main = tk.Frame(self._root, bg=_BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        main.columnconfigure(0, weight=3, minsize=400)
        main.columnconfigure(1, weight=2, minsize=350)
        main.rowconfigure(0, weight=1)

        # ── Left column: Webcam ─────────────────────────────────────
        left = tk.Frame(main, bg=_BG_PANEL, bd=1, relief=tk.FLAT, highlightbackground=_BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        tk.Label(
            left, text="📹  Live Feed — Face ROI", font=self._font_title,
            bg=_BG_PANEL, fg=_FG_ACCENT, anchor="w", padx=10, pady=6,
        ).pack(fill=tk.X)

        self._video_label = tk.Label(left, bg="#000000")
        self._video_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # ── Right column ────────────────────────────────────────────
        right = tk.Frame(main, bg=_BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        right.rowconfigure(0, weight=0)  # Telemetry — fixed height.
        right.rowconfigure(1, weight=1)  # Chat — expandable.
        right.columnconfigure(0, weight=1)

        # ── Panel 2: Telemetry ──────────────────────────────────────
        tele_frame = tk.Frame(right, bg=_BG_PANEL, bd=1, relief=tk.FLAT,
                              highlightbackground=_BORDER, highlightthickness=1)
        tele_frame.grid(row=0, column=0, sticky="new", pady=(0, 4))

        tk.Label(
            tele_frame, text="💓  Physiological Telemetry", font=self._font_title,
            bg=_BG_PANEL, fg=_FG_ACCENT, anchor="w", padx=10, pady=6,
        ).pack(fill=tk.X)

        metrics_row = tk.Frame(tele_frame, bg=_BG_PANEL)
        metrics_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        # BPM card.
        bpm_card = tk.Frame(metrics_row, bg=_BG_INPUT, bd=1, highlightbackground=_BORDER, highlightthickness=1)
        bpm_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(bpm_card, text="HEART RATE", font=self._font_small, bg=_BG_INPUT, fg=_FG_DIM).pack(pady=(6, 0))
        self._bpm_label = tk.Label(bpm_card, text="-- ", font=self._font_value, bg=_BG_INPUT, fg=_FG_ACCENT)
        self._bpm_label.pack()
        tk.Label(bpm_card, text="BPM", font=self._font_small, bg=_BG_INPUT, fg=_FG_DIM).pack(pady=(0, 6))

        # HRV card.
        hrv_card = tk.Frame(metrics_row, bg=_BG_INPUT, bd=1, highlightbackground=_BORDER, highlightthickness=1)
        hrv_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(hrv_card, text="HRV (RMSSD)", font=self._font_small, bg=_BG_INPUT, fg=_FG_DIM).pack(pady=(6, 0))
        self._hrv_label = tk.Label(hrv_card, text="-- ", font=self._font_value, bg=_BG_INPUT, fg=_FG_ACCENT)
        self._hrv_label.pack()
        tk.Label(hrv_card, text="ms", font=self._font_small, bg=_BG_INPUT, fg=_FG_DIM).pack(pady=(0, 6))

        # State indicator.
        state_row = tk.Frame(tele_frame, bg=_BG_PANEL)
        state_row.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(state_row, text="COGNITIVE STATE:", font=self._font_label, bg=_BG_PANEL, fg=_FG_DIM).pack(side=tk.LEFT)
        self._state_label = tk.Label(state_row, text="  Calibrating...  ",
                                     font=self._font_state, bg=_BG_PANEL, fg=_FG_DIM)
        self._state_label.pack(side=tk.LEFT, padx=(8, 0))

        # LLM status.
        llm_status = "🟢 Online" if self._llm.is_available else "🔴 Offline"
        llm_colour = _FG_CALM if self._llm.is_available else _FG_WARN
        self._llm_status_label = tk.Label(
            state_row, text=f"LLM: {llm_status}", font=self._font_small,
            bg=_BG_PANEL, fg=llm_colour,
        )
        self._llm_status_label.pack(side=tk.RIGHT)

        # ── Panel 3: Chat ───────────────────────────────────────────
        chat_frame = tk.Frame(right, bg=_BG_PANEL, bd=1, relief=tk.FLAT,
                              highlightbackground=_BORDER, highlightthickness=1)
        chat_frame.grid(row=1, column=0, sticky="nsew")

        tk.Label(
            chat_frame, text="🤖  SENTIO AI — Adaptive Assistant", font=self._font_title,
            bg=_BG_PANEL, fg=_FG_ACCENT, anchor="w", padx=10, pady=6,
        ).pack(fill=tk.X)

        self._chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            bg=_BG_INPUT,
            fg=_FG_TEXT,
            font=self._font_chat,
            insertbackground=_FG_TEXT,
            selectbackground="#2D333B",
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=8,
            state=tk.DISABLED,
        )
        self._chat_display.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        # Configure text tags for styling.
        self._chat_display.tag_configure("user", foreground=_FG_USER, font=self._font_chat)
        self._chat_display.tag_configure("ai", foreground=_FG_AI, font=self._font_chat)
        self._chat_display.tag_configure("system", foreground=_FG_DIM, font=self._font_small)
        self._chat_display.tag_configure("state_note", foreground=_FG_ACCENT, font=self._font_small)

        # Input row.
        input_row = tk.Frame(chat_frame, bg=_BG_PANEL)
        input_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        self._chat_input = tk.Entry(
            input_row,
            bg=_BG_INPUT,
            fg=_FG_TEXT,
            font=self._font_input,
            insertbackground=_FG_TEXT,
            relief=tk.FLAT,
            bd=0,
        )
        self._chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 4))
        self._chat_input.bind("<Return>", self._on_send)

        send_btn = tk.Button(
            input_row,
            text="Send ▶",
            font=self._font_label,
            bg=_FG_ACCENT,
            fg=_BG_DARK,
            activebackground="#00CC99",
            activeforeground=_BG_DARK,
            relief=tk.FLAT,
            bd=0,
            padx=16,
            pady=4,
            command=self._on_send,
        )
        send_btn.pack(side=tk.RIGHT)

        # Welcome message.
        self._append_chat(
            "SENTIO AI",
            "Welcome to Project SENTIO — Affective AI Dashboard.\n"
            "I'm your adaptive coding assistant. My communication style "
            "automatically adjusts based on your physiological state.\n\n"
            "• When you're CALM → I provide detailed, technical responses.\n"
            "• When you're STRESSED → I keep it concise and supportive.\n\n"
            "Type a message below to start chatting.",
            tag="system",
        )

    # ------------------------------------------------------------------
    # Chat methods
    # ------------------------------------------------------------------

    def _append_chat(
        self, sender: str, message: str, tag: str = "ai"
    ) -> None:
        """Append a message to the chat display (thread-safe via main thread).

        Parameters
        ----------
        sender : str
            Display name for the sender.
        message : str
            The message text.
        tag : str
            Text tag for styling (``"user"``, ``"ai"``, ``"system"``, ``"state_note"``).
        """
        self._chat_display.configure(state=tk.NORMAL)
        self._chat_display.insert(tk.END, f"\n{sender}:\n", tag)
        self._chat_display.insert(tk.END, f"{message}\n", tag)
        self._chat_display.configure(state=tk.DISABLED)
        self._chat_display.see(tk.END)

    def _on_send(self, event=None) -> None:
        """Handle user message submission."""
        message = self._chat_input.get().strip()
        if not message:
            return

        self._chat_input.delete(0, tk.END)
        self._append_chat("You", message, tag="user")

        # Show current state in chat.
        state = self._affect.current_state.value
        self._append_chat(
            "⚡ State",
            f"[Responding in {state} mode]",
            tag="state_note",
        )

        # Submit to LLM asynchronously.
        def on_response(response_text: str):
            self._llm_response_queue.put(response_text)

        self._llm.chat_async(
            user_message=message,
            state=state,
            callback=on_response,
        )

    def _check_llm_queue(self) -> None:
        """Poll the LLM response queue and update chat (runs on main thread)."""
        try:
            while True:
                response = self._llm_response_queue.get_nowait()
                self._append_chat("SENTIO AI", response, tag="ai")
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Processing Thread
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """Background thread: face tracking → signal extraction → vitals.

        Runs continuously until the stop event is set.  Reads frames from
        the CameraSensor, processes them, and updates the shared display
        frame and vital-signs buffers.
        """
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
            if result and result.face_detected:
                mean_r, mean_g, mean_b = self._extractor.extract(
                    frame, result.forehead_mask,
                    result.left_cheek_mask, result.right_cheek_mask,
                )
                self._vitals.add_sample(mean_g, timestamp)
                display = result.annotated_frame if result.annotated_frame is not None else frame
            else:
                display = frame

            # Update display frame.
            with self._frame_lock:
                self._display_frame = display

            # Periodically compute vitals.
            now = time.perf_counter()
            if now - last_vitals_time >= _VITALS_COMPUTE_INTERVAL:
                last_vitals_time = now
                bpm, rmssd = self._vitals.compute()
                self._affect.update(bpm, rmssd)

            time.sleep(0.001)  # Yield CPU.

    # ------------------------------------------------------------------
    # UI Update Loop
    # ------------------------------------------------------------------

    def _update_ui(self) -> None:
        """Periodic UI refresh callback (runs on the main/tkinter thread).

        Updates the video feed, telemetry labels, and checks the LLM
        response queue.  Reschedules itself via ``root.after()``.
        """
        if self._stop_event.is_set():
            return

        # ── Update video feed ────────────────────────────────────────
        with self._frame_lock:
            frame = self._display_frame

        if frame is not None:
            try:
                # Resize to fit the label.
                label_w = self._video_label.winfo_width()
                label_h = self._video_label.winfo_height()

                if label_w > 10 and label_h > 10:
                    h, w = frame.shape[:2]
                    scale = min(label_w / w, label_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = cv2.resize(frame, (new_w, new_h))
                else:
                    resized = frame

                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._photo_image = ImageTk.PhotoImage(image=img)
                self._video_label.configure(image=self._photo_image)
            except Exception:
                pass

        # ── Update telemetry ─────────────────────────────────────────
        bpm = self._vitals.latest_bpm
        rmssd = self._vitals.latest_rmssd
        state = self._affect.current_state

        bpm_text = f"{bpm:.0f}" if not np.isnan(bpm) else "--"
        hrv_text = f"{rmssd:.0f}" if not np.isnan(rmssd) else "--"
        self._bpm_label.configure(text=bpm_text)
        self._hrv_label.configure(text=hrv_text)

        # Colour-code BPM.
        if not np.isnan(bpm):
            bpm_colour = _FG_WARN if bpm > 85 else _FG_CALM
            self._bpm_label.configure(fg=bpm_colour)
        else:
            self._bpm_label.configure(fg=_FG_DIM)

        # Colour-code HRV.
        if not np.isnan(rmssd):
            hrv_colour = _FG_WARN if rmssd < 30 else _FG_CALM
            self._hrv_label.configure(fg=hrv_colour)
        else:
            self._hrv_label.configure(fg=_FG_DIM)

        # State label.
        state_text_map = {
            PhysiologicalState.CALM: ("  ✅ CALM  ", _FG_CALM),
            PhysiologicalState.STRESSED: ("  ⚠️ STRESSED  ", _FG_STRESSED),
            PhysiologicalState.UNKNOWN: ("  ⏳ Calibrating...  ", _FG_DIM),
        }
        label_text, label_fg = state_text_map.get(state, ("  --  ", _FG_DIM))
        self._state_label.configure(text=label_text, fg=label_fg)

        # ── Check LLM responses ─────────────────────────────────────
        self._check_llm_queue()

        # Reschedule.
        self._root.after(_UPDATE_INTERVAL_MS, self._update_ui)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all subsystems and enter the tkinter main loop.

        This method blocks until the window is closed.
        """
        logger.info("Starting SENTIO Affective Dashboard...")

        # Start camera.
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
            name="Processing-Thread",
            daemon=True,
        )
        self._proc_thread.start()

        # Start UI update loop.
        self._root.after(100, self._update_ui)

        # Enter tkinter main loop (blocks).
        self._root.mainloop()

    def _on_close(self) -> None:
        """Graceful shutdown handler."""
        logger.info("Shutting down SENTIO Dashboard...")
        self._stop_event.set()

        self._sensor.stop()
        self._tracker.release()
        self._llm.shutdown()

        self._root.destroy()
        logger.info("All resources released. Goodbye.")


# ======================================================================
# Entry Point
# ======================================================================

def _configure_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler("sentio_affective.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="SENTIO Phase 4",
        description="Affective AI Dashboard — biometric-adaptive coding assistant.",
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS.")
    parser.add_argument("--model", type=str, default="qwen2:0.5b", help="Ollama model name.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(verbose=args.verbose)

    app = SentioGUI(
        camera_device=args.device,
        target_fps=args.fps,
        llm_model=args.model,
        ollama_url=args.ollama_url,
    )
    app.run()
