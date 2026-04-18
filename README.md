<p align="center">
  <img src="https://img.shields.io/badge/AURA-Affective_Computing-00FFD0?style=for-the-badge&labelColor=0D1117" alt="AURA" />
</p>

<h1 align="center">🧠 AURA — Affective Understanding & Responsive Agent</h1>

<p align="center">
  <strong>A real-time, contactless physiological monitoring and affective computing platform</strong><br/>
  <em>Fusing webcam-based rPPG biometrics with an AI assistant that dynamically adapts<br/>its communication style to your stress level — in real time.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Face_Mesh-FF6F00?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=flat-square&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Research-lightgrey?style=flat-square" />
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-research">Research</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎥 **Contactless Vital Signs** | Extracts heart rate (BPM) and heart rate variability (HRV/RMSSD) from a standard webcam — no wearables required |
| 🧬 **rPPG Signal Processing** | 4th-order Butterworth bandpass filtering (0.7–3.0 Hz), linear + rolling detrend, peak detection with IBI validation |
| 🤖 **Adaptive AI Assistant** | LLM chat interface that **changes its personality** based on your physiological state — detailed when calm, concise when stressed |
| 💓 **Real-Time Biometrics Dashboard** | Live BPM, HRV, cognitive state indicator, and pulse waveform chart streaming at 10 Hz via WebSocket |
| 📹 **MJPEG Video Stream** | Live webcam feed with face mesh overlay, ROI annotations, and head pose visualization at ~30 FPS |
| 🔒 **100% Local & Private** | All processing runs on-device — camera feed, biometrics, and AI inference never leave your machine |
| ⚡ **Multi-Threaded Architecture** | 4 concurrent loops for zero-latency UI: camera capture, face processing, LLM inference, and UI rendering |

---

## 🔬 How It Works

### The Science: Remote Photoplethysmography (rPPG)

AURA leverages the principle that **blood volume changes in facial tissue create subtle color variations** invisible to the naked eye but detectable by a camera. The green channel of the RGB signal correlates with hemoglobin absorption, enabling contactless cardiac monitoring.

```
   Webcam Frame → Face Mesh Detection → ROI Isolation → Green Channel Extraction
                                                                    │
            BPM ← Peak Detection ← Bandpass Filter ← Linear Detrend ←
                                                                    │
          RMSSD ← Inter-Beat Interval Analysis ← Valid IBI Selection ←
```

### The Innovation: Affective-Adaptive AI

The AI assistant dynamically switches between **three communication modes** based on your real-time biometric state:

| State | BPM | RMSSD | AI Behavior |
|-------|-----|-------|-------------|
| 🟢 **Calm** | ≤ 85 | ≥ 30 ms | Detailed, technical, explores edge cases, challenges assumptions |
| 🔴 **Stressed** | > 85 | < 30 ms | Concise, supportive, step-by-step, includes breathing reminders |
| ⏳ **Calibrating** | — | — | Balanced, professional, neutral tone |

> **Hysteresis Logic**: State transitions use separate enter/leave thresholds (e.g., enter STRESSED at BPM > 85, but only leave at BPM < 78) to prevent rapid oscillation at boundary values.

---

## 🏗️ Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           AURA System Architecture                           │
│                                                                               │
│  Thread 1 (Daemon)          Thread 2 (Daemon)           Thread 3 (Pool)      │
│  ┌─────────────────┐        ┌──────────────────┐        ┌────────────────┐   │
│  │  CameraSensor   │──────▶ │  FaceTracker     │──────▶ │  OllamaClient  │   │
│  │  • CAP_DSHOW    │        │  • 468-point mesh│        │  • Dynamic     │   │
│  │  • perf_counter │        │  • ROI masking   │        │    system      │   │
│  │  • Lock-guarded │        │  • Head pose     │        │    prompts     │   │
│  └─────────────────┘        │  • Motion detect │        │  • Async chat  │   │
│         │                   └────────┬─────────┘        └───────┬────────┘   │
│         │                            │                          │             │
│         │                   ┌────────▼─────────┐                │             │
│         │                   │ SignalExtractor   │                │             │
│         │                   │ • RGB spatial μ   │                │             │
│         │                   └────────┬─────────┘                │             │
│         │                            │                          │             │
│         │                   ┌────────▼─────────┐       ┌───────▼────────┐   │
│         │                   │ RealtimeVitals   │──────▶│ AffectiveState │   │
│         │                   │ • Butterworth BP │       │ • Hysteresis   │   │
│         │                   │ • Peak detection │       │ • Rolling avg  │   │
│         │                   │ • BPM & RMSSD    │       │ • State FSM    │   │
│         │                   └──────────────────┘       └───────┬────────┘   │
│         │                                                       │             │
│         ▼                                                       ▼             │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                    FastAPI Backend (Port 8000)                        │     │
│  │                                                                      │     │
│  │   GET /video_feed      → MJPEG stream (~30 FPS)                     │     │
│  │   WS  /ws/telemetry    → JSON telemetry (10 Hz)                     │     │
│  │   POST /api/chat       → Affective-aware LLM response               │     │
│  │   GET  /api/health     → System health check                        │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│         │                                                                     │
│         ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                 React + Vite Frontend (Port 5173)                    │     │
│  │                                                                      │     │
│  │   ┌───────────┐  ┌──────────────┐  ┌───────────┐  ┌─────────────┐  │     │
│  │   │ VideoFeed │  │  Biometrics  │  │  Pulse    │  │    Chat     │  │     │
│  │   │ (MJPEG)  │  │   Panel      │  │  Chart    │  │   Panel     │  │     │
│  │   │          │  │  BPM • HRV   │  │ (Recharts)│  │ (Adaptive)  │  │     │
│  │   └───────────┘  └──────────────┘  └───────────┘  └─────────────┘  │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
AURA/
├── backend/                          # Python FastAPI backend
│   ├── app/
│   │   ├── main.py                   # FastAPI app, SentioEngine singleton, endpoints
│   │   ├── api/                      # REST API route modules
│   │   └── core/                     # Core processing pipeline
│   │       ├── camera_sensor.py      #   └─ Threaded capture, perf_counter timestamps
│   │       ├── face_tracker.py       #   └─ MediaPipe Face Mesh, ROI masking, head pose
│   │       ├── signal_extractor.py   #   └─ Spatial-mean RGB from facial ROIs
│   │       ├── realtime_vitals.py    #   └─ Sliding-window BPM & RMSSD computation
│   │       ├── affective_state.py    #   └─ FSM with hysteresis (Calm/Stressed/Unknown)
│   │       ├── affective_agent.py    #   └─ Tkinter desktop dashboard (standalone mode)
│   │       ├── ollama_client.py      #   └─ Async LLM client with dynamic prompting
│   │       └── face_landmarker.task  #   └─ MediaPipe model binary
│   ├── requirements.txt
│   └── setup.py                      # Editable install for clean imports
│
├── frontend/                         # React + Vite + Tailwind
│   ├── src/
│   │   ├── App.jsx                   # Main layout with WebSocket telemetry
│   │   ├── main.jsx                  # React entry point
│   │   ├── index.css                 # Tailwind directives
│   │   └── components/
│   │       ├── VideoFeed.jsx         #   └─ MJPEG stream display
│   │       ├── BiometricsPanel.jsx   #   └─ BPM, HRV, state indicator cards
│   │       ├── PulseChart.jsx        #   └─ Real-time waveform (Recharts)
│   │       ├── MetricCard.jsx        #   └─ Reusable metric display card
│   │       └── ChatPanel.jsx         #   └─ AI chat with state-aware responses
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── research/                         # Experimental validation framework
│   ├── data/                         # Datasets: raw vitals, smartwatch, results
│   ├── experiments/                  # Analysis & experiment scripts
│   │   ├── experiment_runner.py      #   └─ Automated multi-phase pipeline
│   │   ├── phase1_data_collector.py  #   └─ Raw signal acquisition
│   │   ├── signal_processor.py       #   └─ Offline signal processing
│   │   ├── validation_engine.py      #   └─ Statistical validation
│   │   └── sentio_dashboard.py       #   └─ CustomTkinter research GUI
│   └── manuscript/                   # LaTeX paper + figures
│
├── start.bat / start.sh              # One-click full-stack launcher
├── requirements.txt                  # Root Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Backend & signal processing |
| Node.js | 18+ | Frontend build tooling |
| Webcam | Any | Face capture for rPPG |
| Ollama | Latest | Local LLM inference (optional) |

### Option 1: One-Click Launch ⚡

```bash
# Windows
start.bat

# macOS / Linux
chmod +x start.sh && ./start.sh
```

This automatically boots both servers and opens the dashboard.

### Option 2: Manual Setup

<details>
<summary><strong>🔧 Backend Setup</strong></summary>

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install all Python dependencies
pip install -r requirements.txt

# Install backend as editable package (enables clean imports)
pip install -e backend/

# Launch FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend starts the camera, initializes the processing pipeline, and exposes the API at `http://localhost:8000`.

</details>

<details>
<summary><strong>🎨 Frontend Setup</strong></summary>

```bash
cd frontend
npm install
npm run dev
```

The React dashboard will be available at **http://localhost:5173**.

</details>

<details>
<summary><strong>🤖 Ollama Setup (Optional — for AI Chat)</strong></summary>

```bash
# Install Ollama: https://ollama.com

# Pull a lightweight model
ollama pull qwen2:0.5b

# The server auto-starts, or run manually:
ollama serve
```

AURA auto-detects the Ollama server and falls back gracefully if unavailable.

</details>

### Verify Everything Works

| URL | What You See |
|-----|-------------|
| `http://localhost:5173` | 🖥️ Full dashboard with live video, biometrics, and chat |
| `http://localhost:8000/docs` | 📚 Interactive Swagger API documentation |
| `http://localhost:8000/video_feed` | 📹 Raw MJPEG stream with face mesh overlay |
| `http://localhost:8000/api/health` | ✅ System health JSON |

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/video_feed` | MJPEG multipart stream — live webcam with ROI mesh overlay (~30 FPS) |
| `WS` | `/ws/telemetry` | WebSocket — real-time telemetry at 10 Hz (BPM, HRV, state, pulse batch) |
| `POST` | `/api/chat` | AI chat with physiological context injection |
| `GET` | `/api/health` | System health status (camera, LLM, FPS) |

### WebSocket Telemetry Payload

```json
{
  "bpm": 72.3,
  "rmssd": 45.2,
  "state": "Calm",
  "fps": 29.8,
  "pulse": [
    { "t": 12.3401, "v": 142.85 },
    { "t": 12.3734, "v": 143.12 }
  ]
}
```

### Chat Request / Response

```bash
# Request
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain rPPG signal extraction"}'

# Response
{
  "response": "Remote Photoplethysmography extracts cardiac signals from...",
  "state": "Calm"
}
```

---

## 📊 Signal Processing Pipeline

### DSP Chain (RealtimeVitals)

```
Raw Green Channel → Linear Detrend → Rolling Mean Subtraction → Butterworth Bandpass
                                                                  (0.7 – 3.0 Hz, 4th order)
                                                                        │
                                                                        ▼
                                                            Peak Detection (SciPy)
                                                           • min distance: 60/180 × fs
                                                           • prominence: 0.3 × σ(filtered)
                                                                        │
                                                                        ▼
                                                            IBI Validation (0.33–1.5s)
                                                           → BPM = 60 / mean(IBI)
                                                           → RMSSD = √(mean(Δ(IBI)²))
```

### Data Output Schema

The telemetry buffer maintains one row per captured frame:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `float64` | Monotonic `perf_counter` at capture |
| `frame_id` | `int` | Sequential frame number |
| `actual_fps` | `float64` | Rolling FPS estimate |
| `mean_r` / `mean_g` / `mean_b` | `float64` | Spatial mean RGB within facial ROI |
| `head_pitch` / `head_yaw` / `head_roll` | `float64` | Head pose (degrees) |
| `motion_flag` | `bool` | `True` if motion artefact detected |

> **Note:** When no face is detected, RGB and pose columns contain `NaN` to preserve temporal alignment.

---

## 🧪 Research

The `research/` directory contains a complete experimental validation framework:

| Phase | Script | Purpose |
|-------|--------|---------|
| **Phase 1** | `phase1_data_collector.py` | Raw optical signal acquisition from facial ROIs |
| **Phase 2** | `signal_processor.py` | Offline BPF, detrending, BPM/HRV extraction |
| **Phase 3** | `validation_engine.py` | Statistical comparison against smartwatch ground truth |
| **Full Pipeline** | `experiment_runner.py` | Automated end-to-end experiment orchestration |

```bash
# Run the complete experiment pipeline
python -m research.experiments.experiment_runner

# Launch the research dashboard
python -m research.experiments.sentio_dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Computer Vision** | OpenCV (CAP_DSHOW), MediaPipe Face Mesh (468 landmarks) |
| **Signal Processing** | NumPy, SciPy (Butterworth, `find_peaks`), Pandas |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend** | React 18, Vite 5, Tailwind CSS 3, Recharts, Lucide Icons |
| **AI / LLM** | Ollama (local inference), dynamic prompt injection |
| **Desktop GUI** | CustomTkinter (research), Tkinter (affective dashboard) |
| **Data Science** | Matplotlib, Seaborn |

---

## 🧠 Design Philosophy

| Decision | Rationale |
|----------|-----------|
| **No temporal filtering on raw signals** | Preserves signal fidelity for downstream R&D — filtering is applied at the vitals computation stage only |
| **Threaded capture with lock-guarded frames** | Decouples camera driver's blocking I/O from the processing loop, preventing frame-drop aliasing |
| **`perf_counter` timestamps** | Monotonic, nanosecond-resolution, immune to NTP and system clock adjustments |
| **NaN on face loss** | Maintains temporal alignment of the signal vector during occlusion/face-loss events |
| **CAP_DSHOW backend** | Selected for Windows to minimise driver latency (~8ms vs MSMF's ~35ms) |
| **Hysteresis state machine** | Separate enter/leave thresholds prevent rapid calm↔stressed oscillation at boundary values |
| **Dynamic system prompt injection** | LLM personality changes per-request based on live physiology — no fine-tuning needed |
| **Editable package install** | `pip install -e backend/` enables clean cross-module imports without `sys.path` hacks |
| **100% local processing** | Zero external API calls — all biometrics and AI inference run entirely on the user's machine |

---

## 🗂️ CLI Options

### Backend (Affective Agent — Standalone Mode)

```bash
python -m backend.app.core.affective_agent [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index |
| `--fps` | `30` | Target capture FPS |
| `--model` | `qwen2:0.5b` | Ollama model name |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `-v, --verbose` | off | Enable DEBUG-level logging |

---

## 🤝 Contributing

This project is in active research development. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Internal research use only — Project AURA, 2026.

---

<p align="center">
  <sub>Built with ❤️ for advancing contactless affective computing research</sub><br/>
  <sub>
    <strong>AURA</strong> — Because your code assistant should know how you feel.
  </sub>
</p>
