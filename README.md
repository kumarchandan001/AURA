<p align="center">
  <h1 align="center">🧠 AURA</h1>
  <p align="center"><strong>Affective Understanding and Responsive Agent</strong></p>
  <p align="center">
    A real-time, contactless physiological monitoring and affective computing system<br/>
    powered by Remote Photoplethysmography (rPPG), computer vision, and adaptive AI.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/react-18.3-61dafb?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/fastapi-0.111-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/mediapipe-0.10-orange?style=flat-square&logo=google" />
  <img src="https://img.shields.io/badge/license-research-lightgrey?style=flat-square" />
</p>

---

## 📌 Overview

**AURA** (formerly Project SENTIO) is a full-stack affective computing platform that:

- 🎥 **Captures** real-time video via webcam with threaded sub-millisecond timing
- 🧬 **Extracts** facial ROI signals (R/G/B channels) using MediaPipe Face Mesh
- 💓 **Computes** contactless heart rate, stress index, and respiratory rate via rPPG
- 🤖 **Responds** adaptively through an AI agent powered by Ollama LLMs
- 📊 **Visualizes** live biometrics in a modern React dashboard

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AURA System Architecture                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │ CameraSensor│──▶│ FaceTracker │──▶│SignalExtract.│──▶│ Telemetry  │ │
│  │  (threaded) │   │ (Face Mesh) │   │  (RGB mean)  │   │  Manager   │ │
│  └─────────────┘   └─────────────┘   └──────────────┘   └─────┬──────┘ │
│                                                                 │        │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐         │        │
│  │  Realtime   │◀──│ Affective   │◀──│   Ollama     │         │        │
│  │   Vitals    │   │   Agent     │   │   Client     │         │        │
│  └──────┬──────┘   └─────────────┘   └──────────────┘         │        │
│         │                                                       │        │
│         ▼                  FastAPI REST API                     ▼        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    React + Vite Frontend                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────┐  │   │
│  │  │VideoFeed │ │Biometrics│ │PulseChart │ │   ChatPanel      │  │   │
│  │  │          │ │  Panel   │ │(Recharts) │ │  (AI Dialogue)   │  │   │
│  │  └──────────┘ └──────────┘ └───────────┘ └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
AURA/
├── backend/                    # Python backend (FastAPI)
│   ├── app/
│   │   ├── main.py             # FastAPI application entry point
│   │   ├── api/                # REST API routes
│   │   └── core/               # Core processing modules
│   │       ├── camera_sensor.py       # Threaded webcam capture
│   │       ├── face_tracker.py        # MediaPipe Face Mesh + ROI
│   │       ├── signal_extractor.py    # RGB spatial mean extraction
│   │       ├── telemetry_manager.py   # Data buffering & CSV export
│   │       ├── realtime_vitals.py     # rPPG vital sign computation
│   │       ├── affective_state.py     # Emotion/stress state modeling
│   │       ├── affective_agent.py     # Adaptive AI agent with GUI
│   │       └── ollama_client.py       # Ollama LLM integration
│   ├── requirements.txt
│   └── setup.py
│
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx             # Main application layout
│   │   ├── main.jsx            # React entry point
│   │   ├── index.css           # Global styles (Tailwind)
│   │   └── components/
│   │       ├── VideoFeed.jsx          # Live webcam stream
│   │       ├── BiometricsPanel.jsx    # Vital signs display
│   │       ├── PulseChart.jsx         # Real-time pulse graph
│   │       ├── MetricCard.jsx         # Metric display cards
│   │       └── ChatPanel.jsx          # AI chat interface
│   └── package.json
│
├── research/                   # Research & experimentation
│   ├── data/                   # Experimental datasets (CSV/JSON)
│   ├── experiments/            # Experiment scripts & analysis
│   │   ├── experiment_runner.py       # Automated experiment pipeline
│   │   ├── phase1_data_collector.py   # Raw signal acquisition
│   │   ├── signal_processor.py        # Signal processing & filtering
│   │   ├── validation_engine.py       # Statistical validation
│   │   └── sentio_dashboard.py        # Research dashboard (CustomTkinter)
│   └── manuscript/             # LaTeX manuscript & figures
│
├── start.bat / start.sh        # One-click launch scripts
├── requirements.txt            # Root Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** & npm
- **Webcam** (built-in or USB)
- **Ollama** (optional, for AI chat features) — [Install Ollama](https://ollama.com)

### Option 1: One-Click Launch

```bash
# Windows
start.bat

# macOS / Linux
chmod +x start.sh && ./start.sh
```

### Option 2: Manual Setup

#### Backend

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Install backend as editable package
pip install -e backend/

# Start the FastAPI server
uvicorn backend.app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at **http://localhost:5173** and will proxy API requests to the backend on port 8000.

---

## ⚙️ Core Modules

| Module | Description |
|--------|-------------|
| **CameraSensor** | Lock-guarded threaded capture with `perf_counter` timestamps and CAP_DSHOW backend |
| **FaceTracker** | MediaPipe Face Mesh → forehead & cheek ROI masks, PnP head pose estimation, motion detection |
| **SignalExtractor** | Spatial-mean R/G/B extraction within combined facial ROI mask |
| **TelemetryManager** | Live HUD overlay, in-memory ring buffer, Pandas CSV export |
| **RealtimeVitals** | rPPG-based heart rate, stress index, and respiratory rate computation |
| **AffectiveAgent** | Context-aware AI assistant that adapts communication style to user's emotional state |
| **OllamaClient** | Integration with local Ollama LLMs for natural language generation |

---

## 📊 Data Output Schema

The telemetry CSV contains **one row per captured frame**:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `float64` | Monotonic `perf_counter` value at capture |
| `frame_id` | `int` | Sequential frame number |
| `actual_fps` | `float64` | Rolling FPS estimate |
| `mean_r` / `mean_g` / `mean_b` | `float64` | Spatial mean RGB within ROI |
| `head_pitch` / `head_yaw` / `head_roll` | `float64` | Head pose in degrees |
| `motion_flag` | `bool` | `True` if motion artefact detected |

> **Note:** When no face is detected, RGB and pose columns contain `NaN` to maintain temporal alignment.

---

## 🧪 Research

The `research/` directory contains the experimental framework for validating the rPPG pipeline:

- **Phase 1** — Raw signal acquisition and data collection
- **Phase 2** — Signal processing, filtering, and vital sign extraction
- **Phase 3** — Statistical validation against ground-truth measurements

Run the full experiment pipeline:

```bash
python -m research.experiments.experiment_runner
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Signal Processing** | OpenCV, MediaPipe, NumPy, SciPy |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React 18, Vite, Tailwind CSS, Recharts, Lucide Icons |
| **AI / LLM** | Ollama (local inference) |
| **Data & Research** | Pandas, Matplotlib, Seaborn |
| **Desktop GUI** | CustomTkinter |

---

## 📝 Design Decisions

- **No temporal filtering on raw signals** — Preserves signal integrity for downstream algorithmic R&D
- **Threaded capture** — Decouples camera I/O from the processing loop, preventing frame-drop aliasing
- **`perf_counter` timestamps** — Monotonic, high-resolution, immune to NTP/system-clock jumps
- **NaN on face loss** — Maintains temporal alignment during occlusion events
- **CAP_DSHOW backend** — Minimises driver latency on Windows versus MSMF
- **Editable package install** — Backend installed as `pip install -e` for clean cross-module imports

---

## 📄 License

Internal research use only — Project AURA, 2026.

---

<p align="center">
  Built with ❤️ for advancing affective computing research
</p>
