#!/usr/bin/env bash
# =====================================================================
#  SENTIO — Full-Stack Development Launcher (Linux/Mac)
#  Boots FastAPI backend (port 8000) + Vite React frontend (port 5173)
# =====================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# Resolve paths
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv/bin"
FRONTEND="$ROOT/frontend"
BACKEND="$ROOT/backend"

echo "======================================================"
echo "  SENTIO: Contactless Affective Computing Framework"
echo "  Full-Stack Dashboard Launcher"
echo "======================================================"
echo ""

# Verify Python venv
if [ ! -f "$VENV/python" ]; then
    echo "[ERROR] Python virtual environment not found at $VENV"
    echo "        Run: python3 -m venv .venv"
    exit 1
fi

# Verify node_modules
if [ ! -d "$FRONTEND/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd "$FRONTEND"
    npm install
    cd "$ROOT"
    echo ""
fi

# Function to elegantly kill child processes on exit
cleanup() {
    echo ""
    echo "[INFO] Shutting down SENTIO servers..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "[INFO] Shutdown complete."
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting FastAPI backend on http://localhost:8000 ..."
cd "$BACKEND"
"$VENV/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$ROOT"

# Give the backend a moment to initialize the camera
sleep 3

echo "[2/2] Starting Vite frontend on http://localhost:5173 ..."
cd "$FRONTEND"
npm run dev &
FRONTEND_PID=$!
cd "$ROOT"

echo ""
echo "======================================================"
echo "  Both servers are starting in the background."
echo ""
echo "  Frontend : http://localhost:5173"
echo "  Backend  : http://localhost:8000"
echo "  API Docs : http://localhost:8000/docs"
echo "======================================================"
echo ""
echo "Press Ctrl+C to stop both servers."

# Keep script running
wait
