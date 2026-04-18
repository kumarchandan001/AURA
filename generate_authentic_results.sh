#!/usr/bin/env bash
# =====================================================================
#  SENTIO — Authentic Results Generation Pipeline (Linux/Mac)
#  Orchestrates Phase 1-3 in sequence to produce IEEE-ready data & plots
# =====================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

echo ""
echo "================================================================"
echo "  PROJECT SENTIO: Authentic Results Generation Pipeline"
echo "  IEEE Publication Data & Validation Figure Generator"
echo "================================================================"
echo ""

# ── Verify Python venv ─────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python virtual environment not found."
    echo "        Expected: $PYTHON"
    echo "        Run: python3 -m venv .venv"
    exit 1
fi

echo "================================================================"
echo " WARNING"
echo "================================================================"
echo ""
echo " Sit in a well-lit room."
echo " Look directly at the camera for 60 seconds to record"
echo " your authentic pulse."
echo " Press 'q' when finished."
echo ""
echo "================================================================"
echo ""
echo " The pipeline will execute 3 stages:"
echo "   Step 1: Phase 1 Data Collection   (camera capture)"
echo "   Step 2: Signal Processing          (DSP + BPM extraction)"
echo "   Step 3: Validation Engine          (metrics + IEEE figures)"
echo ""
read -p " Press ENTER to begin..."

# ── Step 1: Phase 1 Data Collection ────────────────────────
echo ""
echo "================================================================"
echo " [Step 1/3] Phase 1 — Authentic Pulse Data Collection"
echo "================================================================"
echo " Starting camera sensor... Record your resting pulse data."
echo ""
"$PYTHON" "$ROOT/research/experiments/phase1_data_collector.py"
echo ""
echo " [OK] Phase 1 complete. Raw vitals captured."
echo ""

# ── Step 2: Signal Processing ──────────────────────────────
echo "================================================================"
echo " [Step 2/3] Phase 2 — Signal Processing & BPM Extraction"
echo "================================================================"
echo " Running DSP pipeline on captured data..."
echo ""
"$PYTHON" "$ROOT/research/experiments/signal_processor.py"
echo ""
echo " [OK] Phase 2 complete. Signal analysis finished."
echo ""

# ── Step 3: Validation Engine ──────────────────────────────
echo "================================================================"
echo " [Step 3/3] Phase 3 — Validation Engine & IEEE Figure Generation"
echo "================================================================"
echo " Computing validation metrics and generating publication figures..."
echo ""
"$PYTHON" "$ROOT/research/experiments/validation_engine.py"
echo ""
echo " [OK] Phase 3 complete. Validation metrics and figures generated."
echo ""

# ── Done ───────────────────────────────────────────────────
echo "================================================================"
echo " PIPELINE COMPLETE"
echo "================================================================"
echo ""
echo " Output artifacts:"
echo "   Data   : research/data/final_experimental_results.csv"
echo "   Metrics: research/data/validation_metrics.json"
echo "   Fig 1  : research/manuscript/fig1_lighting_ablation.png"
echo "   Fig 2  : research/manuscript/fig2_stress_response.png"
echo ""
echo "================================================================"
