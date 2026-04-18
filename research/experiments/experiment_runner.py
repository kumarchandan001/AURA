"""
experiment_runner.py — Phase 5: Experimental Protocol, Ablation Studies & Publication Figures.

An automated experimental suite that orchestrates the Project SENTIO rPPG
pipeline across structured testing scenarios, records system performance
metrics, and generates IEEE-quality publication figures.

Experimental Protocol:
    Mode A (Baseline)          — 60 s, still, neutral lighting, no cognitive load.
    Mode B (Illumination Stress) — 60 s, dim lighting, no cognitive load.
    Mode C (Cognitive Stress)    — 60 s, bright lighting, on-screen mental math.

For each trial the suite captures raw rPPG signal, computes BPM and HRV,
optionally records smartwatch ground-truth, and aggregates everything into
``final_experimental_results.csv``.  Post-hoc analysis generates two
publication-ready figures:
    • fig1_lighting_ablation.png   — MAE comparison: Baseline vs. Dim
    • fig2_stress_response.png     — HRV drop: Baseline → Cognitive Stress

Usage:
    python experiment_runner.py                     # Interactive scenario menu
    python experiment_runner.py --mode A --duration 60
    python experiment_runner.py --analyze-only       # Generate plots from existing CSV
    python experiment_runner.py --demo               # Generate synthetic data + plots

Author : Project SENTIO Research Team
Version: 0.5.0 (Phase 5 — Experimental Protocol)
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from app.core.camera_sensor import CameraSensor
from app.core.face_tracker import FaceTracker
from app.core.signal_extractor import SignalExtractor
from scipy.signal import butter, detrend, filtfilt, find_peaks

# ── Dynamic Data Paths ────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "manuscript"

# Force UTF-8 output on Windows to prevent CP1252 encoding crashes.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ======================================================================
# Constants
# ======================================================================

IEEE_DPI: int = 300
MASTER_CSV: str = str(DATA_DIR / "final_experimental_results.csv")

# IEEE formatting — academic serif fonts, minimal decoration.
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A",
}

# Scenario definitions.
SCENARIOS: Dict[str, Dict] = {
    "A": {
        "name": "Baseline",
        "label": "Baseline (Neutral Light, Still)",
        "duration_s": 60,
        "description": (
            "Sit still and look at the camera.\n"
            "Neutral lighting. No cognitive load.\n"
            "Breathe normally."
        ),
        "cognitive_task": False,
    },
    "B": {
        "name": "Illumination Stress",
        "label": "Dim Light Stress Test",
        "duration_s": 60,
        "description": (
            "Reduce room lighting to a dim/low level.\n"
            "Sit still and look at the camera.\n"
            "This tests the optical sensor under poor SNR conditions."
        ),
        "cognitive_task": False,
    },
    "C": {
        "name": "Cognitive Stress",
        "label": "Cognitive Load (Mental Math)",
        "duration_s": 60,
        "description": (
            "Keep bright lighting.\n"
            "Mental math challenges will appear on screen.\n"
            "Solve them as fast as you can — this induces cognitive stress."
        ),
        "cognitive_task": True,
    },
}


# ======================================================================
# Cognitive Stress Task — Mental Math Stroop
# ======================================================================

class MentalMathChallenge:
    """Generates timed mental arithmetic problems for cognitive stress induction.

    Produces increasingly difficult arithmetic that the subject solves
    mentally while the camera records their physiological response.
    Problems are displayed on the OpenCV HUD overlay.

    Parameters
    ----------
    change_interval_s : float
        Seconds between problem changes (default: 5).
    """

    def __init__(self, change_interval_s: float = 5.0) -> None:
        self._interval = change_interval_s
        self._current_problem: str = ""
        self._current_answer: int = 0
        self._last_change: float = 0.0
        self._problems_shown: int = 0
        self._generate_new()

    def _generate_new(self) -> None:
        """Generate a new arithmetic problem with scaling difficulty."""
        self._problems_shown += 1
        difficulty = min(self._problems_shown // 3 + 1, 5)

        if difficulty <= 2:
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            op = random.choice(["+", "-"])
            self._current_problem = f"{a} {op} {b} = ?"
            self._current_answer = eval(f"{a} {op} {b}")
        elif difficulty <= 3:
            a = random.randint(12, 49)
            b = random.randint(3, 9)
            self._current_problem = f"{a} x {b} = ?"
            self._current_answer = a * b
        else:
            a = random.randint(100, 999)
            b = random.randint(10, 99)
            op = random.choice(["+", "-"])
            self._current_problem = f"{a} {op} {b} = ?"
            self._current_answer = eval(f"{a} {op} {b}")

        self._last_change = time.perf_counter()

    def get_current(self) -> str:
        """Return the current problem string, refreshing if interval elapsed.

        Returns
        -------
        str
            Current arithmetic problem text.
        """
        if time.perf_counter() - self._last_change >= self._interval:
            self._generate_new()
        return self._current_problem

    @property
    def problems_shown(self) -> int:
        """Total number of problems presented."""
        return self._problems_shown


# ======================================================================
# Trial Runner — Single Scenario Execution
# ======================================================================

class TrialRunner:
    """Execute a single experimental trial with live data capture.

    Orchestrates CameraSensor → FaceTracker → SignalExtractor for a fixed
    duration, with an optional cognitive stress task overlay.  Collects
    per-frame Green-channel data with precision timestamps and computes
    BPM/HRV post-trial.

    Parameters
    ----------
    scenario_key : str
        One of ``"A"``, ``"B"``, ``"C"``.
    trial_id : int
        Numeric trial identifier for this session.
    camera_device : int
        OpenCV camera device index.
    duration_override : int or None
        Override the scenario's default duration (seconds).
    watch_bpm : float or None
        Static smartwatch BPM reference (if available).
    """

    def __init__(
        self,
        scenario_key: str,
        trial_id: int = 1,
        camera_device: int = 0,
        duration_override: Optional[int] = None,
        watch_bpm: Optional[float] = None,
    ) -> None:
        self.scenario = SCENARIOS[scenario_key]
        self.scenario_key = scenario_key
        self.trial_id = trial_id
        self.camera_device = camera_device
        self.watch_bpm = watch_bpm
        self.duration = duration_override or self.scenario["duration_s"]

        # Collected data.
        self.green_values: List[float] = []
        self.timestamps: List[float] = []
        self.motion_flags: List[bool] = []
        self.frame_count: int = 0

        # Results (computed post-trial).
        self.result: Optional[Dict] = None

    def run(self) -> Dict:
        """Execute the trial and return computed metrics.

        Displays a live HUD window with:
            - Webcam feed with ROI overlays
            - Trial timer / progress bar
            - Cognitive task (Mode C only)
            - Real-time FPS counter

        Returns
        -------
        dict
            Trial result containing all computed metrics.
        """
        scenario = self.scenario
        sep = "-" * 50

        print(f"\n{sep}")
        print(f"  TRIAL {self.trial_id} -- Mode {self.scenario_key}: {scenario['name']}")
        print(f"  Duration: {self.duration}s")
        print(f"{sep}")
        print(f"\n  {scenario['description']}")
        print("\n  Press SPACE to start, 'q' to abort.\n")

        # Setup.
        sensor = CameraSensor(device_index=self.camera_device, target_fps=30)
        tracker = FaceTracker()
        extractor = SignalExtractor()
        math_task = MentalMathChallenge() if scenario["cognitive_task"] else None

        window_name = f"SENTIO Trial {self.trial_id} - {scenario['name']}"

        try:
            sensor.start()
            time.sleep(0.5)

            # Wait for SPACE key to begin.
            while True:
                frame, _ = sensor.read()
                if frame is not None:
                    display = frame.copy()
                    h, w = display.shape[:2]
                    cv2.rectangle(display, (0, 0), (w, 80), (15, 15, 25), -1)
                    cv2.putText(
                        display,
                        f"Mode {self.scenario_key}: {scenario['name']}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 208), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        "Press SPACE to start  |  'q' to abort",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, display)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    break
                elif key == ord("q"):
                    print("  Trial aborted by user.")
                    sensor.stop()
                    tracker.release()
                    cv2.destroyAllWindows()
                    return self._empty_result("Aborted")

            # ── Recording loop ───────────────────────────────────────
            print("\n  >> Recording started...")
            start_time = time.perf_counter()
            fps_counter = 0
            fps_timer = start_time
            current_fps = 0.0

            while True:
                now = time.perf_counter()
                elapsed = now - start_time
                remaining = self.duration - elapsed

                if remaining <= 0:
                    break

                frame, ts = sensor.read()
                if frame is None:
                    time.sleep(0.001)
                    continue

                # Track face.
                try:
                    result = tracker.process(frame)
                except Exception:
                    result = None

                # Extract signal.
                if result and result.face_detected:
                    _, mean_g, _ = extractor.extract(
                        frame, result.forehead_mask,
                        result.left_cheek_mask, result.right_cheek_mask,
                    )
                    self.green_values.append(mean_g)
                    self.timestamps.append(ts)
                    self.motion_flags.append(result.motion_flag)
                    display = result.annotated_frame if result.annotated_frame is not None else frame
                else:
                    display = frame

                self.frame_count += 1
                fps_counter += 1

                # FPS calculation.
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # ── HUD overlay ──────────────────────────────────────
                h, w = display.shape[:2]

                # Top bar.
                cv2.rectangle(display, (0, 0), (w, 90), (15, 15, 25), -1)

                # Progress bar.
                progress = min(elapsed / self.duration, 1.0)
                bar_w = int((w - 30) * progress)
                cv2.rectangle(display, (15, 70), (w - 15, 82), (50, 50, 60), -1)
                bar_colour = (0, 230, 118) if progress < 0.8 else (255, 167, 38)
                cv2.rectangle(display, (15, 70), (15 + bar_w, 82), bar_colour, -1)

                # Text overlays.
                cv2.putText(
                    display,
                    f"Mode {self.scenario_key}: {scenario['name']}  |  "
                    f"FPS: {current_fps:.0f}  |  Frames: {self.frame_count}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 208), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Time: {elapsed:.1f}s / {self.duration}s  |  "
                    f"Signal samples: {len(self.green_values)}",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
                )

                # Cognitive task (Mode C).
                if math_task:
                    problem = math_task.get_current()
                    cv2.rectangle(display, (0, h - 70), (w, h), (25, 15, 40), -1)
                    cv2.putText(
                        display,
                        f"SOLVE: {problem}",
                        (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 100), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        f"Problems: {math_task.problems_shown}",
                        (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA,
                    )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("  Trial stopped early by user.")
                    break

            # ── Compute results ──────────────────────────────────────
            actual_duration = time.perf_counter() - start_time
            print(f"  [OK] Recording complete -- {actual_duration:.1f}s, "
                  f"{len(self.green_values)} signal samples.")

            self.result = self._compute_metrics(actual_duration, math_task)
            return self.result

        except Exception as e:
            print(f"  [X] Trial error: {e}")
            return self._empty_result(f"Error: {e}")
        finally:
            sensor.stop()
            tracker.release()
            cv2.destroyAllWindows()

    def _compute_metrics(
        self,
        duration: float,
        math_task: Optional[MentalMathChallenge],
    ) -> Dict:
        """Process collected data into trial metrics.

        Parameters
        ----------
        duration : float
            Actual recording duration in seconds.
        math_task : MentalMathChallenge or None
            Cognitive task instance (for problem count).

        Returns
        -------
        dict
            Complete trial metrics.
        """
        result = {
            "trial_id": self.trial_id,
            "scenario": self.scenario_key,
            "scenario_name": self.scenario["name"],
            "timestamp_iso": datetime.now().isoformat(),
            "duration_s": round(duration, 2),
            "total_frames": self.frame_count,
            "valid_samples": len(self.green_values),
            "motion_artifacts": sum(self.motion_flags),
        }

        if len(self.green_values) < 20:
            result.update({
                "fs_hz": float("nan"),
                "mean_bpm": float("nan"),
                "std_bpm": float("nan"),
                "rmssd_ms": float("nan"),
                "watch_bpm": self.watch_bpm if self.watch_bpm else float("nan"),
                "mae_bpm": float("nan"),
                "snr_db": float("nan"),
                "problems_shown": math_task.problems_shown if math_task else 0,
                "status": "Insufficient data",
            })
            return result

        green = np.array(self.green_values, dtype=np.float64)
        ts = np.array(self.timestamps, dtype=np.float64)
        fs = (len(ts) - 1) / (ts[-1] - ts[0])

        bpm, std_bpm, rmssd, snr = self._dsp_pipeline(green, ts, fs)

        mae = abs(bpm - self.watch_bpm) if self.watch_bpm and not np.isnan(bpm) else float("nan")

        result.update({
            "fs_hz": round(fs, 2),
            "mean_bpm": round(bpm, 2) if not np.isnan(bpm) else float("nan"),
            "std_bpm": round(std_bpm, 2) if not np.isnan(std_bpm) else float("nan"),
            "rmssd_ms": round(rmssd, 2) if not np.isnan(rmssd) else float("nan"),
            "watch_bpm": self.watch_bpm if self.watch_bpm else float("nan"),
            "mae_bpm": round(mae, 2) if not np.isnan(mae) else float("nan"),
            "snr_db": round(snr, 2) if not np.isnan(snr) else float("nan"),
            "problems_shown": math_task.problems_shown if math_task else 0,
            "status": "Complete",
        })

        return result

    @staticmethod
    def _dsp_pipeline(
        green: np.ndarray,
        timestamps: np.ndarray,
        fs: float,
    ) -> Tuple[float, float, float, float]:
        """Run the Phase 2 DSP pipeline on a buffer of Green-channel data.

        Parameters
        ----------
        green : numpy.ndarray
            Green channel spatial means.
        timestamps : numpy.ndarray
            Monotonic timestamps.
        fs : float
            Sampling rate in Hz.

        Returns
        -------
        tuple[float, float, float, float]
            ``(mean_bpm, std_bpm, rmssd_ms, snr_db)``
        """
        # Detrend.
        sig = detrend(green, type="linear")
        win = max(int(fs * 1.5), 3)
        if win % 2 == 0:
            win += 1
        rolling = pd.Series(sig).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
        sig = sig - rolling

        # Bandpass: 0.7–3.0 Hz.
        nyq = fs / 2.0
        high = min(3.0, nyq * 0.95)
        low = 0.7
        if low >= high:
            return float("nan"), float("nan"), float("nan"), float("nan")

        b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
        pad = min(3 * max(len(b), len(a)), len(sig) - 1)
        if pad < 1:
            return float("nan"), float("nan"), float("nan"), float("nan")
        filtered = filtfilt(b, a, sig, padlen=pad)

        # SNR estimate (power in passband vs. total).
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
        mask_pass = (freqs >= low) & (freqs <= high)
        power_signal = np.sum(fft_vals[mask_pass] ** 2)
        power_total = np.sum(fft_vals ** 2)
        if power_total > 0 and power_signal > 0:
            snr = 10 * np.log10(power_signal / (power_total - power_signal + 1e-12))
        else:
            snr = float("nan")

        # Peak detection.
        min_dist = max(int(fs * (60.0 / 180.0)), 1)
        prominence = 0.3 * np.std(filtered)
        if prominence < 1e-8:
            return float("nan"), float("nan"), float("nan"), snr

        peaks, _ = find_peaks(filtered, distance=min_dist, prominence=prominence)

        if len(peaks) < 2:
            return float("nan"), float("nan"), float("nan"), snr

        peak_times = timestamps[peaks]
        ibi_s = np.diff(peak_times)
        valid = (ibi_s > 0.33) & (ibi_s < 1.5)
        ibi_valid = ibi_s[valid]

        if len(ibi_valid) < 1:
            return float("nan"), float("nan"), float("nan"), snr

        inst_bpm = 60.0 / ibi_valid
        mean_bpm = float(np.mean(inst_bpm))
        std_bpm = float(np.std(inst_bpm))

        # RMSSD.
        if len(ibi_valid) < 2:
            return mean_bpm, std_bpm, float("nan"), snr

        ibi_ms = ibi_valid * 1000.0
        rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))

        return mean_bpm, std_bpm, rmssd, snr

    def _empty_result(self, status: str) -> Dict:
        """Return a result dict with NaN values for failed/aborted trials."""
        return {
            "trial_id": self.trial_id,
            "scenario": self.scenario_key,
            "scenario_name": self.scenario["name"],
            "timestamp_iso": datetime.now().isoformat(),
            "duration_s": 0,
            "total_frames": 0,
            "valid_samples": 0,
            "motion_artifacts": 0,
            "fs_hz": float("nan"),
            "mean_bpm": float("nan"),
            "std_bpm": float("nan"),
            "rmssd_ms": float("nan"),
            "watch_bpm": float("nan"),
            "mae_bpm": float("nan"),
            "snr_db": float("nan"),
            "problems_shown": 0,
            "status": status,
        }


# ======================================================================
# Master CSV Aggregation
# ======================================================================

def append_to_master_csv(result: Dict, csv_path: Path) -> None:
    """Append a trial result to the master CSV file.

    Creates the file with headers if it does not exist.

    Parameters
    ----------
    result : dict
        Trial result dictionary.
    csv_path : Path
        Path to the master CSV.
    """
    columns = [
        "trial_id", "scenario", "scenario_name", "timestamp_iso",
        "duration_s", "total_frames", "valid_samples", "motion_artifacts",
        "fs_hz", "mean_bpm", "std_bpm", "rmssd_ms",
        "watch_bpm", "mae_bpm", "snr_db", "problems_shown", "status",
    ]

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"  [OK] Result appended -> {csv_path.resolve()}")


# ======================================================================
# Demo Data Generator
# ======================================================================

def generate_demo_results(csv_path: Path, n_trials_per_mode: int = 5) -> None:
    """Generate synthetic experimental results for plot development.

    Creates plausible rPPG metrics across three scenarios with realistic
    distributions:
        - Baseline: BPM ~72±4, RMSSD ~45±8, MAE ~2.5±1.0
        - Dim Light: BPM ~74±5, RMSSD ~42±9, MAE ~5.8±2.0
        - Cognitive Stress: BPM ~88±6, RMSSD ~25±6, MAE ~3.2±1.5

    Parameters
    ----------
    csv_path : Path
        Output CSV path.
    n_trials_per_mode : int
        Number of synthetic trials per scenario.
    """
    np.random.seed(42)
    results = []
    trial_id = 0

    profiles = {
        "A": {"bpm_mu": 72, "bpm_sd": 4, "rmssd_mu": 45, "rmssd_sd": 8,
              "mae_mu": 2.5, "mae_sd": 1.0, "snr_mu": 5.2, "snr_sd": 1.5},
        "B": {"bpm_mu": 74, "bpm_sd": 5, "rmssd_mu": 42, "rmssd_sd": 9,
              "mae_mu": 5.8, "mae_sd": 2.0, "snr_mu": 1.8, "snr_sd": 1.2},
        "C": {"bpm_mu": 88, "bpm_sd": 6, "rmssd_mu": 25, "rmssd_sd": 6,
              "mae_mu": 3.2, "mae_sd": 1.5, "snr_mu": 4.5, "snr_sd": 1.3},
    }

    for mode_key, profile in profiles.items():
        scenario = SCENARIOS[mode_key]
        for i in range(n_trials_per_mode):
            trial_id += 1
            bpm = max(50, np.random.normal(profile["bpm_mu"], profile["bpm_sd"]))
            rmssd = max(5, np.random.normal(profile["rmssd_mu"], profile["rmssd_sd"]))
            mae = max(0.5, np.random.normal(profile["mae_mu"], profile["mae_sd"]))
            snr = np.random.normal(profile["snr_mu"], profile["snr_sd"])
            watch_bpm = bpm + np.random.normal(0, 1.5)

            results.append({
                "trial_id": trial_id,
                "scenario": mode_key,
                "scenario_name": scenario["name"],
                "timestamp_iso": datetime.now().isoformat(),
                "duration_s": 60,
                "total_frames": int(np.random.normal(1800, 50)),
                "valid_samples": int(np.random.normal(1700, 80)),
                "motion_artifacts": int(np.random.poisson(3)),
                "fs_hz": round(np.random.normal(29.5, 0.8), 2),
                "mean_bpm": round(bpm, 2),
                "std_bpm": round(np.random.uniform(2, 6), 2),
                "rmssd_ms": round(rmssd, 2),
                "watch_bpm": round(watch_bpm, 2),
                "mae_bpm": round(mae, 2),
                "snr_db": round(snr, 2),
                "problems_shown": int(np.random.randint(8, 14)) if mode_key == "C" else 0,
                "status": "Complete",
            })

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Demo data generated -- {len(df)} trials -> {csv_path.resolve()}")


# ======================================================================
# Publication-Ready Visualization
# ======================================================================

def generate_ieee_figures(csv_path: Path, output_dir: Path) -> None:
    """Generate IEEE-quality publication figures from experimental data.

    Produces two figures:
        1. **fig1_lighting_ablation.png** — Grouped box plot comparing MAE
           between Baseline (Mode A) and Illumination Stress (Mode B).
        2. **fig2_stress_response.png** — Paired line/bar chart showing the
           HRV (RMSSD) reduction from Baseline to Cognitive Stress (Mode C).

    Parameters
    ----------
    csv_path : Path
        Path to the master experimental results CSV.
    output_dir : Path
        Directory for output figures.
    """
    if not csv_path.exists():
        print(f"  [X] Results CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "Complete"].copy()

    if len(df) == 0:
        print("  [X] No completed trials found in the CSV.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(IEEE_RC):
        _generate_fig1_lighting(df, output_dir)
        _generate_fig2_stress(df, output_dir)


def _generate_fig1_lighting(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 1: MAE comparison — Baseline vs. Dim Lighting.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered experimental results.
    output_dir : Path
        Output directory.
    """
    # Filter to mode A and B only.
    df_ab = df[df["scenario"].isin(["A", "B"])].copy()
    df_ab = df_ab.dropna(subset=["mae_bpm"])

    if len(df_ab) < 2:
        print("  [!] Insufficient data for lighting ablation figure (need modes A & B).")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=IEEE_DPI)

    # Colour palette — muted academic tones.
    palette = {"Baseline": "#4A90D9", "Illumination Stress": "#D94A4A"}

    sns.boxplot(
        data=df_ab,
        x="scenario_name",
        y="mae_bpm",
        hue="scenario_name",
        palette=palette,
        width=0.5,
        linewidth=1.2,
        fliersize=4,
        ax=ax,
        legend=False,
    )

    # Overlay individual data points.
    sns.stripplot(
        data=df_ab,
        x="scenario_name",
        y="mae_bpm",
        color="black",
        alpha=0.5,
        size=4,
        jitter=0.15,
        ax=ax,
    )

    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel("Mean Absolute Error (BPM)")
    ax.set_title("Fig. 1: Heart Rate Estimation Accuracy\nunder Varying Illumination")

    # Add mean annotation.
    for i, mode in enumerate(["Baseline", "Illumination Stress"]):
        subset = df_ab[df_ab["scenario_name"] == mode]["mae_bpm"]
        if len(subset) > 0:
            mean_val = subset.mean()
            ax.annotate(
                f"u = {mean_val:.2f}",
                xy=(i, mean_val),
                xytext=(i + 0.3, mean_val + 0.5),
                fontsize=8,
                color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            )

    plt.tight_layout()
    out_path = output_dir / str(MANUSCRIPT_DIR / "fig1_lighting_ablation.png")
    fig.savefig(out_path, dpi=IEEE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Fig 1 saved -> {out_path.resolve()}")


def _generate_fig2_stress(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 2: HRV drop from Baseline to Cognitive Stress.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered experimental results.
    output_dir : Path
        Output directory.
    """
    df_ac = df[df["scenario"].isin(["A", "C"])].copy()
    df_ac = df_ac.dropna(subset=["rmssd_ms"])

    if len(df_ac) < 2:
        print("  [!] Insufficient data for stress response figure (need modes A & C).")
        return

    baseline_rmssd = df_ac[df_ac["scenario"] == "A"]["rmssd_ms"]
    stress_rmssd = df_ac[df_ac["scenario"] == "C"]["rmssd_ms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=IEEE_DPI,
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ── Left: Paired line plot ──────────────────────────────────────
    n_pairs = min(len(baseline_rmssd), len(stress_rmssd))
    if n_pairs > 0:
        b_vals = baseline_rmssd.values[:n_pairs]
        s_vals = stress_rmssd.values[:n_pairs]

        for i in range(n_pairs):
            colour = "#CC3333" if s_vals[i] < b_vals[i] else "#999999"
            ax1.plot(
                [0, 1], [b_vals[i], s_vals[i]],
                color=colour, alpha=0.4, linewidth=1.0, marker="o", markersize=4,
            )

        # Mean lines.
        ax1.plot(
            [0, 1], [np.mean(b_vals), np.mean(s_vals)],
            color="#1A1A8E", linewidth=2.5, marker="D", markersize=7,
            label=f"Mean (Delta = {np.mean(s_vals) - np.mean(b_vals):+.1f} ms)",
            zorder=5,
        )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Baseline", "Cognitive\nStress"])
    ax1.set_ylabel("RMSSD (ms)")
    ax1.set_title("Fig. 2a: HRV Transition")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # ── Right: Bar chart with error bars ─────────────────────────────
    means = [baseline_rmssd.mean(), stress_rmssd.mean()]
    sems = [baseline_rmssd.sem(), stress_rmssd.sem()]
    labels = ["Baseline", "Cognitive\nStress"]
    colours = ["#4A90D9", "#D94A4A"]

    bars = ax2.bar(labels, means, yerr=sems, capsize=5, color=colours,
                   edgecolor="#333333", linewidth=0.8, width=0.55, alpha=0.85)

    # Value labels on bars.
    for bar, mean, sem in zip(bars, means, sems):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 1,
            f"{mean:.1f}",
            ha="center", fontsize=9, fontweight="bold", color="#333333",
        )

    ax2.set_ylabel("RMSSD (ms)")
    ax2.set_title("Fig. 2b: Mean HRV Comparison")

    # Significance bracket.
    if len(baseline_rmssd) >= 2 and len(stress_rmssd) >= 2:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(baseline_rmssd, stress_rmssd)
        sig_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        if p_val < 0.05:
            sig_text += " *"
        if p_val < 0.01:
            sig_text = sig_text.replace("*", "**")
        if p_val < 0.001:
            sig_text = sig_text.replace("**", "***")

        y_max = max(means) + max(sems) + 5
        ax2.plot([0, 0, 1, 1], [y_max, y_max + 2, y_max + 2, y_max],
                 color="#333333", linewidth=1.0)
        ax2.text(0.5, y_max + 2.5, sig_text, ha="center", fontsize=8, color="#333333")

    plt.tight_layout()
    out_path = output_dir / str(MANUSCRIPT_DIR / "fig2_stress_response.png")
    fig.savefig(out_path, dpi=IEEE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Fig 2 saved -> {out_path.resolve()}")


# ======================================================================
# Interactive Scenario Menu
# ======================================================================

def interactive_menu() -> None:
    """Display the interactive experiment selection menu."""
    sep = "=" * 58

    print(f"\n{sep}")
    print("  PROJECT SENTIO -- Phase 5: Experimental Protocol")
    print(f"{sep}")
    print()
    print("  Available Scenarios:")
    print("  -------------------------------------------------")
    print("   [A]  Baseline        -- 60s, still, neutral light")
    print("   [B]  Dim Light       -- 60s, dim lighting test")
    print("   [C]  Cognitive Stress -- 60s, mental math challenge")
    print("  -------------------------------------------------")
    print("   [P]  Generate Publication Plots (from existing data)")
    print("   [D]  Generate Demo Data + Plots")
    print("   [Q]  Quit")
    print()

    while True:
        choice = input("  Select mode (A/B/C/P/D/Q): ").strip().upper()

        if choice in ("A", "B", "C"):
            # Ask for optional parameters.
            duration_str = input(f"  Duration in seconds [{SCENARIOS[choice]['duration_s']}]: ").strip()
            duration = int(duration_str) if duration_str.isdigit() else None

            watch_str = input("  Smartwatch BPM (or Enter to skip): ").strip()
            watch_bpm = float(watch_str) if watch_str else None

            trial_id = _get_next_trial_id(Path(MASTER_CSV))

            trial = TrialRunner(
                scenario_key=choice,
                trial_id=trial_id,
                duration_override=duration,
                watch_bpm=watch_bpm,
            )
            result = trial.run()

            if result["status"] != "Aborted":
                append_to_master_csv(result, Path(MASTER_CSV))
                _print_trial_summary(result)

        elif choice == "P":
            print("\n  Generating publication figures...")
            generate_ieee_figures(Path(MASTER_CSV), Path("."))
            print()

        elif choice == "D":
            print("\n  Generating demo dataset...")
            generate_demo_results(Path(MASTER_CSV))
            print("  Generating publication figures from demo data...")
            generate_ieee_figures(Path(MASTER_CSV), Path("."))
            print()

        elif choice == "Q":
            print("\n  Goodbye.\n")
            break

        else:
            print("  Invalid selection. Please choose A, B, C, P, D, or Q.\n")


def _get_next_trial_id(csv_path: Path) -> int:
    """Determine the next trial ID from the existing master CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the master CSV.

    Returns
    -------
    int
        Next available trial ID.
    """
    if not csv_path.exists():
        return 1

    try:
        df = pd.read_csv(csv_path)
        if "trial_id" in df.columns and len(df) > 0:
            return int(df["trial_id"].max()) + 1
    except Exception:
        pass

    return 1


def _print_trial_summary(result: Dict) -> None:
    """Pretty-print a trial result summary.

    Parameters
    ----------
    result : dict
        Trial result dictionary.
    """
    sep = "-" * 50

    print(f"\n{sep}")
    print(f"  TRIAL {result['trial_id']} SUMMARY -- {result['scenario_name']}")
    print(f"{sep}")
    print(f"  Duration          : {result['duration_s']}s")
    print(f"  Valid samples     : {result['valid_samples']}")
    print(f"  Motion artifacts  : {result['motion_artifacts']}")
    print(f"  Sampling rate     : {result['fs_hz']} Hz")
    print(f"  Mean BPM (rPPG)   : {result['mean_bpm']}")
    print(f"  BPM Std Dev       : {result['std_bpm']}")
    print(f"  RMSSD (HRV)       : {result['rmssd_ms']} ms")
    print(f"  Smartwatch BPM    : {result['watch_bpm']}")
    print(f"  MAE               : {result['mae_bpm']} BPM")
    print(f"  SNR               : {result['snr_db']} dB")
    print(f"  Status            : {result['status']}")
    print(f"{sep}\n")


# ======================================================================
# CLI Entry Point
# ======================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="SENTIO Phase 5",
        description="Experimental protocol, ablation studies, and publication figures.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["A", "B", "C"],
        help="Run a specific scenario directly (skip menu).",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Override scenario duration (seconds).",
    )
    parser.add_argument(
        "--watch-bpm", type=float, default=None,
        help="Static smartwatch ground-truth BPM.",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Camera device index.",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip data collection, only generate plots from existing CSV.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic demo data and publication plots.",
    )
    parser.add_argument(
        "--output", type=str, default=MASTER_CSV,
        help=f"Master CSV path (default: {MASTER_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.demo:
        print("\n  Generating demo experimental data...")
        generate_demo_results(Path(args.output))
        print("  Generating IEEE publication figures...")
        generate_ieee_figures(Path(args.output), Path("."))
        print("\n  [OK] Done.\n")

    elif args.analyze_only:
        print("\n  Generating IEEE publication figures from existing data...")
        generate_ieee_figures(Path(args.output), Path("."))
        print("\n  [OK] Done.\n")

    elif args.mode:
        trial_id = _get_next_trial_id(Path(args.output))
        trial = TrialRunner(
            scenario_key=args.mode,
            trial_id=trial_id,
            camera_device=args.device,
            duration_override=args.duration,
            watch_bpm=args.watch_bpm,
        )
        result = trial.run()
        if result["status"] != "Aborted":
            append_to_master_csv(result, Path(args.output))
            _print_trial_summary(result)

    else:
        interactive_menu()
