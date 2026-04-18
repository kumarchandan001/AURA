"""
validation_engine.py — Phase 3: Cross-Modal Synchronization & Statistical Validation.

This script scientifically validates the rPPG-derived heart rate against a
consumer smartwatch (ground truth) by performing temporal alignment,
resampling, error quantification, and generating IEEE-publication-ready
Bland-Altman and time-series overlay plots.

Processing Pipeline:
    1. Ingest rPPG BPM data (from Phase 2) and smartwatch ground-truth CSV.
    2. Apply a configurable timestamp sync offset to align the two modalities.
    3. Resample the lower-frequency smartwatch signal onto the rPPG time axis
       via cubic interpolation.
    4. Compute validation statistics: MAE, RMSE, Pearson r.
    5. Generate a 2-panel research figure (time-series overlay + Bland-Altman).
    6. Export metrics to ``validation_metrics.json`` for IEEE paper integration.

Usage:
    python validation_engine.py
    python validation_engine.py --rppg phase1_raw_vitals.csv --watch smartwatch_data.csv --offset 2.5
    python validation_engine.py --rppg phase1_raw_vitals.csv --watch smartwatch_data.csv --generate-demo

Author : Project SENTIO Research Team
Version: 0.3.0 (Phase 3 — Statistical Validation)
"""

from __future__ import annotations
from scipy import stats as sp_stats
from scipy import interpolate as sp_interp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

import argparse
import json
import sys
from pathlib import Path

# ── Dynamic Data Paths ────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "manuscript"


# ======================================================================
# Constants
# ======================================================================

# IEEE minimum DPI for figures.
IEEE_DPI: int = 300

# Minimum overlapping data points for meaningful statistics.
MIN_OVERLAP_POINTS: int = 10


# ======================================================================
# Data Ingestion
# ======================================================================

def load_rppg_bpm(csv_path: Path) -> pd.DataFrame:
    """Load rPPG data from the Phase 1/2 pipeline CSV.

    Expects at minimum the columns ``timestamp`` and ``mean_g``.  If the
    Phase 2 signal processor has already been run, instantaneous BPM can
    be recomputed here from the raw signal for maximum flexibility.

    This function recomputes per-frame BPM internally using the filtered
    Green channel and peak detection from Phase 2 logic, so it works
    directly from ``phase1_raw_vitals.csv``.

    Parameters
    ----------
    csv_path : Path
        Path to the Phase 1 raw vitals CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``time_s`` (relative seconds) and ``rppg_bpm``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"rPPG CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    required = {"timestamp", "mean_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rPPG CSV missing columns: {missing}")

    # Drop NaN rows (face-lost frames).
    df = df.dropna(subset=["timestamp", "mean_g"]).reset_index(drop=True)

    if len(df) < MIN_OVERLAP_POINTS:
        raise ValueError(
            f"rPPG CSV has only {len(df)} valid rows; need ≥ {MIN_OVERLAP_POINTS}."
        )

    # Compute relative time axis (start at 0).
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)

    # Re-run lightweight BPM extraction from Green channel.
    raw_green = df["mean_g"].to_numpy(dtype=np.float64)
    fs = _compute_fs(timestamps)
    bpm_time, bpm_values = _extract_instantaneous_bpm(raw_green, timestamps, fs)

    result = pd.DataFrame({"time_s": bpm_time, "rppg_bpm": bpm_values})
    result = result.dropna().reset_index(drop=True)

    return result


def load_smartwatch_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate ground-truth smartwatch HR data.

    Expects columns ``timestamp`` and ``watch_bpm``.  The timestamp can be
    either Unix epoch seconds or relative seconds (auto-detected).
    NaN / missing BPM values are forward-filled then dropped if still NaN.

    Parameters
    ----------
    csv_path : Path
        Path to the smartwatch CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``time_s`` (relative seconds) and ``watch_bpm``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If required columns are missing or data is insufficient.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Smartwatch CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    required = {"timestamp", "watch_bpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Smartwatch CSV missing columns: {missing}")

    # Forward-fill sparse smartwatch readings, then drop remaining NaNs.
    df["watch_bpm"] = df["watch_bpm"].ffill()
    df = df.dropna(subset=["timestamp", "watch_bpm"]).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"Smartwatch CSV has only {len(df)} valid rows; need ≥ 2."
        )

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    t_rel = timestamps - timestamps[0]

    result = pd.DataFrame({
        "time_s": t_rel,
        "watch_bpm": df["watch_bpm"].to_numpy(dtype=np.float64),
    })

    return result


def generate_demo_watch_data(rppg_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Generate synthetic smartwatch ground-truth data for testing.

    Creates a plausible 1 Hz smartwatch signal by downsampling the rPPG BPM
    with added Gaussian noise (σ=2 BPM) and a small systematic bias (+1.5 BPM),
    simulating the typical error profile of a consumer wrist-worn PPG device.

    Parameters
    ----------
    rppg_df : pandas.DataFrame
        rPPG BPM DataFrame with ``time_s`` and ``rppg_bpm`` columns.
    output_path : Path
        Where to save the generated CSV.

    Returns
    -------
    pandas.DataFrame
        Synthetic smartwatch DataFrame.
    """
    t_start = rppg_df["time_s"].iloc[0]
    t_end = rppg_df["time_s"].iloc[-1]

    # 1 Hz smartwatch cadence.
    watch_times = np.arange(t_start, t_end, 1.0)

    # Interpolate rPPG BPM onto 1 Hz grid.
    interp_fn = sp_interp.interp1d(
        rppg_df["time_s"].to_numpy(),
        rppg_df["rppg_bpm"].to_numpy(),
        kind="linear",
        fill_value="extrapolate",
    )
    base_bpm = interp_fn(watch_times)

    # Add realistic noise + systematic bias.
    np.random.seed(42)
    noise = np.random.normal(0, 2.0, size=len(watch_times))
    bias = 1.5
    watch_bpm = base_bpm + noise + bias

    # Clip to physiological range.
    watch_bpm = np.clip(watch_bpm, 40, 200)

    # Inject a few NaNs to test robustness (~5%).
    nan_indices = np.random.choice(len(watch_bpm), size=max(1, len(watch_bpm) // 20), replace=False)
    watch_bpm[nan_indices] = np.nan

    demo_df = pd.DataFrame({"timestamp": watch_times, "watch_bpm": watch_bpm})
    demo_df.to_csv(output_path, index=False)
    print(f"  ✓ Demo smartwatch data generated → {output_path.resolve()}")

    # Return cleaned version.
    demo_df["watch_bpm"] = demo_df["watch_bpm"].ffill()
    demo_df = demo_df.dropna().reset_index(drop=True)
    return pd.DataFrame({
        "time_s": demo_df["timestamp"].to_numpy(),
        "watch_bpm": demo_df["watch_bpm"].to_numpy(),
    })


# ======================================================================
# Temporal Alignment & Resampling
# ======================================================================

def apply_sync_offset(
    watch_df: pd.DataFrame,
    offset_seconds: float,
) -> pd.DataFrame:
    """Apply a manual timestamp offset to align two recording modalities.

    If the smartwatch was started ``offset_seconds`` *after* the webcam,
    a positive offset shifts the watch timeline forward to align.

    Parameters
    ----------
    watch_df : pandas.DataFrame
        Smartwatch data with ``time_s`` column.
    offset_seconds : float
        Seconds to add to the watch timeline (positive = watch started later).

    Returns
    -------
    pandas.DataFrame
        Adjusted DataFrame.
    """
    adjusted = watch_df.copy()
    adjusted["time_s"] = adjusted["time_s"] + offset_seconds
    return adjusted


def resample_to_common_axis(
    rppg_df: pd.DataFrame,
    watch_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample both signals onto a shared time axis via cubic interpolation.

    The common axis spans the *intersection* of the two time ranges.  The
    smartwatch signal (typically 1 Hz) is upsampled via cubic spline to
    match the rPPG cadence, ensuring a one-to-one sample correspondence
    for statistical comparison.

    Parameters
    ----------
    rppg_df : pandas.DataFrame
        rPPG BPM data with ``time_s`` and ``rppg_bpm``.
    watch_df : pandas.DataFrame
        Smartwatch data with ``time_s`` and ``watch_bpm``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(common_time, rppg_bpm_aligned, watch_bpm_aligned)``

    Raises
    ------
    ValueError
        If the overlapping window is too short.
    """
    # Determine overlapping time window.
    t_start = max(rppg_df["time_s"].iloc[0], watch_df["time_s"].iloc[0])
    t_end = min(rppg_df["time_s"].iloc[-1], watch_df["time_s"].iloc[-1])

    if t_end - t_start < 1.0:
        raise ValueError(
            f"Overlap window is only {t_end - t_start:.2f} s — insufficient. "
            "Check your sync offset or input files."
        )

    # Build interpolators.
    rppg_interp = sp_interp.interp1d(
        rppg_df["time_s"].to_numpy(),
        rppg_df["rppg_bpm"].to_numpy(),
        kind="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )
    watch_interp = sp_interp.interp1d(
        watch_df["time_s"].to_numpy(),
        watch_df["watch_bpm"].to_numpy(),
        kind="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Use rPPG time points within the overlap as the common axis.
    rppg_times = rppg_df["time_s"].to_numpy()
    mask = (rppg_times >= t_start) & (rppg_times <= t_end)
    common_time = rppg_times[mask]

    if len(common_time) < MIN_OVERLAP_POINTS:
        raise ValueError(
            f"Only {len(common_time)} overlapping points; "
            f"need ≥ {MIN_OVERLAP_POINTS}."
        )

    rppg_aligned = rppg_interp(common_time)
    watch_aligned = watch_interp(common_time)

    # Drop any NaNs from edge interpolation.
    valid = np.isfinite(rppg_aligned) & np.isfinite(watch_aligned)
    common_time = common_time[valid]
    rppg_aligned = rppg_aligned[valid]
    watch_aligned = watch_aligned[valid]

    if len(common_time) < MIN_OVERLAP_POINTS:
        raise ValueError(
            f"After NaN removal, only {len(common_time)} points remain; "
            f"need ≥ {MIN_OVERLAP_POINTS}."
        )

    return common_time, rppg_aligned, watch_aligned


# ======================================================================
# Statistical Validation
# ======================================================================

def compute_validation_metrics(
    rppg_bpm: np.ndarray,
    watch_bpm: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive agreement statistics between two HR signals.

    Metrics follow IEEE and clinical validation standards:
        - **MAE**: Mean Absolute Error — average magnitude of BPM deviation.
        - **RMSE**: Root Mean Square Error — penalises large deviations.
        - **Pearson r**: Linear correlation coefficient (0–1 = poor–perfect).
        - **Mean Bias**: Systematic over/under-estimation.
        - **LoA Upper/Lower**: 95% Limits of Agreement (Bland-Altman).

    Parameters
    ----------
    rppg_bpm : numpy.ndarray
        rPPG-derived BPM values.
    watch_bpm : numpy.ndarray
        Ground-truth smartwatch BPM values.

    Returns
    -------
    dict[str, float]
        Dictionary of validation metrics.
    """
    diff = rppg_bpm - watch_bpm

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Pearson correlation.
    if np.std(rppg_bpm) > 0 and np.std(watch_bpm) > 0:
        r, p_value = sp_stats.pearsonr(rppg_bpm, watch_bpm)
    else:
        r, p_value = float("nan"), float("nan")

    # Bland-Altman statistics.
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    loa_upper = mean_bias + 1.96 * std_diff
    loa_lower = mean_bias - 1.96 * std_diff

    # Mean Absolute Percentage Error.
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(diff / watch_bpm)) * 100)

    metrics = {
        "mae_bpm": round(mae, 3),
        "rmse_bpm": round(rmse, 3),
        "pearson_r": round(float(r), 4),
        "pearson_p_value": round(float(p_value), 6),
        "mean_bias_bpm": round(mean_bias, 3),
        "loa_upper_bpm": round(loa_upper, 3),
        "loa_lower_bpm": round(loa_lower, 3),
        "mape_percent": round(mape, 2),
        "n_samples": int(len(rppg_bpm)),
    }

    return metrics


# ======================================================================
# Visualization
# ======================================================================

def generate_validation_plots(
    common_time: np.ndarray,
    rppg_bpm: np.ndarray,
    watch_bpm: np.ndarray,
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate an IEEE-publication-ready 2-panel validation figure.

    Panel 1 — **Time-Series Overlay**: Both BPM signals plotted on a shared
    time axis with the inter-signal error shaded.

    Panel 2 — **Bland-Altman Plot**: Difference vs. average with mean bias
    line and 95% Limits of Agreement (LoA) — the gold standard for
    clinical method-comparison studies.

    Parameters
    ----------
    common_time : numpy.ndarray
        Aligned time axis (seconds).
    rppg_bpm : numpy.ndarray
        rPPG BPM values.
    watch_bpm : numpy.ndarray
        Smartwatch BPM values.
    metrics : dict[str, float]
        Pre-computed validation metrics.
    output_path : Path
        Output file path (saved at IEEE_DPI resolution).
    """
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), dpi=IEEE_DPI,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    fig.suptitle(
        "Project SENTIO — Phase 3: rPPG vs. Smartwatch Validation",
        fontsize=15, fontweight="bold", color="#00FFD0", y=0.98,
    )

    # ── Colour palette ───────────────────────────────────────────────
    rppg_colour = "#42A5F5"        # Blue.
    watch_colour = "#EF5350"       # Red.
    fill_colour = "#7E57C2"        # Purple shading.
    bias_colour = "#FFD740"        # Amber.
    loa_colour = "#FF6D00"         # Orange.
    grid_colour = "#2A2A2A"
    scatter_colour = "#26C6DA"     # Teal.

    # ── Panel 1: Time-Series Overlay ─────────────────────────────────
    ax1.plot(
        common_time, rppg_bpm,
        color=rppg_colour, linewidth=1.2, alpha=0.95, label="rPPG (SENTIO)",
    )
    ax1.plot(
        common_time, watch_bpm,
        color=watch_colour, linewidth=1.0, alpha=0.85,
        linestyle="--", label="Smartwatch (Ground Truth)",
    )

    # Shade the error region.
    ax1.fill_between(
        common_time, rppg_bpm, watch_bpm,
        alpha=0.15, color=fill_colour, label="Error margin",
    )

    ax1.set_title(
        "Heart Rate: rPPG vs. Smartwatch",
        fontsize=12, color="#CCCCCC", pad=10,
    )
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Heart Rate (BPM)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax1.grid(True, alpha=0.3, color=grid_colour)
    ax1.tick_params(colors="#999999")

    # Annotate MAE on the plot.
    stats_text = (
        f"MAE = {metrics['mae_bpm']:.1f} BPM  |  "
        f"RMSE = {metrics['rmse_bpm']:.1f} BPM  |  "
        f"r = {metrics['pearson_r']:.3f}"
    )
    ax1.text(
        0.02, 0.95, stats_text,
        transform=ax1.transAxes, fontsize=9, color="#00FFD0",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A2E",
                  edgecolor="#00FFD0", alpha=0.85),
    )

    # ── Panel 2: Bland-Altman Plot ───────────────────────────────────
    mean_vals = (rppg_bpm + watch_bpm) / 2.0
    diff_vals = rppg_bpm - watch_bpm

    mean_bias = metrics["mean_bias_bpm"]
    loa_upper = metrics["loa_upper_bpm"]
    loa_lower = metrics["loa_lower_bpm"]

    ax2.scatter(
        mean_vals, diff_vals,
        color=scatter_colour, s=12, alpha=0.6, edgecolors="none",
    )

    # Mean bias line.
    ax2.axhline(
        y=mean_bias, color=bias_colour, linestyle="-",
        linewidth=1.5, alpha=0.9, label=f"Mean bias: {mean_bias:+.2f} BPM",
    )

    # 95% Limits of Agreement.
    ax2.axhline(
        y=loa_upper, color=loa_colour, linestyle="--",
        linewidth=1.0, alpha=0.8, label=f"+1.96 SD: {loa_upper:+.2f} BPM",
    )
    ax2.axhline(
        y=loa_lower, color=loa_colour, linestyle="--",
        linewidth=1.0, alpha=0.8, label=f"−1.96 SD: {loa_lower:+.2f} BPM",
    )

    # Shade the LoA band.
    ax2.axhspan(loa_lower, loa_upper, alpha=0.08, color=loa_colour)

    # Zero-difference reference.
    ax2.axhline(y=0, color="#555555", linestyle=":", linewidth=0.8, alpha=0.5)

    ax2.set_title(
        "Bland-Altman Plot (Method Agreement Analysis)",
        fontsize=12, color="#CCCCCC", pad=10,
    )
    ax2.set_xlabel("Mean of rPPG & Watch (BPM)", fontsize=10)
    ax2.set_ylabel("Difference: rPPG − Watch (BPM)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax2.grid(True, alpha=0.3, color=grid_colour)
    ax2.tick_params(colors="#999999")

    # ── Footer metrics bar ───────────────────────────────────────────
    footer = (
        f"N = {metrics['n_samples']}  |  "
        f"MAE = {metrics['mae_bpm']:.2f} BPM  |  "
        f"RMSE = {metrics['rmse_bpm']:.2f} BPM  |  "
        f"Pearson r = {metrics['pearson_r']:.4f} (p = {metrics['pearson_p_value']:.2e})  |  "
        f"MAPE = {metrics['mape_percent']:.1f}%"
    )
    fig.text(
        0.5, 0.01, footer, ha="center", va="bottom",
        fontsize=9, color="#00FFD0",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A1A2E",
                  edgecolor="#00FFD0", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(
        output_path, dpi=IEEE_DPI, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  ✓ Validation plot saved → {output_path.resolve()}")


# ======================================================================
# JSON Export
# ======================================================================

def export_metrics_json(
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Export validation metrics to a JSON file for IEEE paper integration.

    Parameters
    ----------
    metrics : dict[str, float]
        Validation statistics.
    output_path : Path
        Output JSON file path.
    """
    payload = {
        "project": "SENTIO",
        "phase": "Phase 3 — Cross-Modal Validation",
        "metrics": metrics,
        "interpretation": {
            "mae_bpm": _interpret_mae(metrics["mae_bpm"]),
            "pearson_r": _interpret_pearson(metrics["pearson_r"]),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Metrics JSON saved → {output_path.resolve()}")


def _interpret_mae(mae: float) -> str:
    """Provide a clinical interpretation of the MAE value."""
    if mae < 3.0:
        return "Excellent — within clinical-grade accuracy (< 3 BPM)"
    elif mae < 5.0:
        return "Good — acceptable for consumer wellness applications"
    elif mae < 10.0:
        return "Moderate — suitable for trend monitoring only"
    else:
        return "Poor — not suitable for heart rate estimation"


def _interpret_pearson(r: float) -> str:
    """Provide a clinical interpretation of the Pearson r value."""
    if np.isnan(r):
        return "Undefined — insufficient variance"
    r_abs = abs(r)
    if r_abs >= 0.9:
        return "Very strong linear agreement"
    elif r_abs >= 0.7:
        return "Strong linear agreement"
    elif r_abs >= 0.5:
        return "Moderate linear agreement"
    elif r_abs >= 0.3:
        return "Weak linear agreement"
    else:
        return "Negligible linear agreement"


# ======================================================================
# Internal DSP helpers (lightweight re-extraction for standalone use)
# ======================================================================

def _compute_fs(timestamps: np.ndarray) -> float:
    """Compute sampling rate from timestamp array."""
    dt = np.diff(timestamps)
    return float(1.0 / np.mean(dt))


def _extract_instantaneous_bpm(
    raw_green: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight BPM extraction (mirrors Phase 2 logic).

    Returns time points and instantaneous BPM values for each inter-beat
    interval detected in the Green channel.

    Parameters
    ----------
    raw_green : numpy.ndarray
        Raw Green channel spatial means.
    timestamps : numpy.ndarray
        Monotonic timestamps.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(bpm_times, bpm_values)`` — midpoint times and instantaneous BPM.
    """
    from scipy.signal import butter, detrend, filtfilt, find_peaks

    # Detrend.
    sig = detrend(raw_green, type="linear")
    window = max(int(fs * 1.5), 3)
    if window % 2 == 0:
        window += 1
    rolling_mean = pd.Series(sig).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    sig = sig - rolling_mean

    # Bandpass filter (0.7–3.0 Hz).
    nyq = fs / 2.0
    high = min(3.0, nyq * 0.95)
    low = 0.7
    if low >= high:
        return np.array([]), np.array([])

    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    pad = min(3 * max(len(b), len(a)), len(sig) - 1)
    filtered = filtfilt(b, a, sig, padlen=pad)

    # Peak detection.
    min_dist = max(int(fs * (60.0 / 180.0)), 1)
    prominence = 0.3 * np.std(filtered)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=prominence)

    if len(peaks) < 2:
        return np.array([]), np.array([])

    # Relative timestamps.
    t_rel = timestamps - timestamps[0]
    peak_times = t_rel[peaks]
    ibi_s = np.diff(peak_times)

    with np.errstate(divide="ignore", invalid="ignore"):
        inst_bpm = 60.0 / ibi_s

    # Midpoint times for each BPM value.
    bpm_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    # Filter physiologically impossible values.
    valid = (inst_bpm >= 35) & (inst_bpm <= 210)
    return bpm_times[valid], inst_bpm[valid]


# ======================================================================
# Main Pipeline
# ======================================================================

def run_validation(
    rppg_csv: Path,
    watch_csv: Path,
    sync_offset: float,
    output_plot: Path,
    output_json: Path,
    generate_demo: bool = False,
) -> Dict[str, float]:
    """Execute the complete Phase 3 validation pipeline.

    Parameters
    ----------
    rppg_csv : Path
        Path to Phase 1 raw vitals CSV.
    watch_csv : Path
        Path to smartwatch ground-truth CSV.
    sync_offset : float
        Timestamp offset in seconds (positive = watch started later).
    output_plot : Path
        Output path for the validation figure.
    output_json : Path
        Output path for the metrics JSON.
    generate_demo : bool
        If True, generate synthetic smartwatch data for testing.

    Returns
    -------
    dict[str, float]
        Computed validation metrics.
    """
    sep = "=" * 62

    print(f"\n{sep}")
    print("  PROJECT SENTIO — Phase 3: Statistical Validation")
    print(f"{sep}\n")

    # ── 1. Load rPPG data ────────────────────────────────────────────
    print("  Loading rPPG data...")
    try:
        rppg_df = load_rppg_bpm(rppg_csv)
        print(f"  ✓ rPPG: {len(rppg_df)} BPM data points loaded")
        print(f"    Time range: {rppg_df['time_s'].iloc[0]:.1f} – {rppg_df['time_s'].iloc[-1]:.1f} s")
    except (FileNotFoundError, ValueError) as e:
        print(f"  ✗ rPPG load failed: {e}")
        sys.exit(1)

    # ── 2. Load or generate smartwatch data ──────────────────────────
    if generate_demo:
        print("\n  Generating demo smartwatch data...")
        watch_df = generate_demo_watch_data(rppg_df, watch_csv)
    else:
        print("\n  Loading smartwatch ground-truth data...")
        try:
            watch_df = load_smartwatch_data(watch_csv)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ Smartwatch load failed: {e}")
            print(
                "\n  TIP: Use --generate-demo to create synthetic smartwatch data"
                "\n       for pipeline testing before real data collection.\n"
            )
            sys.exit(1)

    print(f"  ✓ Smartwatch: {len(watch_df)} data points loaded")
    print(f"    Time range: {watch_df['time_s'].iloc[0]:.1f} – {watch_df['time_s'].iloc[-1]:.1f} s")

    # ── 3. Apply sync offset ─────────────────────────────────────────
    if sync_offset != 0:
        print(f"\n  Applying sync offset: {sync_offset:+.2f} s")
        watch_df = apply_sync_offset(watch_df, sync_offset)

    # ── 4. Resample to common axis ───────────────────────────────────
    print("\n  Resampling to common time axis (cubic interpolation)...")
    try:
        common_time, rppg_aligned, watch_aligned = resample_to_common_axis(
            rppg_df, watch_df
        )
        duration = common_time[-1] - common_time[0]
        print(f"  ✓ Aligned {len(common_time)} samples over {duration:.1f} s")
    except ValueError as e:
        print(f"  ✗ Resampling failed: {e}")
        sys.exit(1)

    # ── 5. Compute validation metrics ────────────────────────────────
    print("\n  Computing validation statistics...")
    metrics = compute_validation_metrics(rppg_aligned, watch_aligned)

    print(f"\n{sep}")
    print("  VALIDATION RESULTS")
    print(f"{sep}")
    print(f"  Mean Absolute Error (MAE)  : {metrics['mae_bpm']:.2f} BPM")
    print(f"  Root Mean Square Error     : {metrics['rmse_bpm']:.2f} BPM")
    print(f"  Pearson Correlation (r)    : {metrics['pearson_r']:.4f}")
    print(f"    p-value                  : {metrics['pearson_p_value']:.2e}")
    print(f"  Mean Bias                  : {metrics['mean_bias_bpm']:+.2f} BPM")
    print(f"  95% LoA                    : [{metrics['loa_lower_bpm']:+.2f}, {metrics['loa_upper_bpm']:+.2f}] BPM")
    print(f"  MAPE                       : {metrics['mape_percent']:.1f}%")
    print(f"  Samples compared           : {metrics['n_samples']}")
    print(f"{sep}")
    print(f"\n  Accuracy    : {_interpret_mae(metrics['mae_bpm'])}")
    print(f"  Agreement   : {_interpret_pearson(metrics['pearson_r'])}\n")

    # ── 6. Generate plots ────────────────────────────────────────────
    print("  Generating IEEE-quality validation plots...")
    generate_validation_plots(
        common_time, rppg_aligned, watch_aligned, metrics, output_plot
    )

    # ── 7. Export JSON ───────────────────────────────────────────────
    export_metrics_json(metrics, output_json)

    print("\n  ✓ Phase 3 validation complete.\n")

    return metrics


# ======================================================================
# Entry point
# ======================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validation engine."""
    parser = argparse.ArgumentParser(
        prog="SENTIO Phase 3",
        description="Cross-modal rPPG vs. smartwatch validation engine.",
    )
    parser.add_argument(
        "--rppg",
        type=str,
        default=str(DATA_DIR / "phase1_raw_vitals.csv"),
        help="Path to Phase 1 rPPG CSV (default: phase1_raw_vitals.csv).",
    )
    parser.add_argument(
        "--watch",
        type=str,
        default=str(DATA_DIR / "smartwatch_data.csv"),
        help="Path to smartwatch ground-truth CSV (default: smartwatch_data.csv).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Sync offset in seconds: positive = watch started later (default: 0.0).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(MANUSCRIPT_DIR / "phase3_validation_plots.png"),
        help="Output path for the validation figure.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DATA_DIR / "validation_metrics.json"),
        help="Output path for the metrics JSON.",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate synthetic smartwatch data from rPPG signal for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_validation(
        rppg_csv=Path(args.rppg),
        watch_csv=Path(args.watch),
        sync_offset=args.offset,
        output_plot=Path(args.output_plot),
        output_json=Path(args.output_json),
        generate_demo=args.generate_demo,
    )
