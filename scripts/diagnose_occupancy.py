"""Audit cached occupancy data and causal controls without downloads or training.

Usage:
    py -3.11 scripts/diagnose_occupancy.py --output-dir results/<run>/occupancy_diagnostics
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import DATASET_CONFIGS, load_occupancy
from src.sequences import create_sequences
from src.split import chronological_split


def target_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarize finite targets, using population variance for metric context."""
    flat = np.asarray(values, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    if not len(finite):
        return {"count": 0, "nonfinite_count": len(flat)}
    return {
        "count": len(finite), "nonfinite_count": int(len(flat) - len(finite)),
        "mean": float(finite.mean()), "std_population": float(finite.std()),
        "variance_population": float(finite.var()), "min": float(finite.min()),
        "max": float(finite.max()),
    }


def original_unit_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Calculate pooled multi-horizon metrics in ppm; constant-target R2 is null."""
    if not target.size:
        return {}
    squared_error = (prediction - target) ** 2
    denominator = float(((target - target.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(squared_error.mean())),
        "mae": float(np.abs(prediction - target).mean()),
        "r2": float(1 - squared_error.sum() / denominator) if denominator > 0 else None,
        "mape": float((np.abs(prediction - target) / np.maximum(np.abs(target), 1)).mean() * 100),
    }


def evaluate_split(
    frame: pd.DataFrame, train_mean: float, lookback: int = 24, horizon: int = 6,
    season_hours: int = 24,
) -> dict[str, Any]:
    """Evaluate controls on shared finite windows within one chronological split.

    Args:
        frame: Sensor rows for a single split with unique ascending timestamps.
        train_mean: Mean target fitted only on training rows.
        lookback: Number of input rows, matching the existing benchmark.
        horizon: Number of target rows, matching the existing benchmark.
        season_hours: Exact elapsed-hour seasonal lag, not a row offset.

    Returns:
        Window coverage, target distributions, elapsed-time diagnostics, and
        metrics on common causal windows and their strictly hourly subset.
    """
    if lookback < 1 or horizon < 1 or season_hours < 1:
        raise ValueError("Window lengths and seasonal lag must be positive")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Timestamps must be unique and ascending")
    x, y, origins = create_sequences(frame, lookback, horizon, target_col="CO2")
    target_column = frame.columns.get_loc("CO2")
    target_series = frame["CO2"]
    seasonal = np.full_like(y, np.nan, dtype=float)
    contiguous = np.ones(len(origins), dtype=bool)
    elapsed_horizons = []
    for row, origin in enumerate(origins):
        target_times = frame.index[origin + 1:origin + 1 + horizon]
        source_times = target_times - pd.Timedelta(hours=season_hours)
        # A bin becomes available at its right edge; only completed input bins
        # may supply forecasts. Never read another split or future target bins.
        available = source_times <= frame.index[origin]
        seasonal[row] = target_series.reindex(source_times).to_numpy()
        seasonal[row, ~available] = np.nan
        times = frame.index[origin - lookback + 1:origin + 1 + horizon]
        contiguous[row] = bool(np.all(np.diff(times.asi8) == pd.Timedelta(hours=1).value))
        elapsed_horizons.append((target_times[-1] - frame.index[origin]).total_seconds() / 3600)
    finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    common = finite & np.isfinite(seasonal).all(axis=1)
    seasonal_name = f"seasonal_{season_hours}h"
    predictions = {
        "persistence": np.repeat(x[:, -1, target_column:target_column + 1], horizon, axis=1),
        seasonal_name: seasonal,
        "train_mean": np.full_like(y, train_mean, dtype=float),
    }
    report = {
        "candidate_row_windows": max(0, len(frame) - lookback - horizon + 1),
        "legacy_nan_valid_windows": len(origins),
        "finite_windows": int(finite.sum()),
        "windows_crossing_time_gaps": int((~contiguous).sum()),
        "last_horizon_elapsed_hours": target_summary(np.asarray(elapsed_horizons)),
        "legacy_evaluation_targets": target_summary(y[finite]),
        "all_finite_window_controls": {
            name: original_unit_metrics(prediction[finite], y[finite])
            for name, prediction in predictions.items() if name != seasonal_name
        },
    }
    for label, selected in [("common_causal_windows", common),
                            ("common_contiguous_hourly_windows", common & contiguous)]:
        report[label] = {
            "windows": int(selected.sum()),
            "first_origin_bin": str(frame.index[origins[selected][0]]) if selected.any() else None,
            "last_origin_bin": str(frame.index[origins[selected][-1]]) if selected.any() else None,
            "target_summary": target_summary(y[selected]),
            "metrics": {name: original_unit_metrics(prediction[selected], y[selected])
                        for name, prediction in predictions.items()},
        }
    return report


def diagnose(data_dir: Path) -> dict[str, Any]:
    """Read an existing occupancy cache and produce a reproducible local audit.

    Args:
        data_dir: Existing data/raw directory containing an extracted cache.

    Returns:
        JSON-compatible input fingerprints, data diagnostics, and controls.

    Raises:
        FileNotFoundError: If the complete local cache is unavailable; no
            download or extraction is attempted.
    """
    cache = data_dir / "occupancy"
    extracted = cache / "extracted"
    files = sorted(extracted.rglob("*.txt")) + sorted(extracted.rglob("*.csv"))
    if not (cache / "occupancy.zip").is_file() or not files:
        raise FileNotFoundError("Occupancy ZIP and extracted CSV/TXT cache required; downloads disabled")
    inputs, raw_frames = [], []
    for path in files:
        raw = pd.read_csv(path)
        if "CO2" not in raw or "date" not in raw:
            continue
        raw["date"] = pd.to_datetime(raw["date"])
        inputs.append({"path": str(path.resolve()), "rows": len(raw),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "start": str(raw["date"].min()), "end": str(raw["date"].max())})
        raw_frames.append(raw.set_index("date"))
    if not raw_frames:
        raise ValueError("No cached occupancy data files with date and CO2 columns")
    raw = pd.concat(raw_frames).sort_index()
    duplicates = int(raw.index.duplicated().sum())
    raw = raw[~raw.index.duplicated(keep="first")]
    frame = load_occupancy(data_dir=data_dir)
    counts = raw["CO2"].resample("1h").count()
    deltas = frame.index.to_series().diff().dt.total_seconds().div(3600)
    gaps = [{"previous_bin": str(frame.index[i - 1]), "next_bin": str(frame.index[i]),
             "elapsed_hours": float(deltas.iloc[i]), "missing_bins": int(deltas.iloc[i] - 1)}
            for i in range(1, len(frame)) if deltas.iloc[i] != 1]
    config = DATASET_CONFIGS["occupancy"]
    splits = chronological_split(frame, ratios=config["split_ratios"])
    train_mean = float(splits[0]["CO2"].mean())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              text=True, capture_output=True, check=True).stdout.strip()
    report = {
        "evidence": "Fresh deterministic diagnostics of local cached data; no neural training",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "code_revision": revision,
                        "source_sha256": {
                            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [Path(__file__), PROJECT_ROOT / "src/data_loading.py",
                                         PROJECT_ROOT / "src/preprocessing.py", PROJECT_ROOT / "src/sequences.py",
                                         PROJECT_ROOT / "src/split.py"]}},
        "config": {"split_ratios": config["split_ratios"], "lookback_rows": config["lookback"],
                   "horizon_rows": config["horizon"], "target": "CO2", "units": "ppm",
                   "seasonal_lag_elapsed_hours": 24, "random_seed": None},
        "inputs": inputs,
        "raw": {"deduplicated_rows": len(raw), "duplicate_timestamps_removed": duplicates,
                "nan_values_by_column": raw.isna().sum().astype(int).to_dict()},
        "hourly": {"rows": len(frame), "expected_bins": len(counts),
                   "empty_co2_bins": int((counts == 0).sum()),
                   "observed_bin_sample_counts": target_summary(counts[counts > 0].to_numpy()),
                   "nan_values_by_column": frame.isna().sum().astype(int).to_dict(), "gaps": gaps},
        "semantics": {
            "hourly_bin": "Left-labeled hourly mean available at label + 1 hour",
            "windows": "Legacy windows count retained rows; gaps do not invalidate them",
            "common_windows": "Finite legacy windows with all exact t-24h sources in this split and known at origin",
            "metrics": "Flattened overlapping horizon targets; repeated targets count repeatedly; R2 null if constant",
            "imputation": "Occupancy loader does not impute; empty hourly bins are dropped",
            "leakage_audit": "No smoothing, interpolation or backfill in occupancy benchmark; normalization fits train only",
            "training_controls": "Training-split metrics are in-sample diagnostics; full-train mean is causal only for held-out splits",
        },
        "splits": {},
    }
    for name, split in zip(["train", "validation", "test"], splits):
        report["splits"][name] = {
            "rows": len(split), "start": str(split.index.min()), "end": str(split.index.max()),
            "target_summary": target_summary(split["CO2"].to_numpy()),
            "controls": evaluate_split(split, train_mean, config["lookback"], config["horizon"]),
        }
    return report


def main() -> None:
    """Write cache-only diagnostics to a new or explicitly selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "occupancy_diagnostics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved diagnostics: {output}")


if __name__ == "__main__":
    main()
