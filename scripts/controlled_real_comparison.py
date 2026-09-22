"""Compare occupancy forecasters on identical contiguous hourly windows.

The protocol retains the historical 70/15/15 row split, but rejects any window
whose input or forecast crosses a missing hour. No smoothing or imputation is
performed. Metrics describe this small fixed split; synthetic acceptance limits
do not apply to real occupancy CO2.
"""

import argparse
import hashlib
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_standard_dataset import _train_model, compute_benchmark_metrics, seed_everything
from scripts.controlled_temporal_comparison import build_model, source_manifest, write_json
from src.baselines import LSTMForecaster
from src.data_loading import load_occupancy
from src.preprocessing import normalize
from src.sequences import create_sequences
from src.split import chronological_split
from src.temporal import extract_angular_time_features

VARIANTS = ["VonMises", "SinCos", "TemporalVM", "VonMisesTime", "SinCosTime", "LSTM", "HandCrafted"]
METRICS = ["rmse", "mae", "r2", "mape"]


def distribution(values: np.ndarray) -> dict[str, float | int]:
    """Summarize finite scalar observations without treating windows as trials."""
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0}
    return {"count": int(finite.size), "mean": float(finite.mean()),
            "std": float(finite.std()), "min": float(finite.min()),
            "max": float(finite.max())}


def contiguous_sequences(
    frame: pd.DataFrame, lookback: int = 24, horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Filter finite sequences to require hourly input AND forecast timestamps.

    Args:
        frame: Sorted, unique hourly-indexed sensor frame with CO2 target.
        lookback: Number of observed hours.
        horizon: Number of future hourly bins to predict.

    Returns:
        Inputs, targets, last-input row positions, and filtering counts.
    """
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("A DatetimeIndex without NaT is required")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Timestamps must be sorted and unique")
    x, y, origins = create_sequences(frame, lookback, horizon, target_col="CO2")
    finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    hourly_steps = np.asarray(frame.index[1:] - frame.index[:-1]) == np.timedelta64(1, "h")
    contiguous = np.array([
        hourly_steps[origin - lookback + 1:origin + horizon].all() for origin in origins
    ], dtype=bool)
    keep = finite & contiguous
    counts = {"candidate_windows": max(0, len(frame) - lookback - horizon + 1),
              "finite_windows": int(finite.sum()), "valid_windows": int(keep.sum()),
              "gap_rejected_finite_windows": int((finite & ~contiguous).sum())}
    return x[keep], y[keep], origins[keep], counts


def prepare_occupancy_data(
    frame: pd.DataFrame, device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit train-only normalization and align sensor/calendar model inputs.

    All variants share the same origins and targets. HandCrafted sees explicit
    sine/cosine daily features; temporal and time-augmented VM models see raw
    daily angles. Calendar features are appended after sensor normalization.
    """
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Sensor values must not contain infinity")
    splits = chronological_split(frame, [0.7, 0.15, 0.15])
    _, stats = normalize(splits[0])
    if not np.isfinite(np.asarray(list(stats.values()))).all():
        raise ValueError("Training normalization statistics must be finite")
    shared = {"lookback": 24, "horizon": 6, "norm_stats": stats,
              "target_col": "CO2", "sensor_size": frame.shape[1], "time_periods": [24],
              "sensor_columns": list(frame.columns)}
    datasets = {name: dict(shared) for name in ("sensor", "temporal", "handcrafted")}
    metadata = {}
    for split_name, raw in zip(("train", "val", "test"), splits):
        normal, _ = normalize(raw, stats)
        angle = extract_angular_time_features(raw.index, [24])[:, 0]
        temporal = normal.assign(angle_24=angle)
        handcrafted = normal.assign(sin_24=np.sin(angle), cos_24=np.cos(angle))
        shared_origins = None
        for mode, processed in (("sensor", normal), ("temporal", temporal), ("handcrafted", handcrafted)):
            x, y, origins, counts = contiguous_sequences(processed)
            if not len(origins):
                raise ValueError(f"No valid contiguous hourly windows in {split_name}")
            if shared_origins is not None and not np.array_equal(origins, shared_origins):
                raise AssertionError("Model inputs have different forecast origins")
            shared_origins = origins
            data = datasets[mode]
            data[f"X_{split_name}"] = torch.tensor(x, device=device, dtype=torch.float32)
            data[f"y_{split_name}"] = torch.tensor(y, device=device, dtype=torch.float32)
            data[f"origins_{split_name}"] = raw.index[origins].to_numpy()
            data[f"target_times_{split_name}"] = raw.index.to_numpy()[origins[:, None] + np.arange(1, 7)]
        metadata[split_name] = {
            "rows": len(raw), "first": str(raw.index[0]), "last": str(raw.index[-1]),
            **counts, "row_target_distribution": distribution(raw["CO2"].to_numpy()),
            "forecast_target_distribution": distribution(y * stats["CO2"][1] + stats["CO2"][0]),
            "sensor_distributions": {col: distribution(raw[col].to_numpy()) for col in raw.columns},
            "first_valid_origin": str(raw.index[shared_origins[0]]),
            "last_valid_origin": str(raw.index[shared_origins[-1]]),
        }
    return datasets, metadata


def references(data: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Make persistence, previous-day, and train-mean predictions in z scores."""
    if data["lookback"] != 24 or data["horizon"] != 6:
        raise ValueError("Occupancy references require the fixed 24-hour lookback and 6-hour horizon")
    history = data["X_test"][..., data["sensor_columns"].index("CO2")]
    horizon = data["horizon"]
    return {"Persistence": history[:, -1:].expand(-1, horizon),
            "SeasonalPersistence24h": history[:, :horizon],
            "TrainMean": torch.zeros_like(data["y_test"])}


def checked_metrics(prediction: torch.Tensor, data: dict[str, Any]) -> dict[str, float]:
    """Compute finite original-scale metrics without synthetic success gates."""
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite predictions")
    result = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite metrics")
    return result


def summarize_real(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report variability across training seeds on the fixed test windows."""
    summary = {}
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [row for row in rows if row["model"] == name]
        summary[name] = {"n_seeds": len(group), "params": group[0]["params"],
                         "numerical_failures": sum(r.get("execution_status") == "numerical_failure" for r in group),
                         "execution_failures": sum(r.get("execution_status") in {"failed", "interrupted"} for r in group)}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in group
                      if row.get("execution_status", "completed") == "completed"
                      and row["metrics"][metric] is not None and np.isfinite(row["metrics"][metric])]
            summary[name][metric] = {"n_finite": len(values), "n_excluded": len(group) - len(values),
                                     "mean": float(np.mean(values)) if values else None,
                                     "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                     "min": float(min(values)) if values else None,
                                     "max": float(max(values)) if values else None}
    return summary


def save_predictions(path: Path, prediction: torch.Tensor, data: dict[str, Any]) -> None:
    """Save aligned forecasts and observed targets in ppm with hourly bin labels."""
    mean, std = data["norm_stats"]["CO2"]
    np.savez_compressed(path, prediction_ppm=(prediction.detach() * std + mean).cpu().numpy(),
                        target_ppm=(data["y_test"].detach() * std + mean).cpu().numpy(),
                        origin_bin_start=data["origins_test"],
                        target_bin_start=data["target_times_test"])


def main() -> None:
    """Run a reproducible occupancy comparison and checkpoint each completed fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--prepare-only", action="store_true", help="Save data audit and reference forecasts without fitting")
    args = parser.parse_args()
    if args.epochs <= 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Epochs must be positive; seeds and models must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "dataset": "occupancy",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        finite_passed = run_comparison(args, manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    if not finite_passed:
        raise SystemExit(1)


def run_comparison(args: argparse.Namespace, manifest: dict[str, Any]) -> bool:
    """Execute the local-cache-only protocol, retaining numerical failure rows."""
    cache = args.data_dir / "occupancy"
    extracted = cache / "extracted"
    if not (cache / "occupancy.zip").is_file() or not extracted.is_dir() or not any(
        path.is_file() and path.suffix.lower() in {".txt", ".csv"} for path in extracted.rglob("*")
    ):
        raise FileNotFoundError("Occupancy cache requires occupancy.zip and extracted CSV/TXT files; downloads are disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_occupancy(args.data_dir)
    datasets, split_metadata = prepare_occupancy_data(frame, device)
    data = datasets["sensor"]
    with zipfile.ZipFile(args.output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ["src/*.py", "scripts/*.py", "tests/*.py", "requirements.txt"]:
            for path in sorted(ROOT.glob(pattern)):
                archive.write(path, path.relative_to(ROOT))
    frame.to_csv(args.output_dir / "hourly_dataset.csv")
    manifest.update({
        "environment": source_manifest(), "dataset": "occupancy", "target": "CO2", "unit": "ppm",
        "training_seeds": args.seeds, "models": args.models, "max_epochs": args.epochs,
        "lookback": 24, "horizon": 6, "split_ratios": [0.7, 0.15, 0.15],
        "split_policy": "chronological retained rows, independently windowed splits",
        "gap_policy": "reject non-hourly input+target windows; no smoothing or imputation",
        "hourly_bin_semantics": "timestamps label bin starts; forecast origin availability is last input bin start + 1h",
        "split": split_metadata, "normalization": data["norm_stats"],
        "sensor_columns": list(frame.columns), "time_periods": [24],
        "time_semantics": "local wall-clock daily angles; fixed Unix epoch for any generic periods; no angle normalization",
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "raw_sha256": {str(path.relative_to(args.data_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((args.data_dir / "occupancy").rglob("*")) if path.is_file()},
        "training": {"rng": "reset seed after construction for each model", "optimizer": "Adam",
                     "lr": 0.001, "weight_decay": 0.0001, "batch_size": 64, "patience": 15,
                     "gradient_clip": 1.0, "validation_selection": "best validation MSE checkpoint"},
        "evaluation": "finite metrics only; no synthetic thresholds; seed SD is optimization variability on one fixed split",
        "status": "running",
    })
    write_json(args.output_dir / "manifest.json", manifest)
    reference_metrics = {}
    for name, prediction in references(data).items():
        reference_metrics[name] = checked_metrics(prediction, data)
        save_predictions(args.output_dir / f"{name}_predictions.npz", prediction, data)
    rows: list[dict[str, Any]] = []
    results = {"runs": rows, "summary": {}, "references": reference_metrics}
    write_json(args.output_dir / "results.json", results)
    if args.prepare_only:
        manifest["status"] = "prepared"
        write_json(args.output_dir / "manifest.json", manifest)
        return True
    try:
        for seed in args.seeds:
            for name in args.models:
                row = {"model": name, "seed": seed, "params": None,
                       "execution_status": "running", "metrics": {metric: None for metric in METRICS}}
                rows.append(row)
                mode = "handcrafted" if name == "HandCrafted" else (
                    "sensor" if name in {"VonMises", "SinCos", "LSTM"} else "temporal")
                model_data = datasets[mode]
                start = time.perf_counter()
                model = None
                try:
                    seed_everything(seed)
                    model = (LSTMForecaster(input_size=frame.shape[1], hidden_size=68,
                                            num_layers=2, dropout=0.1, output_size=6)
                             if name == "LSTM" else build_model(name, model_data)).to(device)
                    row.update(params=sum(p.numel() for p in model.parameters()),
                               architecture=repr(model), input_mode=mode)
                    seed_everything(seed)
                    model = _train_model(model, model_data["X_train"], model_data["y_train"],
                                         model_data["X_val"], model_data["y_val"], max_epochs=args.epochs)
                    row["training"] = model.training_history
                    if not np.isfinite(model.training_history["validation_loss"]).all():
                        raise FloatingPointError("Nonfinite validation history")
                    if not np.isfinite(model.training_history["best_val_loss"]):
                        raise FloatingPointError("No finite validation checkpoint")
                    with torch.no_grad():
                        prediction = model(model_data["X_test"])
                    row["metrics"] = checked_metrics(prediction, model_data)
                    save_predictions(args.output_dir / f"{name}_{seed}_predictions.npz", prediction, model_data)
                    torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                               args.output_dir / f"{name}_{seed}_state.pt")
                    row["execution_status"] = "completed"
                except FloatingPointError as error:
                    row.update(execution_status="numerical_failure", error=str(error))
                except BaseException as error:
                    row.update(execution_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                               error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    row["time_seconds"] = time.perf_counter() - start
                    results["summary"] = summarize_real(rows)
                    write_json(args.output_dir / "results.json", results)
                    del model
                print(f"{name} seed={seed} RMSE={row['metrics']['rmse']} ppm status={row['execution_status']}", flush=True)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    finite_passed = all(row["execution_status"] == "completed" for row in rows)
    manifest.update(status="completed" if finite_passed else "completed_with_numerical_failures",
                    finite_passed=finite_passed, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / "manifest.json", manifest)
    return finite_passed


if __name__ == "__main__":
    main()
