"""Validate persistence residual models against unchanged standard synthetic gates.

This checks implementation readiness, not real-world usefulness. The historical
local-calendar/DST condition is deliberately preserved. The fresh base VM uses
the original standard training routine; residual models additionally allow the
epoch-zero persistence checkpoint to compete on validation data.
"""

import argparse
import hashlib
import math
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

from scripts.benchmark_standard_dataset import (
    THRESHOLDS, _train_model, check_regression, check_thresholds,
    compute_benchmark_metrics, generate_synthetic_co2, prepare_benchmark_data,
    prepare_temporal_data, seed_everything,
)
from scripts.controlled_temporal_comparison import source_manifest, write_json
from src.baselines import LSTMForecaster
from src.forecast_training import batched_predict, train_forecaster
from src.forecasting import PersistenceResidualForecaster
from src.sincos import SinCosForecaster
from src.split import chronological_split
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster

CANDIDATES = ("ResidualLSTM", "ResidualSinCos", "ResidualTemporal")
MODELS = ("VonMises", *CANDIDATES)


def model_config(name: str) -> dict[str, Any]:
    """Return fixed, reconstructable constructor and wrapper settings."""
    common = {"hidden_size": 64, "dropout": 0.1, "output_size": 12}
    if name == "VonMises":
        return {"class": "VonMisesForecaster", "body": {
            **common, "input_size": 4, "num_vm_layers": 2, "init_kappa": 1.0}}
    if name == "ResidualLSTM":
        body = {**common, "hidden_size": 68, "input_size": 6, "num_layers": 2}
        cls = "LSTMForecaster"
    elif name == "ResidualSinCos":
        body = {**common, "input_size": 6, "num_vm_layers": 2, "init_kappa": 1.0}
        cls = "SinCosForecaster"
    elif name == "ResidualTemporal":
        body = {**common, "sensor_size": 4, "time_periods": [24, 168],
                "num_vm_layers": 2, "freq_bands": [0.5, 2.0, 8.0],
                "use_temporal_conv": True, "use_lstm": True,
                "use_time_conditioning": True, "learnable_kappa": False}
        cls = "TemporalVonMisesForecaster"
    else:
        raise ValueError(f"Unknown model: {name}")
    return {"class": cls, "body": body, "wrapper": {
        "sensor_size": 4, "target_index": 2, "zero_init_output": True}}


def build_model(name: str) -> torch.nn.Module:
    """Construct a fixed standard-sized candidate or the original base VM."""
    config = model_config(name)
    classes = {"VonMisesForecaster": VonMisesForecaster,
               "LSTMForecaster": LSTMForecaster,
               "SinCosForecaster": SinCosForecaster,
               "TemporalVonMisesForecaster": TemporalVonMisesForecaster}
    body = classes[config["class"]](**config["body"])
    return (PersistenceResidualForecaster(body, **config["wrapper"])
            if "wrapper" in config else body)


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require completed finite absolute passes and temporal base regression."""
    lookup = {row["model"]: row for row in rows}
    acceptance = {}
    for name in MODELS:
        row = lookup.get(name)
        metrics = {metric: (row or {}).get("metrics", {}).get(metric)
                   for metric in THRESHOLDS}
        metrics = {key: math.nan if value is None else value for key, value in metrics.items()}
        passed, failures = check_thresholds(metrics)
        completed = row is not None and row.get("execution_status") == "completed"
        if not completed:
            failures.append("Run did not complete successfully")
        acceptance[name] = {"passed": passed and completed,
                            "absolute_passed": passed, "failures": failures}
    def get_metrics(name: str) -> dict[str, float]:
        values = lookup.get(name, {}).get("metrics", {})
        return {key: math.nan if values.get(key) is None else values[key]
                for key in THRESHOLDS}
    regression, failures = check_regression(get_metrics("ResidualTemporal"), get_metrics("VonMises"))
    # A numerical/training failure in the reference cannot certify regression.
    regression = regression and acceptance["VonMises"]["passed"]
    if not acceptance["VonMises"]["passed"]:
        failures.append("Fresh base VM did not pass its standard absolute gates")
    temporal = acceptance["ResidualTemporal"]
    temporal["regression_vs_base"] = regression
    temporal["failures"].extend(failures)
    temporal["passed"] = temporal["passed"] and regression
    return {"all_runs_attempted": set(lookup) == set(MODELS),
            "all_passed": all(item["passed"] for item in acceptance.values()),
            "acceptance": acceptance,
            "candidates": {name: acceptance[name]["passed"] for name in CANDIDATES},
            "temporal_regression": {"passed": regression, "failures": failures}}


def _fit(name: str, model: torch.nn.Module, data: dict[str, Any],
         epochs: int, device: torch.device) -> torch.nn.Module:
    """Preserve original base fitting; use epoch-zero selection for residuals."""
    model.to(device)
    keys = ("X_train", "y_train", "X_val", "y_val")
    if name == "VonMises":
        # Identical construction and uninterrupted RNG order to benchmark_von_mises.
        return _train_model(model, *(data[key].to(device) for key in keys), max_epochs=epochs)
    seed_everything(42)
    return train_forecaster(model, *(data[key] for key in keys), max_epochs=epochs,
                            device=device, source_config=model_config(name))


def run_validation(output_dir: Path, epochs: int = 50) -> dict[str, Any]:
    """Train and archive the fixed protocol, preserving partial failure evidence.

    Args:
        output_dir: A new directory; existing paths are rejected.
        epochs: Positive maximum epochs, defaulting to the standard budget.

    Returns:
        Acceptance checks and per-candidate operational eligibility.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"status": "running", "seed": 42, "data_seed": 42,
        "epochs": epochs, "n_days": 90, "lookback": 72, "horizon": 12,
        "target": "CO2", "units": "ppm", "calendar_clock": "local",
        "calendar_timezone": "Europe/Paris", "time_periods": [24, 168],
        "thresholds": THRESHOLDS, "models": {name: model_config(name) for name in MODELS},
        "training_rng": "base: uninterrupted constructor/training; residuals: reset after construction",
        "scope": "synthetic implementation readiness; no real-world usefulness claim"}
    rows: list[dict[str, Any]] = []
    def persist() -> dict[str, Any]:
        validation = validation_summary(rows)
        write_json(output_dir / "results.json", {"runs": rows, **validation})
        write_json(output_dir / "validation.json", validation)
        write_json(output_dir / "manifest.json", manifest)
        return validation
    persist()
    try:
        manifest["environment"] = source_manifest()
        with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for pattern in ["src/*.py", "scripts/*.py", "tests/*.py", "requirements.txt"]:
                for path in sorted(ROOT.glob(pattern)):
                    archive.write(path, path.relative_to(ROOT))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame = generate_synthetic_co2(n_days=90, seed=42)
        cpu = torch.device("cpu")
        base = prepare_benchmark_data(frame, 72, 12, cpu)
        temporal = prepare_temporal_data(frame, 72, 12, cpu, time_periods=[24, 168])
        for split in ("train", "val", "test"):
            torch.testing.assert_close(base[f"y_{split}"], temporal[f"y_{split}"], rtol=0, atol=0)
        manifest.update(normalization=base["norm_stats"], sensor_columns=list(frame.columns),
            data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
            split={name: {"rows": len(part), "first": str(part.index[0]),
                          "last": str(part.index[-1]), "windows": len(base[f"X_{name}"])}
                   for name, part in zip(("train", "val", "test"),
                                         chronological_split(frame, [0.8, 0.1, 0.1]))})
        persist()
        for name in MODELS:
            data = base if name == "VonMises" else temporal
            row: dict[str, Any] = {"model": name, "seed": 42, "execution_status": "running",
                                  "metrics": {}, "failures": [], "config": model_config(name)}
            rows.append(row)
            persist()
            model = None
            start = time.perf_counter()
            try:
                seed_everything(42)
                model = build_model(name)
                row["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model = _fit(name, model, data, epochs, device)
                history = model.training_history
                if not math.isfinite(history["best_val_loss"]) or not all(
                    math.isfinite(value) for value in history["validation_loss"]
                ):
                    raise FloatingPointError("Nonfinite validation history")
                prediction = batched_predict(model, data["X_test"])
                metrics = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
                row["metrics"] = metrics
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite evaluation metrics")
                np.savez_compressed(output_dir / f"{name}_42_predictions.npz",
                                    prediction=prediction.numpy(), target=data["y_test"].numpy())
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                           output_dir / f"{name}_42_state.pt")
                row["execution_status"] = "completed"
            except Exception as exc:
                row["execution_status"] = "numerical_failure" if isinstance(exc, FloatingPointError) else "failed"
                row["failures"].append(f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                row["execution_status"] = "interrupted"
                row["failures"].append(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                row["time_seconds"] = time.perf_counter() - start
                if model is not None:
                    row["training"] = getattr(model, "training_history", None)
                    if row["execution_status"] != "completed":
                        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                                   output_dir / f"{name}_42_failed_state.pt")
                persist()
            print(f"{name}: {row['execution_status']}; metrics={row['metrics']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        completed_utc=datetime.now(timezone.utc).isoformat())
        persist()
        raise
    validation = validation_summary(rows)
    manifest.update(status="completed", validation_status="passed" if validation["all_passed"] else "failed",
                    completed_utc=datetime.now(timezone.utc).isoformat())
    return persist()


def main(argv: list[str] | None = None) -> int:
    """Exit nonzero for any acceptance failure while retaining eligible models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    validation = run_validation(args.output_dir, args.epochs)
    print(f"Residual eligibility: {validation['candidates']}", flush=True)
    return 0 if validation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
