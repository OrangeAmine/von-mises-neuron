"""Validate versioned adapters using the unchanged standard forecasting gates.

This is an engineering regression check, not evidence for stage A/B hypotheses.
The historical local-calendar/DST condition and trainer remain unchanged. Every
model uses the fixed standard 50-epoch ceiling and seed 42. Failed candidates
and interrupted states remain archived; output directories are never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

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
from src.methodology_forecasters import DirectCalendarForecasterV1, PolarForecasterV2
from src.split import chronological_split
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster

CANDIDATES = ("PolarForecasterV2", "DirectCalendarForecasterV1")
MODELS = ("VonMises", *CANDIDATES)


def model_config(name: str) -> dict[str, Any]:
    """Return exact source constructors and explicit versioned conversions."""
    common = {"hidden_size": 64, "num_vm_layers": 2, "dropout": 0.1, "output_size": 12}
    if name in ("VonMises", "PolarForecasterV2"):
        source = {"class": "VonMisesForecaster", "kwargs": {**common, "input_size": 4, "init_kappa": 1.0}}
    elif name == "DirectCalendarForecasterV1":
        source = {"class": "TemporalVonMisesForecaster", "kwargs": {
            **common, "sensor_size": 4, "time_periods": [24, 168],
            "freq_bands": [0.5, 2.0, 8.0], "use_temporal_conv": True,
            "conv_kernel_size": 3, "use_lstm": True,
            "learnable_kappa": False, "use_time_conditioning": True}}
    else:
        raise ValueError(f"Unknown model: {name}")
    return {"class": name if name != "VonMises" else "VonMisesForecaster",
            "source": source, "conversion": None if name == "VonMises" else "from_legacy",
            "implementation": None if name == "VonMises" else "factored"}


def build_model(name: str) -> torch.nn.Module:
    """Initialize the exact standard source and explicitly convert its function."""
    config = model_config(name)
    source_class = {"VonMisesForecaster": VonMisesForecaster,
                    "TemporalVonMisesForecaster": TemporalVonMisesForecaster}[config["source"]["class"]]
    reference = source_class(**config["source"]["kwargs"])
    if name == "PolarForecasterV2":
        return PolarForecasterV2.from_legacy(reference)
    if name == "DirectCalendarForecasterV1":
        return DirectCalendarForecasterV1.from_legacy(reference)
    return reference


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require every absolute gate and temporal regression against fresh base VM."""
    lookup = {row["model"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) - set(MODELS):
        raise ValueError("model rows must have unique declared identifiers")

    def metrics(name: str) -> dict[str, float]:
        values = lookup.get(name, {}).get("metrics", {})
        return {key: math.nan if values.get(key) is None else values[key] for key in THRESHOLDS}

    acceptance = {}
    for name in MODELS:
        passed, failures = check_thresholds(metrics(name))
        completed = lookup.get(name, {}).get("execution_status") == "completed"
        if not completed:
            failures.append("Run did not complete successfully")
        acceptance[name] = {"passed": passed and completed,
                            "absolute_passed": passed, "failures": failures}
    regression, failures = check_regression(metrics("DirectCalendarForecasterV1"), metrics("VonMises"))
    if not acceptance["VonMises"]["passed"]:
        regression = False
        failures.append("Fresh base VM did not pass its standard absolute gates")
    temporal = acceptance["DirectCalendarForecasterV1"]
    temporal["regression_vs_base"] = regression
    temporal["passed"] = temporal["passed"] and regression
    temporal["failures"].extend(failures)
    return {"all_runs_attempted": all(lookup.get(name, {}).get("execution_status") not in (None, "pending") for name in MODELS),
            "all_passed": all(row["passed"] for row in acceptance.values()),
            "acceptance": acceptance,
            "candidates": {name: acceptance[name]["passed"] for name in CANDIDATES},
            "temporal_regression": {"passed": regression, "failures": failures}}


@contextmanager
def _progress(name: str) -> Iterator[None]:
    """Print ongoing fit progress without performing additional model work."""
    stop = threading.Event()
    started = time.perf_counter()

    def update() -> None:
        while not stop.wait(30):
            print(f"{name}: standard fit running ({time.perf_counter() - started:.0f}s)", flush=True)

    thread = threading.Thread(target=update, name="methodology-validator-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _fit(name: str, model: torch.nn.Module, data: dict[str, Any], epochs: int,
         device: torch.device) -> torch.nn.Module:
    """Use the original standard trainer and uninterrupted post-constructor RNG."""
    model.to(device)
    with _progress(name):
        return _train_model(model, *(data[key].to(device) for key in ("X_train", "y_train", "X_val", "y_val")), max_epochs=epochs)


def predict(model: torch.nn.Module, data: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Use the standard full-test forward pass, returning CPU predictions."""
    model.eval()
    with torch.no_grad():
        return model(data["X_test"].to(device)).detach().cpu()


def _save_state(path: Path, model: torch.nn.Module) -> None:
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)


def run_validation(output_dir: Path, epochs: int = 50) -> dict[str, Any]:
    """Run all three fixed-budget engineering fits and preserve every outcome.

    Args:
        output_dir: New output directory, rejected if it already exists.
        epochs: Fixed canonical engineering ceiling, exactly 50.

    Returns:
        Per-model checks and aggregate acceptance, including explicit failures.
    """
    if epochs != 50:
        raise ValueError("the standard methodology adapter budget is exactly 50 epochs")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = [{"model": name, "seed": 42, "execution_status": "pending",
                                "metrics": {}, "failures": [], "config": model_config(name)} for name in MODELS]
    manifest: dict[str, Any] = {
        "protocol": "vm_methodology_v1_engineering_forecasters", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "data_seed": 42, "epochs": epochs, "n_days": 90, "lookback": 72,
        "horizon": 12, "target": "CO2", "units": "ppm", "calendar_clock": "local",
        "calendar_timezone": "Europe/Paris", "time_periods": [24, 168],
        "thresholds": THRESHOLDS, "temporal_regression": {"error_relative_max": 0.1, "r2_drop_max": 0.1},
        "training_workers": 1, "models": {name: model_config(name) for name in MODELS},
        "training_rng": "seed42 before each source constructor; conversion draws no RNG; uninterrupted training RNG",
        "training_protocol": "unchanged benchmark_standard_dataset._train_model including native weight decay",
        "scope": "forecasting integration gates only; not stage A/B methodological evidence"}

    def persist() -> dict[str, Any]:
        summary = validation_summary(rows)
        write_json(output_dir / "results.json", {"runs": rows, **summary})
        write_json(output_dir / "validation.json", summary)
        write_json(output_dir / "manifest.json", manifest)
        return summary

    persist()
    try:
        manifest["environment"] = source_manifest()
        manifest["determinism"] = {"algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                    "cudnn_deterministic": torch.backends.cudnn.deterministic}
        snapshot_paths = sorted({path for pattern in ("src/*.py", "scripts/*.py", "tests/*.py", "configs/*.json", "requirements.txt") for path in ROOT.glob(pattern)})
        manifest["snapshot_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot_paths}
        with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in snapshot_paths:
                archive.write(path, path.relative_to(ROOT))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest["device"] = str(device)
        frame = generate_synthetic_co2(n_days=90, seed=42)
        cpu = torch.device("cpu")
        base = prepare_benchmark_data(frame, 72, 12, cpu)
        temporal = prepare_temporal_data(frame, 72, 12, cpu, time_periods=[24, 168])
        for split in ("train", "val", "test"):
            torch.testing.assert_close(base[f"y_{split}"], temporal[f"y_{split}"], rtol=0, atol=0)
        manifest.update(normalization=base["norm_stats"], sensor_columns=list(frame.columns),
            data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
            split={name: {"rows": len(part), "first": str(part.index[0]), "last": str(part.index[-1]),
                          "windows": len(base[f"X_{name}"])} for name, part in zip(("train", "val", "test"), chronological_split(frame, [0.8, 0.1, 0.1]))})
        persist()
        for row in rows:
            name = row["model"]
            data = temporal if name == "DirectCalendarForecasterV1" else base
            row["execution_status"] = "running"
            started = time.perf_counter()
            model = None
            persist()
            print(f"{name}: starting fixed standard validation", flush=True)
            try:
                seed_everything(42)
                model = build_model(name)
                row["params"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                _save_state(output_dir / f"{name}_42_initial_state.pt", model)
                model = _fit(name, model, data, epochs, device)
                history = model.training_history
                if not math.isfinite(history["best_val_loss"]) or not all(math.isfinite(value) for value in history["validation_loss"]):
                    raise FloatingPointError("Nonfinite validation history")
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                    raise FloatingPointError("Nonfinite model state")
                prediction = predict(model, data, device)
                row["metrics"] = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
                if not all(math.isfinite(value) for value in row["metrics"].values()):
                    raise FloatingPointError("Nonfinite evaluation metrics")
                np.savez_compressed(output_dir / f"{name}_42_predictions.npz", prediction=prediction.numpy(), target=data["y_test"].numpy())
                _save_state(output_dir / f"{name}_42_state.pt", model)
                row["execution_status"] = "completed"
            except Exception as exc:
                row["execution_status"] = "numerical_failure" if isinstance(exc, FloatingPointError) else "failed"
                row["failures"].append(f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                row["execution_status"] = "interrupted"
                row["failures"].append(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                row["time_seconds"] = time.perf_counter() - started
                if model is not None:
                    row["training"] = getattr(model, "training_history", None)
                    if row["execution_status"] != "completed":
                        _save_state(output_dir / f"{name}_42_failed_state.pt", model)
                persist()
            print(f"{name}: {row['execution_status']}; metrics={row['metrics']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_utc=datetime.now(timezone.utc).isoformat())
        persist()
        raise
    summary = validation_summary(rows)
    manifest.update(status="completed", validation_status="passed" if summary["all_passed"] else "failed", completed_utc=datetime.now(timezone.utc).isoformat())
    return persist()


def main(argv: list[str] | None = None) -> int:
    """Execute fixed engineering validation and return nonzero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, choices=[50])
    args = parser.parse_args(argv)
    summary = run_validation(args.output_dir, args.epochs)
    print(f"Methodology adapter gates: {summary['candidates']}", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
