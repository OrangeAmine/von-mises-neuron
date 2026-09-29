"""Run VM-N01 adapter through the unchanged standard synthetic benchmark APIs.

This is a two-fit integration check, not a forecasting comparison study or
evidence for the static smooth/local scientific question. The data, trainer,
budget and absolute acceptance thresholds are imported without modification.
"""

from __future__ import annotations

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
    THRESHOLDS, _train_model, check_thresholds, compute_benchmark_metrics,
    generate_synthetic_co2, prepare_benchmark_data, seed_everything,
)
from scripts.controlled_temporal_comparison import source_manifest, write_json
from src.smooth_local_forecaster_v1 import SmoothLocalForecasterV1
from src.split import chronological_split
from src.von_mises_neuron import VonMisesForecaster

MODELS = ("VonMises", "SmoothLocalForecasterV1")
CANDIDATES = ("SmoothLocalForecasterV1",)


def model_config(name: str) -> dict[str, Any]:
    """Return the fixed constructor configuration, with no adaptive selection."""
    if name == "VonMises":
        return {"class": "VonMisesForecaster", "kwargs": {
            "input_size": 4, "hidden_size": 64, "num_vm_layers": 2,
            "dropout": 0.1, "output_size": 12, "init_kappa": 1.0}}
    if name == "SmoothLocalForecasterV1":
        return {"class": name, "kwargs": {"input_size": 4, "output_size": 12,
            "projection_size": 32, "hidden_size": 64, "dropout": 0.1},
            "basis": {"harmonics": 3, "kernels": 10, "adaptive": True,
                      "concentration": 4.0, "rotation": 0.0,
                      "sharing": "one shared scalar basis for all four sensors"}}
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str) -> torch.nn.Module:
    """Construct a model after the caller resets the standard RNG seed."""
    config = model_config(name)
    cls = VonMisesForecaster if name == "VonMises" else SmoothLocalForecasterV1
    return cls(**config["kwargs"])


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require completed runs and every unchanged absolute metric threshold."""
    lookup = {row["model"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) - set(MODELS):
        raise ValueError("model rows must have unique declared identifiers")
    acceptance = {}
    for name in MODELS:
        row = lookup.get(name, {})
        metrics = {key: math.nan if row.get("metrics", {}).get(key) is None
                   else row["metrics"][key] for key in THRESHOLDS}
        passed, failures = check_thresholds(metrics)
        completed = row.get("execution_status") == "completed"
        if not completed:
            failures.append("Run did not complete successfully")
        acceptance[name] = {"absolute_passed": passed, "passed": passed and completed,
                            "failures": failures}
    return {
        "all_runs_attempted": all(lookup.get(name, {}).get("execution_status")
                                  not in (None, "pending", "running") for name in MODELS),
        "all_passed": all(item["passed"] for item in acceptance.values()),
        "acceptance": acceptance,
        "candidates": {name: acceptance[name]["passed"] for name in CANDIDATES},
        "scope": "standard adapter integration only; no static scientific superiority claim",
    }


def _fit(model: torch.nn.Module, data: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Call the frozen standard trainer with its full unchanged defaults."""
    return _train_model(model.to(device), *(data[key].to(device) for key in
        ("X_train", "y_train", "X_val", "y_val")), max_epochs=50)


def predict(model: torch.nn.Module, data: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return the standard full-test forward pass on CPU."""
    model.eval()
    with torch.no_grad():
        return model(data["X_test"].to(device)).detach().cpu()


def _save_state(path: Path, model: torch.nn.Module) -> None:
    """Archive independent CPU state tensors for replay or failure inspection."""
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)


def run_validation(output_dir: Path, epochs: int = 50) -> dict[str, Any]:
    """Run fixed-budget adapter validation in a fresh evidence directory.

    Args:
        output_dir: New directory; existing paths are rejected.
        epochs: The canonical 50-epoch ceiling; other values are rejected.

    Returns:
        Explicit per-model and aggregate acceptance, retaining failed attempts.
    """
    if epochs != 50:
        raise ValueError("the standard VM-N01 adapter budget is exactly 50 epochs")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = [{"model": name, "seed": 42,
        "execution_status": "pending", "metrics": {}, "failures": [],
        "config": model_config(name)} for name in MODELS]
    manifest: dict[str, Any] = {
        "protocol": "vm_n01_smooth_local_v1_engineering", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "data_seed": 42, "epochs": 50, "n_days": 90, "lookback": 72,
        "horizon": 12, "target": "CO2", "units": "ppm", "dtype": "float32",
        "thresholds": dict(THRESHOLDS), "models": {name: model_config(name) for name in MODELS},
        "training_workers": 1, "training_rng": "seed42 before each constructor; uninterrupted training RNG",
        "training_protocol": "unchanged benchmark_standard_dataset._train_model defaults",
        "scope": "standard adapter integration only; no static scientific superiority claim",
    }

    def persist() -> dict[str, Any]:
        summary = validation_summary(rows)
        write_json(output_dir / "results.json", {"runs": rows, **summary})
        write_json(output_dir / "validation.json", summary)
        write_json(output_dir / "manifest.json", manifest)
        return summary

    persist()
    started = time.perf_counter()
    try:
        manifest["environment"] = source_manifest()
        manifest["torch_threads"] = torch.get_num_threads()
        manifest["determinism"] = {"algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}
        snapshot_paths = sorted({path for pattern in
            ("src/*.py", "scripts/*.py", "tests/*.py", "configs/*.json", "requirements.txt")
            for path in ROOT.glob(pattern)})
        manifest["snapshot_sha256"] = {path.relative_to(ROOT).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot_paths}
        with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in snapshot_paths:
                archive.write(path, path.relative_to(ROOT))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest["device"] = str(device)
        frame = generate_synthetic_co2(n_days=90, seed=42)
        data = prepare_benchmark_data(frame, 72, 12, torch.device("cpu"))
        np.savez_compressed(output_dir / "benchmark_data.npz", **{
            key: data[key].numpy() for key in
            ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")})
        manifest.update(normalization=data["norm_stats"], sensor_columns=list(frame.columns),
            data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
            split={name: {"rows": len(part), "first": str(part.index[0]), "last": str(part.index[-1]),
                          "windows": len(data[f"X_{name}"])} for name, part in zip(
                              ("train", "val", "test"), chronological_split(frame, [0.8, 0.1, 0.1]))})
        persist()
        for row in rows:
            name = row["model"]
            row["execution_status"] = "running"
            fit_start = time.perf_counter()
            model = None
            persist()
            print(f"{name}: starting unchanged standard validation", flush=True)
            try:
                seed_everything(42)
                model = build_model(name)
                row["params"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                _save_state(output_dir / f"{name}_42_initial_state.pt", model)
                model = _fit(model, data, device)
                history = model.training_history
                if not math.isfinite(history["best_val_loss"]) or not all(
                        math.isfinite(value) for value in history["validation_loss"]):
                    raise FloatingPointError("Nonfinite validation history")
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                    raise FloatingPointError("Nonfinite model state")
                prediction = predict(model, data, device)
                row["metrics"] = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
                if not all(math.isfinite(value) for value in row["metrics"].values()):
                    raise FloatingPointError("Nonfinite evaluation metrics")
                np.savez_compressed(output_dir / f"{name}_42_predictions.npz",
                    prediction=prediction.numpy(), target=data["y_test"].numpy())
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
                row["time_seconds"] = time.perf_counter() - fit_start
                if model is not None:
                    row["training"] = getattr(model, "training_history", None)
                    if row["execution_status"] != "completed":
                        _save_state(output_dir / f"{name}_42_failed_state.pt", model)
                persist()
            print(f"{name}: {row['execution_status']}; metrics={row['metrics']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
            completed_utc=datetime.now(timezone.utc).isoformat(), time_seconds=time.perf_counter() - started)
        persist()
        raise
    summary = validation_summary(rows)
    manifest.update(status="completed", validation_status="passed" if summary["all_passed"] else "failed",
        completed_utc=datetime.now(timezone.utc).isoformat(), time_seconds=time.perf_counter() - started)
    return persist()


def main(argv: list[str] | None = None) -> int:
    """Execute the fixed integration check and return nonzero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, choices=[50])
    args = parser.parse_args(argv)
    summary = run_validation(args.output_dir, args.epochs)
    print(f"VM-N01 adapter gates: {summary['candidates']}", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
