"""Run VM-N04 forecasters against unchanged standard synthetic acceptance gates.

Engineering readiness is separate from the prospective dynamics experiment.
The original synthetic generator, chronological pipeline, normalization,
50-epoch trainer, absolute thresholds and base-VM regression limits are imported
unchanged. All four new neural candidates and residual ridge require both absolute and relative gates;
failures remain saved and yield a nonzero exit status. This command never tunes
an architecture or retries a failure to pass the benchmark.
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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_standard_dataset import (
    THRESHOLDS, _train_model, check_finite_metrics, check_regression, check_thresholds,
    compute_benchmark_metrics, generate_synthetic_co2, prepare_benchmark_data,
    prepare_temporal_data, seed_everything,
)
from scripts.controlled_temporal_comparison import source_manifest, write_json
from scripts.validate_methodology_forecasters import _progress
from src.dynamics_models_v1 import StandardDynamicsAdapterV1
from src.dynamics_training_v1 import fit_ridge, ridge_design
from src.split import chronological_split
from src.von_mises_neuron import VonMisesForecaster

CANDIDATE_MODES = {"DynamicsNone": "none", "DynamicsFourier": "fourier",
                   "DynamicsFixed": "fixed", "DynamicsLearned": "learned"}
NEURAL_CANDIDATES = tuple(CANDIDATE_MODES)
CANDIDATES = (*NEURAL_CANDIDATES, "ResidualRidge")
REFERENCES = ("Persistence", "Seasonal")
MODELS = ("VonMises", *CANDIDATES, *REFERENCES)


def model_config(name: str) -> dict[str, Any]:
    """Return constructor settings without altering any historical source."""
    if name == "VonMises":
        return {"class": "VonMisesForecaster", "kwargs": {
            "input_size": 4, "hidden_size": 64, "num_vm_layers": 2,
            "dropout": 0.1, "output_size": 12, "init_kappa": 1.0}}
    if name == "ResidualRidge":
        return {"class": "sensor-only residual ridge", "sensor_column_order": [2, 0, 1, 3],
                "input_size": 6, "lookback": 72, "horizon": 12,
                "penalties": [0.0001, 0.01, 1.0], "penalty_objective": "mean squared residual loss",
                "selection": "validation mean squared error; first on ties"}
    if name in REFERENCES:
        return {"class": name, "definition": "last measured target" if name == "Persistence"
                else "training-only local hour-of-day target mean"}
    if name not in CANDIDATE_MODES:
        raise ValueError(f"Unknown model: {name}")
    return {"class": "StandardDynamicsAdapterV1", "kwargs": {
        "calendar": CANDIDATE_MODES[name], "lookback": 72, "horizon": 12,
        "sensor_size": 4, "target_index": 2, "cadence_hours": 1.0,
        "hidden_size": 32}, "basis_features": 0 if name == "DynamicsNone" else 32,
        "harmonics": 16 if name == "DynamicsFourier" else 0,
        "kernel_initial_kappa": 32.0 if name in ("DynamicsFixed", "DynamicsLearned") else None}


def build_model(name: str) -> torch.nn.Module:
    """Construct the frozen standard-sized candidate or original base VM."""
    if name not in ("VonMises", *NEURAL_CANDIDATES):
        raise ValueError("build_model only constructs the declared neural models")
    config = model_config(name)
    cls = VonMisesForecaster if name == "VonMises" else StandardDynamicsAdapterV1
    return cls(**config["kwargs"])


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require completion, finite absolute gates, and all candidate regressions."""
    lookup = {row["model"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) - set(MODELS):
        raise ValueError("model rows must have unique declared identifiers")

    def metrics(name: str) -> dict[str, float]:
        values = lookup.get(name, {}).get("metrics", {})
        return {key: math.nan if values.get(key) is None else values[key]
                for key in THRESHOLDS}

    acceptance = {}
    for name in MODELS:
        absolute, absolute_failures = check_thresholds(metrics(name))
        passed, failures = (check_finite_metrics(metrics(name)) if name in REFERENCES
                            else (absolute, absolute_failures))
        completed = lookup.get(name, {}).get("execution_status") == "completed"
        if not completed:
            failures.append("Run did not complete successfully")
        acceptance[name] = {"passed": passed and completed,
                            "absolute_passed": absolute, "failures": failures,
                            "gate_policy": "finite-only reference" if name in REFERENCES
                            else "absolute plus base regression" if name in CANDIDATES else "absolute"}
    for name in CANDIDATES:
        regression, failures = check_regression(metrics(name), metrics("VonMises"))
        if not acceptance["VonMises"]["passed"]:
            regression = False
            failures.append("Fresh base VM did not pass its standard absolute gates")
        acceptance[name]["regression_vs_base"] = regression
        acceptance[name]["passed"] &= regression
        acceptance[name]["failures"].extend(failures)
    for name in REFERENCES:
        regression, failures = check_regression(metrics(name), metrics("VonMises"))
        acceptance[name]["diagnostic_regression_vs_base"] = regression
        acceptance[name]["diagnostic_regression_failures"] = failures
    return {
        "all_runs_attempted": all(lookup.get(name, {}).get("execution_status")
                                  not in (None, "pending") for name in MODELS),
        "all_passed": all(item["passed"] for item in acceptance.values()),
        "acceptance": acceptance,
        "candidates": {name: acceptance[name]["passed"] for name in CANDIDATES}}


def _sync(device: torch.device) -> None:
    """Complete queued device work before wall-clock timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit(name: str, model: torch.nn.Module, data: dict[str, Any],
         device: torch.device) -> torch.nn.Module:
    """Use the exact original standard 50-epoch training routine and defaults."""
    model.to(device)
    # Pair new-arm minibatch RNG despite none having no calendar-head draws.
    # The base retains its exact uninterrupted constructor -> trainer sequence.
    if name != "VonMises":
        seed_everything(42)
    with _progress(name):
        return _train_model(model, *(data[key].to(device) for key in
                                     ("X_train", "y_train", "X_val", "y_val")),
                            max_epochs=50)


def predict(model: torch.nn.Module, data: dict[str, Any],
            device: torch.device) -> torch.Tensor:
    """Return the unchanged full-test forward pass in normalized target units."""
    model.eval()
    with torch.no_grad():
        return model(data["X_test"].to(device)).detach().cpu()


def _save_state(path: Path, model: torch.nn.Module) -> None:
    """Archive reconstructable CPU state, including buffers."""
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)


def _reference_result(name: str, data: dict[str, Any], frame: pd.DataFrame,
                      output_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit fixed simple controls with identical standard splits and label access."""
    record: dict[str, Any] = {}
    if name == "ResidualRidge":
        batches = {}
        for split in ("train", "val", "test"):
            sensors = data[f"X_{split}"].numpy()[..., [2, 0, 1, 3]]
            available = np.ones_like(sensors[..., :1])
            sensor = np.concatenate((sensors, available, np.zeros_like(available)), axis=-1)
            batches[split] = SimpleNamespace(sensor=sensor, y=data[f"y_{split}"].numpy())
        fitted = fit_ridge(batches["train"], batches["val"], model_config(name)["penalties"])
        np.savez_compressed(output_dir / "ResidualRidge_coefficients.npz",
                            weight=fitted["weight"], bias=fitted["bias"])
        record.update(params=fitted["parameters"], training_seconds=fitted["training_seconds"],
                      training={key: value for key, value in fitted.items() if key not in ("weight", "bias")})
        inference_started = time.perf_counter()
        batch = batches["test"]
        result = batch.sensor[:, -1, 0, None] + ridge_design(batch.sensor) @ fitted["weight"] + fitted["bias"]
    elif name == "Persistence":
        record.update(params=0, training_seconds=0.0, training=None)
        inference_started = time.perf_counter()
        result = np.repeat(data["X_test"].numpy()[:, -1, 2, None], 12, axis=1)
    elif name == "Seasonal":
        fit_started = time.perf_counter()
        train, _, _ = chronological_split(frame, [0.8, 0.1, 0.1])
        mean, std = data["norm_stats"]["CO2"]
        hour_mean = ((train["CO2"] - mean) / std).groupby(train.index.hour).mean().reindex(range(24)).to_numpy()
        np.savez_compressed(output_dir / "Seasonal_hour_means.npz", hour_mean=hour_mean)
        record.update(params=24, training_seconds=time.perf_counter() - fit_started,
                      training={"source": "training rows only", "hours": 24})
        inference_started = time.perf_counter()
        last_angle = data["X_test"].numpy()[:, -1, 4, None]
        # Identical final-daily-phase plus regular hourly extension to the arms.
        future_hours = np.rint(last_angle * 24 / (2 * math.pi) + np.arange(1, 13)).astype(int) % 24
        result = hour_mean[future_hours]
    else:
        raise ValueError(f"Unknown reference {name}")
    record["inference_seconds"] = time.perf_counter() - inference_started
    return torch.as_tensor(result), record


def run_validation(output_dir: Path, epochs: int = 50) -> dict[str, Any]:
    """Run and retain every prescribed engineering fit in a new directory.

    Args:
        output_dir: Fresh artifact directory; existing paths are rejected.
        epochs: Must equal the original standard default, 50.

    Returns:
        Acceptance for four neural candidates, ridge, base VM, and references.
    """
    if epochs != 50:
        raise ValueError("the unchanged standard benchmark budget is exactly 50 epochs")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = [
        {"model": name, "seed": 42, "execution_status": "pending",
         "metrics": {}, "failures": [], "config": model_config(name)} for name in MODELS]
    manifest: dict[str, Any] = {
        "protocol": "vm_n04_dynamics_v1_engineering", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42, "data_seed": 42, "epochs": 50, "n_days": 90,
        "lookback": 72, "horizon": 12, "target": "CO2", "units": "ppm",
        "calendar_clock": "legacy local Europe/Paris",
        "future_calendar_adapter": "last daily angle plus hourly increments; no DST-transition inference",
        "information": "four historical sensors, all-observed mask, age zero; no future measured sensors",
        "thresholds": THRESHOLDS,
        "regression_for_all_new_arms": {"error_relative_max": 0.1, "r2_drop_max": 0.1},
        "training_workers": 1, "models": {name: model_config(name) for name in MODELS},
        "training_rng": "seed42 before each constructor; base uninterrupted; new arms reset42 before unchanged trainer",
        "training_protocol": "benchmark_standard_dataset._train_model with exact defaults",
        "scope": "engineering acceptance only, separate from VM-N04 scientific evidence"}

    def persist() -> dict[str, Any]:
        summary = validation_summary(rows)
        write_json(output_dir / "results.json", {"runs": rows, **summary})
        write_json(output_dir / "validation.json", summary)
        write_json(output_dir / "manifest.json", manifest)
        return summary

    persist()
    try:
        manifest["environment"] = source_manifest()
        paths = sorted({path for pattern in (
            "src/*.py", "scripts/*.py", "tests/*.py", "configs/*.json", "requirements.txt")
                        for path in ROOT.glob(pattern)})
        manifest["source_sha256"] = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
        with zipfile.ZipFile(output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest["device"] = str(device)
        frame = generate_synthetic_co2(n_days=90, seed=42)
        base = prepare_benchmark_data(frame, 72, 12, torch.device("cpu"))
        temporal = prepare_temporal_data(frame, 72, 12, torch.device("cpu"), time_periods=[24, 168])
        for split in ("train", "val", "test"):
            torch.testing.assert_close(base[f"y_{split}"], temporal[f"y_{split}"], rtol=0, atol=0)
        manifest.update(
            normalization=base["norm_stats"], sensor_columns=list(frame.columns),
            data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
            split={name: {"rows": len(part), "first": str(part.index[0]),
                          "last": str(part.index[-1]), "windows": len(base[f"X_{name}"])}
                   for name, part in zip(("train", "val", "test"),
                                         chronological_split(frame, [0.8, 0.1, 0.1]))})
        persist()
        for row in rows:
            name = row["model"]
            data = base if name == "VonMises" else temporal
            row["execution_status"] = "running"
            model = None
            started = time.perf_counter()
            persist()
            print(f"{name}: starting unchanged standard validation", flush=True)
            try:
                if name in ("ResidualRidge", *REFERENCES):
                    prediction, reference_record = _reference_result(name, data, frame, output_dir)
                    row.update(reference_record)
                    row["metrics"] = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
                    if not all(math.isfinite(value) for value in row["metrics"].values()):
                        raise FloatingPointError("Nonfinite reference metrics")
                    np.savez_compressed(output_dir / f"{name}_42_predictions.npz",
                                        prediction=prediction.numpy(), target=data["y_test"].numpy())
                    row["execution_status"] = "completed"
                    print(f"{name}: completed; metrics={row['metrics']}", flush=True)
                    continue
                seed_everything(42)
                model = build_model(name)
                row["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if isinstance(model, StandardDynamicsAdapterV1):
                    row["capacity"] = model.body.get_config()
                _save_state(output_dir / f"{name}_42_initial_state.pt", model)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _sync(device)
                fit_started = time.perf_counter()
                model = _fit(name, model, data, device)
                _sync(device)
                row["training_seconds"] = time.perf_counter() - fit_started
                history = model.training_history
                if not math.isfinite(history["best_val_loss"]) or not all(
                        math.isfinite(value) for value in history["validation_loss"]):
                    raise FloatingPointError("Nonfinite validation history")
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                    raise FloatingPointError("Nonfinite model state")
                inference_started = time.perf_counter()
                prediction = predict(model, data, device)
                _sync(device)
                row["inference_seconds"] = time.perf_counter() - inference_started
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
                _sync(device)
                row["time_seconds"] = time.perf_counter() - started
                if device.type == "cuda" and model is not None:
                    row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                if model is not None:
                    row["training"] = getattr(model, "training_history", None)
                    if row["execution_status"] != "completed":
                        _save_state(output_dir / f"{name}_42_failed_state.pt", model)
                persist()
            print(f"{name}: {row['execution_status']}; metrics={row['metrics']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        completed_utc=datetime.now(timezone.utc).isoformat())
        persist()
        raise
    summary = validation_summary(rows)
    manifest.update(status="completed", validation_status="passed" if summary["all_passed"] else "failed",
                    completed_utc=datetime.now(timezone.utc).isoformat())
    return persist()


def main(argv: list[str] | None = None) -> int:
    """Execute the fixed readiness gate and fail truthfully on any failed arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, choices=[50])
    args = parser.parse_args(argv)
    summary = run_validation(args.output_dir, args.epochs)
    print(f"VM-N04 engineering acceptance: {summary['candidates']}", flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
