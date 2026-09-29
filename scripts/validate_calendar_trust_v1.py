"""Full tests and fresh, unchanged standard benchmark for the N07 mixtures."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_calendar_trust_v1 import digest, verify_lock, write_json
from scripts.benchmark_standard_dataset import (
    THRESHOLDS, check_regression, check_thresholds, compute_benchmark_metrics,
    generate_synthetic_co2, prepare_benchmark_data, prepare_temporal_data, seed_everything,
)
from scripts.validate_dynamics_forecasters_v1 import _fit, build_model, model_config
from src.calendar_trust_v1 import delayed_mix
from src.dynamics_training_v1 import fit_ridge, ridge_design
from src.split import chronological_split


def run_worker(output: Path) -> dict:
    """Archive full-suite output and execute exactly one fresh benchmark fit."""
    lock = verify_lock(output)
    config = lock["config"]
    development_path = output / "development/summary.json"
    development = json.loads(development_path.read_text())
    if development["status"] != "complete":
        raise ValueError("completed development required")
    eta = development["selected_eta"]
    directory = output / "engineering"
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    record = {"status": "running", "eta": eta, "thresholds": THRESHOLDS,
              "lock_sha256": digest(output / "lock.json"), "development_sha256": digest(development_path),
              "scope": "engineering acceptance; online feedback is available only at observation event time",
              "data_seed": 42, "model_seed": 42, "days": 90, "lookback": 72, "horizon": 12,
              "split_ratios": [0.8, 0.1, 0.1], "epochs_max": 50, "base_model": model_config("VonMises"),
              "pytest_command": [sys.executable, "-m", "pytest", "tests/", "-q"]}
    write_json(directory / "results.json", record)
    try:
        print("Running full test suite", flush=True)
        with (directory / "pytest.log").open("w", encoding="utf-8") as log:
            suite = subprocess.run(record["pytest_command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        record["pytest_returncode"] = suite.returncode
        record["pytest_log_sha256"] = digest(directory / "pytest.log")
        print((directory / "pytest.log").read_text()[-1000:], flush=True)
        if suite.returncode:
            raise RuntimeError("full tests failed")
        verify_lock(output)
        frame = generate_synthetic_co2(n_days=90, seed=42)
        cpu = torch.device("cpu")
        base = prepare_benchmark_data(frame, 72, 12, cpu)
        temporal = prepare_temporal_data(frame, 72, 12, cpu, time_periods=[24, 168])
        parts = dict(zip(("train", "val", "test"), chronological_split(frame, [0.8, 0.1, 0.1])))
        mean, std = base["norm_stats"]["CO2"]
        record["normalization"] = base["norm_stats"]
        record["splits"] = {name: {"rows": len(part), "first": str(part.index[0]), "last": str(part.index[-1]),
                                   "windows": len(base[f"X_{name}"])} for name, part in parts.items()}
        batches = {}
        for split in parts:
            torch.testing.assert_close(base[f"y_{split}"], temporal[f"y_{split}"], rtol=0, atol=0)
            sensors = temporal[f"X_{split}"].numpy()[..., [2, 0, 1, 3]]
            sensor = np.concatenate((sensors, np.ones_like(sensors[..., :1]), np.zeros_like(sensors[..., :1])), axis=-1)
            batches[split] = SimpleNamespace(sensor=sensor, y=base[f"y_{split}"].numpy())
        fitted = fit_ridge(batches["train"], batches["val"], config["ridge_penalties"])
        table = ((parts["train"]["CO2"] - mean) / std).groupby(parts["train"].index.hour).mean().reindex(range(24)).to_numpy()
        experts = {}
        for split in ("val", "test"):
            batch = batches[split]
            sensor = batch.sensor[:, -1, 0, None] + ridge_design(batch.sensor) @ fitted["weight"] + fitted["bias"]
            last = temporal[f"X_{split}"].numpy()[:, -1, 4, None]
            hours = np.rint(last * 24 / (2 * np.pi) + np.arange(1, 13)).astype(int) % 24
            experts[split] = np.stack((sensor, table[hours]), axis=-1)
        initial = ((experts["val"] - batches["val"].y[..., None]) ** 2).mean(axis=0)
        count = len(batches["test"].y)
        indices = np.arange(count) + 71
        if count != len(parts["test"]) - 72 - 12 + 1:
            raise ValueError("standard data has unexpected gaps")
        issue = parts["test"].index.asi8[indices]
        target = parts["test"].index.asi8[indices[:, None] + np.arange(1, 13)]
        mix = delayed_mix(experts["test"], batches["test"].y, issue, target, target, initial,
                          int(parts["val"].index[-1].value), 1.0, eta, config["feedback_half_life_updates"])
        predictions = {"Sensor": experts["test"][..., 0], "Seasonal": experts["test"][..., 1],
                       "FixedMixture": experts["test"].mean(axis=-1), "AdaptiveMixture": mix.prediction}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(42)
        model = build_model("VonMises")
        print("Fitting fresh base VM on", device, flush=True)
        fit_started = time.perf_counter()
        model = _fit("VonMises", model, base, device)
        model.eval()
        with torch.no_grad():
            predictions["VonMises"] = model(base["X_test"].to(device)).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        record["base_training_and_inference_seconds"] = time.perf_counter() - fit_started
        record["device"] = str(device)
        record["base_parameters"] = sum(p.numel() for p in model.parameters())
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "base_vm.pt")
        np.savez_compressed(directory / "predictions.npz", **predictions, truth=batches["test"].y,
                            experts=experts["test"], initial_loss=initial, weights=mix.weights,
                            feedback_count=mix.feedback_count, latest_feedback_ns=mix.latest_feedback_ns,
                            issue_ns=issue, target_ns=target, initialization_cutoff_ns=int(parts["val"].index[-1].value))
        np.savez_compressed(directory / "component_fit.npz", weight=fitted["weight"], bias=fitted["bias"], seasonal=table)
        record["component_fit"] = {k: v for k, v in fitted.items() if k not in ("weight", "bias")}
        record["seasonal_parameters"] = 24
        record["adaptive_state_values"] = int(initial.size)
        metrics = {name: compute_benchmark_metrics(torch.as_tensor(p), base["y_test"], base["norm_stats"])
                   for name, p in predictions.items()}
        base_ok, _ = check_thresholds(metrics["VonMises"])
        acceptance = {}
        for name, values in metrics.items():
            absolute, failures = check_thresholds(values)
            regression, regression_failures = check_regression(values, metrics["VonMises"])
            required_regression = name in ("Sensor", "FixedMixture", "AdaptiveMixture")
            acceptance[name] = {"metrics": values, "absolute_passed": absolute,
                                "regression_passed": regression, "regression_required": required_regression,
                                "passed": bool(absolute and (not required_regression or (regression and base_ok))),
                                "failures": failures + (regression_failures if required_regression else [])}
        record.update(status="complete", acceptance=acceptance,
                      all_new_mixtures_passed=all(acceptance[n]["passed"] for n in ("FixedMixture", "AdaptiveMixture")),
                      artifacts_sha256={p.name: digest(p) for p in directory.iterdir() if p.suffix in (".npz", ".pt")},
                      seconds=time.perf_counter() - started)
        print(json.dumps(acceptance, indent=2), flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error), seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(directory / "results.json", record)
    return record


def main() -> None:
    """Run the worker under an external wall-clock cap, preserving its records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = run_worker(args.output)
        raise SystemExit(0 if result["all_new_mixtures_passed"] else 2)
    lock = verify_lock(args.output)
    if (args.output / "engineering").exists():
        raise FileExistsError("engineering stage already exists")
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--worker"],
                                 cwd=ROOT, timeout=lock["config"]["engineering_cap_seconds"])
    except subprocess.TimeoutExpired:
        path = args.output / "engineering/results.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        result.update(status="failed", error="prospective engineering runtime cap exceeded")
        write_json(path, result)
        raise
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
