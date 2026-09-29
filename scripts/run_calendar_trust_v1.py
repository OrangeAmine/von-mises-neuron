"""Freeze, run and replay the prospective N07 calendar-trust study."""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.calendar_trust_v1 import NEVER, delayed_mix, paired_summary
from src.dynamics_data_v1 import DynamicsConfig, fit_normalizer, generate_trajectory, make_windows
from src.dynamics_training_v1 import error_metrics, fit_ridge, ridge_design
from scripts.run_vm_n04_v1 import seasonal_table, batch_receipt

CONFIG = ROOT / "configs/calendar_trust_v1.json"


def digest(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write strict JSON with a terminating newline."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_paths() -> list[Path]:
    """Include executable sources, tests, configuration and prospective protocol."""
    return sorted({p for pattern in ("src/*.py", "scripts/*.py", "tests/*.py", "configs/*.json", "requirements.txt")
                   for p in ROOT.glob(pattern)} | {ROOT / "docs/calendar_trust_protocol_v1.md",
                                                 ROOT / "docs/source_interval_contract_v1.md"})


def freeze(output: Path) -> None:
    """Create an immutable-input receipt before generating any study room."""
    output.mkdir(parents=True, exist_ok=False)
    paths = source_paths()
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "config": json.loads(CONFIG.read_text()),
            "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "worktree_differs_from_revision": True,
            "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                            "ridge_device": "CPU, unchanged N04 NumPy closed-form solver"}}
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    lock["source_archive_sha256"] = digest(output / "source_snapshot.zip")
    write_json(output / "lock.json", lock)
    print("Frozen", len(paths), "source files", flush=True)


def verify_lock(output: Path) -> dict:
    """Fail on any change to a frozen input."""
    lock = json.loads((output / "lock.json").read_text())
    for name, expected in lock["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(output / "source_snapshot.zip") != lock["source_archive_sha256"]:
        raise ValueError("Source archive changed")
    return lock


def experts_for(batch: object, fit: dict, table: np.ndarray, normalizer: object) -> np.ndarray:
    """Return sensor and seasonal forecasts in ppm using common frozen fits."""
    normalized = batch.sensor[:, -1, 0, None] + ridge_design(batch.sensor) @ fit["weight"] + fit["bias"]
    sensor = normalized * normalizer.std + normalizer.mean
    step_ns = 15 * 60 * 1_000_000_000
    seasonal = table[(batch.target_times // step_ns) % len(table)]
    return np.stack((sensor, seasonal), axis=-1)


def metric_record(prediction: np.ndarray, truth: np.ndarray, issue: np.ndarray,
                  test_start_ns: int, std: float, seed: int, regime: str, arm: str,
                  config: dict) -> dict:
    """Compute fixed endpoint and early/late diagnostics in physical units."""
    endpoints = {str(step): error_metrics(prediction[:, step - 1], truth[:, step - 1])
                 for step in [config["primary_step"], *config["diagnostic_steps"]]}
    j = config["primary_step"] - 1
    early = issue < test_start_ns + 48 * 3_600_000_000_000
    return {"seed": seed, "regime": regime, "arm": arm, "normalization_std": std,
            "endpoints": endpoints,
            "early_48h": error_metrics(prediction[early, j], truth[early, j]),
            "later": error_metrics(prediction[~early, j], truth[~early, j])}


def run_stage(output: Path, stage: str) -> None:
    """Execute one bounded stage once; archive failures instead of retrying."""
    lock = verify_lock(output)
    config = lock["config"]
    etas = config["eta_candidates"]
    dependencies = {"lock_sha256": digest(output / "lock.json")}
    if stage == "confirmation":
        development = json.loads((output / "development/summary.json").read_text())
        engineering = json.loads((output / "engineering/results.json").read_text())
        if development["status"] != "complete" or engineering["status"] != "complete" or engineering["pytest_returncode"] != 0:
            raise ValueError("completed development and engineering with passing full tests required")
        etas = [development["selected_eta"]]
        dependencies.update(development_sha256=digest(output / "development/summary.json"),
                            engineering_sha256=digest(output / "engineering/results.json"))
    directory = output / stage
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    deadline = started + config[f"{stage}_cap_seconds"]
    manifest = {"stage": stage, "status": "running", "dependencies": dependencies,
                "etas": etas, "seeds": config[f"{stage}_seeds"], "rooms": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(directory / "manifest.json", manifest)
    grouped = {str(eta): [] for eta in etas}
    try:
        for seed in config[f"{stage}_seeds"]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("prospective stage runtime cap reached")
            room_started = time.perf_counter()
            trajectory = generate_trajectory(seed, DynamicsConfig(**config["data"]))
            normalizer = fit_normalizer(trajectory)
            train, val = [make_windows(trajectory, split, normalizer) for split in ("train", "validation")]
            fitted = fit_ridge(train, val, config["ridge_penalties"])
            table = seasonal_table(trajectory)
            val_experts = experts_for(val, fitted, table, normalizer)
            initial = np.mean(((val_experts - val.y_ppm[..., None]) / normalizer.std) ** 2, axis=0)
            split_start = sum(config["data"]["split_days"][:2]) * 96
            cutoff = int(trajectory.event_times[split_start - 1].value)
            test_start = int(trajectory.event_times[split_start].value)
            np.savez_compressed(directory / f"room_{seed}_fit.npz", weight=fitted["weight"], bias=fitted["bias"],
                                seasonal=table, initial_loss=initial, normalization_mean=normalizer.mean,
                                normalization_std=normalizer.std)
            for regime in config["regimes"]:
                case = generate_trajectory(seed, DynamicsConfig(**config["data"], shifted=regime == "shifted"))
                np.testing.assert_array_equal(case.observed_ppm[:split_start], trajectory.observed_ppm[:split_start])
                batch = make_windows(case, "test", normalizer)
                experts = experts_for(batch, fitted, table, normalizer)
                ids = batch.origin_indices[:, None] + np.arange(1, case.config.horizon + 1)
                arrival = case.arrival_times.asi8[ids].copy()
                arrival[~case.sensor_available[ids]] = NEVER
                issue = batch.origin_times.asi8
                for eta in etas:
                    mix = delayed_mix(experts, batch.y_ppm, issue, batch.target_times, arrival,
                                      initial, cutoff, normalizer.std, eta, config["feedback_half_life_updates"])
                    arms = {"sensor": experts[..., 0], "seasonal": experts[..., 1],
                            "fixed": experts.mean(axis=-1), "adaptive": mix.prediction}
                    for arm, pred in arms.items():
                        grouped[str(eta)].append(metric_record(pred, batch.y_ppm, issue, test_start,
                                                              normalizer.std, seed, regime, arm, config))
                    np.savez_compressed(directory / f"room_{seed}_{regime}_eta_{eta:g}.npz",
                                        experts=experts, truth=batch.y_ppm, issue_ns=issue,
                                        target_ns=batch.target_times, arrival_ns=arrival,
                                        initial_loss=initial, initialization_cutoff_ns=cutoff,
                                        scale=normalizer.std, test_start_ns=test_start,
                                        prediction=mix.prediction, weights=mix.weights,
                                        feedback_count=mix.feedback_count, latest_feedback_ns=mix.latest_feedback_ns)
            manifest["rooms"].append({"seed": seed, "data_config": asdict(trajectory.config),
                                      "fit": {k: v for k, v in fitted.items() if k not in ("weight", "bias")},
                                      "normalization": asdict(normalizer), "seasonal_parameters": len(table),
                                      "adaptive_state_values": int(initial.size), "train": batch_receipt(train),
                                      "validation": batch_receipt(val), "test": batch_receipt(batch),
                                      "seconds": time.perf_counter() - room_started})
            write_json(directory / "manifest.json", manifest)
            print(stage, "room", seed, "complete", f"{time.perf_counter()-started:.2f}s", flush=True)
        if time.perf_counter() >= deadline:
            raise TimeoutError("prospective stage runtime cap reached")
        scores = {eta: float(np.mean([r["endpoints"][str(config["primary_step"])]["rmse"] / r["normalization_std"]
                                     for r in records if r["arm"] == "adaptive"])) for eta, records in grouped.items()}
        selected = min(etas, key=lambda eta: (scores[str(eta)], eta))
        summary = {"status": "complete", "stage": stage, "selected_eta": selected,
                   "selection_scores": scores, "records": grouped, "seconds": time.perf_counter() - started}
        if stage == "confirmation":
            summary["analysis"] = paired_summary(grouped[str(selected)], config["primary_step"],
                config["bootstrap_seed"], config["bootstrap_replicates"], config["minimum_gain_percent"],
                config["maximum_regime_regression_percent"])
        summary["artifacts_sha256"] = {p.name: digest(p) for p in sorted(directory.glob("*.npz"))}
        write_json(directory / "summary.json", summary)
        manifest.update(status="complete", seconds=time.perf_counter() - started)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(directory / "manifest.json", manifest)


def audit(output: Path) -> dict:
    """Recompute all confirmation forecasts and statistics from saved arrays."""
    config = verify_lock(output)["config"]
    directory = output / "confirmation"
    summary = json.loads((directory / "summary.json").read_text())
    eta = summary["selected_eta"]
    manifest = json.loads((directory / "manifest.json").read_text())
    for key, path in (("lock_sha256", "lock.json"), ("development_sha256", "development/summary.json"),
                      ("engineering_sha256", "engineering/results.json")):
        if digest(output / path) != manifest["dependencies"][key]:
            raise ValueError(f"changed dependency: {path}")
    for name, expected in summary["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"changed artifact: {name}")
    records = []
    for seed in config["confirmation_seeds"]:
        for regime in config["regimes"]:
            with np.load(directory / f"room_{seed}_{regime}_eta_{eta:g}.npz") as data:
                mix = delayed_mix(data["experts"], data["truth"], data["issue_ns"], data["target_ns"],
                                  data["arrival_ns"], data["initial_loss"], int(data["initialization_cutoff_ns"]),
                                  float(data["scale"]), eta, config["feedback_half_life_updates"])
                for name in ("prediction", "weights", "feedback_count", "latest_feedback_ns"):
                    np.testing.assert_array_equal(data[name], getattr(mix, name))
                np.testing.assert_allclose(mix.weights.sum(axis=-1), 1, atol=1e-14)
                if np.any(mix.latest_feedback_ns > data["issue_ns"][:, None]):
                    raise AssertionError("future feedback")
                for arm, prediction in (("sensor", data["experts"][..., 0]), ("seasonal", data["experts"][..., 1]),
                                        ("fixed", data["experts"].mean(axis=-1)), ("adaptive", mix.prediction)):
                    records.append(metric_record(prediction, data["truth"], data["issue_ns"], int(data["test_start_ns"]),
                                                 float(data["scale"]), seed, regime, arm, config))
    if records != summary["records"][str(eta)]:
        raise AssertionError("metric replay mismatch")
    analysis = paired_summary(records, config["primary_step"], config["bootstrap_seed"],
                              config["bootstrap_replicates"], config["minimum_gain_percent"],
                              config["maximum_regime_regression_percent"])
    if analysis != summary["analysis"]:
        raise AssertionError("paired analysis replay mismatch")
    result = {"status": "passed", "source_and_dependency_hashes": "passed",
              "cells_replayed": len(config["confirmation_seeds"]) * len(config["regimes"]),
              "metrics_replayed": len(records), "future_feedback_count": 0,
              "scope": "same-agent artifact replay, not independent review",
              "confirmation_summary_sha256": digest(directory / "summary.json")}
    write_json(output / "audit.json", result)
    return result


def main() -> None:
    """Dispatch one explicit freeze/run/audit operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "confirmation", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "audit":
        print(json.dumps(audit(args.output), indent=2))
    else:
        run_stage(args.output, args.stage)


if __name__ == "__main__":
    main()
