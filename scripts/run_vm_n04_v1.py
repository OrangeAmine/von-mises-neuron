"""Freeze and run the finite VM-N04 first-order forecasting study.

Examples:
    py -3.11 scripts/run_vm_n04_v1.py freeze --output-dir results/vm_n04/study_v1
    py -3.11 scripts/run_vm_n04_v1.py pilot --output-dir results/vm_n04/study_v1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import torch

from src.dynamics_data_v1 import DynamicsConfig, fit_normalizer, generate_trajectory, make_windows
from src.dynamics_models_v1 import DynamicsForecasterV1
from src.dynamics_training_v1 import (check_deadline, error_metrics, fit_neural,
                                      fit_ridge, predict_neural, ridge_design)

SOURCES = ["configs/vm_n04_v1.json", "docs/vm_n04_protocol_v1.md",
           "src/dynamics_data_v1.py", "src/dynamics_models_v1.py",
           "src/dynamics_training_v1.py", "scripts/run_vm_n04_v1.py",
           "scripts/analyze_vm_n04_v1.py", "src/circular_layers_v2.py"]


def digest(path: Path) -> str:
    """Hash the exact consumed bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write a human-readable, strict JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_lock(root: Path) -> dict:
    """Require every frozen source byte before accessing any study trajectory."""
    lock = json.loads((root / "source_lock.json").read_text())
    if set(lock["sources"]) != set(SOURCES):
        raise ValueError("Source lock must contain the exact declared source set")
    for name, sha in lock["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source mismatch: {name}")
    return lock


def verify_stage(output: Path, stage: str, config: dict) -> dict:
    """Verify a completed stage, its full identity matrix and saved artifacts."""
    stage_dir = output / stage
    summary_path = stage_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    terminal = json.loads((stage_dir / "terminal.json").read_text())
    access = json.loads((stage_dir / "access_lock.json").read_text())
    if (terminal.get("status") != "complete" or summary.get("status") != "complete"
            or terminal["summary_sha256"] != digest(summary_path)
            or access["source_lock_sha256"] != digest(output / "source_lock.json")):
        raise ValueError("Stage completion/source receipt mismatch")
    for dependency, sha in access["dependencies"].items():
        if digest(output / dependency) != sha:
            raise ValueError("Prior stage dependency changed")
    rates = config["training"]["learning_rates"] if stage == "pilot" else [summary["records"][0]["rate"]]
    expected = stage_plan(config, stage, rates)
    keys = ("room", "tau", "irregular", "seed", "rate", "arm")
    expected_ids = {tuple(row[k] for k in keys) for row in expected}
    actual = {}
    for row in summary["records"]:
        identity = tuple(row[k] for k in keys)
        actual.setdefault(identity, []).append((row["shifted"], row["outage"]))
    variants = list(itertools.product(config["test_shift"], config["test_outage"]))
    if (summary["planned_fits"] != len(expected) or summary["completed_fits"] != len(expected)
            or set(actual) != expected_ids or any(sorted(v) != sorted(variants) for v in actual.values())):
        raise ValueError("Incomplete or duplicate prior fit matrix")
    artifact_path = stage_dir / "artifacts.json"
    if terminal["artifacts_sha256"] != digest(artifact_path):
        raise ValueError("Stage artifact manifest changed")
    for name, sha in json.loads(artifact_path.read_text()).items():
        path = (stage_dir / name).resolve()
        if not path.is_relative_to(stage_dir.resolve()) or digest(path) != sha:
            raise ValueError("Stage artifact changed or escaped its root")
    return summary


def environment() -> dict:
    """Record the actual local numerical runtime and code revision."""
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "threads": 1}


def stage_plan(config: dict, stage: str, rates: list[float]) -> list[dict]:
    """Enumerate every fit in advance; test interventions reuse fitted states."""
    spec = config["stages"][stage]
    return [dict(room=room, tau=tau, irregular=irregular, seed=seed, rate=rate, arm=arm)
            for room, tau, irregular, seed, rate, arm in itertools.product(
                spec["room_seeds"], config["tau_ratios"], config["attendance"],
                spec["initialization_seeds"], rates, config["arms"])]


def batch_receipt(batch: object) -> dict:
    """Disclose overlapping label uses separately from unique measurement times."""
    targets = np.asarray(batch.target_times)
    return {"windows": len(batch.sensor), "label_uses": int(batch.y.size),
            "unique_target_times": int(np.unique(targets).size),
            "exclusions": batch.exclusions,
            "first_origin": str(batch.origin_times[0]), "last_origin": str(batch.origin_times[-1])}


def evaluate(pred: np.ndarray, batch: object, std: float, endpoint_steps: list[int]) -> dict:
    """Compute endpoint and whole-horizon diagnostics on common eligible cases."""
    result = {"all_horizon": error_metrics(pred, batch.y_ppm),
              "clean_all_horizon": error_metrics(pred, batch.clean_y_ppm), "endpoints": {}}
    for step in endpoint_steps:
        index = step - 1
        endpoint = error_metrics(pred[:, index], batch.y_ppm[:, index])
        endpoint["normalized_rmse"] = endpoint["rmse"] / std
        endpoint["clean"] = error_metrics(pred[:, index], batch.clean_y_ppm[:, index])
        for label, mask in (("boundary", batch.target_boundary[:, index]),
                            ("event", batch.target_event[:, index]),
                            ("origin_outage", batch.origin_outage),
                            ("target_outage", batch.target_outage[:, index])):
            mask = np.asarray(mask, dtype=bool)
            endpoint[label] = error_metrics(pred[mask, index], batch.y_ppm[mask, index])
        result["endpoints"][str(step)] = endpoint
    return result


def seasonal_table(trajectory: object) -> np.ndarray:
    """Fit 96 UTC-bin means from only observations received before train cutoff."""
    cfg = trajectory.config
    ntrain = cfg.split_days[0] * (1440 // cfg.dt_minutes)
    ids = np.arange(ntrain)
    cutoff = trajectory.event_times[ntrain - 1]
    ok = trajectory.sensor_available[:ntrain] & (trajectory.arrival_times[:ntrain] <= cutoff)
    table = np.full(1440 // cfg.dt_minutes, np.nan)
    bins = ids % len(table)
    for b in range(len(table)):
        table[b] = trajectory.observed_ppm[:ntrain][ok & (bins == b)].mean()
    if not np.isfinite(table).all():
        raise ValueError("Training seasonal table has uncovered bins")
    return table


def run_stage(config: dict, stage: str, output: Path) -> None:
    """Execute one immutable stage after its source and dependency preflight."""
    started = time.perf_counter()
    deadline = started + config["stages"][stage]["cap_seconds"]
    lock = verify_lock(output)
    if config != lock["config"]:
        raise ValueError("Runtime config differs from the frozen configuration")
    stage_dir = output / stage
    if stage_dir.exists():
        raise FileExistsError("Stage directory already exists; formal runs cannot be overwritten/retried")
    spec = config["stages"][stage]
    dependencies = {}
    if stage == "pilot":
        rates = config["training"]["learning_rates"]
    else:
        pilot_path = output / "pilot/summary.json"
        pilot = verify_stage(output, "pilot", config)
        if pilot["status"] != "complete":
            raise ValueError("Completed pilot required")
        rates = [pilot["selected_rate"]]
        dependencies["pilot/summary.json"] = digest(pilot_path)
        if stage == "confirmation":
            development_path = output / "development/summary.json"
            development = verify_stage(output, "development", config)
            if development["status"] != "complete":
                raise ValueError("Completed development required")
            dependencies["development/summary.json"] = digest(development_path)
            proof = output / "engineering.json"
            engineering = json.loads(proof.read_text())
            if not engineering.get("full_suite_completed") or not engineering.get("standard_benchmarks_completed"):
                raise ValueError("Completed engineering checks required; failures must be disclosed")
            dependencies["engineering.json"] = digest(proof)
        projection = pilot["projected_stage_seconds"][stage]
        if projection > spec["cap_seconds"]:
            raise TimeoutError(f"Projected {stage} runtime {projection:.1f}s exceeds prospective cap")
    plan = stage_plan(config, stage, rates)
    stage_dir.mkdir(parents=True)
    write_json(stage_dir / "plan.json", plan)
    write_json(stage_dir / "access_lock.json", {"source_lock_sha256": digest(output / "source_lock.json"),
                                               "dependencies": dependencies, "stage": stage,
                                               "created_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    summary = {"stage": stage, "status": "running", "environment": environment(),
               "source_lock": lock, "planned_fits": len(plan), "completed_fits": 0,
               "records": [], "datasets": [], "references": []}
    write_json(stage_dir / "summary.json", summary)
    current = None
    try:
        for room, tau, irregular in itertools.product(spec["room_seeds"], config["tau_ratios"], config["attendance"]):
            check_deadline(deadline)
            common = dict(config["data"], tau_ratio=tau, irregular=irregular)
            trajectory = generate_trajectory(room, DynamicsConfig(**common))
            normalizer = fit_normalizer(trajectory)
            train = make_windows(trajectory, "train", normalizer)
            val = make_windows(trajectory, "validation", normalizer)
            cell = f"room{room}_tau{tau}_irregular{int(irregular)}"
            testsets = []
            for shift, outage in itertools.product(config["test_shift"], config["test_outage"]):
                variant = generate_trajectory(room, DynamicsConfig(**common, shifted=shift, outage=outage))
                batch = make_windows(variant, "test", normalizer)
                # Test-only interventions must never alter fitting information.
                ntrainval = sum(variant.config.split_days[:2]) * (1440 // variant.config.dt_minutes)
                if not np.array_equal(variant.observed_ppm[:ntrainval], trajectory.observed_ppm[:ntrainval]):
                    raise ValueError("Test intervention changed fitting observations")
                testsets.append((shift, outage, batch))
                summary["datasets"].append(dict(room=room, tau=tau, irregular=irregular, shifted=shift,
                    outage=outage, normalizer=dataclasses.asdict(normalizer), metadata=variant.metadata,
                    train=batch_receipt(train), validation=batch_receipt(val), test=batch_receipt(batch)))
                np.savez_compressed(stage_dir / f"{cell}_shift{int(shift)}_outage{int(outage)}_cases.npz",
                    origin_indices=batch.origin_indices, origin_times=np.asarray(batch.origin_times).astype(str),
                    target_times=np.asarray(batch.target_times).astype(str), y_ppm=batch.y_ppm,
                    clean_y_ppm=batch.clean_y_ppm, sensor=batch.sensor, future_angles=batch.future_angles)
            ridge = fit_ridge(train, val, config["training"]["ridge_penalties"])
            seasonal_started = time.perf_counter()
            season = seasonal_table(trajectory)
            seasonal_seconds = time.perf_counter() - seasonal_started
            np.savez_compressed(stage_dir / f"{cell}_references.npz", ridge_weight=ridge["weight"],
                                ridge_bias=ridge["bias"], seasonal=season)
            for reference in config["references"]:
                for shift, outage, batch in testsets:
                    before = time.perf_counter()
                    if reference == "ridge":
                        p = batch.sensor[:, -1, 0, None] + ridge_design(batch.sensor) @ ridge["weight"] + ridge["bias"]
                        p = p * normalizer.std + normalizer.mean
                        fit_seconds, parameters = ridge["training_seconds"], ridge["parameters"]
                    elif reference == "persistence":
                        p = np.repeat(batch.sensor[:, -1, 0, None], config["data"]["horizon"], axis=1)
                        p = p * normalizer.std + normalizer.mean
                        fit_seconds, parameters = 0.0, 0
                    else:
                        bins = np.rint(batch.future_angles[..., 0] / (2*np.pi) * len(season)).astype(int) % len(season)
                        p = season[bins]
                        fit_seconds, parameters = seasonal_seconds, len(season)
                    infer_seconds = time.perf_counter() - before
                    record = dict(room=room, tau=tau, irregular=irregular, shifted=shift, outage=outage,
                                  arm=reference, seed=None, rate=None, training_seconds=fit_seconds,
                                  inference_seconds=infer_seconds, parameters=parameters,
                                  metrics=evaluate(p, batch, normalizer.std, config["analysis"]["endpoint_steps"]))
                    summary["references"].append(record)
                    np.savez_compressed(stage_dir / f"{cell}_{reference}_shift{int(shift)}_outage{int(outage)}.npz", prediction=p)
            for seed, rate, arm in itertools.product(spec["initialization_seeds"], rates, config["arms"]):
                current = dict(room=room, tau=tau, irregular=irregular, seed=seed, rate=rate, arm=arm)
                check_deadline(deadline)
                torch.manual_seed(seed)
                model = DynamicsForecasterV1(calendar=arm, lookback=config["data"]["lookback"],
                    input_size=3, horizon=config["data"]["horizon"], hidden_size=config["model"]["hidden_size"],
                    num_features=config["model"]["calendar_features"], init_kappa=config["model"]["initial_kappa"])
                history = fit_neural(model, train, val, config["training"], rate, seed, device, deadline)
                fit_id = f"{cell}_seed{seed}_lr{rate}_{arm}"
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "identity": current, "normalizer": dataclasses.asdict(normalizer), "history": history},
                           stage_dir / f"{fit_id}.pt")
                records = []
                for shift, outage, batch in testsets:
                    normalized, inference_seconds = predict_neural(model, batch, device)
                    p = normalized * normalizer.std + normalizer.mean
                    record = dict(current, shifted=shift, outage=outage, **history,
                                  inference_seconds=inference_seconds,
                                  metrics=evaluate(p, batch, normalizer.std, config["analysis"]["endpoint_steps"]))
                    if not shift and not outage:
                        offset = config["analysis"]["clock_defect_minutes"] * 2*np.pi / 1440
                        defective_batch = dataclasses.replace(batch,
                            future_angles=(batch.future_angles.astype(np.float64) + offset).astype(np.float32))
                        restored_batch = dataclasses.replace(batch,
                            future_angles=(defective_batch.future_angles.astype(np.float64) - offset).astype(np.float32))
                        defective, _ = predict_neural(model, defective_batch, device)
                        restored, _ = predict_neural(model, restored_batch, device)
                        record["clock_diagnostic"] = {
                            "offset_minutes": config["analysis"]["clock_defect_minutes"],
                            "restored_max_abs_normalized": float(np.max(np.abs(restored-normalized))),
                            "offset_metrics": evaluate(defective * normalizer.std + normalizer.mean, batch,
                                                       normalizer.std, config["analysis"]["endpoint_steps"])}
                    records.append(record)
                    np.savez_compressed(stage_dir / f"{fit_id}_shift{int(shift)}_outage{int(outage)}.npz", prediction=p)
                write_json(stage_dir / f"{fit_id}.json", {"identity": current, "status": "complete", "records": records})
                summary["records"].extend(records)
                summary["completed_fits"] += 1
                print(f"{stage} {summary['completed_fits']}/{len(plan)} {fit_id} val={history['validation_mse']:.5f}", flush=True)
                check_deadline(deadline)
        summary["status"] = "complete"
        if stage == "pilot":
            scores = {str(rate): float(np.mean([r["validation_mse"] for r in summary["records"] if r["rate"] == rate])) for rate in rates}
            summary["pilot_rate_scores"] = scores
            summary["selected_rate"] = min(rates, key=lambda r: (scores[str(r)], r))
        artifact_hashes = {p.name: digest(p) for p in sorted(stage_dir.iterdir()) if p.name != "summary.json"}
        write_json(stage_dir / "artifacts.json", artifact_hashes)
        elapsed = time.perf_counter() - started
        summary["elapsed_seconds"] = elapsed
        if stage == "pilot":
            summary["projected_stage_seconds"] = {
                name: elapsed * len(stage_plan(config, name, [summary["selected_rate"]])) / len(plan) * 1.5
                for name in ("development", "confirmation")}
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        try:
            import psutil
            summary["process_rss_bytes_at_end"] = psutil.Process().memory_info().rss
        except ImportError:
            summary["process_rss_bytes_at_end"] = None
        write_json(stage_dir / "summary.json", summary)
        check_deadline(deadline)
        write_json(stage_dir / "terminal.json", {"status": "complete", "elapsed_seconds_including_summary_write": time.perf_counter()-started,
                                               "summary_sha256": digest(stage_dir / "summary.json"),
                                               "artifacts_sha256": digest(stage_dir / "artifacts.json")})
        check_deadline(deadline)
    except BaseException as error:
        summary.update(status="failed", failed_fit=current, error=repr(error), traceback=traceback.format_exc(),
                       elapsed_seconds=time.perf_counter()-started)
        write_json(stage_dir / "summary.json", summary)
        raise


def main() -> None:
    """Dispatch explicit freeze/pilot/development/confirmation commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "pilot", "development", "confirmation"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/vm_n04_v1.json").read_text())
    if args.stage == "freeze":
        target = args.output_dir / "source_lock.json"
        if target.exists():
            raise FileExistsError("Source lock already exists")
        write_json(target, {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "baseline_revision": config["baseline_revision"],
                            "sources": {path: digest(ROOT / path) for path in SOURCES}, "config": config})
        print(f"Frozen {len(SOURCES)} source files: {digest(target)}")
    else:
        run_stage(config, args.stage, args.output_dir)


if __name__ == "__main__":
    main()
