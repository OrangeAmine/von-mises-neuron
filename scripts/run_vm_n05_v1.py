"""Reproducible, source-locked VM-N05 fixture, checks, study and analysis CLI."""

from __future__ import annotations

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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.uncertainty_data_v1 import (
    DelayedErrorLedger, NEVER, build_rows, scenario_trajectory, timeline_ns,
)
from src.uncertainty_calibration_v1 import CONDITIONAL, calibrate, fit_scale
from src.uncertainty_analysis_v1 import cell_table, paired_summary, validate_alignment

CONFIG = ROOT / "configs/vm_n05_v1.json"
SOURCES = [
    "configs/vm_n05_v1.json", "configs/vm_n04_v1.json", "docs/vm_n05_protocol_v1.md", "requirements.txt", "src/__init__.py",
    "src/dynamics_data_v1.py", "src/uncertainty_data_v1.py",
    "src/uncertainty_calibration_v1.py", "src/uncertainty_analysis_v1.py",
    "scripts/run_vm_n05_v1.py", "tests/test_vm_n05_v1.py",
]


def digest(path: Path) -> str:
    """Return exact on-disk byte identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    """Collect the full scientific and focused-test dependency closure."""
    return {p: digest(ROOT / p) for p in SOURCES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    """Write portable strict JSON; scientific nonfinite results must be explicit."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment() -> dict:
    """Record the selected interpreter, libraries, numerical device and revision."""
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "blas_threads": 1, "torch_threads": torch.get_num_threads()}


def check_config(config: dict) -> None:
    """Reject unintended changes to the versioned contract rather than ignoring them."""
    if config["version"] != "vm_n05_v1" or config["point_predictor"] != "latest_arrived_persistence":
        raise ValueError("unsupported contract")
    if config["arms"] != ["static", "static_matched", "rolling", "fourier", "hour_bin", "vm"]:
        raise ValueError("arm matrix differs from the declared contract")
    if config["scale"]["basis_count"] != 24 or config["regimes"]["phase_bins"] != 6:
        raise ValueError("version 1 requires 24 clock coordinates and six reporting bins")
    if config["compute"]["threads"] != 1 or config["compute"]["search_trials_per_conditional_arm"] != 1:
        raise ValueError("version 1 has one thread and no hyperparameter search")
    t = config["timeline_days"]
    if not (0 < t["scale_start"] < t["scale_end"] == t["calibration_start"] < t["evaluation_start"] < t["end"]):
        raise ValueError("invalid timeline")
    if t["end"] != sum(config["data"]["split_days"]) or t["evaluation_start"] != sum(config["data"]["split_days"][:2]):
        raise ValueError("N04 intervention boundary must match evaluation")
    inherited = json.loads((ROOT / "configs/vm_n04_v1.json").read_text())
    consumed = {s for stage in inherited["stages"].values() for s in stage["room_seeds"]}
    seeds = config["room_seeds"]
    if len(set(seeds)) != len(seeds) or set(seeds) & consumed or config["fixture_seed"] in seeds:
        raise ValueError("reserved seeds must be fresh, unique, and separate from fixtures")


def budget_record(ledger: DelayedErrorLedger, config: dict) -> dict:
    """Count label rows and distinct target measurements by permitted use."""
    times, r = timeline_ns(config), ledger.rows
    output = {}
    unions = {}
    for h in config["horizon_steps"]:
        entries = {
            "scale": ledger.indices(times["scale_end"] - 1, times["scale_start"], times["scale_end"], h),
            "static_calibration": ledger.indices(times["evaluation_start"], times["calibration_start"], times["evaluation_start"], h),
            "online_ever_eligible": ledger.indices(int(r.issue_ns.max()), times["calibration_start"], int(r.issue_ns.max()), h),
        }
        output[str(h)] = {name: {"label_rows": len(idx), "unique_targets": len(np.unique(r.target_ns[idx]))} for name, idx in entries.items()}
        raw_scale = (r.horizon == h) & (r.issue_ns >= times["scale_start"]) & (r.issue_ns < times["scale_end"])
        raw_cal = (r.horizon == h) & (r.issue_ns >= times["calibration_start"]) & (r.issue_ns < times["evaluation_start"])
        output[str(h)]["scale_origins_excluded_by_maturity"] = int(raw_scale.sum() - len(entries["scale"]))
        output[str(h)]["calibration_origins_not_in_static"] = int(raw_cal.sum() - len(entries["static_calibration"]))
        timed = ledger.timed_indices(int(r.issue_ns.max()), times["calibration_start"], int(r.issue_ns.max()), h)
        output[str(h)]["arrived_nonfinite_labels_excluded"] = int((~np.isfinite(r.truth[timed])).sum())
        for name, idx in entries.items():
            unions.setdefault(name, set()).update(r.target_ns[idx].tolist())
    output["distinct_targets_across_horizons"] = {k: len(v) for k, v in unions.items()}
    output["point_fit_labels"] = output["tuning_labels"] = 0
    return output


def run_rooms(output: Path, config: dict, seeds: list[int], cap: float, source_id: str) -> dict:
    """Fit nine scales per room and issue all four paired scenarios once."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + cap
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    progress = {"status": "running", "started_utc": utc_now(), "rooms": [], "source_id": source_id,
                "config_sha256": digest(CONFIG), "environment": environment()}
    save_json(output / "progress.json", progress)
    try:
        for seed in seeds:
            if time.perf_counter() >= deadline:
                raise TimeoutError("N05 prospective study cap reached")
            room_dir = output / str(seed)
            room_dir.mkdir()
            models, prefix = {}, None
            scenario_records = []
            for scenario in config["scenarios"]:
                trajectory = scenario_trajectory(seed, config, scenario)
                rows, exclusions = build_rows(trajectory, config)
                ledger = DelayedErrorLedger(rows, config["data"]["dt_minutes"] * 60 * 1_000_000_000)
                earlier = rows.issue_ns < timeline_ns(config)["evaluation_start"]
                # Include all pre-evaluation row arrays: their future truth can
                # cross the evaluation boundary, so compare only matured targets.
                earlier &= rows.target_ns < timeline_ns(config)["evaluation_start"]
                current = {k: v[earlier] for k, v in vars(rows).items()}
                if prefix is None:
                    prefix = current
                    for h in config["horizon_steps"]:
                        for arm in CONDITIONAL:
                            if time.perf_counter() >= deadline:
                                raise TimeoutError("N05 fit budget expired")
                            models[(h, arm)] = fit_scale(ledger, h, arm, config, device)
                    save_json(room_dir / "factors.json", {f"{h}:{arm}": m.record() for (h, arm), m in models.items()})
                elif any(not np.array_equal(prefix[k], v) for k, v in current.items()):
                    raise ValueError("paired scenarios unexpectedly differ before evaluation")
                name = scenario["name"]
                np.savez_compressed(room_dir / f"{name}_rows.npz", **vars(rows))
                intervals = pd.DataFrame(calibrate(ledger, models, config, deadline))
                intervals["room_seed"], intervals["scenario"] = seed, name
                intervals["units"], intervals["point_model_id"] = "ppm", config["point_predictor"]
                intervals["source_id"], intervals["config_id"] = source_id, digest(CONFIG)
                validate_alignment(intervals, config)
                path = room_dir / f"{name}_intervals.csv.gz"
                intervals.to_csv(path, index=False, float_format="%.17g", compression="gzip")
                scenario_records.append({"name": name, "interval_rows": len(intervals), "exclusions": exclusions,
                                         "label_budget": budget_record(ledger, config), "trajectory_metadata": trajectory.metadata,
                                         "data_config": asdict(trajectory.config),
                                         "evaluation_noise_multiplier": scenario["evaluation_noise_multiplier"]})
            progress["rooms"].append({"seed": seed, "scenarios": scenario_records,
                                       "fitted_parameters_per_scale": 29, "fitted_scales": len(models),
                                       "total_scale_fit_seconds": sum(m.fit_seconds for m in models.values())})
            progress["elapsed_seconds"] = time.perf_counter() - started
            save_json(output / "progress.json", progress)
            print(f"Completed room {seed}: {progress['elapsed_seconds']:.2f} seconds", flush=True)
        progress.update(status="complete", completed_utc=utc_now(), elapsed_seconds=time.perf_counter() - started,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        if progress["elapsed_seconds"] > cap:
            raise TimeoutError("N05 stage exceeded declared cap")
        progress["artifacts"] = {str(p.relative_to(output)).replace("\\", "/"): digest(p)
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "progress.json"}
        save_json(output / "manifest.json", progress)
        save_json(output / "progress.json", progress)
        return progress
    except Exception as error:
        progress.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        save_json(output / "progress.json", progress)
        raise


def verify_manifest(directory: Path) -> dict:
    """Validate completed artifact identities before using their content."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("incomplete study")
    for name, expected in manifest["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("artifact identity mismatch")
    return manifest


def export_contract(frame: pd.DataFrame, output: Path, config: dict) -> None:
    """Export a day and a small outage example; truth stays explicitly offline."""
    sample = frame[(frame.room_seed == config["room_seeds"][0]) & (frame.scenario == "stable")].copy()
    origins = sorted(sample.issue_ns.unique())[:24]
    sample = sample[sample.issue_ns.isin(origins)].sort_values(["issue_ns", "horizon_steps", "arm"])
    outage = frame[(frame.room_seed == config["room_seeds"][0]) & (frame.scenario == "shifted_noisy_outage")]
    absent_origins = sorted(outage[outage.observation_available_ns == NEVER].issue_ns.unique())[:4]
    sample = pd.concat([sample, outage[outage.issue_ns.isin(absent_origins)]], ignore_index=True)
    for source, target in (("issue_ns", "issue_time_utc"), ("target_ns", "target_time_utc"),
                           ("observation_available_ns", "observation_available_time_utc"),
                           ("calibration_cutoff_ns", "calibration_cutoff_utc")):
        sample[target] = [pd.Timestamp(int(v), tz="UTC").isoformat() if int(v) != NEVER else ""
                          for v in sample[source]]
    sample["horizon_minutes"] = sample.horizon_steps * config["data"]["dt_minutes"]
    columns = ["room_seed", "scenario", "arm", "issue_time_utc", "target_time_utc", "observation_available_time_utc",
               "horizon_minutes", "point_model_id", "point_ppm", "lower_ppm", "upper_ppm", "units",
               "calibration_cutoff_utc", "calibration_count", "source_id", "config_id", "truth_ppm_offline"]
    sample[columns].to_csv(output / "interval_contract_sample.csv", index=False, float_format="%.17g")


def analyze(study: Path, output: Path, config: dict) -> dict:
    """Verify saved intervals, score every cell and create compact report artifacts."""
    started = time.perf_counter()
    lock = json.loads((study / "source_lock.json").read_text())
    if lock["sources"] != sources():
        raise ValueError("scientific sources changed after freeze")
    manifest = verify_manifest(study / "run")
    if manifest["source_id"] != digest(study / "source_lock.json"):
        raise ValueError("run source identifier differs from frozen lock")
    if [r["seed"] for r in manifest["rooms"]] != config["room_seeds"]:
        raise ValueError("room matrix incomplete")
    frames = []
    for seed in config["room_seeds"]:
        for scenario in config["scenarios"]:
            frames.append(pd.read_csv(study / "run" / str(seed) / f"{scenario['name']}_intervals.csv.gz", float_precision="round_trip"))
    frame = pd.concat(frames, ignore_index=True)
    validate_alignment(frame, config)
    cells = cell_table(frame, config)
    summary = paired_summary(frame, cells, config)
    summary.update(version="vm_n05_v1", source_id=digest(study / "source_lock.json"),
                   config_id=digest(CONFIG), manifest_sha256=digest(study / "run/manifest.json"),
                   run_seconds=manifest["elapsed_seconds"], interval_rows=len(frame),
                   total_scale_fits=sum(r["fitted_scales"] for r in manifest["rooms"]),
                   total_scale_fit_seconds=sum(r["total_scale_fit_seconds"] for r in manifest["rooms"]),
                   environment=manifest["environment"],
                   flags={k: int(frame[k].sum()) for k in ("scale_fallback", "scale_floor", "scale_ceiling", "small_pool_fallback", "infinite_interval")})
    output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(output / "cells.csv", index=False, float_format="%.10g")
    group = ["scenario", "arm", "horizon_steps", "stratum"]
    metrics = ["interval_score", "coverage", "width", "lower_miss", "upper_miss", "point_mae", "point_rmse"]
    balanced = cells.groupby(group, sort=True)[metrics].mean()
    balanced["n"] = cells.groupby(group).n.sum()
    balanced["nonempty_rooms"] = cells.groupby(group).n.apply(lambda v: int((v > 0).sum()))
    balanced.to_csv(output / "comparison.csv", float_format="%.10g")
    export_contract(frame, output, config)
    # Static scientific figure; no browser or sibling runtime dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"static": "#9b9b9b", "static_matched": "#555555", "rolling": "#2274a5",
              "fourier": "#ed9c32", "hour_bin": "#5d9c59", "vm": "#9856a6"}
    primary = balanced.reset_index()
    primary = primary[(primary.stratum == "all") & (primary.horizon_steps == config["analysis"]["primary_horizon"])]
    x = np.arange(len(config["scenarios"]))
    names = [s["name"] for s in config["scenarios"]]
    for i, arm in enumerate(config["arms"]):
        series = primary[primary.arm == arm].set_index("scenario").loc[names]
        axes[0].bar(x + (i - 2.5) * .13, series.interval_score, .13, label=arm, color=colors[arm])
        axes[1].plot(x, series.coverage * 100, marker="o", label=arm, color=colors[arm])
    for ax in axes:
        ax.set_xticks(x, ["Stable", "Shifted", "Noisy", "Shift + noise\n+ outage"])
        ax.grid(axis="y", alpha=.2)
    axes[0].set(ylabel="Interval score (ppm; lower is better)", title="One-hour forecasts: point predictions fixed")
    axes[1].set(ylabel="Empirical coverage (%)", title="Coverage does not establish decision value")
    axes[1].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    summary["analysis_seconds"] = time.perf_counter() - started
    if summary["analysis_seconds"] > config["compute"]["analysis_cap_seconds"]:
        raise TimeoutError("analysis exceeded declared cap")
    save_json(output / "summary.json", summary)
    save_json(output / "manifest.json", {"artifacts": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main() -> None:
    """Dispatch isolated stages; reject implicit overwrite and formal retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["engineering", "fixture", "freeze", "run", "analyze"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engineering-dir", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--study-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    check_config(config)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.stage == "engineering":
            args.output_dir.mkdir(parents=True, exist_ok=False)
            before = sources()
            started = time.perf_counter()
            command = [sys.executable, "-m", "pytest", "tests/", "-q", "--junitxml=" + str(args.output_dir / "full_suite.xml")]
            completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.output_dir / "full_suite.log").write_text(completed.stdout, encoding="utf-8")
            print(completed.stdout, flush=True)
            if sources() != before:
                raise ValueError("sources changed during engineering checks")
            suites = list(ET.parse(args.output_dir / "full_suite.xml").getroot().iter("testsuite"))
            counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
            save_json(args.output_dir / "receipt.json", {"sources": before, "exit_code": completed.returncode,
                      "counts": counts, "seconds": time.perf_counter() - started,
                      "junit_sha256": digest(args.output_dir / "full_suite.xml"), "command": command,
                      "environment": environment(), "created_utc": utc_now()})
            if completed.returncode:
                raise SystemExit(completed.returncode)
        elif args.stage == "fixture":
            before = sources()
            record = run_rooms(args.output_dir, config, [config["fixture_seed"]], config["compute"]["fixture_cap_seconds"], "nonreserved-fixture")
            if sources() != before:
                raise ValueError("sources changed during fixture")
            projection = record["elapsed_seconds"] * len(config["room_seeds"]) * 1.5
            save_json(args.output_dir / "feasibility.json", {"sources": before, "created_utc": utc_now(),
                      "projected_formal_seconds": projection, "within_cap": projection <= config["compute"]["study_cap_seconds"]})
        elif args.stage == "freeze":
            if args.engineering_dir is None or args.fixture_dir is None:
                parser.error("freeze requires --engineering-dir and --fixture-dir")
            engineering = json.loads((args.engineering_dir / "receipt.json").read_text())
            fixture = json.loads((args.fixture_dir / "feasibility.json").read_text())
            verify_manifest(args.fixture_dir)
            if engineering["exit_code"] or engineering["counts"]["failures"] or engineering["counts"]["errors"]:
                raise ValueError("engineering checks must pass")
            if engineering["junit_sha256"] != digest(args.engineering_dir / "full_suite.xml"):
                raise ValueError("engineering artifact mismatch")
            if engineering["sources"] != sources() or fixture["sources"] != sources() or not fixture["within_cap"]:
                raise ValueError("source/feasibility prerequisite failed")
            args.output_dir.mkdir(parents=True, exist_ok=False)
            save_json(args.output_dir / "source_lock.json", {"version": config["version"], "created_utc": utc_now(),
                      "sources": sources(), "config": config, "environment": environment(),
                      "engineering": engineering, "fixture": fixture,
                      "engineering_receipt_sha256": digest(args.engineering_dir / "receipt.json"),
                      "fixture_manifest_sha256": digest(args.fixture_dir / "manifest.json")})
            print("Source lock " + digest(args.output_dir / "source_lock.json"), flush=True)
        elif args.stage == "run":
            lock = json.loads((args.output_dir / "source_lock.json").read_text())
            if lock["sources"] != sources() or lock["config"] != config:
                raise ValueError("source/config differs from frozen contract")
            run_rooms(args.output_dir / "run", config, config["room_seeds"], config["compute"]["study_cap_seconds"], digest(args.output_dir / "source_lock.json"))
        else:
            if args.study_dir is None:
                parser.error("analyze requires --study-dir")
            summary = analyze(args.study_dir, args.output_dir, config)
            print(json.dumps({k: summary[k] for k in ("primary_improvement", "paired_room_95_interval", "scientific_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
