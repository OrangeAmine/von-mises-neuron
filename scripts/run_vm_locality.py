"""Run the prospective locality study with immutable cells and bounded phases."""

from __future__ import annotations

import argparse
import _thread
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.locality_protocol import (
    CONFIG_PATH, create_confirmation_lock, environment, file_hash, object_hash,
    planned_runs, read_json, snapshot_sources, source_manifest,
    verify_confirmation_lock, verify_run_manifest, verify_pilot_selection,
    verify_feasibility, write_json,
)


def relative(path: Path) -> str:
    """Store portable evidence paths beneath the repository root."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


@contextmanager
def watchdog(output: Path, deadline: float):
    """Preserve pending/running cells even if a native call exceeds the phase ceiling."""
    done = threading.Event()
    def watch() -> None:
        if not done.wait(max(0, deadline - time.time())):
            print("Declared phase deadline reached; retaining incomplete evidence", flush=True)
            _thread.interrupt_main()
            if not done.wait(3):
                write_json(output / "hard_timeout.json", {"status": "hard_timeout", "deadline": deadline})
                os._exit(124)
    worker = threading.Thread(target=watch, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join(timeout=1)


def run_phase(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Fit every registered cell once; existing invocation directories are rejected."""
    from src.locality_tasks import generate_locality_task, initializer
    from src.locality_training import fit_locality, warmup_locality

    phase, output = args.phase, args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Study outputs must be under this repository")
    rates, lock = None, None
    prerequisite_hashes: dict[str, str] = {}
    if phase == "development":
        if args.selection is None or args.feasibility is None:
            raise ValueError("Development requires selected pilot rates and feasibility")
        selected = verify_pilot_selection(args.selection, config)
        feasible = verify_feasibility(args.feasibility, config)
        if feasible["pilot_selection"] != relative(args.selection):
            raise ValueError("Feasibility must bind this exact pilot selection")
        rates = selected["learning_rates"]
        for path in (args.selection, args.feasibility):
            prerequisite_hashes[relative(path)] = file_hash(path)
    elif phase == "confirmation":
        lock = verify_confirmation_lock(args.lock, config)
        rates = lock["learning_rates"]
        prerequisite_hashes[relative(args.lock)] = file_hash(args.lock)
    cells = planned_runs(config, phase, rates)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ceiling = config["phases"][f"{phase}_ceiling_seconds"]
    started = time.time()
    manifest = {"kind": "locality_run_v2", "phase": phase, "status": "running",
                "complete": False, "config_hash": object_hash(config),
                "source_hashes": snapshot_sources(output), "environment": environment(),
                "started_unix": started, "deadline_unix": started + ceiling,
                "phase_ceiling_seconds": ceiling, "learning_rates": rates,
                "lock_hash": file_hash(args.lock) if lock else None,
                "prerequisite_hashes": prerequisite_hashes, "evidence_hashes": {},
                "planned_count": len(cells), "rows": [{"cell": c, "status": "pending"} for c in cells]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", config)
    tasks: dict[tuple[str, int], Any] = {}
    inits: dict[tuple[str, int, int], Any] = {}
    current = None
    try:
        with threadpool_limits(limits=1), watchdog(output, started + ceiling):
            warm_start = time.perf_counter()
            warmup_locality(device=device)
            manifest["warmup_seconds"] = time.perf_counter() - warm_start
            for index, current in enumerate(manifest["rows"]):
                if time.time() >= started + ceiling:
                    raise TimeoutError("Phase ceiling reached before next fit")
                cell_start = time.perf_counter()
                cell = current["cell"]
                current["status"] = "running"
                # Keep the full registered matrix on disk, then journal each fit.
                # Rewriting its growing evidence closure twice per fit is quadratic.
                write_json(output / "progress.json", {"index": index, "cell_id": cell["id"],
                           "status": "running", "completed_count": manifest.get("completed_count", 0)},
                           exclusive=False)
                task_key = (cell["family"], cell["replicate"])
                if task_key not in tasks:
                    task = generate_locality_task(config, phase, *task_key,
                                                  confirmation_lock=args.lock, root=ROOT)
                    tasks[task_key] = task
                    task_path = output / "tasks" / f"{task_key[0]}_{task_key[1]:03d}.json"
                    write_json(task_path, task.to_dict())
                    manifest["evidence_hashes"][relative(task_path)] = file_hash(task_path)
                task = tasks[task_key]
                init_key = (*task_key, cell["initialization"])
                if init_key not in inits:
                    inits[init_key] = initializer(config, phase, *init_key)
                arrays = task.for_budget(cell["total_labels"], cell["noise_std"])
                fit_start = time.perf_counter()
                bundle = fit_locality(cell["model_id"], arrays, inits[init_key], config,
                                      learning_rate=cell["learning_rate"], device=device)
                fit_call_seconds = time.perf_counter() - fit_start
                cell_dir = output / "fits" / cell["id"]
                cell_dir.mkdir(parents=True, exist_ok=False)
                artifact_path = cell_dir / "artifacts.pt"
                torch.save(bundle["artifacts"], artifact_path)
                summary_path = cell_dir / "summary.json"
                write_json(summary_path, bundle["summary"])
                current.update(status=bundle["summary"]["status"],
                               summary_path=relative(summary_path), artifact_path=relative(artifact_path),
                               task_path=relative(output / "tasks" / f"{task_key[0]}_{task_key[1]:03d}.json"),
                               summary_hash=file_hash(summary_path), artifact_hash=file_hash(artifact_path),
                               fit_call_seconds=fit_call_seconds)
                record_path = cell_dir / "record.json"
                write_json(record_path, current)
                current["record_path"] = relative(record_path)
                for path in (summary_path, artifact_path):
                    manifest["evidence_hashes"][relative(path)] = file_hash(path)
                manifest["evidence_hashes"][relative(record_path)] = file_hash(record_path)
                manifest["completed_count"] = sum(r["status"] == "completed" for r in manifest["rows"])
                write_json(output / "progress.json", {"index": index, "cell_id": cell["id"],
                           "status": current["status"], "completed_count": manifest["completed_count"]},
                           exclusive=False)
                # Include progress I/O in measured per-cell orchestration cost.
                current["wall_seconds"] = time.perf_counter() - cell_start
                if current["status"] == "interrupted":
                    raise InterruptedError("Fit interrupted; retaining this cell and all remaining pending cells")
                if index % 10 == 0 or index + 1 == len(cells):
                    print(f"{phase}: {index+1}/{len(cells)}, {cell['model_id']}, {current['status']}", flush=True)
        manifest["complete"] = all(row["status"] == "completed" for row in manifest["rows"])
        manifest["status"] = "completed" if manifest["complete"] else "incomplete"
    except BaseException as error:
        if current is not None and current["status"] == "running":
            current["status"] = "interrupted"
        manifest["status"] = "interrupted"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        manifest["ended_unix"] = time.time()
        manifest["elapsed_seconds"] = manifest["ended_unix"] - started
        manifest["deadline_respected"] = manifest["ended_unix"] <= manifest["deadline_unix"]
        if not manifest["deadline_respected"]:
            manifest["complete"] = False
        manifest["completed_count"] = sum(r["status"] == "completed" for r in manifest["rows"])
        write_json(output / "manifest.json", manifest, exclusive=False)
        # The small terminal receipt times the full matrix flush and hashing.
        # Its own atomic metadata write is the only excluded finalization I/O.
        manifest_digest = file_hash(output / "manifest.json")
        completed_at = time.time()
        receipt = {"kind": "locality_completion_v2", "manifest_hash": manifest_digest,
                   "ended_unix": completed_at, "elapsed_seconds": completed_at-started,
                   "deadline_respected": completed_at <= manifest["deadline_unix"]}
        write_json(output / "completion.json", receipt)
    print(f"{phase}: {manifest['status']}; {receipt['elapsed_seconds']:.2f}s", flush=True)
    return 0 if manifest["complete"] and receipt["deadline_respected"] else 1


def completed_run(path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate exact planned cells and load immutable summaries without refitting."""
    manifest, summaries = verify_run_manifest(path, config)
    if (manifest.get("kind") != "locality_run_v2" or manifest.get("complete") is not True
            or manifest.get("status") != "completed" or not manifest.get("deadline_respected")):
        raise ValueError("A complete, within-deadline invocation is required")
    expected = planned_runs(config, manifest["phase"], manifest.get("learning_rates"))
    if [r["cell"] for r in manifest["rows"]] != expected:
        raise ValueError("Experiment matrix differs from the protocol")
    return manifest, summaries


def select_pilots(path: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Choose rates using only pilot noisy validation, never timing-panel/test errors."""
    manifest, summaries = completed_run(path, config)
    if manifest["phase"] != "pilot":
        raise ValueError("Pilot invocation required")
    scores, choices = {}, {}
    for model in config["models"]:
        if model["kind"] == "ridge":
            continue
        scores[model["id"]] = []
        for lr in config["training"]["pilot_learning_rates"]:
            vals = [float(summary["best_validation_mse"]) for row, summary in zip(manifest["rows"], summaries)
                    if row["cell"]["role"] == "selection" and row["cell"]["model_id"] == model["id"]
                    and row["cell"]["learning_rate"] == lr]
            expected = len(config["tasks"]["families"]) * len(config["tasks"]["noise_levels"])
            if len(vals) != expected or not np.isfinite(vals).all():
                raise ValueError("Incomplete pilot candidate")
            scores[model["id"]].append({"learning_rate": lr, "mean_validation_mse": float(np.mean(vals)), "count": len(vals)})
        choices[model["id"]] = min(scores[model["id"]], key=lambda r: (r["mean_validation_mse"], r["learning_rate"]))["learning_rate"]
    result = {"kind": "locality_pilot_selection_v2", "complete": True,
              "input_manifest": relative(path),
              "config_hash": object_hash(config), "source_hashes": source_manifest(),
              "evidence_hashes": {relative(path): file_hash(path), **manifest["evidence_hashes"]},
              "learning_rates": choices, "scores": scores,
              "selection_data": "128-label pilot noisy validation only"}
    write_json(output, result)
    return result


def feasibility(pilot: Path, selection: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Project complete matrices using full adaptive allowances and measured overhead."""
    manifest, summaries = completed_run(pilot, config)
    selected = verify_pilot_selection(selection, config)
    if manifest["phase"] != "pilot" or selected["input_manifest"] != relative(pilot):
        raise ValueError("Complete pilots and choices required")
    fixed = [row["fit_call_seconds"] for row in manifest["rows"]
             if row["cell"]["role"] == "timing" and row["cell"]["kind"] == "ridge"]
    # Entire fit-call noncompute + I/O counts as overhead; fit time key is fixed by trainer.
    overhead = [max(0., row["wall_seconds"] - float(summary["fit_seconds"]))
                for row, summary in zip(manifest["rows"], summaries)]
    overshoot = max(float(s.get("overshoot_seconds", 0.)) for row, s in zip(manifest["rows"], summaries)
                    if row["cell"]["kind"] != "ridge")
    p90fixed, p90overhead = float(np.quantile(fixed, .9)), float(np.quantile(overhead, .9))
    estimates = {}
    for phase in ("development", "confirmation"):
        rows = planned_runs(config, phase, selected["learning_rates"])
        adaptive = sum(row["kind"] != "ridge" for row in rows)
        seconds = adaptive * (config["training"]["primary_time_seconds"] + overshoot)
        seconds += (len(rows)-adaptive)*p90fixed + len(rows)*p90overhead + 60
        estimates[phase] = {"records": len(rows), "adaptive": adaptive, "fixed": len(rows)-adaptive,
                            "projected_seconds": seconds, "ceiling_seconds": config["phases"][f"{phase}_ceiling_seconds"],
                            "passed": seconds <= config["phases"][f"{phase}_ceiling_seconds"]}
    result = {"kind": "locality_feasibility_v2", "complete": True,
              "input_manifest": relative(pilot), "pilot_selection": relative(selection),
              "all_passed": all(row["passed"] for row in estimates.values()),
              "config_hash": object_hash(config), "source_hashes": source_manifest(),
              "evidence_hashes": {relative(pilot): file_hash(pilot), relative(selection): file_hash(selection),
                                  **manifest["evidence_hashes"]}, "estimates": estimates,
              "measured": {"fixed_maxshape_call_p90": p90fixed, "nonfit_overhead_p90": p90overhead,
                           "max_overshoot": overshoot}, "interpretation": "projection, not a worst-case runtime bound"}
    write_json(output, result)
    return result


def main() -> int:
    """Dispatch run, pilot selection, measured feasibility and lock creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--phase", choices=("pilot", "development", "confirmation"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--selection", type=Path)
    run.add_argument("--feasibility", type=Path)
    run.add_argument("--lock", type=Path)
    select = commands.add_parser("select-pilots")
    select.add_argument("--pilot", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    feasible = commands.add_parser("feasibility")
    feasible.add_argument("--pilot", type=Path, required=True)
    feasible.add_argument("--selection", type=Path, required=True)
    feasible.add_argument("--output", type=Path, required=True)
    lock = commands.add_parser("lock")
    for name in ("pilot-selection", "development-analysis", "feasibility", "independent-audit", "validation", "output"):
        lock.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "run":
        return run_phase(args, config)
    if args.command == "select-pilots":
        result = select_pilots(args.pilot, args.output, config)
    elif args.command == "feasibility":
        result = feasibility(args.pilot, args.selection, args.output, config)
    else:
        inputs = {name: getattr(args, name) for name in ("pilot_selection", "development_analysis", "feasibility", "independent_audit", "validation")}
        result = create_confirmation_lock(args.output, config, inputs)
    print({k: result[k] for k in ("kind", "complete", "all_passed", "learning_rates", "estimates") if k in result})
    return 0 if result.get("all_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
