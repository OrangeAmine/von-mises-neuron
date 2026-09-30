"""Run the frozen VM-N01 phases with serialized fitting and immutable evidence.

Use ``--help`` for source freeze, pilot, development, lock, and confirmation.
The report uses recorded predictions; no stage retries or historical reruns.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.smooth_local_protocol_v1 import (digest, file_hash, inventory, load_config,
    scientific_paths, verify_inventory, verify_lock, verify_source_lock, write_json)


class _PhaseFailure:
    """Capture outer-loop failures so completed attempts receive a terminal receipt."""

    error: str | None = None

    def __enter__(self) -> "_PhaseFailure":
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        if exception is not None:
            self.error = f"{exception_type.__name__}: {exception}"
        return exception is not None


def complete_row(row: dict[str, Any]) -> bool:
    """Require successful execution and finite scientific/selection metrics."""
    return row.get("status") in {"complete", "completed", "ok", "success"} and all(
        isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        and math.isfinite(row[key]) and row[key] >= 0
        for key in ("normalized_mse", "noisy_validation_mse"))


def utc() -> str:
    """Return an unambiguous UTC receipt timestamp."""
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> Any:
    """Load a UTF-8 JSON receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def environment() -> dict[str, Any]:
    """Record hardware/software and worktree identity, not just Git HEAD."""
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))}


def planned(config: dict[str, Any], phase: str, rates: dict[str, float]) -> list[dict[str, Any]]:
    """Enumerate every prospective fit before accessing any target."""
    result = []
    budgets = config["tasks"]["pilot_label_budgets" if phase == "pilot" else "label_budgets"]
    for family in config["tasks"]["families"]:
        for replicate in range(config["tasks"]["replicates"][phase]):
            for labels in budgets:
                for initialization in range(config["tasks"]["initializations"][phase]):
                    for spec in config["models"]:
                        candidates = ([None] if spec["kind"] == "ridge" else
                            config["training"]["pilot_learning_rates"] if phase == "pilot" else [rates[spec["id"]]])
                        for learning_rate in candidates:
                            result.append({"fit_id": f"fit_{len(result):05d}", "phase": phase,
                                "family": family, "replicate": replicate, "labels": labels,
                                "initialization": initialization, "model_id": spec["id"], "learning_rate": learning_rate})
    if len(result) != config["phases"]["planned_fits"][phase]:
        raise ValueError("Planned matrix differs from prospective fit count")
    return result


def feasibility(config: dict[str, Any], pilot: dict[str, Any], phase: str,
                verification_seconds: float = 0.) -> dict[str, Any]:
    """Project full adaptive caps and observed maximum fixed/nonfit costs."""
    rows = planned(config, phase, pilot["selected_rates"])
    kinds = {s["id"]: s["kind"] for s in config["models"]}
    adaptive = sum(kinds[r["model_id"]] != "ridge" for r in rows)
    fixed = len(rows) - adaptive
    timing = pilot["timing"]
    projected = (adaptive * (max(config["training"]["time_budgets_seconds"]) + timing["maximum_adaptive_overshoot"])
                 + fixed * timing["maximum_fixed_fit_seconds"]
                 + len(rows) * timing["maximum_nonfit_seconds"] + 60 + verification_seconds)
    ceiling = config["phases"]["ceiling_seconds"][phase]
    return {"phase": phase, "adaptive_fits": adaptive, "fixed_fits": fixed,
            "projected_seconds": projected, "ceiling_seconds": ceiling,
            "verification_seconds": verification_seconds, "passed": projected <= ceiling,
            "pilot_timing": timing}


def rows_from_bundle(item: dict[str, Any], spec: dict[str, Any], bundle: dict[str, Any],
                     config: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep a row for every allowance, including failed or unavailable states."""
    rows = []
    for allowance in config["training"]["time_budgets_seconds"]:
        key = format(allowance, "g")
        selected = bundle.get("selected", {}).get(key)
        checkpoint = bundle.get("checkpoints", [])[selected] if selected is not None else {}
        metric = bundle.get("metrics", {}).get(key, {})
        status = bundle.get("status", "failed")
        if selected is None:
            status = "no_eligible_checkpoint" if status in ("complete", "completed", "ok", "success") else status
        rows.append(dict(item, allowance=allowance, status=status,
            normalized_mse=metric.get("normalized_mse"), noisy_validation_mse=checkpoint.get("validation_mse"),
            selected_step=checkpoint.get("step"), selected_seconds=checkpoint.get("elapsed"),
            fit_seconds=bundle.get("fit_seconds"), total_seconds=bundle.get("total_seconds"),
            features=spec["features"], parameters=spec["parameters"]))
    return rows


def run_phase(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Execute one immutable planned matrix, retaining failures and late states."""
    phase_start = time.perf_counter()
    from src.smooth_local_tasks_v1 import generate_task, cell_arrays, initializer_rotation
    from src.smooth_local_training_v1 import fit_smooth_local
    from src.smooth_local_analysis_v1 import select_learning_rates, select_models

    phase = args.command
    source = verify_source_lock(args.source_lock, config)
    pilot = read(args.pilot_dir / "summary.json") if phase != "pilot" else None
    rates = pilot["selected_rates"] if pilot else {}
    if pilot and not pilot["complete"]:
        raise ValueError("Incomplete pilot cannot authorize another phase")
    if pilot:
        pilot_receipt = read(args.pilot_dir / "receipt.json")
        if not pilot_receipt["complete"] or pilot_receipt["summary_sha256"] != file_hash(args.pilot_dir / "summary.json"):
            raise ValueError("Pilot terminal receipt failed or was altered")
    if phase == "confirmation":
        lock = verify_lock(args.confirmation_lock, config)
        if rates != lock["selected_rates"]:
            raise ValueError("Pilot rates differ from lock")
    elif pilot and not feasibility(config, pilot, phase)["passed"]:
        raise ValueError("Measured development feasibility failed")
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "results/vm_n01_v1").resolve()):
        raise ValueError("Use a new contained results/vm_n01_v1/ directory")
    output.mkdir(parents=True, exist_ok=False)
    matrix = planned(config, phase, rates)
    write_json(output / "plan.json", {"created_utc": utc(), "phase": phase, "matrix": matrix,
        "environment": environment(), "source_lock_sha256": file_hash(args.source_lock),
        "config_sha256": digest(config)})
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Warm the actual numerical backend before any timed fit, without labels.
    with threadpool_limits(limits=1):
        warm = torch.eye(40, device=device, dtype=torch.float64)
        torch.linalg.cholesky(warm)
        torch.optim.Adam([torch.nn.Parameter(warm)], lr=.01)
        if device == "cuda":
            torch.cuda.synchronize()
    start = phase_start
    deadline = config["phases"]["ceiling_seconds"][phase]
    specs = {s["id"]: s for s in config["models"]}
    bundles, flat_rows, task, task_key, nonfit_durations = [], [], None, None, []
    interrupted = None
    with _PhaseFailure() as phase_failure, threadpool_limits(limits=1):
        for number, item in enumerate(matrix):
            if time.perf_counter() - start >= deadline:
                interrupted = "phase_deadline"
                break
            nonfit_start = time.perf_counter()
            key = (item["family"], item["replicate"])
            if key != task_key:
                verify_source_lock(args.source_lock, config)
                task = generate_task(config, phase, *key, confirmation_lock=args.confirmation_lock)
                task_key = key
                prefix = f"task_{key[0]}_{key[1]}"
                np.savez_compressed(output / f"{prefix}.npz", **task["arrays"])
                write_json(output / f"{prefix}.json", task["metadata"])
            arrays = cell_arrays(task, item["labels"])
            rotation = initializer_rotation(config, phase, *key, item["initialization"])
            spec = specs[item["model_id"]]
            fit_start = time.perf_counter()
            try:
                bundle = fit_smooth_local(spec, arrays, rotation, item["learning_rate"], config["training"], device=device)
            except Exception as error:
                bundle = {"status": "failed", "failures": [repr(error)], "checkpoints": [],
                          "selected": {}, "metrics": {}, "fit_seconds": time.perf_counter()-fit_start,
                          "total_seconds": time.perf_counter()-fit_start}
            bundle["identity"] = dict(item, rotation=rotation, task_sha256=digest(task["metadata"]))
            checkpoint_path = output / f"{item['fit_id']}.pt"
            torch.save(bundle, checkpoint_path)
            rows = rows_from_bundle(item, spec, bundle, config)
            compact = {**item, "status": bundle["status"], "rows": rows,
                "fit_seconds": bundle["fit_seconds"], "overshoot_seconds": bundle.get("overshoot_seconds", 0.),
                "max_block_seconds": bundle.get("max_block_seconds", 0.),
                "checkpoint_count": len(bundle["checkpoints"]), "failures": bundle.get("failures", []),
                "artifact_sha256": file_hash(checkpoint_path)}
            compact["nonfit_seconds_before_record_write"] = max(0., time.perf_counter()-nonfit_start-bundle["fit_seconds"])
            write_json(output / f"{item['fit_id']}.json", compact)
            bundles.append(compact)
            flat_rows.extend(rows)
            if number % 25 == 0 or number + 1 == len(matrix):
                print(f"{phase}: {number+1}/{len(matrix)} fits, {time.perf_counter()-start:.1f}s", flush=True)
            nonfit_durations.append(max(0., time.perf_counter()-nonfit_start-bundle["fit_seconds"]))
            if bundle["status"] == "interrupted":
                interrupted = "fit_interrupted"
                break
    if phase_failure.error:
        interrupted = f"{phase_failure.error}; current_fit={item['fit_id']}"
    complete = (interrupted is None and len(bundles) == len(matrix)
                and len(flat_rows) == len(matrix)*len(config["training"]["time_budgets_seconds"])
                and all(complete_row(r) for r in flat_rows))
    timing = {
        "maximum_adaptive_overshoot": max((r["max_block_seconds"] for r in bundles if specs[r["model_id"]]["kind"] != "ridge"), default=0.),
        "maximum_realized_overshoot": max((r["overshoot_seconds"] for r in bundles), default=0.),
        "maximum_fixed_fit_seconds": max((r["fit_seconds"] for r in bundles if specs[r["model_id"]]["kind"] == "ridge"), default=0.),
        "maximum_nonfit_seconds": max(nonfit_durations, default=0.),
        "nonfit_seconds_by_attempt": nonfit_durations,
        "sum_fit_seconds": sum(r["fit_seconds"] for r in bundles)}
    summary = {"phase": phase, "planned_count": len(matrix), "attempted_count": len(bundles),
        "complete": complete, "interrupted": interrupted, "timing": timing,
        "selected_rates": rates, "rows": flat_rows, "finished_utc": utc(), "config_sha256": digest(config),
        "sources": source["sources"]}
    if complete and phase == "pilot":
        summary["selected_rates"] = select_learning_rates(flat_rows, config)
    if complete and phase == "development":
        summary["selected_models"] = select_models(flat_rows, config)
    try:
        verify_inventory(source["sources"])
    except Exception as error:
        summary["complete"] = complete = False
        summary["interrupted"] = f"{summary['interrupted'] or ''}; final_source_check={error}"
    write_json(output / "summary.json", summary)
    # Includes final summary serialization/hash in phase wall time.
    summary_hash = file_hash(output / "summary.json")
    elapsed = time.perf_counter()-start
    receipt = {"elapsed_seconds": elapsed, "ceiling_seconds": deadline,
        "within_ceiling": elapsed <= deadline, "complete": complete and elapsed <= deadline,
        "summary_sha256": summary_hash, "finished_utc": utc()}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"phase": phase, **receipt}, indent=2), flush=True)
    if not receipt["complete"]:
        raise SystemExit(1)


def verify_phase(directory: Path, phase: str, config: dict[str, Any],
                 source_lock_path: Path, rates: dict[str, float]) -> dict[str, Any]:
    """Check the exact matrix, identities, states and selections before locking."""
    from src.smooth_local_analysis_v1 import select_learning_rates, select_models

    source = verify_source_lock(source_lock_path, config)
    plan, summary, receipt = (read(directory / name) for name in ("plan.json", "summary.json", "receipt.json"))
    matrix = planned(config, phase, rates)
    if (plan["phase"] != phase or summary["phase"] != phase
            or plan["config_sha256"] != digest(config) or summary["config_sha256"] != digest(config)
            or plan["source_lock_sha256"] != file_hash(source_lock_path)
            or summary["sources"] != source["sources"] or plan["matrix"] != matrix):
        raise ValueError("Prior phase has wrong stage, matrix, configuration or source identity")
    if (not summary["complete"] or not receipt["complete"] or not receipt["within_ceiling"]
            or receipt["elapsed_seconds"] > config["phases"]["ceiling_seconds"][phase]
            or file_hash(directory / "summary.json") != receipt["summary_sha256"]
            or summary["attempted_count"] != len(matrix) or summary["planned_count"] != len(matrix)):
        raise ValueError("Prior phase terminal receipt is incomplete or altered")
    expected_files = {f"{row['fit_id']}.json" for row in matrix}
    if {p.name for p in directory.glob("fit_*.json")} != expected_files:
        raise ValueError("Prior phase contains missing or extra fit records")
    rows = []
    for item in matrix:
        compact = read(directory / f"{item['fit_id']}.json")
        if any(compact.get(key) != value for key, value in item.items()):
            raise ValueError("Stored fit identity differs from prospective matrix")
        if file_hash(directory / f"{item['fit_id']}.pt") != compact["artifact_sha256"]:
            raise ValueError("Stored fit state hash mismatch")
        if len(compact["rows"]) != len(config["training"]["time_budgets_seconds"]):
            raise ValueError("Missing stored allowance")
        for row, allowance in zip(compact["rows"], config["training"]["time_budgets_seconds"]):
            if (any(row.get(key) != value for key, value in item.items())
                    or row["allowance"] != allowance or not complete_row(row)):
                raise ValueError("Invalid stored metric or cell identity")
        rows.extend(compact["rows"])
    if rows != summary["rows"]:
        raise ValueError("Summary differs from immutable per-fit records")
    selected_rates = select_learning_rates(rows, config) if phase == "pilot" else rates
    if summary["selected_rates"] != selected_rates:
        raise ValueError("Prior phase rate selection mismatch")
    if phase == "development" and summary["selected_models"] != select_models(rows, config):
        raise ValueError("Development model selection mismatch")
    return summary


def freeze(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Bind passed required checks and all study sources before pilot access."""
    checks = read(args.checks)
    if not all(checks.get(k) is True for k in ("full_tests", "standard_adapter_gate", "independent_review")):
        raise ValueError("All required checks must have passed")
    verify_inventory(checks["evidence"])
    paths = scientific_paths()
    write_json(args.output, {"kind": "vm_n01_source_lock_v1", "created_utc": utc(),
        "config_sha256": digest(config), "sources": inventory(paths),
        "checks": {**checks["evidence"], args.checks.resolve().relative_to(ROOT).as_posix(): file_hash(args.checks)},
        "environment": environment()})
    print(f"Frozen {len(paths)} scientific files at {args.output}")


def lock_confirmation(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Lock choices and complete prior evidence before reserved generation."""
    source = verify_source_lock(args.source_lock, config)
    pilot = verify_phase(args.pilot_dir, "pilot", config, args.source_lock, {})
    development = verify_phase(args.development_dir, "development", config, args.source_lock, pilot["selected_rates"])
    evidence = inventory([p for directory in (args.pilot_dir, args.development_dir) for p in directory.glob("*") if p.is_file()])
    evidence.update(source["checks"])
    source_lock_relative = args.source_lock.resolve().relative_to(ROOT).as_posix()
    evidence[source_lock_relative] = file_hash(args.source_lock)
    repeats = 1 + len(config["tasks"]["families"]) * config["tasks"]["replicates"]["confirmation"]
    projected = feasibility(config, pilot, "confirmation")
    if not projected["passed"]:
        write_json(args.output.with_suffix(".failed_feasibility.json"), projected)
        raise ValueError("Confirmation feasibility failed; no reserved access")
    candidate = {"kind": "vm_n01_confirmation_lock_v1", "created_utc": utc(),
        "config_sha256": digest(config), "sources": source["sources"], "evidence": evidence,
        "source_lock_path": source_lock_relative,
        "phase_directories": {"pilot": args.pilot_dir.resolve().relative_to(ROOT).as_posix(),
                              "development": args.development_dir.resolve().relative_to(ROOT).as_posix()},
        "selected_rates": pilot["selected_rates"], "selected_models": development["selected_models"],
        "feasibility": projected,
        "prerequisites": {k: True for k in ("full_tests", "standard_adapter_gate", "independent_review", "complete_development", "complete_pilot", "feasibility")}}
    # A timing-only proof cannot authorize target generation. Measure the
    # complete guard, including matrix validation and selection replay.
    candidate_path = args.output.with_suffix(".timing_only.json")
    write_json(candidate_path, dict(candidate, preflight_only=True))
    before = time.perf_counter()
    verify_lock(candidate_path, config, allow_preflight=True)
    verification = time.perf_counter()-before
    projected = feasibility(config, pilot, "confirmation", verification * repeats)
    if not projected["passed"]:
        write_json(args.output.with_suffix(".failed_feasibility.json"), projected)
        raise ValueError("Full guard makes confirmation infeasible; no reserved access")
    candidate.update(feasibility=projected, verification_seconds_once=verification,
                     preflight_only=False, timing_proof_sha256=file_hash(candidate_path))
    write_json(args.output, candidate)
    verify_lock(args.output, config)
    print(json.dumps({"lock_sha256": file_hash(args.output), "feasibility": projected}, indent=2))


def main() -> None:
    """Parse the bounded study commands without implicit continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--checks", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    for phase in ("pilot", "development", "confirmation"):
        command = commands.add_parser(phase)
        command.add_argument("--source-lock", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--pilot-dir", type=Path, required=phase != "pilot")
        command.add_argument("--confirmation-lock", type=Path, required=phase == "confirmation")
    locked = commands.add_parser("lock")
    locked.add_argument("--source-lock", type=Path, required=True)
    locked.add_argument("--pilot-dir", type=Path, required=True)
    locked.add_argument("--development-dir", type=Path, required=True)
    locked.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    if args.command == "freeze":
        freeze(args, config)
    elif args.command == "lock":
        lock_confirmation(args, config)
    else:
        run_phase(args, config)


if __name__ == "__main__":
    main()
