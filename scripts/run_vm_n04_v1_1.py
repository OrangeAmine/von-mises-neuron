"""Storage-only successor for VM-N04, retaining every frozen scientific source.

The original pilot completed all fits but failed summary serialization on
NumPy integer metadata. This controller handles numeric JSON scalars and can
recover that specific completed-fit pilot without training or model selection
changes. The original failed directory is read-only throughout recovery.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_vm_n04_v1 as original

ORIGINAL_SOURCES = tuple(original.SOURCES)
EXTRA_SOURCES = ["scripts/run_vm_n04_v1_1.py", "docs/vm_n04_storage_recovery_v1_1.md"]


def write_json(path: Path, value: object) -> None:
    """Write strict JSON with explicit NumPy scalar conversion only."""
    def scalar(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON object {type(item).__name__}")
    text = json.dumps(value, indent=2, allow_nan=False, default=scalar) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def activate() -> None:
    """Install only the serializer and expanded source identity in old runner."""
    original.write_json = write_json
    original.SOURCES = [*ORIGINAL_SOURCES, *EXTRA_SOURCES]


def recover_pilot(source: Path, output: Path) -> dict:
    """Recover completed pilot fits, verifying every copied case and prediction.

    Args:
        source: Original failed study root, never modified.
        output: New root with a successor source lock, no existing pilot.

    Returns:
        Completed reconstructed summary with explicit recovery/timing caveats.
    """
    started = time.perf_counter()
    activate()
    lock = original.verify_lock(output)
    config = lock["config"]
    old_lock = json.loads((source / "source_lock.json").read_text())
    if old_lock["config"] != config or set(old_lock["sources"]) != set(ORIGINAL_SOURCES):
        raise ValueError("Recovery must preserve the exact original scientific configuration")
    for name, sha in old_lock["sources"].items():
        if original.digest(ROOT / name) != sha:
            raise ValueError("Frozen original scientific source changed")
    old = source / "pilot"
    new = output / "pilot"
    if new.exists():
        raise FileExistsError("Recovered stage cannot be overwritten or retried")
    if (old / "terminal.json").exists():
        raise ValueError("Recovery is restricted to the failed-finalization pilot")
    artifact_map = json.loads((old / "artifacts.json").read_text())
    for name, sha in artifact_map.items():
        path = (old / name).resolve()
        if not path.is_relative_to(old.resolve()) or original.digest(path) != sha:
            raise ValueError("Original pilot evidence changed")
    plan = original.stage_plan(config, "pilot", config["training"]["learning_rates"])
    if json.loads((old / "plan.json").read_text()) != plan:
        raise ValueError("Original planned matrix changed")
    new.mkdir()
    for name in artifact_map:
        if name not in ("plan.json", "access_lock.json"):
            shutil.copy2(old / name, new / name)
    for name in ("summary.json", "access_lock.json", "artifacts.json"):
        shutil.copy2(old / name, new / f"inherited_{name}")
    write_json(new / "plan.json", plan)
    summary = {"stage": "pilot", "status": "running", "environment": original.environment(),
               "source_lock": lock, "planned_fits": len(plan), "completed_fits": 0,
               "records": [], "datasets": [], "references": []}
    for identity in plan:
        room, tau, irregular = (identity[k] for k in ("room", "tau", "irregular"))
        cell = f"room{room}_tau{tau}_irregular{int(irregular)}"
        fit_id = f"{cell}_seed{identity['seed']}_lr{identity['rate']}_{identity['arm']}"
        journal = json.loads((old / f"{fit_id}.json").read_text())
        if journal["identity"] != identity or journal["status"] != "complete" or len(journal["records"]) != 4:
            raise ValueError("Original fit is incomplete; recovery cannot create fitted evidence")
        variants = {(r["shifted"], r["outage"]) for r in journal["records"]}
        if variants != set(itertools.product(config["test_shift"], config["test_outage"])):
            raise ValueError("Original fit variant coverage is incomplete")
        for record in journal["records"]:
            if not all(record[k] == v for k, v in identity.items()):
                raise ValueError("Original fit identity mismatch")
            history = record["history"]
            best = min(history, key=lambda r: r["validation_mse"])
            if best["epoch"] != record["best_epoch"] or best["validation_mse"] != record["validation_mse"]:
                raise ValueError("Checkpoint selection does not match saved validation history")
        summary["records"].extend(journal["records"])
        summary["completed_fits"] += 1
    # Regenerate only already-consumed pilot covariates/metadata; every saved
    # model input, timestamp and target is verified exactly before use.
    for room, tau, irregular in itertools.product(config["stages"]["pilot"]["room_seeds"], config["tau_ratios"], config["attendance"]):
        common = dict(config["data"], tau_ratio=tau, irregular=irregular)
        trajectory = original.generate_trajectory(room, original.DynamicsConfig(**common))
        norm = original.fit_normalizer(trajectory)
        train = original.make_windows(trajectory, "train", norm)
        val = original.make_windows(trajectory, "validation", norm)
        cell = f"room{room}_tau{tau}_irregular{int(irregular)}"
        for shift, outage in itertools.product(config["test_shift"], config["test_outage"]):
            variant = original.generate_trajectory(room, original.DynamicsConfig(**common, shifted=shift, outage=outage))
            batch = original.make_windows(variant, "test", norm)
            suffix = f"_shift{int(shift)}_outage{int(outage)}"
            with np.load(old / f"{cell}{suffix}_cases.npz", allow_pickle=False) as cases:
                for key in ("sensor", "future_angles", "y_ppm", "clean_y_ppm", "origin_indices"):
                    np.testing.assert_array_equal(cases[key], getattr(batch, key))
                np.testing.assert_array_equal(cases["origin_times"], np.asarray(batch.origin_times).astype(str))
                np.testing.assert_array_equal(cases["target_times"], np.asarray(batch.target_times).astype(str))
            summary["datasets"].append(dict(room=room, tau=tau, irregular=irregular, shifted=shift, outage=outage,
                normalizer=dataclasses.asdict(norm), metadata=variant.metadata, train=original.batch_receipt(train),
                validation=original.batch_receipt(val), test=original.batch_receipt(batch)))
            for record in [r for r in summary["records"] if r["room"] == room and r["tau"] == tau
                           and r["irregular"] == irregular and r["shifted"] == shift and r["outage"] == outage]:
                fit_id = f"{cell}_seed{record['seed']}_lr{record['rate']}_{record['arm']}"
                prediction = np.load(old / f"{fit_id}{suffix}.npz", allow_pickle=False)["prediction"]
                if original.evaluate(prediction, batch, norm.std, config["analysis"]["endpoint_steps"]) != record["metrics"]:
                    raise ValueError("Saved predictions do not reproduce journal metrics")
            for reference in config["references"]:
                prediction = np.load(old / f"{cell}_{reference}{suffix}.npz", allow_pickle=False)["prediction"]
                count = {"ridge": 24*3*16+16, "persistence": 0, "seasonal": 96}[reference]
                summary["references"].append(dict(room=room,tau=tau,irregular=irregular,shifted=shift,outage=outage,
                    arm=reference,seed=None,rate=None,training_seconds=None,inference_seconds=None,parameters=count,
                    timing_status="not recoverable; original reference timing existed only in failed in-memory summary",
                    metrics=original.evaluate(prediction,batch,norm.std,config["analysis"]["endpoint_steps"])))
    scores = {str(rate): float(np.mean([r["validation_mse"] for r in summary["records"] if r["rate"] == rate]))
              for rate in config["training"]["learning_rates"]}
    selected = min(config["training"]["learning_rates"], key=lambda r: (scores[str(r)],r))
    original_access = json.loads((old / "access_lock.json").read_text())
    observed_span = (old / "artifacts.json").stat().st_mtime - pd.Timestamp(original_access["created_utc"]).timestamp()
    if not 0 < observed_span < config["stages"]["pilot"]["cap_seconds"]:
        raise ValueError("Original timing reconstruction is inconsistent with the pilot ceiling")
    recovery_seconds = time.perf_counter()-started
    conservative = observed_span + 60.0 + recovery_seconds
    if conservative > config["stages"]["pilot"]["cap_seconds"]:
        raise TimeoutError("Conservative recovered pilot runtime exceeds its unchanged ceiling")
    receipt = {"original_status": "all_48_fits_complete_but_summary_finalization_failed",
        "error": "TypeError: Object of type int32 is not JSON serializable",
        "original_root": str(source.resolve()), "original_source_lock_sha256": original.digest(source/"source_lock.json"),
        "original_artifacts_sha256": original.digest(old/"artifacts.json"),
        "original_initial_summary_sha256": original.digest(old/"summary.json"),
        "all_original_fits_reused": True, "fits_retrained": 0, "validation_selection_changed": False,
        "original_access_to_last_artifact_wall_seconds": observed_span,
        "conservative_margin_seconds": 60.0, "recovery_seconds_to_selection": recovery_seconds,
        "timing_caveat": "Original monotonic stage elapsed was not flushed; feasibility uses wall-clock artifact span plus60s margin and measured recovery work. Original reference fit/inference timings are unavailable.",
        "selected_rate": selected, "pilot_rate_scores": scores}
    write_json(output/"pilot_recovery.json",receipt)
    write_json(new/"access_lock.json",dict(source_lock_sha256=original.digest(output/"source_lock.json"),
        dependencies={"pilot_recovery.json":original.digest(output/"pilot_recovery.json")},stage="pilot",
        created_utc=pd.Timestamp.now(tz="UTC").isoformat()))
    summary.update(status="complete",recovery=receipt,pilot_rate_scores=scores,selected_rate=selected,
        elapsed_seconds=conservative,elapsed_seconds_basis="conservative_reconstructed_upper_estimate_not_original_monotonic",
        projected_stage_seconds={stage: conservative * len(original.stage_plan(config,stage,[selected]))/len(plan)*1.5
                                 for stage in ("development","confirmation")},cuda_peak_allocated_bytes=None,
        process_rss_bytes_at_end=None)
    write_json(new/"artifacts.json",{p.name:original.digest(p) for p in sorted(new.iterdir())})
    write_json(new/"summary.json",summary)
    write_json(new/"terminal.json",dict(status="complete",summary_sha256=original.digest(new/"summary.json"),
        artifacts_sha256=original.digest(new/"artifacts.json"),
        elapsed_seconds_including_summary_write=observed_span+60.0+time.perf_counter()-started))
    original.verify_stage(output,"pilot",config)
    print(json.dumps({"recovered_fits":len(plan),"selected_rate":selected,"projected_stage_seconds":summary["projected_stage_seconds"]}))
    return summary


def main() -> None:
    """Run the versioned serializer controller or explicit completed-fit recovery."""
    activate()
    if len(sys.argv)>1 and sys.argv[1]=="recover-pilot":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("stage",choices=["recover-pilot"])
        parser.add_argument("--source-root",type=Path,required=True)
        parser.add_argument("--output-dir",type=Path,required=True)
        args=parser.parse_args()
        recover_pilot(args.source_root,args.output_dir)
    else:
        original.main()


if __name__=="__main__":
    main()
