"""Summarize saved secondary diagnostics from complete audited methodology runs.

This reporting utility never generates tasks, fits models, loads checkpoints or
uses GPU APIs. It adds no gates. Fit-level pooled descriptions are kept separate
from arithmetic task-balanced summaries; neither supplies a new inferential
claim. All planned fit/task rows remain in the output JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ERROR_METRICS = ("noisy_rmse", "mean_absolute_error", "max_absolute_error",
                 "boundary_mse", "boundary_mean_absolute_error")
A_FINAL_METRICS = (*ERROR_METRICS, "clean_mse", "normalized_clean_mse", "amplitude",
                   "amplitude_error", "coefficient_l2_error", "wrapped_phase_error", "bias")
B_TASK_METRICS = (*ERROR_METRICS, "updates_completed", "fit_runtime_seconds", "wall_seconds",
                  "peak_cuda_allocated_bytes", "mean_gradient_norm_before_clip",
                  "train_near_zero_feature_fraction", "grid_near_zero_feature_fraction",
                  "center_circular_resultant", "concentration_min", "concentration_median", "concentration_max")


def file_hash(path: Path) -> str:
    """Hash exact input or reporting source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read strict finite JSON without importing scientific execution modules."""
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def describe(values: Sequence[float | None]) -> dict[str, Any]:
    """Return explicitly counted descriptions; nulls never silently become zeros.

    These quantiles describe the supplied correlated records, not sampling
    uncertainty. Missing values remain counted separately from defined values.
    """
    available = [float(value) for value in values if value is not None]
    if not np.isfinite(available).all():
        raise ValueError("Nonfinite saved diagnostic")
    result = {"records": len(values), "defined_count": len(available),
              "null_count": len(values) - len(available), "mean": None, "sample_sd": None,
              "min": None, "median": None, "p90": None, "max": None}
    if available:
        result.update(mean=float(np.mean(available)), sample_sd=float(np.std(available, ddof=1)) if len(available) > 1 else None,
                      min=min(available), median=float(np.median(available)), p90=float(np.percentile(available, 90)), max=max(available))
    return result


def _complete_mean(values: Sequence[float | None]) -> float | None:
    """Average a complete diagnostic set, or retain null for missing members."""
    if not values or any(value is None for value in values):
        return None
    numbers = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError("Nonfinite secondary metric")
    return float(np.mean(numbers))


def load_inputs(inputs: Sequence[Path], analysis_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require complete A/B audits and their exact unchanged manifest bytes."""
    analysis = read_json(analysis_path)
    if analysis.get("complete") is not True or analysis.get("phase") not in ("development", "confirmation"):
        raise ValueError("Secondary reporting requires complete development or confirmation analysis")
    if len(inputs) != 2 or {entry["stage"] for entry in analysis.get("audits", [])} != {"A", "B"}:
        raise ValueError("Exactly one audited A directory and one audited B directory are required")
    if len(analysis["audits"]) != 2 or not all(entry.get("complete") is True for entry in analysis["audits"]):
        raise ValueError("Both stage audits must be complete")
    if not all((analysis.get(stage) or {}).get("complete") is True for stage in ("A", "B")):
        raise ValueError("Both audited stage summaries must be complete")
    expected = {str(Path(name).resolve()): digest for name, digest in analysis["input_manifest_hashes"].items()}
    observed = {str((path / "manifest.json").resolve()) for path in inputs}
    if set(expected) != observed:
        raise ValueError("Reporting inputs differ from the audited manifest set")
    manifests = {}
    for directory in inputs:
        path = directory / "manifest.json"
        if file_hash(path) != expected[str(path.resolve())]:
            raise ValueError(f"Input manifest changed after its audit: {path}")
        manifest = read_json(path)
        if manifest["phase"] != analysis["phase"] or manifest["stage"] in manifests:
            raise ValueError("Distinct A/B stages with a shared audited phase are required")
        if manifest["status"] != "completed" or any(row["status"] != "completed" for row in manifest["runs"]):
            raise ValueError("Failed/incomplete runs cannot be excluded from secondary reporting")
        if len({row["run_id"] for row in manifest["runs"]}) != len(manifest["runs"]):
            raise ValueError("Duplicate fit rows")
        if any("summary" not in row for row in manifest["runs"]):
            raise ValueError("Reporting requires full terminal summaries, not running references")
        manifests[manifest["stage"]] = manifest
    if set(manifests) != {"A", "B"} or manifests["A"]["config"] != manifests["B"]["config"]:
        raise ValueError("A/B configurations do not match")
    return manifests, analysis


def _gradient_descriptions(summary: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind != "gradient":
        return {"available": False, "reason": "Closed-form solve has no gradient-update history"}
    history = summary["update_history"]
    if len(history) != summary["updates_completed"]:
        raise ValueError("Incomplete gradient update diagnostics")
    parameters = sorted({name for item in history for name in item["parameter_gradient_norms"]})
    return {"available": True, "scope": "Pooled within-fit update descriptions before clipping; updates are dependent",
            "total_norm": describe([item["gradient_norm_before_clip"] for item in history]),
            "per_parameter": {name: describe([item["parameter_gradient_norms"].get(name) for item in history]) for name in parameters}}


def _feature_fraction(diagnostics: Mapping[str, Any], split: str) -> float | None:
    # Read both historical spelling and the corrected explicit exponent label.
    return diagnostics.get(f"{split}_feature_absolute_below_1e_minus12_fraction",
                           diagnostics.get(f"{split}_feature_absolute_below_1e12_fraction"))


def _geometry(diagnostics: Mapping[str, Any], model: str) -> dict[str, Any]:
    if model not in ("learned_vm", "learned_geodesic"):
        return {"available": False, "reason": "Only the learned-center/width models receive learned geometry summaries"}
    centers = np.asarray(diagnostics["centers_wrapped"], dtype=np.float64)
    widths = np.asarray(diagnostics["concentrations"], dtype=np.float64)
    if centers.shape != (16,) or widths.shape != (16,) or not np.isfinite(centers).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("Incomplete or invalid learned kernel geometry")
    cosine, sine = float(np.mean(np.cos(centers))), float(np.mean(np.sin(centers)))
    resultant = math.hypot(cosine, sine)
    direction = math.atan2(sine, cosine) if resultant > 1e-12 else None
    return {"available": True, "centers_wrapped_radians": centers.tolist(), "concentrations": widths.tolist(),
            "center_circular_resultant": resultant, "center_circular_mean_radians": direction,
            "center_circular_mean_null_reason": "Resultant <= 1e-12" if direction is None else None,
            "concentration_min": float(np.min(widths)), "concentration_median": float(np.median(widths)),
            "concentration_max": float(np.max(widths)),
            "scope": "Within-fit learned basis geometry; not causal directions or independent kernel replications"}


def _base_record(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    result = {key: row.get(key) for key in ("run_id", "stage", "phase", "family", "replicate", "kind", "model", "optimizer", "penalty", "learning_rate", "n_train", "initialization_index")}
    result.update(status=row["status"], parameter_count=summary.get("parameter_count"),
                  fit_runtime_seconds=summary.get("runtime_seconds"), wall_seconds=row.get("wall_seconds"),
                  peak_cuda_allocated_bytes=summary.get("peak_cuda_allocated_bytes"), device=summary.get("device"),
                  gradient_diagnostics=_gradient_descriptions(summary, row["kind"]))
    if row["kind"] == "gradient":
        result.update(updates_completed=summary["updates_completed"], selected_step=summary.get("selected_step"), stop_reason=summary["stop_reason"])
    else:
        result.update(updates_completed=None, selected_step=None, stop_reason="closed_form_diagnostic" if row["kind"] == "reference" else "closed_form_validation_selection")
    gradient = result["gradient_diagnostics"]
    result["mean_gradient_norm_before_clip"] = gradient["total_norm"]["mean"] if gradient["available"] else None
    return result


def summarize_a(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve every A fit and summarize final/phase diagnostics per design cell."""
    records = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    spec = manifest["config"]["A"]
    expected_count = spec[manifest["phase"]]["task_pairs"]
    for row in manifest["runs"]:
        record, summary = _base_record(row), row["summary"]
        if row["kind"] == "gradient":
            final = summary["final_metrics"]
            history = summary["history"]
            record.update(final_metrics=final, final_phase_error_defined=final.get("phase_error_defined"),
                          final_phase_error_null_reason=final.get("phase_error_undefined_reason"),
                          recorded_curve_points=len(history),
                          trajectory_phase_defined_count=sum(item.get("phase_error_defined") is True for item in history),
                          trajectory_phase_undefined_count=sum(item.get("phase_error_defined") is False for item in history))
        else:
            record.update(final_metrics=summary["metrics"], final_phase_error_defined=None,
                          final_phase_error_null_reason="Closed-form diagnostic does not record phase errors",
                          recorded_curve_points=0, trajectory_phase_defined_count=0, trajectory_phase_undefined_count=0)
        records.append(record)
        groups[(row["family"], row.get("optimizer"), row["penalty"], row["model"])].append(record)
    expected_groups = {(family, optimizer, penalty, model) for family in spec["family_ids"]
                       for optimizer in spec["optimizers"] for penalty in spec["explicit_amplitude_penalties"] for model in spec["models"]}
    expected_groups.update((family, None, penalty, "reference") for family in spec["family_ids"] for penalty in spec["explicit_amplitude_penalties"])
    if set(groups) != expected_groups:
        raise ValueError("Missing, duplicate or unplanned A secondary summary cells")
    summaries = []
    for (family, optimizer, penalty, model), rows in groups.items():
        if len(rows) != expected_count or {row["replicate"] for row in rows} != set(range(expected_count)):
            raise ValueError("Incomplete paired A tasks")
        summaries.append({"family": family, "optimizer": optimizer, "penalty": penalty, "model": model,
            "task_count": len(rows), "final_metrics_across_tasks": {metric: describe([row["final_metrics"].get(metric) for row in rows]) for metric in A_FINAL_METRICS},
            "final_phase_error_counts": {"defined": sum(row["final_phase_error_defined"] is True for row in rows),
                                          "undefined": sum(row["final_phase_error_defined"] is False for row in rows),
                                          "not_recorded": sum(row["final_phase_error_defined"] is None for row in rows)},
            "trajectory_phase_error_counts_descriptive": {"defined": sum(row["trajectory_phase_defined_count"] for row in rows),
                                                          "undefined": sum(row["trajectory_phase_undefined_count"] for row in rows)},
            "updates": describe([row["updates_completed"] for row in rows]),
            "fit_runtime_seconds": describe([row["fit_runtime_seconds"] for row in rows]),
            "wall_seconds": describe([row["wall_seconds"] for row in rows]),
            "peak_cuda_allocated_bytes": describe([row["peak_cuda_allocated_bytes"] for row in rows]),
            "fit_mean_gradient_norm": describe([row["mean_gradient_norm_before_clip"] for row in rows]),
            "stop_reason_counts": dict(Counter(row["stop_reason"] for row in rows))})
    return {"records": records, "cells": summaries, "planned_fit_rows_preserved": len(records),
            "interpretation": "Arithmetic summaries across tasks within the same condition/optimizer/penalty/model; phase-error means condition on explicitly counted defined phase errors. Trajectory counts are descriptive, not independent tasks."}


def summarize_b(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve B fit rows and separately summarize task-balanced and pooled data."""
    spec = manifest["config"]["B"]
    count = spec[manifest["phase"]]["task_pairs"]
    model_specs = {model["id"]: model for model in spec["models"]}
    records, task_rows, groups = [], [], defaultdict(list)
    for row in manifest["runs"]:
        record, summary = _base_record(row), row["summary"]
        features = summary["feature_diagnostics"]
        geometry = _geometry(features, row["model"])
        record.update(metrics=summary["metrics"], geometry=geometry, feature_count=features["feature_count"],
                      train_near_zero_feature_fraction=_feature_fraction(features, "train"),
                      grid_near_zero_feature_fraction=_feature_fraction(features, "grid"),
                      initialization_selected=(summary["selected_step"] == 0) if row["kind"] == "gradient" else None)
        for metric in ERROR_METRICS:
            record[metric] = summary["metrics"][metric]
        for metric in ("center_circular_resultant", "concentration_min", "concentration_median", "concentration_max"):
            record[metric] = geometry.get(metric)
        records.append(record)
        groups[(row["model"], row["family"])].append(record)
    if set(groups) != {(model, family) for model in model_specs for family in spec["family_ids"]}:
        raise ValueError("Missing or unplanned B model/family groups")
    summaries = []
    for (model, family), rows in groups.items():
        seed_ids = {None} if model_specs[model]["kind"] == "ridge" else set(range(spec[manifest["phase"]]["initialization_seeds"]))
        expected = {(task, size, seed) for task in range(count) for size in spec["train_observations"] for seed in seed_ids}
        identities = [(row["replicate"], row["n_train"], row["initialization_index"]) for row in rows]
        if set(identities) != expected or len(identities) != len(expected):
            raise ValueError("Missing, duplicate or unplanned B task/size/initialization rows")
        local_tasks = []
        for task in range(count):
            by_size = []
            for size in spec["train_observations"]:
                selected = [row for row in rows if row["replicate"] == task and row["n_train"] == size]
                by_size.append({"n_train": size, "initializations": len(selected),
                                "diagnostics": {metric: _complete_mean([row[metric] for row in selected]) for metric in B_TASK_METRICS}})
            task_row = {"model": model, "family": family, "replicate": task,
                        "per_size_initialization_means": by_size,
                        "task_balanced_diagnostics": {metric: _complete_mean([row["diagnostics"][metric] for row in by_size]) for metric in B_TASK_METRICS}}
            local_tasks.append(task_row)
            task_rows.append(task_row)
        gradient_rows = [row for row in rows if row["kind"] == "gradient"]
        learned = [row for row in rows if row["geometry"]["available"]]
        pooled_widths = [value for row in learned for value in row["geometry"]["concentrations"]]
        summaries.append({"model": model, "family": family, "task_count": count,
            "negative_control_excluded_from_periodic_gates": family == "nonperiodic",
            "task_balanced_secondary_descriptions": {metric: describe([row["task_balanced_diagnostics"][metric] for row in local_tasks]) for metric in B_TASK_METRICS},
            "per_fit_pooled_descriptions": {metric: describe([row[metric] for row in rows]) for metric in B_TASK_METRICS},
            "stop_reason_counts_per_fit": dict(Counter(row["stop_reason"] for row in rows)),
            "initialization_selection_counts_per_fit": {"eligible_gradient_fits": len(gradient_rows),
                                                        "selected_step_zero": sum(row["initialization_selected"] for row in gradient_rows),
                                                        "selected_after_updates": sum(not row["initialization_selected"] for row in gradient_rows)},
            "pooled_learned_kernel_concentrations_descriptive": describe(pooled_widths),
            "pooled_geometry_scope": "Kernel values and fit replicates are dependent descriptive records, not inferential units; center directions are not pooled across independent task coordinate origins"})
    return {"records": records, "task_rows": task_rows, "model_family_summaries": summaries,
            "planned_fit_rows_preserved": len(records),
            "task_balancing": "For each model/family/task: arithmetic initialization mean within sample size; equal arithmetic mean of 32/128/512 endpoints; then arithmetic mean over whole tasks. Any missing diagnostic yields null for that complete task diagnostic, with explicit counts.",
            "pooled_description_scope": "Fit/update/kernel/memory quantiles describe recorded dependent values. They are not confidence intervals, new gates or population-inference results."}


def summarize(inputs: Sequence[Path], analysis_path: Path, output_path: Path) -> dict[str, Any]:
    """Write a new secondary report, refusing overwritten or incomplete evidence."""
    if output_path.exists():
        raise FileExistsError(output_path)
    manifests, analysis = load_inputs(inputs, analysis_path)
    a_report, b_report = summarize_a(manifests["A"]), summarize_b(manifests["B"])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": analysis["protocol_id"],
              "phase": analysis["phase"], "complete_audited_inputs": True, "new_gates": None,
              "claim_scope": "Saved secondary diagnostics only; descriptive task-balanced summaries remain separate from pooled fit/update/kernel descriptions; no new inferential or superiority claim",
              "source_script": str(Path(__file__).resolve()), "source_script_sha256": file_hash(Path(__file__)),
              "analysis_path": str(analysis_path.resolve()), "analysis_sha256": file_hash(analysis_path),
              "input_manifest_hashes": analysis["input_manifest_hashes"],
              "units": {"A": "Unstandardized signal; angular errors in radians; runtime in seconds; recorded CUDA peak allocation in bytes",
                        "B": "Generator-standardized signal; MSE in squared signal units; center angles in radians; dimensionless widths/resultants/fractions; runtime seconds; CUDA allocation bytes"},
              "memory_scope": "Recorded absolute torch.cuda peak allocated bytes per fit, not isolated incremental model memory. Null denotes unavailable/CPU, not zero.",
              "runtime_scope": "fit_runtime_seconds includes trainer evaluation/diagnostics; wall_seconds includes fit dispatch and artifact work; invocation elapsed includes progress/source/data overhead. No timing quantity is an inferential performance claim.",
              "original_full_histories": "Remain in the audited, hashed per-fit artifacts; this report preserves every fit/task row with reduced within-fit diagnostic descriptions.",
              "invocation_runtime_seconds": {stage: manifests[stage]["time_seconds"] for stage in ("A", "B")},
              "A": a_report, "B": b_report}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose a read-only-input secondary-report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, type=Path, required=True, metavar=("A_DIR", "B_DIR"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.inputs, args.analysis, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "phase": result["phase"],
                      "A_fit_rows": result["A"]["planned_fit_rows_preserved"],
                      "B_fit_rows": result["B"]["planned_fit_rows_preserved"],
                      "B_task_rows": len(result["B"]["task_rows"]), "new_gates": None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
