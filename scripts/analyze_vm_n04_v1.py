"""Analyze the frozen VM-N04 confirmation using paired room-level replication.

This observer checks complete summary identities and provenance, then aggregates
saved metrics. It does not refit models or claim to replay prediction arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGIME_FIELDS = ("tau", "irregular", "shifted", "outage")
SUBGROUPS = ("boundary", "event", "origin_outage", "target_outage")


def digest(path: Path) -> str:
    """Return the exact file SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _contained(root: Path, relative: str) -> Path:
    """Resolve a saved relative identity without allowing path escape."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def load_study(study: Path) -> tuple[dict, dict, dict]:
    """Verify frozen source, prerequisite, terminal and summary bindings.

    Failed engineering acceptance is retained in the output; incomplete
    engineering execution or incomplete scientific execution is rejected.
    """
    lock_path = study / "source_lock.json"
    lock = read_json(lock_path)
    for name, expected in lock["sources"].items():
        if digest(_contained(ROOT, name)) != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
    stage = study / "confirmation"
    access = read_json(stage / "access_lock.json")
    if access.get("stage") != "confirmation" or access["source_lock_sha256"] != digest(lock_path):
        raise ValueError("Confirmation access lock mismatch")
    required = {"pilot/summary.json", "development/summary.json", "engineering.json"}
    if not required.issubset(access["dependencies"]):
        raise ValueError("Confirmation prerequisite binding missing")
    for name, expected in access["dependencies"].items():
        if digest(_contained(study, name)) != expected:
            raise ValueError(f"Confirmation dependency mismatch: {name}")
    for phase in ("pilot", "development"):
        if read_json(study / phase / "summary.json").get("status") != "complete":
            raise ValueError(f"Incomplete prerequisite: {phase}")
    engineering = read_json(study / "engineering.json")
    if not engineering.get("full_suite_completed") or not engineering.get("standard_benchmarks_completed"):
        raise ValueError("Engineering execution incomplete")
    summary_path = stage / "summary.json"
    summary = read_json(summary_path)
    terminal = read_json(stage / "terminal.json")
    if terminal.get("status") != "complete" or terminal["summary_sha256"] != digest(summary_path):
        raise ValueError("Confirmation terminal receipt mismatch")
    artifacts_path = stage / "artifacts.json"
    if terminal["artifacts_sha256"] != digest(artifacts_path):
        raise ValueError("Confirmation artifact manifest mismatch")
    artifacts = read_json(artifacts_path)
    for name, expected in artifacts.items():
        if digest(_contained(stage, name)) != expected:
            raise ValueError(f"Confirmation artifact mismatch: {name}")
    if summary["source_lock"] != lock:
        raise ValueError("Summary source lock mismatch")
    cap = lock["config"]["stages"]["confirmation"]["cap_seconds"]
    if not 0 <= terminal["elapsed_seconds_including_summary_write"] <= cap:
        raise ValueError("Confirmation terminal runtime exceeds its frozen cap")
    pilot = read_json(study / "pilot/summary.json")
    if {r["rate"] for r in summary["records"]} != {pilot["selected_rate"]}:
        raise ValueError("Confirmation learning rate differs from pilot selection")
    provenance = {"source_lock_sha256": digest(lock_path), "summary_sha256": digest(summary_path),
                  "access_lock_sha256": digest(stage / "access_lock.json"),
                  "artifacts_sha256": digest(artifacts_path), "artifacts_verified": len(artifacts),
                  "terminal": terminal, "source_lock": lock,
                  "prerequisite_hashes": access["dependencies"], "engineering": engineering,
                  "stage_seconds": {p: read_json(study / p / "summary.json")["elapsed_seconds"]
                                    for p in ("pilot", "development", "confirmation")}}
    return summary, lock["config"], provenance


def _regime(record: dict) -> tuple:
    return tuple(record[field] for field in REGIME_FIELDS)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if value is None or not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {label}: {value}")
    return float(value)


def _metric(metric: dict, expected_n: int | None = None) -> None:
    n = metric["n"]
    if not isinstance(n, int) or n < 0 or (expected_n is not None and n != expected_n):
        raise ValueError("Metric count disagrees with common eligible cases")
    if n == 0:
        if metric["rmse"] is not None or metric["mae"] is not None:
            raise ValueError("Empty subgroup must have explicit null errors")
    else:
        _number(metric["rmse"], "RMSE")
        _number(metric["mae"], "MAE")
        if metric["mae"] > metric["rmse"] + 1e-8:
            raise ValueError("MAE exceeds RMSE")


def validate_summary(summary: dict, config: dict) -> tuple[list, list, dict, dict]:
    """Require every prespecified room/regime/arm/initialization exactly once."""
    if summary.get("stage") != "confirmation" or summary.get("status") != "complete":
        raise ValueError("Scientific completion requires a complete confirmation stage")
    spec = config["stages"]["confirmation"]
    rooms, seeds = spec["room_seeds"], spec["initialization_seeds"]
    regimes = list(itertools.product(config["tau_ratios"], config["attendance"],
                                    config["test_shift"], config["test_outage"]))
    if len(set(rooms)) != len(rooms) or len(set(seeds)) != len(seeds) or len(set(regimes)) != len(regimes):
        raise ValueError("Duplicate configured replication identities")
    expected_fits = len(rooms) * len(config["tau_ratios"]) * len(config["attendance"]) * len(seeds) * len(config["arms"])
    if summary["planned_fits"] != expected_fits or summary["completed_fits"] != expected_fits:
        raise ValueError("Incomplete confirmation fit count")
    _number(summary["elapsed_seconds"], "stage runtime")
    if summary["elapsed_seconds"] > spec["cap_seconds"]:
        raise ValueError("Confirmation exceeds prospective runtime cap")
    datasets = {}
    for row in summary["datasets"]:
        key = (row["room"], _regime(row))
        if key in datasets:
            raise ValueError("Duplicate dataset identity")
        _number(row["normalizer"]["std"], "training standard deviation", positive=True)
        if not np.isfinite(row["normalizer"]["mean"]):
            raise ValueError("Invalid training mean")
        for split in ("train", "validation", "test"):
            receipt = row[split]
            if receipt["windows"] <= 0 or receipt["label_uses"] != receipt["windows"] * config["data"]["horizon"]:
                raise ValueError("Invalid split label/window receipt")
            if not 0 < receipt["unique_target_times"] <= receipt["label_uses"]:
                raise ValueError("Invalid unique-label receipt")
            for count in receipt["exclusions"].values():
                if not isinstance(count, int) or count < 0:
                    raise ValueError("Invalid exclusion count")
            excluded = receipt["exclusions"]
            if excluded.get("retained", receipt["windows"]) != receipt["windows"]:
                raise ValueError("Exclusion ledger disagrees with retained windows")
            if "candidates" in excluded and excluded["candidates"] != sum(v for k, v in excluded.items() if k != "candidates"):
                raise ValueError("Exclusion ledger does not partition candidate origins")
        datasets[key] = row
    if set(datasets) != set(itertools.product(rooms, regimes)):
        raise ValueError("Missing or unexpected confirmation dataset")
    # Test-only interventions must not introduce extra training labels or refits.
    for room, tau, irregular in itertools.product(rooms, config["tau_ratios"], config["attendance"]):
        related = [datasets[(room, r)] for r in regimes if r[:2] == (tau, irregular)]
        for row in related[1:]:
            if any(row[k] != related[0][k] for k in ("normalizer", "train", "validation")):
                raise ValueError("Test variants have inconsistent training information")
    indexed, count_checks = {}, {}
    for record in summary["records"] + summary["references"]:
        reference = record["arm"] in config["references"]
        if reference and record["seed"] is not None:
            raise ValueError("Reference must not be duplicated across initialization seeds")
        key = (record["room"], _regime(record), record["arm"], record["seed"])
        if key in indexed:
            raise ValueError("Duplicate model/room/regime/initialization record")
        if key[:2] not in datasets:
            raise ValueError("Record has an unknown dataset")
        data = datasets[key[:2]]
        n = data["test"]["windows"]
        std = data["normalizer"]["std"]
        metrics = record["metrics"]
        for field in ("all_horizon", "clean_all_horizon"):
            _metric(metrics[field], n * config["data"]["horizon"])
        if set(metrics["endpoints"]) != set(map(str, config["analysis"]["endpoint_steps"])):
            raise ValueError("Endpoint matrix differs from protocol")
        for step in config["analysis"]["endpoint_steps"]:
            endpoint = metrics["endpoints"][str(step)]
            _metric(endpoint, n)
            if not np.isclose(endpoint["normalized_rmse"], endpoint["rmse"] / std, rtol=1e-10, atol=1e-12):
                raise ValueError("Endpoint normalization disagrees with training SD")
            _metric(endpoint["clean"], n)
            for group in SUBGROUPS:
                _metric(endpoint[group])
                group_key = key[:2] + (step, group)
                if endpoint[group]["n"] > n:
                    raise ValueError("Subgroup exceeds eligible cases")
                if group_key in count_checks and count_checks[group_key] != endpoint[group]["n"]:
                    raise ValueError("Methods use different diagnostic cases")
                count_checks[group_key] = endpoint[group]["n"]
        for name in ("training_seconds", "inference_seconds", "parameters"):
            _number(record[name], name)
        if not isinstance(record["parameters"], int):
            raise ValueError("Parameter count must be integral")
        if not reference and not record["shifted"] and not record["outage"]:
            clock = record["clock_diagnostic"]
            if clock["offset_minutes"] != config["analysis"]["clock_defect_minutes"]:
                raise ValueError("Clock defect differs from protocol")
            # An add/subtract correction in float32 need not be bit-identical.
            # Preserve its measured discrepancy without making a post-hoc gate.
            _number(clock["restored_max_abs_normalized"], "clock restoration")
            for step in config["analysis"]["endpoint_steps"]:
                _metric(clock["offset_metrics"]["endpoints"][str(step)], n)
        indexed[key] = record
    expected = {(room, regime, arm, seed) for room, regime, arm in itertools.product(rooms, regimes, config["arms"])
                for seed in seeds}
    expected |= {(room, regime, arm, None) for room, regime, arm in itertools.product(rooms, regimes, config["references"])}
    if set(indexed) != expected:
        raise ValueError("Missing or unexpected model/reference confirmation records")
    if len({r["rate"] for r in summary["records"]}) != 1:
        raise ValueError("Confirmation must use one common learning rate")
    for arm in config["arms"] + config["references"]:
        if len({r["parameters"] for r in indexed.values() if r["arm"] == arm}) != 1:
            raise ValueError("Parameter count varies within an arm")
    return rooms, regimes, datasets, indexed


def paired_contrast(candidate: np.ndarray, control: np.ndarray, config: dict) -> dict:
    """Bootstrap paired whole-room means and retain the per-regime guards.

    Args:
        candidate: Array of room by regime errors, after averaging seeds.
        control: Identically ordered paired room by regime errors.
        config: Frozen analysis settings.
    """
    if candidate.shape != control.shape or candidate.ndim != 2 or len(candidate) < 2:
        raise ValueError("Paired contrast requires matching room-by-regime arrays")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all() or np.any(control <= 0):
        raise ValueError("Paired contrast requires finite positive control errors")
    a, b = candidate.mean(axis=1), control.mean(axis=1)
    improvement = float(1 - a.mean() / b.mean())
    rng = np.random.default_rng(config["bootstrap_seed"])
    indices = rng.integers(0, len(a), size=(config["bootstrap_draws"], len(a)))
    effects = 1 - a[indices].mean(axis=1) / b[indices].mean(axis=1)
    low, high = np.quantile(effects, [0.025, 0.975])
    regime_effects = 1 - candidate.mean(axis=0) / control.mean(axis=0)
    room_effects = 1 - a / b
    practical = improvement >= config["minimum_improvement"]
    interval = bool(low > 0)
    guards = regime_effects >= -config["maximum_regime_regression"]
    return {"improvement": improvement, "paired_95_interval": [float(low), float(high)],
            "practical_pass": bool(practical), "interval_pass": interval,
            "regime_guard_pass": bool(guards.all()), "passed": bool(practical and interval and guards.all()),
            "regime_improvements": regime_effects.tolist(), "regime_passes": guards.tolist(),
            "room_improvements": room_effects.tolist(),
            "room_candidate_minus_control": (a - b).tolist(),
            "room_improvement_range": [float(room_effects.min()), float(room_effects.max())],
            "rooms_improved": int(np.count_nonzero(a < b)), "rooms_tied": int(np.count_nonzero(a == b)),
            "rooms_worse": int(np.count_nonzero(a > b)),
            "candidate_mean_normalized_rmse": float(a.mean()), "control_mean_normalized_rmse": float(b.mean())}


def _summary(values: np.ndarray) -> dict:
    """Summarize room by initialization values without treating seeds as units."""
    means = values.mean(axis=1)
    return {"mean": float(means.mean()), "room_sd": float(means.std(ddof=1)),
            "mean_within_room_initialization_sd": float(values.std(axis=1, ddof=1).mean()) if values.shape[1] > 1 else 0.0,
            "room_means": means.tolist()}


def analyze_summary(summary: dict, config: dict) -> tuple[dict, list[dict]]:
    """Aggregate the full saved matrix with the prospective nested weighting."""
    rooms, regimes, datasets, indexed = validate_summary(summary, config)
    arms = config["arms"] + config["references"]
    horizons = [*config["analysis"]["endpoint_steps"], "all"]
    cubes, cells, aggregate = {}, [], []
    for arm, horizon in itertools.product(arms, horizons):
        seeds = config["stages"]["confirmation"]["initialization_seeds"] if arm in config["arms"] else [None]
        arrays = {name: np.zeros((len(rooms), len(regimes), len(seeds))) for name in ("rmse", "mae", "normalized_rmse", "clean_rmse", "clean_mae")}
        for j, regime in enumerate(regimes):
            row = dict(zip(REGIME_FIELDS, regime))
            row.update(arm=arm, horizon="whole_horizon" if horizon == "all" else f"step_{horizon}",
                       endpoint_minutes=None if horizon == "all" else horizon * config["data"]["dt_minutes"],
                       n_rooms=len(rooms), n_initializations=len(seeds))
            counts = []
            diagnostics = {group: {"rmse": [], "mae": [], "n": []} for group in SUBGROUPS}
            for i, room in enumerate(rooms):
                std = datasets[(room, regime)]["normalizer"]["std"]
                for k, seed in enumerate(seeds):
                    metrics = indexed[(room, regime, arm, seed)]["metrics"]
                    value = metrics["all_horizon"] if horizon == "all" else metrics["endpoints"][str(horizon)]
                    clean = metrics["clean_all_horizon"] if horizon == "all" else value["clean"]
                    for field in ("rmse", "mae"):
                        arrays[field][i, j, k] = value[field]
                        arrays[f"clean_{field}"][i, j, k] = clean[field]
                    arrays["normalized_rmse"][i, j, k] = value["rmse"] / std
                counts.append(value["n"])
                if horizon != "all":
                    for group in SUBGROUPS:
                        group_values = [indexed[(room, regime, arm, seed)]["metrics"]["endpoints"][str(horizon)][group] for seed in seeds]
                        diagnostics[group]["n"].append(group_values[0]["n"])
                        if group_values[0]["n"]:
                            for field in ("rmse", "mae"):
                                diagnostics[group][field].append(float(np.mean([v[field] for v in group_values])))
            for field, values in arrays.items():
                stats = _summary(values[:, j, :])
                row[field] = stats["mean"]
                row[f"{field}_room_sd"] = stats["room_sd"]
                row[f"{field}_init_sd"] = stats["mean_within_room_initialization_sd"]
            row["scored_labels_across_rooms"] = int(sum(counts))
            row["scored_labels_min_per_room"] = int(min(counts))
            row["scored_labels_max_per_room"] = int(max(counts))
            for group, values in diagnostics.items():
                row[f"{group}_n_rooms"] = len(values["rmse"])
                row[f"{group}_n_labels"] = sum(values["n"]) if horizon != "all" else None
                for field in ("rmse", "mae"):
                    row[f"{group}_{field}"] = float(np.mean(values[field])) if values[field] else None
            cells.append(row)
        cubes[(arm, horizon)] = arrays
        aggregate.append({"arm": arm, "horizon": "whole_horizon" if horizon == "all" else f"step_{horizon}",
                          "endpoint_minutes": None if horizon == "all" else horizon * config["data"]["dt_minutes"],
                          "metrics": {field: _summary(values.mean(axis=1)) for field, values in arrays.items()}})
    lookup = {(tuple(row[f] for f in REGIME_FIELDS), row["arm"], row["horizon"]): row for row in cells}
    for row in cells:
        regime = tuple(row[f] for f in REGIME_FIELDS)
        for reference in ["none", *config["references"]]:
            denom = lookup[(regime, reference, row["horizon"])]["normalized_rmse"]
            row[f"normalized_rmse_skill_vs_{reference}"] = 1 - row["normalized_rmse"] / denom if denom > 0 else None
    aggregate_lookup = {(row["arm"], row["horizon"]): row for row in aggregate}
    for row in aggregate:
        row["normalized_rmse_skill"] = {}
        for reference in ["none", *config["references"]]:
            denom = aggregate_lookup[(reference, row["horizon"])]["metrics"]["normalized_rmse"]["mean"]
            row["normalized_rmse_skill"][reference] = 1 - row["metrics"]["normalized_rmse"]["mean"] / denom if denom > 0 else None
    primary_step = config["analysis"]["primary_step"]
    primary = {control: paired_contrast(cubes[("learned", primary_step)]["normalized_rmse"].mean(axis=2),
                                        cubes[(control, primary_step)]["normalized_rmse"].mean(axis=2), config["analysis"])
               for control in config["analysis"]["primary_comparators"]}
    runtime = {}
    for arm in arms:
        records = [r for r in indexed.values() if r["arm"] == arm]
        fits = {}
        for r in records:
            key = (r["room"], r["tau"], r["irregular"], r["seed"], r["rate"])
            identity = (r["training_seconds"], r["parameters"])
            if key in fits and fits[key] != identity:
                raise ValueError("Repeated test variants disagree on fitting runtime/parameters")
            fits[key] = identity
        times = [v[0] for v in fits.values()]
        runtime[arm] = {"unique_fits": len(fits), "parameters": records[0]["parameters"],
                        "training_seconds_total": float(sum(times)), "training_seconds_median": float(np.median(times)),
                        "inference_seconds_total": float(sum(r["inference_seconds"] for r in records)),
                        "inference_calls": len(records), "inference_seconds_median": float(np.median([r["inference_seconds"] for r in records]))}
    clock = {}
    for arm in config["arms"]:
        records = [r for r in summary["records"] if r["arm"] == arm and "clock_diagnostic" in r]
        offset = [r["clock_diagnostic"]["offset_metrics"]["endpoints"][str(primary_step)]["rmse"] for r in records]
        correct = [r["metrics"]["endpoints"][str(primary_step)]["rmse"] for r in records]
        clock[arm] = {"diagnostic_records": len(records), "offset_minutes": config["analysis"]["clock_defect_minutes"],
                      "correct_primary_rmse_ppm": float(np.mean(correct)), "offset_primary_rmse_ppm": float(np.mean(offset)),
                      "offset_minus_correct_rmse_ppm": float(np.mean(np.asarray(offset) - correct)),
                      "max_restored_abs_normalized": max(r["clock_diagnostic"]["restored_max_abs_normalized"] for r in records)}
    # Train/validation budgets occur once per fitting cell, not once per test variant.
    budgets = []
    exclusions = []
    for room, tau, irregular in itertools.product(rooms, config["tau_ratios"], config["attendance"]):
        data = datasets[(room, next(r for r in regimes if r[:2] == (tau, irregular)))]
        budgets.append({"room": room, "tau": tau, "irregular": irregular,
                        "train": data["train"], "validation": data["validation"], "normalizer": data["normalizer"]})
    for (room, regime), data in datasets.items():
        exclusions.append({"room": room, **dict(zip(REGIME_FIELDS, regime)), "test": data["test"]})
    result = {"version": "vm_n04_analysis_v1", "status": "complete", "scientific_gate_passed": all(v["passed"] for v in primary.values()),
              "independent_unit": "generated room/trajectory draw", "rooms": rooms, "regimes": [dict(zip(REGIME_FIELDS, r)) for r in regimes],
              "initialization_seeds": config["stages"]["confirmation"]["initialization_seeds"],
              "aggregation": "Mean endpoint RMSE over initializations within room/regime; equal regimes within room; equal rooms. Whole-horizon RMSE is a separate flattened-target error. Skill uses the ratio of balanced normalized RMSE means.",
              "diagnostic_labels": {"origin_outage": "Sensor history contains an outage; not necessarily an outage exactly at issuance.",
                                    "target_outage": "Target event time lies within the outage interval; scoring truth remains available offline.",
                                    "empty_groups": "Null errors, zero contributing rooms, and zero labels are retained; whole-horizon subgroup diagnostics were not computed."},
              "uncertainty": "Paired whole-room percentile bootstrap; six-room synthetic precision is limited. Initialization SD is nested and is not an independent replication count.",
              "primary_step": primary_step, "analysis_settings": config["analysis"], "primary": primary,
              "aggregate": aggregate, "runtime_parameters": runtime, "clock_diagnostic": clock,
              "information_budgets": budgets, "test_eligibility": exclusions,
              "environment": summary.get("environment", {}), "elapsed_seconds": summary["elapsed_seconds"],
              "cuda_peak_allocated_bytes": summary.get("cuda_peak_allocated_bytes"),
              "process_rss_bytes_at_end": summary.get("process_rss_bytes_at_end"),
              "record_counts": {"neural": len(summary["records"]), "references": len(summary["references"]), "datasets": len(datasets), "cells": len(cells)},
              "verification_scope": "Complete saved-summary identities, normalization, counts and aggregation; no independent prediction-array replay or building-population validation."}
    return result, cells


def write_outputs(result: dict, cells: list[dict], output: Path) -> None:
    """Save strict compact JSON, complete cell CSV and a static scientific plot."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "vm_n04_results_v1.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "vm_n04_cells_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"none": "#555555", "fourier": "#0072B2", "fixed": "#009E73", "learned": "#D55E00",
              "ridge": "#CC79A7", "persistence": "#999933", "seasonal": "#7777BB"}
    for arm, color in colors.items():
        selected = sorted([r for r in result["aggregate"] if r["arm"] == arm and r["endpoint_minutes"] is not None], key=lambda r: r["endpoint_minutes"])
        axes[0].plot([r["endpoint_minutes"] for r in selected], [r["metrics"]["normalized_rmse"]["mean"] for r in selected], "o-", label=arm, color=color)
    axes[0].set(xlabel="Forecast endpoint (minutes)", ylabel="Balanced normalized RMSE", xticks=[15, 60, 240], title="All regimes; mean within rooms first")
    axes[0].legend(fontsize=8, ncol=2)
    for offset, (control, contrast) in enumerate(result["primary"].items()):
        x = np.arange(len(result["regimes"])) + (offset - 0.5) * 0.25
        axes[1].scatter(x, np.asarray(contrast["regime_improvements"]) * 100, s=22, label=f"learned vs {control}", color=colors[control])
    axes[1].axhline(-10, color="#990000", linestyle="--", label="-10% regime guard")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Prespecified regime index (CSV/JSON order)", ylabel="One-hour RMSE improvement (%)", title="Every regime retained")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"VM-N04 | {len(result['rooms'])} independent simulated rooms | scientific gate {'passed' if result['scientific_gate_passed'] else 'failed'}")
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures/vm_n04_v1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Analyze the locked completed study without fitting or selecting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    summary, config, provenance = load_study(args.study_dir)
    result, cells = analyze_summary(summary, config)
    result["provenance"] = provenance
    write_outputs(result, cells, args.output_dir)
    print(json.dumps({"status": result["status"], "scientific_gate_passed": result["scientific_gate_passed"], "cells": len(cells), "primary": result["primary"]}, indent=2))


if __name__ == "__main__":
    main()
