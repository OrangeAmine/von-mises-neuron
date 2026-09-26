"""Manufactured metric checks for nested VM-N04 confirmation aggregation."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_vm_n04_v1 as audit


@pytest.fixture
def manufactured():
    """Build paired rooms with deliberately different scales and seed errors."""
    config = audit.read_json(audit.ROOT / "configs/vm_n04_v1.json")
    config["tau_ratios"] = [0.25, 4.0]
    config["attendance"] = [False]
    config["stages"]["confirmation"]["room_seeds"] = [91001, 91002, 91003]
    config["analysis"]["bootstrap_draws"] = 400
    rooms = config["stages"]["confirmation"]["room_seeds"]
    seeds = config["stages"]["confirmation"]["initialization_seeds"]
    fits = len(rooms) * 2 * len(seeds) * 4
    summary = dict(stage="confirmation", status="complete", planned_fits=fits, completed_fits=fits,
                   records=[], references=[], datasets=[], elapsed_seconds=20.0)
    regimes = list(itertools.product(config["tau_ratios"], config["attendance"], [False, True], [False, True]))
    arm_errors = dict(none=1.3, fourier=1.0, fixed=1.05, learned=0.8, ridge=1.1, persistence=1.5, seasonal=1.4)
    for ri, room in enumerate(rooms):
        std = 10.0 * (ri + 1)
        for regime in regimes:
            tau, irregular, shifted, outage = regime
            identity = dict(room=room, tau=tau, irregular=irregular, shifted=shifted, outage=outage)
            windows = 5 if outage else 8
            def receipt(n):
                return dict(windows=n, label_uses=n * 16, unique_target_times=n * 4,
                            exclusions=dict(candidates=n + 3, insufficient_history=2, insufficient_future=1, retained=n),
                            first_origin="2030-01-01 06:00:00+00:00", last_origin="2030-01-01 14:00:00+00:00")
            summary["datasets"].append(dict(identity, normalizer=dict(mean=420.0, std=std),
                                             train=receipt(20), validation=receipt(10), test=receipt(windows)))
            for arm in arm_errors:
                for si, seed in enumerate(seeds if arm in config["arms"] else [None]):
                    # Seed arithmetic averages to the arm value; room errors
                    # differ so pooled-label and room-balanced means differ.
                    factor = arm_errors[arm] + 0.1 * ri + (0.02 * (2 * si - 1) if seed is not None else 0)
                    factor += 0.1 * shifted + 0.15 * outage + 0.02 * tau
                    error = std * factor
                    def metric(n, value=error):
                        return dict(n=n, rmse=value if n else None, mae=0.75 * value if n else None)
                    endpoints = {}
                    for step in (1, 4, 16):
                        e = error * (1 + step / 100)
                        endpoints[str(step)] = dict(metric(windows, e), normalized_rmse=e / std, clean=metric(windows, e * 0.9),
                                                    boundary=metric(2, e * 1.3), event=metric(1, e * 1.2),
                                                    origin_outage=metric(2 if outage else 0, e * 1.4),
                                                    target_outage=metric(1 if outage else 0, e * 1.5))
                    metrics = dict(all_horizon=metric(windows * 16), clean_all_horizon=metric(windows * 16, error * 0.9), endpoints=endpoints)
                    record = dict(identity, arm=arm, seed=seed, rate=0.003 if seed is not None else None,
                                  training_seconds=2.0, inference_seconds=0.1, parameters=164 if arm == "learned" else 100,
                                  metrics=metrics)
                    if seed is not None and not shifted and not outage:
                        altered = copy.deepcopy(metrics)
                        for ep in altered["endpoints"].values():
                            ep["rmse"] += 1
                            ep["normalized_rmse"] = ep["rmse"] / std
                        record["clock_diagnostic"] = dict(offset_minutes=60, restored_max_abs_normalized=0.0, offset_metrics=altered)
                    summary["records" if seed is not None else "references"].append(record)
    return summary, config


def test_nested_aggregation_and_reference_replication(manufactured):
    summary, config = manufactured
    result, cells = audit.analyze_summary(summary, config)
    assert len(cells) == 8 * 7 * 4
    assert result["scientific_gate_passed"]
    assert result["record_counts"]["neural"] == 3 * 8 * 4 * 2
    learned = next(r for r in result["aggregate"] if r["arm"] == "learned" and r["horizon"] == "step_4")
    independent_room_values = []
    for room in result["rooms"]:
        records = [r for r in summary["records"] if r["room"] == room and r["arm"] == "learned"]
        independent_room_values.append(np.mean([r["metrics"]["endpoints"]["4"]["normalized_rmse"] for r in records]))
    assert learned["metrics"]["normalized_rmse"]["room_means"] == pytest.approx(independent_room_values)
    assert learned["metrics"]["normalized_rmse"]["mean"] == pytest.approx(np.mean(independent_room_values))
    assert learned["metrics"]["normalized_rmse"]["mean_within_room_initialization_sd"] == pytest.approx(0.02 * np.sqrt(2) * 1.04)
    assert result["runtime_parameters"]["learned"]["unique_fits"] == 3 * 2 * 2
    assert result["runtime_parameters"]["ridge"]["unique_fits"] == 3 * 2
    assert result["runtime_parameters"]["learned"]["training_seconds_total"] == 24
    assert result["runtime_parameters"]["ridge"]["training_seconds_total"] == 12
    assert result["runtime_parameters"]["ridge"]["inference_calls"] == 3 * 8
    assert len(result["information_budgets"]) == 3 * 2
    assert len(result["test_eligibility"]) == 3 * 8
    assert result["clock_diagnostic"]["learned"]["offset_minus_correct_rmse_ppm"] == pytest.approx(1)
    reference = next(r for r in cells if r["arm"] == "ridge")
    assert reference["n_initializations"] == 1
    assert reference["normalized_rmse_init_sd"] == 0


def test_empty_subgroups_and_endpoint_distinction(manufactured):
    result, cells = audit.analyze_summary(*manufactured)
    row = next(r for r in cells if r["arm"] == "learned" and r["horizon"] == "step_1" and not r["outage"])
    assert row["origin_outage_rmse"] is None
    assert row["origin_outage_n_rooms"] == 0
    assert row["origin_outage_n_labels"] == 0
    assert row["boundary_n_rooms"] == 3
    assert row["boundary_n_labels"] == 6
    assert row["scored_labels_across_rooms"] == 24
    whole = next(r for r in cells if all(r[k] == row[k] for k in audit.REGIME_FIELDS) and r["arm"] == "learned" and r["horizon"] == "whole_horizon")
    assert whole["scored_labels_across_rooms"] == 24 * 16
    assert whole["rmse"] != row["rmse"]
    assert whole["boundary_n_labels"] is None
    assert row["normalized_rmse_skill_vs_none"] > 0
    assert result["primary"]["fourier"]["rooms_improved"] == 3


def test_bootstrap_matches_separate_room_draw_calculation():
    candidate = np.array([[1, 2], [2, 3], [3, 3]], dtype=float)
    control = np.array([[2, 3], [4, 3], [4, 5]], dtype=float)
    settings = dict(bootstrap_seed=94004, bootstrap_draws=10000, minimum_improvement=.05, maximum_regime_regression=.10)
    result = audit.paired_contrast(candidate, control, settings)
    rng = np.random.default_rng(94004)
    effects = []
    for _ in range(10000):
        sample = rng.integers(0, 3, size=3)
        effects.append(1 - candidate[sample].mean() / control[sample].mean())
    assert result["paired_95_interval"] == pytest.approx(np.percentile(effects, [2.5, 97.5]))
    assert result["improvement"] == pytest.approx(1 - candidate.mean() / control.mean())


def test_one_regime_failure_cannot_be_averaged_away():
    settings = dict(bootstrap_seed=2, bootstrap_draws=200, minimum_improvement=.05, maximum_regime_regression=.10)
    candidate = np.tile([1.11, .2, .2, .2], (6, 1))
    result = audit.paired_contrast(candidate, np.ones((6, 4)), settings)
    assert result["practical_pass"] and result["interval_pass"]
    assert not result["regime_guard_pass"] and not result["passed"]


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s.update(status="failed"), "complete confirmation"),
    (lambda s: s.update(completed_fits=0), "fit count"),
    (lambda s: s["records"].append(copy.deepcopy(s["records"][0])), "Duplicate model"),
    (lambda s: s["references"].append(copy.deepcopy(s["references"][0])), "Duplicate model"),
    (lambda s: s["records"].pop(), "Missing or unexpected model"),
    (lambda s: s["references"][0].update(seed=701), "Reference must not"),
    (lambda s: s["datasets"].pop(), "Missing or unexpected confirmation dataset"),
    (lambda s: s["records"][0]["metrics"]["endpoints"]["4"].update(normalized_rmse=99), "normalization"),
    (lambda s: s["records"][0]["metrics"]["endpoints"]["4"].update(n=99), "count"),
    (lambda s: s["records"][0]["metrics"]["endpoints"]["4"]["boundary"].update(n=1), "different diagnostic"),
    (lambda s: s["records"][0]["metrics"]["endpoints"]["4"]["origin_outage"].update(rmse=0), "Empty subgroup"),
    (lambda s: s["records"][0]["clock_diagnostic"].update(restored_max_abs_normalized=float("nan")), "restoration"),
    (lambda s: s["records"][0].update(parameters=9), "Parameter count varies"),
    (lambda s: s["records"][0].update(training_seconds=9), "runtime/parameters"),
    (lambda s: s["records"][0].update(rate=.01), "one common learning rate"),
    (lambda s: s["datasets"][0]["test"]["exclusions"].update(retained=0), "retained windows"),
    (lambda s: s["datasets"][0]["test"]["exclusions"].update(candidates=999), "partition candidate"),
])
def test_reject_invalid_evidence(manufactured, mutation, match):
    summary, config = manufactured
    mutation(summary)
    with pytest.raises(ValueError, match=match):
        audit.analyze_summary(summary, config)


def test_output_roundtrip_and_figure(manufactured, tmp_path):
    result, cells = audit.analyze_summary(*manufactured)
    audit.write_outputs(result, cells, tmp_path)
    restored = json.loads((tmp_path / "vm_n04_results_v1.json").read_text())
    assert restored["scientific_gate_passed"] == result["scientific_gate_passed"]
    assert len((tmp_path / "vm_n04_cells_v1.csv").read_text().splitlines()) == len(cells) + 1
    assert (tmp_path / "figures/vm_n04_v1.png").stat().st_size > 1000


def test_contained_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        audit._contained(tmp_path, "../outside.json")


def test_float32_clock_restore_discrepancy_remains_reported(manufactured):
    summary, config = manufactured
    summary["records"][0]["clock_diagnostic"]["restored_max_abs_normalized"] = 3e-6
    result, _ = audit.analyze_summary(summary, config)
    assert result["clock_diagnostic"][summary["records"][0]["arm"]]["max_restored_abs_normalized"] == 3e-6


def test_lock_manifest_verification_and_tamper_rejection(manufactured, tmp_path, monkeypatch):
    summary, config = manufactured
    source = tmp_path / "source"
    source.mkdir()
    (source / "implementation.py").write_text("frozen fixture\n")
    monkeypatch.setattr(audit, "ROOT", source)
    study = tmp_path / "study"
    stage = study / "confirmation"
    stage.mkdir(parents=True)
    def write(relative, value):
        path = study / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    lock = dict(config=config, sources={"implementation.py": audit.digest(source / "implementation.py")})
    write("source_lock.json", lock)
    summary["source_lock"] = lock
    write("pilot/summary.json", dict(status="complete", selected_rate=.003, elapsed_seconds=1))
    write("development/summary.json", dict(status="complete", elapsed_seconds=2))
    write("engineering.json", dict(full_suite_completed=True, standard_benchmarks_completed=True,
                                   all_models_accepted=False))
    dependencies = {name: audit.digest(study / name) for name in ("pilot/summary.json", "development/summary.json", "engineering.json")}
    write("confirmation/access_lock.json", dict(stage="confirmation", source_lock_sha256=audit.digest(study / "source_lock.json"), dependencies=dependencies))
    write("confirmation/summary.json", summary)
    payload = stage / "prediction.bin"
    payload.write_bytes(b"saved fixture")
    write("confirmation/artifacts.json", {"prediction.bin": audit.digest(payload)})
    write("confirmation/terminal.json", dict(status="complete", elapsed_seconds_including_summary_write=21,
                                             summary_sha256=audit.digest(stage / "summary.json"), artifacts_sha256=audit.digest(stage / "artifacts.json")))
    loaded, frozen, provenance = audit.load_study(study)
    assert loaded == summary and frozen == config
    assert provenance["artifacts_verified"] == 1
    assert not provenance["engineering"]["all_models_accepted"]
    payload.write_bytes(b"altered fixture")
    with pytest.raises(ValueError, match="artifact mismatch"):
        audit.load_study(study)
