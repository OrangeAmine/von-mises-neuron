"""Numerical aggregation tests use fabricated records, never reserved targets."""

from __future__ import annotations

import copy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from src.smooth_local_analysis_v1 import paired_contrast, select_learning_rates, select_models, summarize


@pytest.fixture
def config():
    value = json.loads((Path(__file__).resolve().parents[1] / "configs/vm_n01_v1.json").read_text())
    value["analysis"]["bootstrap_resamples"] = 1000
    return value


def fake_records(config, phase="confirmation"):
    records = []
    task = config["tasks"]
    budgets = task["pilot_label_budgets"] if phase == "pilot" else task["label_budgets"]
    for spec, family, rep, initialization, labels, cap in product(config["models"], task["families"],
            range(task["replicates"][phase]), range(task["initializations"][phase]),
            budgets, config["training"]["time_budgets_seconds"]):
        rates = config["training"]["pilot_learning_rates"] if phase == "pilot" and spec["kind"] != "ridge" else [None]
        for rate in rates:
            value = (1 + task["families"].index(family)) * (rep+1) * (1 + initialization) * (labels/64)
            if spec["id"] == "hybrid_joint":
                value *= .8
            records.append({"phase": phase, "model_id": spec["id"], "family": family,
                            "replicate": rep, "initialization": initialization, "labels": labels,
                            "allowance": cap, "normalized_mse": value,
                            "noisy_validation_mse": 1 if rate == .05 else 2,
                            "learning_rate": rate, "status": "complete", "fit_id": str(len(records))})
    return records


def test_balanced_paired_summary_and_function_bootstrap(config):
    records = fake_records(config)
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert result["integrity"]["complete"]
    assert result["integrity"]["expected_records"] == 2640
    assert result["gate"]["passed"]
    # Mean family multiplier 2.5, function multiplier 3.5, initialization 1.5,
    # label multiplier 2.5. Integration points never enter the sample count.
    assert result["models"]["fourier16"]["balanced_error"] == pytest.approx(2.5*3.5*1.5*2.5)
    assert result["models"]["hybrid_joint"]["balanced_error"] == pytest.approx(.8*2.5*3.5*1.5*2.5)
    assert len(result["cells"]) == 11*5*2*2
    assert len(result["functions"]) == 11*5*6*2
    for contrast in result["contrasts"].values():
        assert contrast["relative_improvement"] == pytest.approx(.2)
        np.testing.assert_allclose(contrast["interval"], [.2, .2], atol=2e-15)
        assert len(contrast["paired_functions"]) == 24
        assert contrast["improving_local_families"] == 3
    first = next(row for row in result["functions"] if row["model_id"] == "fourier16" and row["family"] == "smooth" and row["replicate"] == 0)
    assert first["normalized_mse"] == pytest.approx(1*1.5*2.5)


def test_nonperiodic_excluded_from_effect_but_failed_diagnostic_blocks_claim(config):
    records = fake_records(config)
    for row in records:
        if row["family"] == "nonperiodic" and row["model_id"] == "hybrid_joint":
            row["normalized_mse"] = 1e9
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert result["gate"]["passed"]
    failed = next(row for row in records if row["family"] == "nonperiodic")
    failed["status"] = "failed"
    failed["normalized_mse"] = None
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert not result["gate"]["passed"]
    assert result["integrity"]["failed_count"] == 1
    assert any(row["status"] == "incomplete" and row["normalized_mse"] is None for row in result["cells"])


def test_missing_duplicate_and_nonfinite_cells_never_survivor_average(config):
    records = fake_records(config)
    records.pop()
    records.append(copy.deepcopy(records[0]))
    records[1]["normalized_mse"] = float("nan")
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert not result["gate"]["passed"]
    assert result["integrity"]["missing_count"] == result["integrity"]["duplicate_count"] == result["integrity"]["failed_count"] == 1
    assert any(row["observed_normalized_mse"] is not None and row["normalized_mse"] is None for row in result["cells"])
    # Every declared result still appears, and the report itself is strict JSON.
    assert len(result["cells"]) == 220
    json.dumps(result, allow_nan=False)


def test_smooth_guard_cannot_be_overruled_by_large_local_gains(config):
    records = fake_records(config)
    for row in records:
        if row["model_id"] == "hybrid_joint":
            row["normalized_mse"] *= (1.3/.8 if row["family"] == "smooth" else .1)
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert not result["gate"]["passed"]
    for contrast in result["contrasts"].values():
        assert contrast["checks"]["practical_effect"]
        assert not contrast["checks"]["smooth_guard"]
        assert contrast["smooth_regression"] == pytest.approx(.3)


def test_all_three_comparisons_required(config):
    records = fake_records(config)
    for row in records:
        if row["model_id"] == "fourier36":
            row["normalized_mse"] *= .5
    result = summarize(records, config, "hybrid_joint", "fourier36")
    assert result["contrasts"]["fourier16"]["passed"]
    assert result["contrasts"]["hybrid_fixed16"]["passed"]
    assert not result["contrasts"]["fourier36"]["passed"]
    assert not result["gate"]["passed"]


def test_pilot_rate_selection_uses_noisy_validation_all_families_and_tie_rule(config):
    records = fake_records(config, "pilot")
    for row in records:
        row["normalized_mse"] = 0 if row["learning_rate"] == .01 else 10000
    assert set(select_learning_rates(records, config).values()) == {.05}
    for row in records:
        row["noisy_validation_mse"] = 2
    assert set(select_learning_rates(records, config).values()) == {.01}
    # Nonperiodic is excluded from the scientific gate but included in pilot
    # rate selection under the prospectively declared equal-family rule.
    for row in records:
        if row["family"] == "nonperiodic" and row["learning_rate"] == .01:
            row["noisy_validation_mse"] = 100
    assert set(select_learning_rates(records, config).values()) == {.05}


def test_pilot_missing_or_failed_candidate_rejected(config):
    records = fake_records(config, "pilot")
    candidate = next(row for row in records if row["model_id"] == "hybrid_joint" and row["allowance"] == 1.)
    records.remove(candidate)
    with pytest.raises(ValueError, match="Incomplete or failed"):
        select_learning_rates(records, config)
    records.append(candidate)
    candidate["status"] = "no_eligible"
    with pytest.raises(ValueError, match="Incomplete or failed"):
        select_learning_rates(records, config)


def test_development_selects_only_declared_candidates_primary_and_balanced(config):
    records = fake_records(config, "development")
    selected = select_models(records, config)
    assert selected["selected_hybrid"] == "hybrid_joint"
    assert selected["capacity_comparator"] == "fourier36"  # Registry-order tie.
    for row in records:
        if row["model_id"] == "hybrid_joint" and row["family"] == "nonperiodic":
            row["normalized_mse"] = 1e9
        if row["model_id"] == "hybrid_vp" and row["allowance"] == .25:
            row["normalized_mse"] = 0
    assert select_models(records, config) == selected
    candidate = next(row for row in records if row["model_id"] == "fourier36" and row["family"] == "smooth" and row["allowance"] == 1.)
    candidate["normalized_mse"] = None
    with pytest.raises(ValueError):
        select_models(records, config)


def test_pairing_and_bootstrap_reproducibility(config):
    summary = summarize(fake_records(config), config, "hybrid_joint", "fourier36")
    first = paired_contrast(summary["functions"], config, "hybrid_joint", "fourier16")
    np.random.seed(982)
    second = paired_contrast(list(reversed(summary["functions"])), config, "hybrid_joint", "fourier16")
    assert first == second
    altered = copy.deepcopy(summary["functions"])
    for row in altered:
        if row["model_id"] == "hybrid_joint":
            row["normalized_mse"] *= (6-row["replicate"])/(row["replicate"]+1)
    crossed = paired_contrast(altered, config, "hybrid_joint", "fourier16")
    assert crossed["relative_improvement"] == pytest.approx(.2)
    assert crossed["interval"][1] - crossed["interval"][0] > .1


def test_zero_reference_and_incomplete_bundles_fail_without_nan(config):
    summary = summarize(fake_records(config), config, "hybrid_joint", "fourier36")
    functions = summary["functions"]
    for row in functions:
        if row["model_id"] == "fourier16":
            row["normalized_mse"] = 0
    contrast = paired_contrast(functions, config, "hybrid_joint", "fourier16")
    assert not contrast["passed"] and contrast["reason"] == "nonpositive_reference_error"
    functions.pop(next(i for i, row in enumerate(functions) if row["model_id"] == "hybrid_joint" and row["family"] == "smooth" and row["allowance"] == 1.))
    contrast = paired_contrast(functions, config, "hybrid_joint", "fourier16")
    assert not contrast["complete"]
    json.dumps(contrast, allow_nan=False)
