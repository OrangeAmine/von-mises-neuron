"""Meaningful delayed-feedback, causal-feature and interval-contract checks."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch

from src.dynamics_data_v1 import DynamicsConfig, generate_trajectory
from src.uncertainty_data_v1 import (
    DAY_NS, NEVER, DelayedErrorLedger, ForecastRows, build_rows, origin_features,
    scenario_trajectory, timeline_ns,
)
from src.uncertainty_calibration_v1 import (
    CONDITIONAL, ScaleModel, calibrate, design, empirical_radius, fit_scale, phase_hours,
)
from src.uncertainty_analysis_v1 import metric_arrays, validate_alignment, cell_table, paired_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/vm_n05_v1.json").read_text())


def toy_rows():
    """Shuffled targets include delayed, absent and future labels."""
    issue = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    return ForecastRows(issue, issue + 2, np.array([3, 9, NEVER, 5, 7], dtype=np.int64),
                        np.full(5, 2, dtype=np.int64), np.full(5, 100.),
                        np.array([101., 103., 109., 115., 124.]), np.zeros((5, 4)),
                        np.zeros(5), issue.copy())


def test_delayed_ledger_requires_both_clocks_and_accepts_arrival_equality():
    ledger = DelayedErrorLedger(toy_rows(), 1)
    assert ledger.indices(2, 0, 5, 2).tolist() == []
    assert ledger.indices(3, 0, 5, 2).tolist() == [0]
    assert ledger.indices(5, 0, 5, 2).tolist() == [0, 3]
    assert ledger.indices(8, 0, 5, 2).tolist() == [0, 3, 4]
    assert ledger.indices(9, 0, 5, 2).tolist() == [0, 1, 3, 4]


def test_pre_split_forecast_still_has_unmatured_target():
    ledger = DelayedErrorLedger(toy_rows(), 1)
    assert ledger.indices(4, 0, 4, 2).tolist() == [0]
    with pytest.raises(ValueError, match="unmatured"):
        ledger.errors(np.array([1]), 4)
    with pytest.raises(ValueError, match="unmatured"):
        ledger.errors(np.array([2]), 999)


def test_row_order_independent_latest_selection():
    original = toy_rows()
    shuffled = original.subset(np.array([4, 2, 1, 0, 3]))
    left, right = DelayedErrorLedger(original, 1), DelayedErrorLedger(shuffled, 1)
    a, b = left.indices(9, 0, 9, 2, 2), right.indices(9, 0, 9, 2, 2)
    np.testing.assert_array_equal(original.target_ns[a], shuffled.target_ns[b])
    np.testing.assert_array_equal(left.errors(a, 9), right.errors(b, 9))


def test_unavailable_label_mutation_does_not_change_ledger_values():
    original = toy_rows()
    changed = original.subset(np.arange(5))
    changed.truth[[1, 2, 4]] = [1e9, np.nan, -1e9]
    for rows in (original, changed):
        ledger = DelayedErrorLedger(rows, 1)
        np.testing.assert_array_equal(ledger.errors(ledger.indices(5, 0, 5, 2), 5), [1., 15.])


@pytest.mark.parametrize("defect", ["arrival", "horizon", "duplicate", "point", "shape"])
def test_ledger_rejects_invalid_identity(defect):
    rows = toy_rows()
    if defect == "arrival":
        rows.available_ns[0] = 1
    elif defect == "horizon":
        rows.target_ns[0] += 1
    elif defect == "duplicate":
        rows.issue_ns[1], rows.target_ns[1] = rows.issue_ns[0], rows.target_ns[0]
    elif defect == "point":
        rows.point[0] = np.nan
    else:
        rows.features = np.zeros((4, 4))
    with pytest.raises(ValueError):
        DelayedErrorLedger(rows, 1)


def test_nonfinite_arrived_labels_are_excluded_not_turned_into_zero():
    rows = toy_rows()
    rows.truth[0] = np.nan
    ledger = DelayedErrorLedger(rows, 1)
    assert ledger.indices(3, 0, 5, 2).size == 0
    assert ledger.timed_indices(3, 0, 5, 2).tolist() == [0]
    with pytest.raises(ValueError, match="nonfinite"):
        ledger.errors(np.array([0]), 3)


def test_sensor_mutations_after_issuance_or_arrival_are_invisible():
    trajectory = generate_trajectory(15007, DynamicsConfig(split_days=(2, 1, 1)))
    origin = 100
    arrivals = trajectory.arrival_times.asi8.copy()
    arrivals[origin - 3] = trajectory.event_times[origin + 1].value
    trajectory = replace(trajectory, arrival_times=pd.to_datetime(arrivals, utc=True))
    before = origin_features(trajectory, origin, 4, 50.)
    changed = trajectory.observed_ppm.copy()
    changed[trajectory.arrival_times > trajectory.event_times[origin]] = np.nan
    after = origin_features(replace(trajectory, observed_ppm=changed), origin, 4, 50.)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


def test_outage_hidden_truth_cannot_enter_feature_or_ledger(config):
    cfg = deepcopy(config)
    trajectory = scenario_trajectory(15011, cfg, cfg["scenarios"][-1])
    rows, _ = build_rows(trajectory, cfg)
    changed = trajectory.observed_ppm.copy()
    changed[~trajectory.sensor_available] += 100000
    other, _ = build_rows(replace(trajectory, observed_ppm=changed), cfg)
    np.testing.assert_array_equal(rows.point, other.point)
    np.testing.assert_array_equal(rows.features, other.features)
    absent = rows.available_ns == NEVER
    assert absent.any() and np.any(rows.truth[absent] != other.truth[absent])
    ledger = DelayedErrorLedger(rows, 15 * 60 * 10**9)
    idx = ledger.indices(int(rows.issue_ns.max()), 0, int(rows.issue_ns.max()), 4)
    assert not absent[idx].any()


def test_known_target_clock_is_future_time_not_future_sensor(config):
    trajectory = generate_trajectory(15013, DynamicsConfig(split_days=(2, 1, 1)))
    rows, _ = build_rows(trajectory, config)
    expected = 2 * np.pi * (rows.target_ns % DAY_NS) / DAY_NS
    np.testing.assert_array_equal(rows.phase, expected)
    changed = replace(trajectory, observed_ppm=np.full(len(trajectory.event_times), 500.))
    other, _ = build_rows(changed, config)
    np.testing.assert_array_equal(rows.phase, other.phase)
    assert np.any(rows.phase != 2 * np.pi * (rows.issue_ns % DAY_NS) / DAY_NS)


def test_offline_nonfinite_target_never_filters_issuance(config):
    trajectory = generate_trajectory(15017, DynamicsConfig(split_days=(2, 1, 1)))
    rows, _ = build_rows(trajectory, config)
    origin = rows.origin_step[0]
    altered = trajectory.observed_ppm.copy()
    altered[origin + 1] = np.nan
    other, _ = build_rows(replace(trajectory, observed_ppm=altered), config)
    mask = rows.origin_step == origin
    compare = other.origin_step == origin
    np.testing.assert_array_equal(rows.issue_ns[mask], other.issue_ns[compare])
    np.testing.assert_array_equal(rows.point[mask], other.point[compare])
    assert np.isnan(other.truth[compare]).any()


@pytest.mark.parametrize("arm", CONDITIONAL)
def test_wrap_and_matched_information(arm, config):
    angles = np.array([0., 2*np.pi, -2*np.pi, 23/24*2*np.pi, .23])
    features = np.arange(20).reshape(5, 4).astype(float)
    a = design(features, angles, arm, config["scale"])
    b = design(features, angles + 2 * np.pi, arm, config["scale"])
    assert a.shape == (5, 28)
    np.testing.assert_array_equal(a[:, :4], features)
    np.testing.assert_allclose(a, b, atol=2e-14)
    assert phase_hours(angles).tolist() == [0, 0, 0, 23, 0]


@pytest.mark.parametrize("n,alpha,expected", [(19,.1,18.),(9,.1,9.),(8,.1,np.inf),(0,.1,np.inf),(99,.1,90.),(9,.2,8.)])
def test_exact_empirical_order_statistic(n, alpha, expected):
    assert empirical_radius(np.arange(n, 0, -1, dtype=float), alpha) == expected


@pytest.mark.parametrize("scores,alpha", [([np.nan],.1),([np.inf],.1),([-1],.1),([1],0),([1],1)])
def test_bad_quantile_inputs_fail(scores, alpha):
    with pytest.raises(ValueError):
        empirical_radius(np.array(scores), alpha)


def test_positive_scale_sparse_and_nonfinite_fallback_and_clipping(config):
    settings = config["scale"]
    model = ScaleModel("vm", settings, np.zeros(28), 2000., 10., np.full(24, 10), 100, 0., "cpu")
    angles = np.array([0., .3, .6, .9])
    features = np.zeros((4, 4))
    features[1, 0] = np.nan
    model.phase_counts[phase_hours(angles)[2]] = 0
    scale, flags = model.predict(features, angles)
    assert scale[0] == pytest.approx(1000.)
    assert flags["ceiling"][0]
    assert scale[1] == pytest.approx(10.) and flags["fallback"][1]
    assert scale[2] == pytest.approx(10.) and flags["fallback"][2]
    model.bias = -2000
    scale, flags = model.predict(np.zeros((1, 4)), np.zeros(1))
    assert scale[0] == 1 and flags["floor"][0]


@pytest.fixture(scope="module")
def fitted_fixture():
    cfg = json.loads((ROOT / "configs/vm_n05_v1.json").read_text())
    trajectory = scenario_trajectory(15019, cfg, cfg["scenarios"][0])
    rows, _ = build_rows(trajectory, cfg)
    ledger = DelayedErrorLedger(rows, 15 * 60 * 10**9)
    models = {(h, arm): fit_scale(ledger, h, arm, cfg, torch.device("cpu"))
              for h in cfg["horizon_steps"] for arm in CONDITIONAL}
    return cfg, rows, ledger, models


def test_scale_fit_excludes_all_future_and_unmatured_labels(fitted_fixture):
    cfg, rows, ledger, models = fitted_fixture
    t = timeline_ns(cfg)
    changed = rows.subset(np.arange(len(rows)))
    unavailable = (changed.target_ns >= t["scale_end"]) | (changed.available_ns >= t["scale_end"])
    changed.truth[unavailable] = 1e15
    other = DelayedErrorLedger(changed, 15 * 60 * 10**9)
    actual = fit_scale(other, 16, "vm", cfg, torch.device("cpu"))
    np.testing.assert_array_equal(actual.weight, models[(16, "vm")].weight)
    assert actual.bias == models[(16, "vm")].bias
    assert models[(16, "vm")].fit_count < 14 * 24


def test_no_scale_labels_defined_fallback(fitted_fixture):
    cfg, rows, _, _ = fitted_fixture
    rows = rows.subset(np.arange(len(rows)))
    rows.available_ns[:] = NEVER
    model = fit_scale(DelayedErrorLedger(rows, 15 * 60 * 10**9), 4, "vm", cfg, torch.device("cpu"))
    value, flags = model.predict(np.zeros((2, 4)), np.array([0., 1.]))
    assert model.fit_count == 0
    np.testing.assert_array_equal(value, np.ones(2))
    assert flags["fallback"].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_scale_cpu_cuda_parity(fitted_fixture):
    cfg, rows, ledger, models = fitted_fixture
    model = fit_scale(ledger, 4, "vm", cfg, torch.device("cuda"))
    left = models[(4, "vm")].predict(rows.features, rows.phase)[0]
    right = model.predict(rows.features, rows.phase)[0]
    np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)


@pytest.fixture(scope="module")
def intervals_fixture(fitted_fixture):
    cfg, _, ledger, models = fitted_fixture
    frame = pd.DataFrame(calibrate(ledger, models, cfg))
    frame["room_seed"], frame["scenario"] = 15019, "stable"
    return frame


def test_all_arms_points_matched_and_pools_time_valid(intervals_fixture, fitted_fixture):
    cfg, _, _, _ = fitted_fixture
    validate_alignment(intervals_fixture, cfg)
    grouped = intervals_fixture.groupby(["issue_ns", "horizon_steps"])
    assert (grouped.point_ppm.nunique() == 1).all()
    assert (intervals_fixture.calibration_latest_arrival_ns <= intervals_fixture.calibration_cutoff_ns).all()
    rolling = intervals_fixture[intervals_fixture.arm == "rolling"]
    assert (rolling.calibration_count == 168).all()
    static = intervals_fixture[intervals_fixture.arm == "static"]
    assert (static.calibration_count > 168).all()


def test_future_label_mutation_cannot_change_already_issued_intervals(fitted_fixture, intervals_fixture):
    cfg, rows, _, models = fitted_fixture
    cutoff = timeline_ns(cfg)["evaluation_start"] + 3 * 3600 * 10**9
    altered = rows.subset(np.arange(len(rows)))
    altered.truth[(altered.available_ns > cutoff) | (altered.target_ns > cutoff)] = 1e9
    # Limit evaluation to four issue times for a cheap end-to-end causality check.
    short = deepcopy(cfg)
    short["timeline_days"]["end"] = cfg["timeline_days"]["evaluation_start"] + 1
    after = pd.DataFrame(calibrate(DelayedErrorLedger(altered, 15 * 60 * 10**9), models, short))
    keys = ["issue_ns", "horizon_steps", "arm"]
    fields = ["point_ppm", "lower_ppm", "upper_ppm", "calibration_count"]
    left = intervals_fixture[intervals_fixture.issue_ns <= cutoff].sort_values(keys)[fields].to_numpy()
    right = after[after.issue_ns <= cutoff].sort_values(keys)[fields].to_numpy()
    np.testing.assert_array_equal(left, right)


def test_small_online_pool_falls_back_without_finite_cap(fitted_fixture):
    cfg, rows, _, models = fitted_fixture
    config = deepcopy(cfg)
    config["timeline_days"]["calibration_start"] = config["timeline_days"]["evaluation_start"]
    config["timeline_days"]["end"] = config["timeline_days"]["evaluation_start"] + 1
    frame = pd.DataFrame(calibrate(DelayedErrorLedger(rows, 15 * 60 * 10**9), models, config))
    first = frame[frame.issue_ns == timeline_ns(cfg)["evaluation_start"]]
    assert first.infinite_interval.all()
    assert first[first.arm.isin(CONDITIONAL)].small_pool_fallback.all()
    assert np.isneginf(first.lower_ppm).all() and np.isposinf(first.upper_ppm).all()


def test_scalar_interval_score_miss_rates_and_infinite_pool():
    metrics = metric_arrays(np.array([910., 950., 850.]), np.full(3, 900.),
                             np.full(3, 860.), np.full(3, 940.), .1)
    np.testing.assert_array_equal(metrics["interval_score"], [80., 280., 280.])
    np.testing.assert_array_equal(metrics["lower_miss"], [0., 0., 1.])
    np.testing.assert_array_equal(metrics["upper_miss"], [0., 1., 0.])
    infinite = metric_arrays(np.array([0.]), np.array([0.]), np.array([-np.inf]), np.array([np.inf]), .1)
    assert infinite["interval_score"][0] == np.inf and infinite["coverage"][0] == 1.


@pytest.mark.parametrize("defect", ["truth", "shape", "bounds"])
def test_invalid_metrics_fail(defect):
    args = [np.array([1.]), np.array([1.]), np.array([0.]), np.array([2.])]
    if defect == "truth":
        args[0][0] = np.nan
    elif defect == "shape":
        args[1] = np.array([1., 2.])
    else:
        args[2][0] = 3.
    with pytest.raises(ValueError):
        metric_arrays(*args, .1)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "point", "target", "cutoff"])
def test_alignment_failures_detected(intervals_fixture, config, defect):
    frame = intervals_fixture.head(36).copy()
    if defect == "missing":
        frame = frame.iloc[1:]
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "point":
        frame.loc[frame.index[0], "point_ppm"] += 1
    elif defect == "target":
        frame.loc[frame.index[0], "target_ns"] += 100
    else:
        frame.loc[frame.index[0], "calibration_cutoff_ns"] += DAY_NS
    with pytest.raises(ValueError):
        validate_alignment(frame, config)


def test_empty_subgroups_and_balance_are_explicit(intervals_fixture, config):
    cells = cell_table(intervals_fixture, config)
    empty = cells[cells.stratum == "stale_sensor"]
    assert (empty.n == 0).all()
    assert empty.interval_score.isna().all()
    total = cells[cells.stratum == "all"].n.sum()
    assert total == len(intervals_fixture)
    for _, group in cells[cells.stratum == "all"].groupby("horizon_steps"):
        assert group.point_mae.nunique() == 1


def test_paired_room_and_block_bootstrap_preserve_identical_arms(intervals_fixture, config):
    cfg = deepcopy(config)
    cfg["analysis"]["bootstrap_draws"] = 100
    frames = []
    for room in (15021, 15023):
        for scenario in cfg["scenarios"]:
            f = intervals_fixture.copy()
            f["room_seed"], f["scenario"] = room, scenario["name"]
            fourier = f[f.arm == "fourier"].sort_values(["issue_ns", "horizon_steps"])
            vm_idx = f[f.arm == "vm"].sort_values(["issue_ns", "horizon_steps"]).index
            f.loc[vm_idx, ["lower_ppm", "upper_ppm"]] = fourier[["lower_ppm", "upper_ppm"]].to_numpy()
            frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    result = paired_summary(frame, cell_table(frame, cfg), cfg)
    assert result["primary_improvement"] == 0
    assert result["paired_room_95_interval"] == [0., 0.]
    np.testing.assert_allclose(result["conditional_fixed_interval_block_95_interval"], [0., 0.], atol=1e-14)
    assert not result["scientific_gate_passed"]
    assert result["independent_rooms"] == 2


def test_deadline_checked_before_issuance(fitted_fixture):
    config, _, ledger, models = fitted_fixture
    with pytest.raises(TimeoutError):
        calibrate(ledger, models, config, time.perf_counter() - 1)


def test_artifact_export_handles_absent_arrival_exactly(intervals_fixture, config, tmp_path):
    from scripts.run_vm_n05_v1 import export_contract
    frame = intervals_fixture.copy()
    frame["room_seed"] = config["room_seeds"][0]
    frame["point_model_id"] = config["point_predictor"]
    frame["units"], frame["source_id"], frame["config_id"] = "ppm", "source", "config"
    outage = frame.copy()
    outage["scenario"], outage["observation_available_ns"] = "shifted_noisy_outage", NEVER
    export_contract(pd.concat([frame, outage]), tmp_path, config)
    exported = pd.read_csv(tmp_path / "interval_contract_sample.csv")
    assert len(exported) == (24 + 4) * 3 * 6
    assert exported[exported.scenario == "shifted_noisy_outage"].observation_available_time_utc.isna().all()
    first = exported.iloc[0]
    assert pd.Timestamp(first.target_time_utc) - pd.Timestamp(first.issue_time_utc) == pd.Timedelta(minutes=first.horizon_minutes)


def test_reserved_seed_config_disjoint_and_matrix_frozen(config):
    from scripts.run_vm_n05_v1 import check_config
    check_config(config)
    config["room_seeds"][0] = 430101
    with pytest.raises(ValueError, match="fresh"):
        check_config(config)
