"""Behavioral causality and paired-unit checks for discounted expert mixing."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.calendar_trust_v1 import NEVER, delayed_mix, paired_summary


def fixture(n=12, h=3):
    """Return a disposable fixture without using any reserved room seed."""
    issue = np.arange(n, dtype=np.int64) * 10 + 100
    target = issue[:, None] + np.arange(1, h + 1) * 10
    return dict(experts=np.tile([0., 10.], (n, h, 1)), truth=np.zeros((n, h)),
                issue_ns=issue, target_ns=target, arrival_ns=target+5,
                initial_loss=np.ones((h, 2)), initialization_cutoff_ns=99,
                scale=10., eta=2., half_life_updates=2.)


def test_feedback_arrives_after_target_and_before_use():
    data = fixture()
    result = delayed_mix(**data)
    np.testing.assert_array_equal(result.weights[:2], .5)
    assert result.weights[2, 0, 0] > .5
    assert result.weights[2, 1, 0] == .5
    assert np.all(result.latest_feedback_ns <= data["issue_ns"][:, None])
    np.testing.assert_allclose(result.weights.sum(-1), 1)
    assert np.all((result.prediction >= 0) & (result.prediction <= 10))


def test_future_truth_and_predictions_cannot_change_earlier_outputs():
    data = fixture()
    expected = delayed_mix(**data)
    cutoff = data["issue_ns"][5]
    data["truth"][data["arrival_ns"] > cutoff] = 1e9
    data["experts"][6:] = 500
    actual = delayed_mix(**data)
    np.testing.assert_array_equal(actual.prediction[:6], expected.prediction[:6])
    np.testing.assert_array_equal(actual.weights[:6], expected.weights[:6])


def test_missing_truth_is_never_inspected_and_weights_do_not_decay():
    data = fixture()
    data["arrival_ns"][:] = NEVER
    data["truth"][:] = np.nan
    result = delayed_mix(**data)
    np.testing.assert_array_equal(result.weights, .5)
    np.testing.assert_array_equal(result.feedback_count, 0)


def test_zero_delay_updates_at_target_time():
    data = fixture()
    data["arrival_ns"] = data["target_ns"].copy()
    result = delayed_mix(**data)
    assert result.feedback_count[1, 0] == 1
    assert result.feedback_count[1, 1] == 0


def test_per_horizon_losses_and_input_state_are_independent():
    data = fixture()
    data["truth"][:, 1] = 10
    initial = data["initial_loss"].copy()
    result = delayed_mix(**data)
    assert result.weights[-1, 0, 0] > .5
    assert result.weights[-1, 1, 0] < .5
    np.testing.assert_array_equal(data["initial_loss"], initial)


def test_affine_change_of_units_preserves_weights():
    data = fixture()
    original = delayed_mix(**data)
    data["experts"] = data["experts"] * 7 + 400
    data["truth"] = data["truth"] * 7 + 400
    data["scale"] *= 7
    changed = delayed_mix(**data)
    np.testing.assert_allclose(changed.weights, original.weights)
    np.testing.assert_allclose(changed.prediction, original.prediction * 7 + 400)


@pytest.mark.parametrize("field,value", [("eta", 0), ("eta", np.nan), ("scale", -1),
                                          ("half_life_updates", 0), ("initialization_cutoff_ns", 101)])
def test_invalid_parameters_rejected(field, value):
    data = fixture()
    data[field] = value
    with pytest.raises(ValueError):
        delayed_mix(**data)


@pytest.mark.parametrize("change", ["early_arrival", "same_time_target", "unordered_origins", "bad_shape", "nonfinite_expert", "negative_loss"])
def test_invalid_inputs_rejected(change):
    data = fixture()
    if change == "early_arrival":
        data["arrival_ns"][0, 0] = 99
    elif change == "same_time_target":
        data["target_ns"][0, 0] = 100
    elif change == "unordered_origins":
        data["issue_ns"][2] = 100
    elif change == "bad_shape":
        data["truth"] = np.zeros((1, 1))
    elif change == "nonfinite_expert":
        data["experts"][0, 0, 0] = np.inf
    else:
        data["initial_loss"][0, 0] = -1
    with pytest.raises(ValueError):
        delayed_mix(**data)


def test_arrived_nonfinite_truth_is_an_error():
    data = fixture()
    data["truth"][0, 0] = np.nan
    with pytest.raises(ValueError, match="arrived observation"):
        delayed_mix(**data)


def records():
    """Construct two room-level effects with known arithmetic mean."""
    return [{"seed": seed, "regime": regime, "arm": arm, "normalization_std": scale,
             "endpoints": {"4": {"rmse": scale * (effect if arm == "adaptive" else 1.)}}}
            for seed, scale, effect in ((1, 1, .8), (2, 100, .6))
            for regime in ("stable", "shifted") for arm in ("sensor", "seasonal", "fixed", "adaptive")]


def test_room_paired_analysis_normalizes_before_pooling_and_rejects_missing_pairs():
    rows = records()
    result = paired_summary(rows, 4, 11, 1000, 5, 10)
    assert result["overall_passed"]
    assert result["contrasts"]["sensor"]["mean_gain_percent"] == pytest.approx(30.)
    assert result["contrasts"]["sensor"]["ci95"] == pytest.approx([20., 40.])
    for invalid in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            paired_summary(invalid, 4, 11, 1000, 5, 10)


def test_intersection_gate_does_not_hide_one_failed_control():
    rows = records()
    for row in rows:
        if row["arm"] == "seasonal":
            row["endpoints"]["4"]["rmse"] *= .2
    result = paired_summary(rows, 4, 11, 1000, 5, 10)
    assert not result["overall_passed"]
    assert not result["contrasts"]["seasonal"]["passed"]


def test_prospective_seed_sets_and_selection_budget():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/calendar_trust_v1.json").read_text())
    assert not set(cfg["development_seeds"]) & set(cfg["confirmation_seeds"])
    assert len(set(cfg["confirmation_seeds"])) == 12
    assert cfg["eta_candidates"] == sorted(set(cfg["eta_candidates"]))


def test_disposable_runner_selection_confirmation_replay_and_no_overwrite(tmp_path, monkeypatch):
    """Exercise the real data/fit/artifact path on non-study seeds and short data."""
    from scripts import run_calendar_trust_v1 as runner
    cfg = json.loads(runner.CONFIG.read_text())
    cfg.update(development_seeds=[9876501], confirmation_seeds=[9876502], bootstrap_replicates=20)
    cfg["data"]["split_days"] = [3, 2, 2]
    runner.write_json(tmp_path / "lock.json", {"disposable_test": True})
    monkeypatch.setattr(runner, "verify_lock", lambda output: {"config": cfg})
    runner.run_stage(tmp_path, "development")
    dev = json.loads((tmp_path / "development/summary.json").read_text())
    assert dev["selected_eta"] == min(cfg["eta_candidates"], key=lambda e: (dev["selection_scores"][str(e)], e))
    (tmp_path / "engineering").mkdir()
    runner.write_json(tmp_path / "engineering/results.json", {"status": "complete", "pytest_returncode": 0})
    runner.run_stage(tmp_path, "confirmation")
    assert runner.audit(tmp_path)["cells_replayed"] == 2
    with pytest.raises(FileExistsError):
        runner.run_stage(tmp_path, "development")
    with (tmp_path / "lock.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="dependency"):
        runner.audit(tmp_path)
