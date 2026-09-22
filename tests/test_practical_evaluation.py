"""Small deterministic checks for causal controls and practical selection."""

import numpy as np
import pytest

from src.practical_evaluation import (
    fit_ridge_residual, forecast_metrics, select_candidate, simple_predictions,
)


def test_metrics_compute_original_units_and_horizon_errors():
    target = np.array([[1., 2.], [3., 4.]])
    pred = np.array([[2., 2.], [1., 4.]])
    metrics = forecast_metrics(pred, target)
    assert metrics["sse"] == 5 and metrics["n_targets"] == 4
    assert metrics["rmse"] == np.sqrt(1.25)
    assert metrics["mae"] == .75
    assert metrics["r2"] == 0
    assert metrics["wape"] == 30
    np.testing.assert_allclose(metrics["per_horizon_rmse"], [np.sqrt(2.5), 0])


def test_constant_zero_targets_have_undefined_r2_wape_and_clamped_mape():
    metrics = forecast_metrics(np.ones((2, 1)), np.zeros((2, 1)))
    assert metrics["r2"] is None and metrics["wape"] is None
    assert metrics["mape"] == 1000
    assert metrics["mape_denominator_floor_kw"] == .1


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[np.inf]])])
def test_nonfinite_metrics_are_rejected(bad):
    with pytest.raises(FloatingPointError):
        forecast_metrics(bad, np.ones((1, 1)))


def test_metrics_refuse_broadcasting():
    with pytest.raises(ValueError, match="exactly"):
        forecast_metrics(np.ones((2, 1)), np.ones((2, 3)))


def test_seasonal_controls_retrieve_only_correct_observed_lags():
    x = np.zeros((2, 168, 3))
    x[0, :, 1] = np.arange(168)
    x[1, :, 1] = np.arange(168) + 1000
    predictions = simple_predictions(x, target_index=1, horizon=24)
    np.testing.assert_array_equal(predictions["Persistence"][0], np.repeat(167, 24))
    np.testing.assert_array_equal(predictions["Seasonal24"][0], np.arange(144, 168))
    np.testing.assert_array_equal(predictions["Seasonal168"][0], np.arange(24))
    np.testing.assert_array_equal(predictions["Seasonal168"][1], np.arange(24) + 1000)
    predictions["Seasonal168"][:] = -1
    assert x[0, 0, 1] == 0


@pytest.mark.parametrize("lookback,horizon", [(167, 24), (168, 25), (168, 0)])
def test_unavailable_seasonal_history_or_future_is_rejected(lookback, horizon):
    with pytest.raises(ValueError):
        simple_predictions(np.ones((2, lookback, 1)), 0, horizon)


def test_ridge_fits_train_residuals_and_preserves_level_shift_behavior():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 3, 3))
    y = x[:, -1, 0, None] + (x[:, 0, 0] - x[:, -1, 0])[:, None] * np.array([[2., -1.]])
    before = x.copy()
    model, metadata = fit_ridge_residual(x[:35], y[:35], x[35:], y[35:], 0, 2)
    assert not metadata["persistence_selected"]
    assert metadata["selected_alpha"] == .1
    assert len(metadata["validation_scores"]) == 6
    assert metadata["validation_mse"] < metadata["validation_scores"][0]["validation_mse"]
    shifted = x[35:].copy()
    shifted[:, :, 0] += 20
    np.testing.assert_allclose(model.predict(shifted), model.predict(x[35:]) + 20, atol=1e-10)
    np.testing.assert_array_equal(x, before)


def test_ridge_validation_can_retain_persistence_even_when_training_fits():
    x = np.zeros((10, 3, 3))
    train_y = np.ones((10, 2))
    val_y = -np.ones((10, 2))
    model, metadata = fit_ridge_residual(x, train_y, x, val_y, 0)
    assert metadata["selected_alpha"] is None
    assert model.estimator is None
    assert metadata["persistence_selected"] and metadata["passed"]
    np.testing.assert_array_equal(model.predict(x), np.zeros((10, 2)))


def test_ridge_never_fits_validation_targets():
    x = np.zeros((8, 2, 3))
    y_train = np.ones((8, 1))
    first, _ = fit_ridge_residual(x, y_train, x, np.ones((8, 1)) * 1.1, 0)
    second, _ = fit_ridge_residual(x, y_train, x, np.ones((8, 1)) * 1.9, 0)
    assert first.estimator is not None and second.estimator is not None
    np.testing.assert_array_equal(first.estimator.coef_, second.estimator.coef_)
    np.testing.assert_array_equal(first.estimator.intercept_, second.estimator.intercept_)
    np.testing.assert_allclose(second.predict(x), 1)


def row(rmse, n=100):
    return {"sse": rmse ** 2 * n, "n_targets": n}


def folds(first, second):
    return {
        "winter": {"Persistence": row(10), "Seasonal24": row(11), "Seasonal168": row(12), **first},
        "summer": {"Persistence": row(10), "Seasonal24": row(11), "Seasonal168": row(12), **second},
    }


def test_learned_requires_five_percent_and_each_fold_gate():
    scores = folds({"SmallGain": row(9.6), "Unsafe": row(1), "Useful": row(9)},
                   {"SmallGain": row(9.6), "Unsafe": row(11.1), "Useful": row(9)})
    selected = select_candidate(scores)
    assert selected["selected_family"] == "Useful"
    assert selected["per_family_gates"]["Unsafe"]["aggregate_5pct_improvement"]
    assert not selected["per_family_gates"]["Unsafe"]["per_fold_within_10pct"]["summer"]
    assert not selected["per_family_gates"]["SmallGain"]["passed"]


def test_no_qualifying_learned_family_falls_back_to_best_simple():
    selection = select_candidate(folds({"Learned": row(9.6)}, {"Learned": row(10)}))
    assert selection["selected_family"] == "Persistence"
    assert not selection["learned_selected"]


def test_pooling_uses_target_count_not_mean_fold_rmse():
    scores = {
        "short": {"Persistence": row(1, 1), "Seasonal24": row(10, 1), "Seasonal168": row(20, 1)},
        "long": {"Persistence": row(10, 100), "Seasonal24": row(1, 100), "Seasonal168": row(20, 100)},
    }
    selected = select_candidate(scores)
    assert selected["selected_family"] == "Seasonal24"
    assert selected["aggregate"]["Seasonal24"]["rmse"] == np.sqrt(200 / 101)


def test_equal_scores_favor_declared_simplicity_order():
    scores = folds({"B": row(9), "A": row(9)}, {"B": row(9), "A": row(9)})
    assert select_candidate(scores, family_order=["B", "A"])["selected_family"] == "B"
    for rows in scores.values():
        for name in rows:
            rows[name] = row(10)
    assert select_candidate(scores)["selected_family"] == "Persistence"


def test_per_fold_gate_uses_that_folds_best_simple_control():
    scores = folds({"Learned": row(8)}, {"Learned": row(9)})
    scores["winter"]["Seasonal168"] = row(7)
    result = select_candidate(scores)
    assert not result["per_family_gates"]["Learned"]["per_fold_within_10pct"]["winter"]


def test_mismatched_target_counts_and_missing_family_are_rejected():
    scores = folds({}, {})
    scores["winter"]["Persistence"] = row(10, 99)
    with pytest.raises(ValueError, match="same target count"):
        select_candidate(scores)
    scores = folds({"Learned": row(9)}, {})
    with pytest.raises(ValueError, match="same candidate families"):
        select_candidate(scores)


def test_perfect_simple_control_does_not_yield_learned_improvement():
    scores = {"fold": {name: row(0) for name in ["Persistence", "Seasonal24", "Seasonal168", "Learned"]}}
    assert select_candidate(scores)["selected_family"] == "Persistence"
