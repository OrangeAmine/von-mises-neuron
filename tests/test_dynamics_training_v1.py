"""Numerical selection and finite-budget checks for the new forecasting runner."""

from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from src.dynamics_models_v1 import DynamicsForecasterV1
from src.dynamics_training_v1 import (check_deadline, error_metrics, fit_neural,
                                      fit_ridge, predict_neural, ridge_design)


def fixture_batch(seed: int = 19) -> SimpleNamespace:
    """Build a disposable fixture independent of all prospective study seeds."""
    rng = np.random.default_rng(seed)
    sensor = rng.normal(size=(30, 4, 3)).astype(np.float32)
    angles = rng.uniform(0, 2*np.pi, size=(30, 3, 1)).astype(np.float32)
    y = np.repeat(sensor[:, -1, 0, None], 3, axis=1)
    y += 0.2 * np.sin(angles[..., 0])
    return SimpleNamespace(sensor=sensor, future_angles=angles, y=y)


def test_ridge_matches_independent_augmented_least_squares():
    train, val = fixture_batch(), fixture_batch(23)
    fitted = fit_ridge(train, val, [0.01])
    x = ridge_design(train.sensor)
    y = train.y.astype(float) - train.sensor[:, -1, 0, None]
    design = np.c_[x, np.ones(len(x))]
    regularizer = np.diag(np.r_[np.full(x.shape[1], np.sqrt(len(x)*.01)), 0])
    expected = np.linalg.lstsq(np.r_[design, regularizer],
                              np.r_[y, np.zeros((design.shape[1], 3))], rcond=None)[0]
    np.testing.assert_allclose(fitted["weight"], expected[:-1], atol=1e-10)
    np.testing.assert_allclose(fitted["bias"], expected[-1], atol=1e-10)


def test_selection_uses_validation_and_keeps_epoch_zero():
    train, val = fixture_batch(), fixture_batch(23)
    settings = dict(epochs=4, validation_every=2, batch_size=11, weight_decay=1e-4, gradient_clip=1.)
    histories, predictions = [], []
    for _ in range(2):
        torch.manual_seed(59)
        model = DynamicsForecasterV1("learned", lookback=4, horizon=3)
        histories.append(fit_neural(model, train, val, settings, .01, 73, torch.device("cpu"), time.perf_counter()+30))
        predictions.append(predict_neural(model, val, torch.device("cpu"))[0])
    assert [h["epoch"] for h in histories[0]["history"]] == [0, 2, 4]
    assert histories[0]["validation_mse"] == min(h["validation_mse"] for h in histories[0]["history"])
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert abs(np.mean((predictions[0].astype(float)-val.y)**2)-histories[0]["validation_mse"]) < 1e-12


def test_deadline_is_enforced():
    with pytest.raises(TimeoutError):
        check_deadline(time.perf_counter()-1)
    model = DynamicsForecasterV1("none", lookback=4, horizon=3)
    with pytest.raises(TimeoutError):
        fit_neural(model, fixture_batch(), fixture_batch(), dict(epochs=1,validation_every=1,batch_size=10,
            weight_decay=0.,gradient_clip=1.), .01, 1, torch.device("cpu"), time.perf_counter()-1)


def test_invalid_and_empty_errors():
    assert error_metrics(np.array([]),np.array([])) == {"n":0,"rmse":None,"mae":None}
    with pytest.raises(ValueError):
        error_metrics(np.zeros(3),np.zeros((3,1)))
    with pytest.raises(FloatingPointError):
        error_metrics(np.array([np.nan]),np.zeros(1))
