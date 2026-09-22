"""Checks that the independent artifact audit detects meaningful corruption."""

import numpy as np
import pytest

from scripts.audit_practical_results import _compare, audit_history, independent_metrics


def history():
    return {"status": "completed", "validation_loss": [2., 1., 1.2],
            "validation_epochs": [0, 1, 2], "epochs_completed": 2,
            "training_loss": [1.1, .9], "learning_rate": [.001, .001],
            "best_epoch": 1, "best_val_loss": 1., "min_delta": 1e-6}


def test_checkpoint_audit_accepts_best_and_rejects_last_epoch_substitution():
    record = history()
    assert audit_history(record)["best_epoch"] == 1
    record.update(best_epoch=2, best_val_loss=1.2)
    with pytest.raises(AssertionError):
        audit_history(record)


def test_checkpoint_audit_requires_initialization_to_compete():
    record = history()
    record.update(validation_loss=[.5, 1., 1.2], best_epoch=0, best_val_loss=.5)
    assert audit_history(record)["best_epoch"] == 0
    record.update(best_epoch=1, best_val_loss=1.)
    with pytest.raises(AssertionError):
        audit_history(record)


def test_independent_metric_audit_detects_changed_horizon_error():
    metrics = independent_metrics(np.array([[1., 2.], [3., 4.]]), np.ones((2, 2)))
    assert metrics["sse"] == 14
    assert metrics["r2"] is None
    _compare(metrics, metrics)
    corrupt = {**metrics, "per_horizon_rmse": [1., 1.]}
    with pytest.raises(AssertionError, match="per_horizon_rmse"):
        _compare(metrics, corrupt)


def test_metric_audit_refuses_nonfinite_prediction():
    with pytest.raises(AssertionError):
        independent_metrics(np.array([[np.nan]]), np.ones((1, 1)))
