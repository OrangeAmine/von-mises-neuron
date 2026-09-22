"""Small CPU checks of checkpoint selection and bounded forecast evaluation."""

import pytest
import torch
from torch import nn

from src.forecast_training import batched_predict, train_forecaster


def zero_linear():
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    return model


def test_initial_checkpoint_survives_training_that_worsens_validation():
    model = zero_linear()
    x = torch.ones(4, 1)
    train_forecaster(model, x, x, x, -x, max_epochs=8, patience=2,
                     lr=0.1, device="cpu", source_config={"seed": 3})
    history = model.training_history
    assert history["best_epoch"] == 0
    assert history["epochs_completed"] == 2
    assert history["validation_epochs"] == [0, 1, 2]
    assert history["best_val_loss"] == 1
    assert all(loss > 1 for loss in history["validation_loss"][1:])
    assert history["source_config"] == {"seed": 3}
    assert model.weight.item() == 0
    assert not model.training


def test_improving_fit_selects_later_checkpoint_and_preserves_inputs():
    model = zero_linear()
    x = torch.ones(7, 1)
    targets = torch.ones_like(x)
    before = (x.clone(), targets.clone())
    train_forecaster(model, x, targets, x, targets, max_epochs=3,
                     lr=0.05, batch_size=3, validation_batch_size=3, device="cpu")
    history = model.training_history
    assert history["best_epoch"] == 3
    assert history["epochs_completed"] == 3
    assert len(history["training_loss"]) == 3
    assert history["best_val_loss"] == history["validation_loss"][3]
    assert history["best_val_loss"] < history["validation_loss"][0]
    assert history["status"] == "completed"
    torch.testing.assert_close(model(x).sub(targets).square().mean(),
                               torch.tensor(history["best_val_loss"]))
    assert torch.equal(x, before[0]) and torch.equal(targets, before[1])


def test_zero_epoch_budget_still_selects_initialization():
    model = zero_linear()
    x = torch.ones(2, 1)
    train_forecaster(model, x, x, x, x, max_epochs=0, device="cpu")
    assert model.training_history["validation_loss"] == [1.0]
    assert model.training_history["epochs_completed"] == 0
    assert model.training_history["best_epoch"] == 0


@pytest.mark.parametrize("location", ["training", "validation"])
def test_nonfinite_data_fails_explicitly_with_partial_history(location):
    model = zero_linear()
    x = torch.ones(3, 1)
    bad = x.clone()
    bad[0] = float("nan")
    train_x, val_x = (bad, x) if location == "training" else (x, bad)
    with pytest.raises(FloatingPointError, match=f"Nonfinite {location} inputs"):
        train_forecaster(model, train_x, x, val_x, x, max_epochs=1, device="cpu")
    assert model.training_history["status"] == "failed"
    assert "FloatingPointError" in model.training_history["failure"]


def test_nonfinite_gradient_is_not_hidden_by_restoring_initial_checkpoint():
    model = zero_linear()
    model.weight.register_hook(lambda grad: grad * float("nan"))
    x = torch.ones(2, 1)
    with pytest.raises(FloatingPointError, match="gradient norm"):
        train_forecaster(model, x, x, x, x, max_epochs=1, device="cpu")
    assert model.training_history["status"] == "failed"
    assert model.training_history["failed_epoch"] == 1


def test_batched_prediction_matches_full_eval_and_restores_mode():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.9), nn.Linear(4, 1))
    x = torch.randn(11, 2)
    model.eval()
    expected = model(x).detach()
    model.train()
    actual = batched_predict(model, x, batch_size=3)
    torch.testing.assert_close(actual, expected)
    assert actual.device.type == "cpu" and not actual.requires_grad
    assert model.training


def test_validation_mse_weights_uneven_batches_by_elements():
    model = zero_linear()
    x = torch.ones(5, 1)
    y = torch.tensor([[1.], [2.], [3.], [4.], [10.]])
    train_forecaster(model, x, y, x, y, max_epochs=0,
                     validation_batch_size=2, device="cpu")
    assert model.training_history["best_val_loss"] == y.square().double().mean().item()


def test_broadcastable_wrong_forecast_shape_is_rejected():
    model = zero_linear()
    x = torch.ones(3, 1)
    y = torch.ones(3, 2)
    with pytest.raises(ValueError, match="exactly match"):
        train_forecaster(model, x, y, x, y, max_epochs=0, device="cpu")


def test_nonfinite_prediction_restores_training_mode_and_fails():
    model = zero_linear()
    model.weight.data.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="predictions"):
        batched_predict(model, torch.ones(2, 1), batch_size=1)
    assert model.training
