"""Persistence anchoring, level shifts, and training of residual forecasters."""

import pytest
import torch
from torch import nn

from src.baselines import LSTMForecaster
from src.forecasting import PersistenceResidualForecaster
from src.sincos import SinCosForecaster
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster


def make_body(kind):
    """Build small representatives with three sensors and two clock columns."""
    if kind == "temporal":
        return TemporalVonMisesForecaster(
            sensor_size=3, time_periods=[24, 168], hidden_size=8, output_size=4,
        )
    cls = {"vm": VonMisesForecaster, "sincos": SinCosForecaster,
           "lstm": LSTMForecaster}[kind]
    return cls(input_size=5, hidden_size=8, output_size=4)


@pytest.mark.parametrize("kind", ["vm", "sincos", "lstm", "temporal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_persistence_device_dtype_and_no_added_parameters(kind, dtype, device):
    """Zero head starts at persistence without hiding unused trainable weights."""
    body = make_body(kind).to(device=device, dtype=dtype).eval()
    parameter_ids = {id(p) for p in body.parameters()}
    model = PersistenceResidualForecaster(body, 3, 1)
    x = torch.randn(2, 7, 5, dtype=dtype, device=device)
    actual = model(x)
    assert not model.training
    assert model.body is body
    assert model.input_size == 5 and model.output_size == 4
    assert model.sensor_size == 3 and model.target_index == 1
    assert model.zero_init_output is True
    assert actual.shape == (2, 4)
    assert actual.dtype == dtype and actual.device == x.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, x[:, -1, 1:2].expand(-1, 4), rtol=0, atol=0)
    assert {id(p) for p in model.parameters()} == parameter_ids


@pytest.mark.parametrize("kind", ["vm", "sincos", "lstm", "temporal"])
def test_trained_increment_is_equivariant_to_independent_sensor_level_shifts(kind):
    """A nonzero increment sees identical centered histories after level shifts."""
    torch.manual_seed(21)
    body = make_body(kind).double().eval()
    saved = {name: p.clone() for name, p in body.state_dict().items()}
    model = PersistenceResidualForecaster(body, 3, 1, zero_init_output=False)
    for name, tensor in body.state_dict().items():
        torch.testing.assert_close(tensor, saved[name], atol=0, rtol=0)
    x = torch.randn(2, 7, 5, dtype=torch.float64)
    offsets = torch.tensor([[[13., -6., 32., 0., 0.]],
                            [[-2., 9., -17., 0., 0.]]], dtype=torch.float64)
    expected = model(x) + offsets[:, 0, 1:2]
    torch.testing.assert_close(model(x + offsets), expected, atol=1e-12, rtol=1e-12)


def test_body_receives_centered_sensors_and_exact_original_time_angles():
    """Centering neither mutates the input nor changes calendar phases."""
    body = make_body("temporal")
    captured = []
    handle = body.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    model = PersistenceResidualForecaster(body, 3, 1)
    x = torch.randn(2, 7, 5)
    original = x.clone()
    model(x)
    handle.remove()
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    torch.testing.assert_close(captured[0][..., 3:], x[..., 3:], atol=0, rtol=0)
    torch.testing.assert_close(captured[0][..., :3], x[..., :3] - x[:, -1:, :3])
    assert torch.count_nonzero(captured[0][:, -1, :3]) == 0


@pytest.mark.parametrize("kind", ["vm", "sincos", "lstm", "temporal"])
def test_gradients_reach_body_after_first_head_learning_step(kind, device):
    """Zero initialization delays upstream updates for exactly the first step."""
    torch.manual_seed(10)
    model = PersistenceResidualForecaster(make_body(kind), 3, 1).to(device)
    x = torch.randn(3, 7, 5, device=device, requires_grad=True)
    target = x[:, -1, 1:2].detach() + torch.randn(3, 4, device=device) + 2
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    (model(x) - target).square().mean().backward()
    assert model.body.output_head.weight.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in model.body.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        if not name.startswith("output_head."):
            assert torch.count_nonzero(parameter.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model(x) - target).square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("sensor_size,target_index", [
    (0, 0), (-1, 0), (6, 0), (2.5, 0), (True, 0),
    (3, -1), (3, 3), (3, 1.5), (3, True),
])
def test_invalid_sensor_feature_contract(sensor_size, target_index):
    with pytest.raises(ValueError):
        PersistenceResidualForecaster(make_body("vm"), sensor_size, target_index)


def test_temporal_sensor_split_must_match_body():
    with pytest.raises(ValueError, match="temporal"):
        PersistenceResidualForecaster(make_body("temporal"), 2, 1)


@pytest.mark.parametrize("shape", [(2, 5), (2, 7, 4), (2, 7, 6), (0, 7, 5), (2, 0, 5)])
def test_rejects_malformed_input(shape):
    model = PersistenceResidualForecaster(make_body("vm"), 3, 1)
    with pytest.raises(ValueError):
        model(torch.empty(shape))


def test_rejects_nonfloating_input():
    model = PersistenceResidualForecaster(make_body("vm"), 3, 1)
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(2, 7, 5, dtype=torch.long))


def test_rejects_invalid_body_and_head():
    with pytest.raises(TypeError, match="Module"):
        PersistenceResidualForecaster(None, 3, 1)
    with pytest.raises(ValueError, match="input_size"):
        PersistenceResidualForecaster(nn.Identity(), 3, 1)
    body = make_body("vm")
    body.output_head = nn.Identity()
    with pytest.raises(TypeError, match="output_head"):
        PersistenceResidualForecaster(body, 3, 1)
    body.output_head = nn.Linear(8, 2)
    with pytest.raises(ValueError, match="width"):
        PersistenceResidualForecaster(body, 3, 1)


def test_rejects_silently_broadcasting_body_output():
    body = make_body("vm")
    body.forward = lambda x: torch.zeros(x.shape[0], 1)
    model = PersistenceResidualForecaster(body, 3, 1)
    with pytest.raises(ValueError, match="body must return"):
        model(torch.randn(2, 7, 5))
