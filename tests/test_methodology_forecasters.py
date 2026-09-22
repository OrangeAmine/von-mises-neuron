"""Engineering conversion checks for complete deterministic forecasters."""

import copy
import math

import pytest
import torch

from src.methodology_forecasters import DirectCalendarForecasterV1, PolarForecasterV2
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster


@pytest.mark.parametrize("use_lstm", [False, True])
@pytest.mark.parametrize("use_layernorm", [False, True])
@pytest.mark.parametrize("log_output", [False, True])
def test_base_complete_function_all_state_ownership_and_rng(use_lstm, use_layernorm, log_output):
    torch.manual_seed(3)
    reference = VonMisesForecaster(3, 8, output_size=4, init_kappa=0.2, dropout=0.2, use_lstm=use_lstm, use_layernorm=use_layernorm, log_output=log_output).double().eval()
    source_states = copy.deepcopy(reference.state_dict())
    rng = torch.get_rng_state().clone()
    adapter = PolarForecasterV2.from_legacy(reference)
    assert torch.equal(rng, torch.get_rng_state())
    assert not adapter.training
    x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    actual, expected = adapter(x), reference(x)
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    first = torch.autograd.grad(actual.sum(), x)[0]
    second = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in adapter.parameters())
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(1)
    for name, original in source_states.items():
        torch.testing.assert_close(reference.state_dict()[name], original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_lstm", [False, True])
@pytest.mark.parametrize("use_conv", [False, True])
@pytest.mark.parametrize("conditioning", [False, True])
@pytest.mark.parametrize("learnable_kappa", [False, True])
def test_temporal_complete_conversion_and_gradient_flow(dtype, use_lstm, use_conv, conditioning, learnable_kappa):
    torch.manual_seed(1130)
    reference = TemporalVonMisesForecaster(3, time_periods=[24, 168], hidden_size=8, output_size=4, dropout=0.2, use_lstm=use_lstm, use_temporal_conv=use_conv, use_time_conditioning=conditioning, learnable_kappa=learnable_kappa).to(dtype=dtype).eval()
    before = torch.get_rng_state().clone()
    converted = DirectCalendarForecasterV1.from_legacy(reference)
    assert torch.equal(before, torch.get_rng_state())
    assert not converted.training
    x = torch.randn(2, 5, 5, dtype=dtype, requires_grad=True)
    actual, expected = converted(x), reference(x)
    tolerance = dict(atol=1e-10, rtol=1e-10) if dtype == torch.float64 else dict(atol=1e-5, rtol=1e-4)
    assert actual.shape == (2, 4) and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, **tolerance)
    actual_grad = torch.autograd.grad(actual.square().mean(), [x, *converted.parameters()])
    original_input_grad = torch.autograd.grad(expected.square().mean(), x)[0]
    torch.testing.assert_close(actual_grad[0], original_input_grad, **tolerance)
    assert all(torch.isfinite(gradient).all() for gradient in actual_grad)
    original_ptrs = {v.data_ptr() for v in reference.state_dict().values()}
    assert not any(value.data_ptr() in original_ptrs for value in converted.state_dict().values())
    for name in ("lstm", "output_head"):
        source = getattr(reference, name)
        if source is not None:
            for key, value in source.state_dict().items():
                torch.testing.assert_close(getattr(converted, name).state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("conv", [False, True])
def test_temporal_periodicity_at_correct_input_boundary(conv):
    torch.manual_seed(774)
    reference = TemporalVonMisesForecaster(3, time_periods=[24, 168], hidden_size=8, output_size=3, use_temporal_conv=conv).double().eval()
    converted = DirectCalendarForecasterV1.from_legacy(reference)
    x = torch.randn(2, 4, 5, dtype=torch.float64)
    for index in (3, 4):
        shifted = x.clone()
        shifted[..., index] += 2 * math.pi
        torch.testing.assert_close(converted(x), converted(shifted), atol=1e-10, rtol=1e-10)
    shifted = x.clone()
    shifted[..., 0] += 2 * math.pi
    if not conv:
        torch.testing.assert_close(converted(x), converted(shifted), atol=1e-10, rtol=1e-10)
    else:
        assert not torch.allclose(converted(x), converted(shifted), atol=1e-5, rtol=1e-5)


def test_conversion_retains_heterogeneous_module_modes_and_state_loading():
    reference = TemporalVonMisesForecaster(3, hidden_size=8, dropout=0.3).double()
    reference.encoder.dropout.eval()
    adapter = DirectCalendarForecasterV1.from_legacy(reference)
    assert adapter.training and not adapter.encoder.dropout.training
    copy_adapter = DirectCalendarForecasterV1.from_legacy(reference)
    copy_adapter.load_state_dict(adapter.state_dict())
    adapter.eval()
    copy_adapter.eval()
    x = torch.randn(2, 4, 6, dtype=torch.float64)
    torch.testing.assert_close(adapter(x), copy_adapter(x), atol=0, rtol=0)
