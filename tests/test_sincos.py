"""Functional and experimental controls for the Cartesian VM comparator."""

import math

import pytest
import torch

from src.sincos import SinCosForecaster, SinCosLinear
from src.von_mises_neuron import VonMisesForecaster, VonMisesLinear


@pytest.mark.parametrize("log_output", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_layer_function_and_input_gradient_equivalence(log_output, bias):
    """The cosine identity reproduces both values and derivatives in x."""
    torch.manual_seed(7)
    vm = VonMisesLinear(3, 5, bias=bias, log_output=log_output).double()
    with torch.no_grad():
        vm.log_kappa.normal_(-1, 0.5)
        if bias:
            vm.bias.normal_()
    baseline = SinCosLinear.from_von_mises(vm)
    x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    expected, actual = vm(x), baseline(x)
    assert actual.shape == (2, 4, 5)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    assert sum(p.numel() for p in vm.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )


@pytest.mark.parametrize("use_lstm", [True, False])
@pytest.mark.parametrize("use_layernorm", [True, False])
@pytest.mark.parametrize("log_output", [True, False])
def test_forecaster_conversion_matches_all_ablations(use_lstm, use_layernorm, log_output):
    """Non-encoder architecture and parameters are held fixed by conversion."""
    torch.manual_seed(13)
    vm = VonMisesForecaster(
        input_size=3, hidden_size=8, num_vm_layers=2, dropout=0.25,
        output_size=4, init_kappa=0.2, use_lstm=use_lstm,
        use_layernorm=use_layernorm, log_output=log_output,
    ).double().eval()
    baseline = SinCosForecaster.from_von_mises(vm)
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    assert not baseline.training
    assert baseline(x).shape == (2, 4)
    torch.testing.assert_close(baseline(x), vm(x), atol=1e-11, rtol=1e-11)
    assert sum(p.numel() for p in vm.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )
    for source, copied in zip(vm.vm_encoder, baseline.sincos_encoder):
        if not isinstance(source, VonMisesLinear):
            assert type(source) is type(copied)
            for name, tensor in source.state_dict().items():
                torch.testing.assert_close(copied.state_dict()[name], tensor, rtol=0, atol=0)
    for module_name in ["lstm", "output_head"]:
        source, copied = getattr(vm, module_name), getattr(baseline, module_name)
        if source is not None:
            for name, tensor in source.state_dict().items():
                torch.testing.assert_close(copied.state_dict()[name], tensor, rtol=0, atol=0)
                assert copied.state_dict()[name].data_ptr() != tensor.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_device_dtype_and_finite_gradient_flow(device, dtype):
    """The full model trains with finite input and parameter derivatives."""
    torch.manual_seed(11)
    vm = VonMisesForecaster(3, 8, output_size=4).to(device=device, dtype=dtype)
    baseline = SinCosForecaster.from_von_mises(vm)
    x = torch.randn(2, 6, 3, device=device, dtype=dtype, requires_grad=True)
    y = baseline(x)
    assert y.dtype == dtype
    assert y.device == x.device
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    for parameter in [x, *baseline.parameters()]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_per_feature_periodicity_at_layer_and_full_model_boundary():
    """First-layer periodicity survives deterministic downstream processing."""
    torch.manual_seed(9)
    baseline = SinCosForecaster(3, 8, output_size=4, dropout=0.3).double().eval()
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    layer = baseline.get_sincos_layers()[0]
    for feature in range(3):
        shifted = x.clone()
        shifted[..., feature] += 2 * math.pi
        torch.testing.assert_close(layer(x), layer(shifted), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(baseline(x), baseline(shifted), atol=1e-12, rtol=1e-12)


def test_conversion_preserves_rng_and_does_not_share_parameters():
    """Pairing cannot silently change the next draw or mutate its reference."""
    vm = VonMisesForecaster(3, 8, output_size=4)
    saved_state = {name: value.clone() for name, value in vm.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    baseline = SinCosForecaster.from_von_mises(vm)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.add_(1)
    for name, value in vm.state_dict().items():
        torch.testing.assert_close(value, saved_state[name], atol=0, rtol=0)


def test_seeded_constructor_and_dropout_pairing():
    """Same constructor seed and dropout seed match the initial training function."""
    kwargs = dict(input_size=4, hidden_size=64, output_size=12, dropout=0.2)
    torch.manual_seed(42)
    vm = VonMisesForecaster(**kwargs)
    torch.manual_seed(42)
    baseline = SinCosForecaster(**kwargs)
    assert sum(p.numel() for p in vm.parameters()) == 43148
    assert sum(p.numel() for p in baseline.parameters()) == 43148
    x = torch.randn(2, 6, 4)
    torch.manual_seed(19)
    expected = vm(x)
    torch.manual_seed(19)
    actual = baseline(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
