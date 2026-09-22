"""Numerical contracts for new layers; legacy implementations remain intact."""

import copy
import io
import math

import pytest
import torch
from torch import nn
from torch.func import functional_call

from src.circular_layers_v2 import (
    CartesianCircularLinearV1,
    DirectCalendarPhaseV1,
    PeakCircularKernelBankV1,
    PolarCircularLinearV2,
    amplitude_regularizer,
    inverse_softplus,
)
from src.temporal import TemporalVonMisesEncoder
from src.von_mises_neuron import VonMisesLinear


def tolerances(dtype):
    """Return unchanged canonical dtype tolerances."""
    return {"atol": 1e-10, "rtol": 1e-10} if dtype == torch.float64 else {"atol": 1e-5, "rtol": 1e-4}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [0.001, 0.01, 0.05, 1, 1000, 1e-20])
@pytest.mark.parametrize("policy", ["constant", "unit_variance"])
def test_initialization_recovers_requested_concentration(dtype, value, policy):
    layer = PolarCircularLinearV2(7, 3, init_kappa=value, init_policy=policy, dtype=dtype)
    expected = value * (math.sqrt(2 / 7) if policy == "unit_variance" else 1)
    torch.testing.assert_close(layer.kappa, torch.full((3, 7), expected, dtype=dtype), rtol=2e-6 if dtype == torch.float32 else 1e-13, atol=0)
    assert layer.requested_init_kappa == value
    assert layer.effective_init_kappa > 0
    assert layer.get_config()["class"] == "PolarCircularLinearV2"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), 1e-100, 1e100])
def test_invalid_or_unrepresentable_requests_fail(value):
    with pytest.raises(ValueError):
        PolarCircularLinearV2(3, 2, init_kappa=value, dtype=torch.float32)


@pytest.mark.parametrize("kwargs", [{"in_features": 0}, {"out_features": -1}, {"in_features": True}, {"out_features": 2.5}, {"response": "density"}, {"init_policy": "guess"}, {"implementation": "guess"}, {"dtype": torch.int64}])
def test_invalid_configuration_fails(kwargs):
    arguments = {"in_features": 2, "out_features": 3, **kwargs}
    with pytest.raises(ValueError):
        PolarCircularLinearV2(**arguments)


def test_variance_formula_over_independent_phase_initialization():
    torch.manual_seed(114)
    layer = PolarCircularLinearV2(8, 30000, init_kappa=0.4, init_policy="unit_variance", dtype=torch.float64)
    expected = layer.kappa.square().sum(-1) / 2
    torch.testing.assert_close(expected, torch.full_like(expected, 0.4 ** 2), atol=1e-12, rtol=1e-12)
    empirical = layer(torch.zeros(8, dtype=torch.float64)).var(unbiased=True)
    assert abs(empirical.item() / 0.4 ** 2 - 1) < 0.025


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("response", ["sum", "exp"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("kappa", [0.001, 0.4])
@pytest.mark.parametrize("leading", [(), (4,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("shift_mode", ["none", "full", "shared"])
def test_reference_factored_output_and_all_gradient_parity(dtype, response, bias, kappa, leading, shift_mode):
    torch.manual_seed(113)
    reference = PolarCircularLinearV2(3, 4, bias=bias, response=response, init_kappa=kappa, dtype=dtype, implementation="reference")
    factored = copy.deepcopy(reference)
    factored.implementation = "factored"
    x = torch.randn(*leading, 3, dtype=dtype, requires_grad=True)
    shifted = None
    if shift_mode != "none":
        shape = (*leading, 4, 1) if shift_mode == "full" else (4, 1)
        shifted = torch.randn(shape, dtype=dtype, requires_grad=True) * 0.3
    expected, actual = reference(x, shifted), factored(x, shifted)
    assert actual.shape == (*leading, 4)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    weight = torch.randn_like(expected)
    inputs = [x] + ([] if shifted is None else [shifted])
    gradients_ref = torch.autograd.grad((expected * weight).sum(), [*inputs, *reference.parameters()])
    gradients_factored = torch.autograd.grad((actual * weight).sum(), [*inputs, *factored.parameters()])
    for observed, intended in zip(gradients_factored, gradients_ref):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, intended, **tolerances(dtype))


@pytest.mark.parametrize("implementation", ["reference", "factored"])
@pytest.mark.parametrize("response", ["sum", "exp"])
def test_gradcheck_inputs_phase_raw_width_dynamic_offset_and_bias(implementation, response):
    layer = PolarCircularLinearV2(2, 2, dtype=torch.float64, init_kappa=0.3, implementation=implementation, response=response)
    parameters = dict(layer.named_parameters())
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    offset = torch.randn(2, 2, 1, dtype=torch.float64, requires_grad=True)

    def evaluate(x_arg, offset_arg, *values):
        return functional_call(layer, dict(zip(parameters, values)), (x_arg, offset_arg))

    assert torch.autograd.gradcheck(evaluate, (x, offset, *parameters.values()), atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(2, 4, 3), (2, 4), (3, 2, 4, 1), (5, 4, 1)])
def test_reject_general_edges_and_expanding_dynamic_offsets(shape):
    layer = PolarCircularLinearV2(3, 4)
    with pytest.raises(ValueError, match="phase_shift"):
        layer(torch.zeros(2, 3), torch.zeros(shape))


def test_reject_input_shape_and_shift_dtype():
    layer = PolarCircularLinearV2(3, 4)
    with pytest.raises(ValueError, match="input"):
        layer(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="dtype"):
        layer(torch.zeros(2, 3), torch.zeros(2, 4, 1, dtype=torch.float64))


@pytest.mark.parametrize("log_output", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legacy_conversion_function_gradients_rng_state_and_independence(log_output, bias, dtype):
    torch.manual_seed(92)
    legacy = VonMisesLinear(3, 4, bias=bias, log_output=log_output, init_kappa=0.001).to(dtype=dtype).eval()
    snapshot = copy.deepcopy(legacy.state_dict())
    before_rng = torch.get_rng_state().clone()
    converted = PolarCircularLinearV2.from_legacy(legacy)
    cartesian = CartesianCircularLinearV1.from_polar(converted)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not converted.training and not cartesian.training
    assert converted.kappa.min() > 0.04  # Preserve the old initializer's actual function.
    assert converted.init_policy == "legacy_copy"
    x = torch.randn(2, 5, 3, dtype=dtype, requires_grad=True)
    shift = torch.randn(2, 5, 4, 1, dtype=dtype, requires_grad=True)
    expected = legacy(x, shift)
    for actual in (converted(x, shift), cartesian(x, shift)):
        torch.testing.assert_close(actual, expected, **tolerances(dtype))
        expected_grad = torch.autograd.grad(expected.sum(), [x, shift], retain_graph=True)
        actual_grad = torch.autograd.grad(actual.sum(), [x, shift], retain_graph=True)
        for first, second in zip(expected_grad, actual_grad):
            torch.testing.assert_close(first, second, **tolerances(dtype))
    expected_parameter_grad = torch.autograd.grad(expected.sum(), list(legacy.parameters()))
    converted_parameter_grad = torch.autograd.grad(converted(x, shift).sum(), list(converted.parameters()))
    for first, second in zip(expected_parameter_grad, converted_parameter_grad):
        torch.testing.assert_close(first, second, **tolerances(dtype))
    assert sum(p.numel() for p in converted.parameters()) == sum(p.numel() for p in cartesian.parameters()) == 24 + (4 if bias else 0)
    with torch.no_grad():
        for parameter in converted.parameters():
            parameter.add_(1)
    for name, original in snapshot.items():
        torch.testing.assert_close(legacy.state_dict()[name], original, atol=0, rtol=0)


def test_conversion_preserves_gradient_flags_and_strict_loading_rejects_legacy_keys():
    legacy = VonMisesLinear(2, 3)
    legacy.requires_grad_(False)
    converted = PolarCircularLinearV2.from_legacy(legacy)
    assert not any(p.requires_grad for p in converted.parameters())
    with pytest.raises(RuntimeError):
        converted.load_state_dict(legacy.state_dict())


@pytest.mark.parametrize("response", ["sum", "exp"])
def test_per_input_periodicity_preferred_centers_and_amplitude_penalty(response):
    torch.manual_seed(810)
    polar = PolarCircularLinearV2(3, 4, response=response, init_kappa=0.2, dtype=torch.float64)
    cartesian = CartesianCircularLinearV1.from_polar(polar)
    x = torch.randn(2, 3, dtype=torch.float64)
    for index in range(3):
        shifted = x.clone()
        shifted[..., index] += 2 * math.pi
        torch.testing.assert_close(polar(x), polar(shifted), **tolerances(torch.float64))
        torch.testing.assert_close(cartesian(x), cartesian(shifted), **tolerances(torch.float64))
    assert ((polar.preferred_centers() >= 0) & (polar.preferred_centers() < 2 * math.pi)).all()
    torch.testing.assert_close(amplitude_regularizer(polar, 0.001), amplitude_regularizer(cartesian, 0.001), **tolerances(torch.float64))
    torch.testing.assert_close(amplitude_regularizer([polar, polar], 0.001), 2 * amplitude_regularizer(polar, 0.001))
    penalty_grad = torch.autograd.grad(amplitude_regularizer(polar, 1), [polar.raw_kappa, polar.phase, polar.bias], allow_unused=True)
    assert penalty_grad[0] is not None and penalty_grad[1] is None and penalty_grad[2] is None


def test_exponential_overflow_and_kernel_underflow_remain_explicit():
    layer = PolarCircularLinearV2(2, 1, response="exp", init_kappa=1000)
    with torch.no_grad():
        layer.phase.zero_()
    assert torch.isinf(layer(torch.zeros(1, 2))).all()
    bank = PeakCircularKernelBankV1(1, 1, centers=torch.zeros(1, 1), concentrations=torch.full((1, 1), 1000.), learn_centers=True, learn_concentrations=True)
    assert bank(torch.tensor([[math.pi]])).item() == 0


@pytest.mark.parametrize("sharing", ["per_edge", "per_kernel", "global"])
@pytest.mark.parametrize("learn_centers", [False, True])
@pytest.mark.parametrize("learn_widths", [False, True])
def test_kernel_freeze_counts_buffers_state_roundtrip_and_signed_readout(sharing, learn_centers, learn_widths):
    torch.manual_seed(712)
    centers = torch.randn(4, 2, dtype=torch.float64)
    shape = {"per_edge": (4, 2), "per_kernel": (4,), "global": ()}[sharing]
    widths = torch.full(shape, 4., dtype=torch.float64)
    bank = PeakCircularKernelBankV1(2, 4, centers=centers, concentrations=widths, learn_centers=learn_centers, learn_concentrations=learn_widths, width_sharing=sharing)
    expected_count = (8 if learn_centers else 0) + (widths.numel() if learn_widths else 0)
    assert sum(p.numel() for p in bank.parameters()) == expected_count
    assert ("centers" in dict(bank.named_buffers())) != learn_centers
    assert ("fixed_kappa" in dict(bank.named_buffers())) != learn_widths
    assert bank.centers.data_ptr() != centers.data_ptr()
    x = torch.randn(3, 2, dtype=torch.float64)
    saved = io.BytesIO()
    torch.save(bank.state_dict(), saved)
    saved.seek(0)
    clone = copy.deepcopy(bank)
    clone.load_state_dict(torch.load(saved, weights_only=True))
    torch.testing.assert_close(clone(x), bank(x), atol=0, rtol=0)
    assert all(buffer.dtype == torch.float32 for buffer in clone.float().buffers())
    readout = nn.Linear(4, 1, dtype=torch.float64)
    with torch.no_grad():
        readout.weight.fill_(-1)
        readout.bias.fill_(-1)
    assert (readout(bank(x)) < 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sharing", ["per_edge", "per_kernel", "global"])
def test_peak_chordal_output_input_center_and_width_gradient_parity(dtype, sharing):
    torch.manual_seed(100)
    shape = {"per_edge": (4, 2), "per_kernel": (4,), "global": ()}[sharing]
    bank = PeakCircularKernelBankV1(2, 4, centers=torch.randn(4, 2, dtype=dtype), concentrations=torch.full(shape, 4., dtype=dtype), learn_centers=True, learn_concentrations=True, width_sharing=sharing)
    x = torch.randn(3, 2, dtype=dtype, requires_grad=True)
    stable, chordal = bank(x), bank.chordal_features(x)
    torch.testing.assert_close(stable, chordal, **tolerances(dtype))
    first = torch.autograd.grad(stable.sum(), [x, bank.centers, bank.raw_kappa])
    second = torch.autograd.grad(chordal.sum(), [x, bank.centers, bank.raw_kappa])
    for observed, expected in zip(first, second):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, expected, **tolerances(dtype))
    maxima = bank(bank.centers).diagonal()
    torch.testing.assert_close(maxima, torch.ones_like(maxima), atol=0, rtol=0)
    assert (bank.log_features(x) <= 0).all()
    for index in range(2):
        shifted = x.detach().clone()
        shifted[:, index] += 2 * math.pi
        torch.testing.assert_close(bank(x), bank(shifted), **tolerances(dtype))


def test_kernel_gradcheck_and_49_parameter_primary_capacity():
    bank = PeakCircularKernelBankV1(1, 16, centers=torch.linspace(-math.pi, math.pi, 17, dtype=torch.float64)[:-1, None], concentrations=torch.full((16, 1), 4., dtype=torch.float64), learn_centers=True, learn_concentrations=True)
    model = nn.Sequential(bank, nn.Linear(16, 1, dtype=torch.float64))
    assert sum(p.numel() for p in model.parameters()) == 49
    x = torch.randn(3, 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value, centers, raw: functional_call(bank, {"centers": centers, "raw_kappa": raw}, (value,)), (x, bank.centers, bank.raw_kappa))


@pytest.mark.parametrize("invalid", ["centers_shape", "nonfinite_centers", "width_shape", "zero_width", "dtype", "sharing"])
def test_kernel_rejects_invalid_initial_tensors(invalid):
    kwargs = dict(centers=torch.zeros(4, 2), concentrations=torch.ones(4, 2), learn_centers=True, learn_concentrations=True)
    if invalid == "centers_shape":
        kwargs["centers"] = torch.zeros(4)
    elif invalid == "nonfinite_centers":
        kwargs["centers"][0, 0] = float("nan")
    elif invalid == "width_shape":
        kwargs["concentrations"] = torch.ones(4)
    elif invalid == "zero_width":
        kwargs["concentrations"][0, 0] = 0
    elif invalid == "dtype":
        kwargs["concentrations"] = kwargs["concentrations"].double()
    else:
        kwargs["width_sharing"] = "unknown"
    with pytest.raises(ValueError):
        PeakCircularKernelBankV1(2, 4, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("learnable_kappa", [False, True])
def test_direct_calendar_projection_function_input_gradient_counts_and_rng(dtype, learnable_kappa):
    torch.manual_seed(718)
    encoder = TemporalVonMisesEncoder(3, 2, hidden_size=64, num_vm_layers=2, learnable_kappa=learnable_kappa).to(dtype=dtype).eval()
    rng = torch.get_rng_state().clone()
    direct = DirectCalendarPhaseV1.from_legacy_encoder(encoder)
    assert torch.equal(rng, torch.get_rng_state())
    assert not direct.training
    legacy_count = sum(p.numel() for p in encoder.freq_bank.parameters()) + sum(p.numel() for p in encoder.delta_mu_projections.parameters())
    assert legacy_count == 8512 + (64 if learnable_kappa else 0)
    assert sum(p.numel() for p in direct.parameters()) == 640
    angles = torch.randn(2, 3, 2, dtype=dtype, requires_grad=True)
    expected = tuple(projection(encoder.freq_bank(angles)) for projection in encoder.delta_mu_projections)
    for actual, original in zip(direct(angles), expected):
        torch.testing.assert_close(actual, original, **tolerances(dtype))
        first = torch.autograd.grad(actual.square().mean(), angles, retain_graph=True)[0]
        second = torch.autograd.grad(original.square().mean(), angles, retain_graph=True)[0]
        torch.testing.assert_close(first, second, **tolerances(dtype))
    source_ptrs = {p.data_ptr() for p in encoder.parameters()}
    assert not any(p.data_ptr() in source_ptrs for p in direct.parameters())


def test_layer_state_roundtrip_and_metadata():
    polar = PolarCircularLinearV2(2, 3, init_kappa=0.001)
    restored = PolarCircularLinearV2(2, 3, init_kappa=0.001)
    restored.load_state_dict(polar.state_dict())
    x = torch.randn(3, 2)
    torch.testing.assert_close(restored(x), polar(x), atol=0, rtol=0)
    cartesian = CartesianCircularLinearV1.from_polar(polar)
    assert cartesian.get_config()["class"] == "CartesianCircularLinearV1"
    assert inverse_softplus(torch.tensor([1000.])).item() == 1000
