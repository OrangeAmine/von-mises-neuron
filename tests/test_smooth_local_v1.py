"""Equation, capacity, numerical gradient and replay contracts for VM-N01."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.locality_models import reduced_objective, solve_readout
from src.smooth_local_v1 import (
    LineRBFBasisV1, SmoothLocalBasisV1, make_model, predict_smooth_local_state,
)


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/vm_n01_v1.json").read_text())
SPECS = CONFIG["models"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,expected", [((), (16,)), ((7,), (7, 16)), ((7, 1), (7, 16)), ((2, 3, 4, 1), (2, 3, 4, 16))])
def test_equation_shapes_dtype_and_finite_gradients(dtype, shape, expected):
    basis = SmoothLocalBasisV1(3, 10, True, rotation=.23, dtype=dtype)
    theta = torch.linspace(-2.7, 2.7, math.prod(shape) or 1, dtype=dtype).reshape(shape).requires_grad_()
    features = basis(theta)
    assert features.shape == expected and features.dtype == dtype
    angles = theta[..., 0] if theta.ndim >= 2 else theta
    phase = angles[..., None] * torch.arange(1, 4, dtype=dtype)
    expected_features = torch.cat((phase.cos(), phase.sin(),
        (basis.bank.concentrations * ((angles[..., None] - basis.bank.centers[:, 0]).cos() - 1)).exp()), -1)
    tolerance = 4e-7 if dtype == torch.float32 else 2e-15
    torch.testing.assert_close(features, expected_features, atol=tolerance, rtol=tolerance)
    features.square().sum().backward()
    assert torch.isfinite(theta.grad).all()
    for parameter in basis.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("harmonics,kernels,adaptive", [(3, 10, True), (3, 10, False), (8, 0, False), (0, 16, True)])
def test_full_signed_model_wrap_and_gradient_wrap(harmonics, kernels, adaptive):
    basis = SmoothLocalBasisV1(harmonics, kernels, adaptive)
    theta = torch.tensor([[-math.pi], [.12], [math.pi]], dtype=torch.float64, requires_grad=True)
    weight = torch.linspace(-1, 1, basis.basis_count, dtype=torch.float64)
    output = basis(theta) @ weight + .7
    shifted_theta = (theta.detach() + 2 * math.pi).requires_grad_()
    shifted = basis(shifted_theta) @ weight + .7
    torch.testing.assert_close(shifted, output, atol=3e-14, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(output.sum(), theta)[0],
                               torch.autograd.grad(shifted.sum(), shifted_theta)[0], atol=1e-13, rtol=1e-11)
    if basis.bank is not None:
        with torch.no_grad():
            basis.bank.centers.add_(2 * math.pi)
        torch.testing.assert_close(basis(theta) @ weight + .7, output, atol=3e-14, rtol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=lambda v: v["id"])
def test_registered_capacity_rng_and_state_replay(spec):
    before = torch.get_rng_state().clone()
    model = make_model(spec, rotation=.39)
    assert torch.equal(before, torch.get_rng_state())
    assert model.basis_count == spec["features"]
    assert model.fitted_parameter_count == spec["parameters"]
    geometry_count = 2 * spec["kernels"] if spec["kind"] != "ridge" else 0
    assert sum(p.numel() for p in model.parameters()) == geometry_count + (spec["features"] + 1 if spec["kind"] == "joint" else 0)
    theta = torch.linspace(-math.pi, math.pi, 61, dtype=torch.float64)[:, None]
    y = .3 + theta.sin() + .2 * (3 * theta).cos()
    model.refresh_readout(model.features(theta), y, .001)
    state = model.snapshot()
    expected = model(theta).detach()
    torch.testing.assert_close(predict_smooth_local_state(state, theta.numpy()), expected, atol=0, rtol=0)
    with torch.no_grad():
        model.readout_weight.add_(1)
    torch.testing.assert_close(predict_smooth_local_state(state, theta), expected, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())


def test_multiscale_fixed_geometry_order_and_buffers():
    basis = SmoothLocalBasisV1(3, 30, False, concentration_bands=[1., 4., 16.])
    assert len(list(basis.parameters())) == 0
    assert "bank.fixed_kappa" in dict(basis.named_buffers())
    torch.testing.assert_close(basis.bank.concentrations, torch.tensor([1., 4., 16.], dtype=torch.float64).repeat(10))
    assert torch.equal(basis.bank.centers[:3], basis.bank.centers[:1].expand(3, 1))
    assert basis.float()(torch.zeros(2, 1)).dtype == torch.float32


def test_line_rbf_is_unrestricted_and_equation_matches():
    basis = LineRBFBasisV1()
    theta = torch.tensor([[-math.pi], [0.], [math.pi]], dtype=torch.float64)
    features = basis(theta)
    expected = torch.cat((theta / math.pi, (-.5 * ((theta - basis.centers) / .45).square()).exp()), 1)
    torch.testing.assert_close(features, expected, atol=0, rtol=0)
    assert not torch.allclose(features, basis(theta + 2 * math.pi))
    torch.testing.assert_close(features[:, 0], torch.tensor([-1., 0., 1.], dtype=torch.float64))


def test_hybrid_ridge_matches_independent_augmented_svd():
    basis = SmoothLocalBasisV1(3, 10, False)
    x = torch.linspace(-3, 3, 43, dtype=torch.float64)[:, None]
    y = (x.sin() + .3 * x.cos())[:, 0]
    features = basis(x)
    weight, bias = solve_readout(features, y, .001)
    matrix = np.column_stack((features.numpy(), np.ones(len(x))))
    penalty_rows = np.column_stack((np.sqrt(len(x) * .001) * np.eye(16), np.zeros(16)))
    expected = np.linalg.lstsq(np.vstack((matrix, penalty_rows)), np.concatenate((y.numpy(), np.zeros(16))), rcond=None)[0]
    np.testing.assert_allclose(weight.numpy(), expected[:-1], atol=2e-13, rtol=1e-11)
    np.testing.assert_allclose(bias.numpy(), expected[-1], atol=2e-13, rtol=1e-11)


def test_hybrid_vp_gradient_matches_envelope_and_finite_differences():
    basis = SmoothLocalBasisV1(2, 3, True, rotation=.2)
    x = torch.linspace(-2.9, 2.8, 29, dtype=torch.float64)[:, None]
    y = (x.sin() + .5 * (5 * x).cos())[:, 0]
    features = basis(x)
    reduced = reduced_objective(features, y, .001)
    params = tuple(basis.parameters())
    automatic = torch.autograd.grad(reduced, params, retain_graph=True)
    weight, bias = solve_readout(features, y, .001)
    envelope = .5 * (features @ weight.detach() + bias.detach() - y).square().mean()
    envelope_grad = torch.autograd.grad(envelope, params)
    for left, right in zip(automatic, envelope_grad):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)
    eps = 1e-5
    for parameter, gradient in zip(params, automatic):
        for index in range(parameter.numel()):
            initial = parameter.detach().clone()
            with torch.no_grad():
                parameter.reshape(-1)[index] += eps
            plus = reduced_objective(basis(x), y, .001).item()
            with torch.no_grad():
                parameter.copy_(initial)
                parameter.reshape(-1)[index] -= eps
            minus = reduced_objective(basis(x), y, .001).item()
            with torch.no_grad():
                parameter.copy_(initial)
            assert gradient.reshape(-1)[index].item() == pytest.approx((plus-minus)/(2*eps), abs=1e-9, rel=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"harmonics": -1}, {"harmonics": True}, {"kernels": -1}, {"adaptive": "yes"},
    {"harmonics": 0, "kernels": 0}, {"concentration": 0}, {"concentration": math.inf},
    {"rotation": math.nan}, {"concentration_bands": [1, 2, 3]},
    {"concentration_bands": [1, 4]}, {"dtype": torch.int64},
])
def test_invalid_basis_configuration(kwargs):
    arguments = {"harmonics": 3, "kernels": 10, "adaptive": True, **kwargs}
    with pytest.raises(ValueError):
        SmoothLocalBasisV1(**arguments)


def test_invalid_dtype_shape_and_capacity_rejected():
    basis = SmoothLocalBasisV1(3, 10, True)
    with pytest.raises(ValueError, match="dtype"):
        basis(torch.zeros(3, 1))
    with pytest.raises(ValueError, match="singleton"):
        basis(torch.zeros(3, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="counts"):
        make_model({**SPECS[0], "parameters": 19})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 3e-5), (torch.float64, 1e-11)])
def test_cpu_cuda_value_and_gradient_parity(dtype, tolerance):
    cpu = SmoothLocalBasisV1(3, 10, True, rotation=.3, dtype=dtype)
    gpu = copy.deepcopy(cpu).cuda()
    x = torch.linspace(-math.pi, math.pi, 41, dtype=dtype)[:, None]
    a, b = cpu(x), gpu(x.cuda())
    torch.testing.assert_close(a, b.cpu(), atol=tolerance, rtol=tolerance)
    a.square().mean().backward()
    b.square().mean().backward()
    for left, right in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(left.grad, right.grad.cpu(), atol=tolerance, rtol=tolerance)
