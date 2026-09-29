"""Numerical and integration contracts for the narrow VM-N01 adapter."""

import copy
import math

import pytest
import torch

from src.smooth_local_forecaster_v1 import SmoothLocalForecasterV1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_dtype_finite_gradients_and_state_replay(dtype):
    torch.manual_seed(251)
    model = SmoothLocalForecasterV1(input_size=3, output_size=5).to(dtype=dtype).eval()
    inputs = torch.randn(2, 7, 3, dtype=dtype, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2, 5) and output.dtype == dtype
    gradients = torch.autograd.grad(output.square().mean(), [inputs, *model.parameters()])
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    restored = SmoothLocalForecasterV1(input_size=3, output_size=5).to(dtype=dtype).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(inputs), output, atol=0, rtol=0)


def test_every_sensor_wrap_is_preserved_end_to_end():
    torch.manual_seed(99)
    model = SmoothLocalForecasterV1().double().eval()
    inputs = torch.randn(2, 6, 4, dtype=torch.float64)
    original = model(inputs)
    for column in range(4):
        shifted = inputs.clone()
        shifted[..., column] += 2 * math.pi
        torch.testing.assert_close(model(shifted), original, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float32_output_and_gradient_parity():
    torch.manual_seed(129)
    # cuDNN RNN backward requires training mode; zero dropout pairs the paths.
    cpu_model = SmoothLocalForecasterV1(dropout=0.0).train()
    gpu_model = copy.deepcopy(cpu_model).cuda()
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    gpu_inputs = inputs.detach().cuda().requires_grad_()
    cpu_output, gpu_output = cpu_model(inputs), gpu_model(gpu_inputs)
    torch.testing.assert_close(gpu_output.cpu(), cpu_output, atol=1e-5, rtol=1e-4)
    cpu_gradients = torch.autograd.grad(cpu_output.square().mean(), [inputs, *cpu_model.parameters()])
    gpu_gradients = torch.autograd.grad(gpu_output.square().mean(), [gpu_inputs, *gpu_model.parameters()])
    for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients):
        assert gpu_gradient.is_cuda and torch.isfinite(gpu_gradient).all()
        torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 5), (2, 0, 4)])
def test_invalid_input_shape_fails_clearly(shape):
    with pytest.raises(ValueError, match="input must have shape"):
        SmoothLocalForecasterV1()(torch.randn(*shape))


@pytest.mark.parametrize("kwargs", [{"input_size": 0}, {"output_size": True},
    {"projection_size": -2}, {"hidden_size": 2.5}, {"dropout": -0.1}, {"dropout": 1.0}])
def test_invalid_constructor_rejected(kwargs):
    with pytest.raises(ValueError):
        SmoothLocalForecasterV1(**kwargs)
