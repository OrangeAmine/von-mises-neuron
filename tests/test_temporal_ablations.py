"""Behavioral checks for temporal conditioning and amplitude ablations."""

import math

import pytest
import torch

from src.temporal import FrequencyBankVonMises, TemporalVonMisesForecaster


def make_model(**kwargs) -> TemporalVonMisesForecaster:
    """Construct a small forecaster retaining stacked layers and residuals."""
    return TemporalVonMisesForecaster(
        sensor_size=3, time_periods=[24, 168], hidden_size=8,
        num_vm_layers=2, output_size=4, dropout=0, **kwargs,
    )


def test_learned_bank_matches_frozen_initial_function_and_rng():
    """Changing amplitude parameterization preserves phases and initial output."""
    torch.manual_seed(7)
    frozen = FrequencyBankVonMises(2, 8)
    frozen_rng = torch.get_rng_state()
    torch.manual_seed(7)
    learned = FrequencyBankVonMises(2, 8, learnable_kappa=True)
    assert torch.equal(torch.get_rng_state(), frozen_rng)
    torch.testing.assert_close(frozen.mu, learned.mu, rtol=0, atol=0)
    torch.testing.assert_close(frozen.get_kappa(), learned.get_kappa())
    x = torch.randn(3, 5, 2)
    torch.testing.assert_close(frozen(x), learned(x))
    assert "kappa" in dict(frozen.named_buffers())
    assert "kappa" not in dict(frozen.named_parameters())
    assert "raw_kappa" in dict(learned.named_parameters())
    assert "kappa" not in dict(learned.named_buffers())
    learned(x).square().mean().backward()
    assert torch.isfinite(learned.raw_kappa.grad).all()
    assert torch.count_nonzero(learned.raw_kappa.grad) == 8
    with torch.no_grad():
        learned.raw_kappa.sub_(0.1 * learned.raw_kappa.grad)
    assert (learned.get_kappa() > 0).all()
    assert not torch.equal(frozen.get_kappa(), learned.get_kappa())


def test_explicit_default_flags_preserve_state_and_rng():
    """Optional flags do not consume random draws or alter frozen state keys."""
    torch.manual_seed(13)
    implicit = make_model()
    implicit_rng = torch.get_rng_state()
    torch.manual_seed(13)
    explicit = make_model(learnable_kappa=False, use_time_conditioning=True)
    assert torch.equal(torch.get_rng_state(), implicit_rng)
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("use_lstm", [False, True])
@pytest.mark.parametrize("use_temporal_conv", [False, True])
@pytest.mark.parametrize("use_time_conditioning", [False, True])
@pytest.mark.parametrize("learnable_kappa", [False, True])
def test_ablation_shapes_and_finite_gradients(
    use_lstm, use_temporal_conv, use_time_conditioning, learnable_kappa,
):
    """Every supported combination uses all parameters and backpropagates."""
    torch.manual_seed(19)
    model = make_model(
        use_lstm=use_lstm, use_temporal_conv=use_temporal_conv,
        use_time_conditioning=use_time_conditioning, learnable_kappa=learnable_kappa,
    ).double()
    x = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 4)
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[..., :3]) > 0
    if use_time_conditioning:
        assert torch.count_nonzero(x.grad[..., 3:]) > 0
    else:
        assert torch.count_nonzero(x.grad[..., 3:]) == 0
        assert model.get_freq_bank() is None
        assert len(model.encoder.delta_mu_projections) == 0
        assert not any("freq_bank" in name or "delta_mu" in name
                       for name, _ in model.named_parameters())


@pytest.mark.parametrize("learnable_kappa", [False, True])
def test_full_forecaster_time_angle_periodicity(learnable_kappa):
    """Time-angle periodicity survives sensor residuals, convolutions and LSTM."""
    torch.manual_seed(31)
    model = make_model(learnable_kappa=learnable_kappa).double().eval()
    x = torch.randn(2, 9, 5, dtype=torch.float64)
    with torch.no_grad():
        expected = model(x)
        for column in (3, 4):
            shifted = x.clone()
            shifted[..., column] += 2 * math.pi
            torch.testing.assert_close(model(shifted), expected, rtol=1e-10, atol=1e-10)


def test_disabled_conditioning_ignores_time_values_entirely():
    """Disabling conditioning removes the entire calendar input pathway."""
    model = make_model(use_time_conditioning=False).eval()
    x = torch.randn(2, 9, 5)
    changed = x.clone()
    changed[..., 3:] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model(changed), model(x), rtol=0, atol=0)
    enabled = make_model()
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in enabled.parameters())


@pytest.mark.parametrize("learnable_kappa", [False, True])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bank_amplitudes_follow_dtype_and_device(learnable_kappa, device):
    bank = FrequencyBankVonMises(2, 8, learnable_kappa=learnable_kappa).to(
        device=device, dtype=torch.float64,
    )
    x = torch.randn(2, 3, 2, device=device, dtype=torch.float64, requires_grad=True)
    y = bank(x)
    assert y.device == x.device == bank.get_kappa().device
    assert y.dtype == x.dtype == bank.get_kappa().dtype
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("bands", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_rejects_invalid_amplitude_initialization(bands):
    with pytest.raises(ValueError, match="positive finite"):
        FrequencyBankVonMises(2, 8, freq_bands=bands, learnable_kappa=True)
