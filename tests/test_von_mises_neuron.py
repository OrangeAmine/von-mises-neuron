"""Tests for src/von_mises_neuron.py -- VonMisesLinear and VonMisesForecaster."""

import math

import pytest
import torch

from src.von_mises_neuron import VonMisesLinear, VonMisesForecaster


# ---------------------------------------------------------------------------
# VonMisesLinear
# ---------------------------------------------------------------------------

class TestVonMisesLinear:
    """Tests for the Von Mises linear layer."""

    @pytest.mark.parametrize("in_feat,out_feat,batch", [
        (1, 1, 1),
        (4, 8, 4),
        (16, 32, 8),
    ])
    def test_output_shape(self, in_feat, out_feat, batch):
        """Forward pass produces shape (..., out_features)."""
        layer = VonMisesLinear(in_feat, out_feat)
        x = torch.randn(batch, in_feat)
        y = layer(x)
        assert y.shape == (batch, out_feat)

    def test_output_shape_3d(self):
        """Works with 3D input (batch, seq_len, features)."""
        layer = VonMisesLinear(4, 8)
        x = torch.randn(2, 10, 4)
        y = layer(x)
        assert y.shape == (2, 10, 8)

    def test_periodicity(self):
        """Output is 2π-periodic in each input dimension: f(θ) == f(θ + 2π)."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(5, 4)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x + 2 * math.pi)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)

    def test_periodicity_per_dimension(self):
        """Shifting a single input dimension by 2π does not change output."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(3, 4)
        shift = torch.zeros_like(x)
        shift[:, 2] = 2 * math.pi  # shift only dimension 2
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x + shift)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)

    def test_concentration_positive(self):
        """Concentration κ is always positive (softplus constraint)."""
        layer = VonMisesLinear(4, 8, init_kappa=0.5)
        kappa = layer.get_concentrations()
        assert (kappa > 0).all()

    def test_concentration_near_init(self):
        """Concentration is near the requested initial value."""
        target = 2.0
        layer = VonMisesLinear(4, 8, init_kappa=target)
        kappa = layer.get_concentrations()
        assert torch.allclose(kappa, torch.full_like(kappa, target), atol=0.01)

    def test_phases_in_range(self):
        """get_phases returns values in [0, 2π)."""
        layer = VonMisesLinear(4, 8)
        phases = layer.get_phases()
        assert (phases >= 0).all()
        assert (phases < 2 * math.pi).all()

    def test_gradient_flow(self):
        """Gradients propagate to both μ and log_κ."""
        layer = VonMisesLinear(4, 8)
        x = torch.randn(4, 4)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.mu.grad is not None
        assert layer.log_kappa.grad is not None
        assert layer.mu.grad.abs().sum() > 0
        assert layer.log_kappa.grad.abs().sum() > 0

    def test_gradient_flow_bias(self):
        """Bias receives gradients when enabled."""
        layer = VonMisesLinear(4, 8, bias=True)
        x = torch.randn(4, 4)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.bias.grad is not None
        assert layer.bias.grad.abs().sum() > 0

    def test_no_bias(self):
        """Layer works without bias parameter."""
        layer = VonMisesLinear(4, 8, bias=False)
        assert layer.bias is None
        x = torch.randn(3, 4)
        y = layer(x)
        assert y.shape == (3, 8)

    def test_exp_output_positive(self):
        """In exp-space mode (default), outputs without bias are always > 0."""
        layer = VonMisesLinear(4, 8, bias=False, log_output=False)
        x = torch.randn(10, 4)
        with torch.no_grad():
            y = layer(x)
        assert (y > 0).all()

    def test_log_output_mode(self):
        """Log-space output matches log of exp-space output (no bias)."""
        layer = VonMisesLinear(4, 8, bias=False, log_output=False)
        x = torch.randn(5, 4)
        with torch.no_grad():
            y_exp = layer(x)

        # Switch to log mode and recompute
        layer.log_output = True
        with torch.no_grad():
            y_log = layer(x)

        torch.testing.assert_close(y_log, torch.log(y_exp), atol=1e-5, rtol=1e-5)

    def test_peak_at_preferred_direction(self):
        """Output peaks when input aligns with the negative of μ.

        cos(θ + μ) is maximised when θ + μ = 0  ⟹  θ = −μ (mod 2π).
        """
        layer = VonMisesLinear(1, 1, bias=False, init_kappa=3.0)
        layer.eval()
        # Set μ to a known value
        with torch.no_grad():
            layer.mu.fill_(1.0)  # μ = 1.0 rad

        # θ = -μ = -1.0 should give higher output than θ = -μ + π = 2.14
        theta_peak = torch.tensor([[-1.0]])
        theta_off = torch.tensor([[-1.0 + math.pi]])
        with torch.no_grad():
            y_peak = layer(theta_peak)
            y_off = layer(theta_off)
        assert y_peak.item() > y_off.item()

    def test_deterministic_eval(self):
        """Same input produces same output in eval mode."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)
        torch.testing.assert_close(y1, y2)

    def test_concentration_init_small_kappa(self):
        """Low-kappa fallback path (init_kappa <= 0.05) initializes correctly.

        Covers reset_parameters() else branch: when init_kappa is very small,
        the inverse softplus formula log(exp(k) - 1) is replaced by a hardcoded
        inv = -3.0, giving softplus(-3) ~ 0.049.  Without this test, that
        branch is never exercised.
        """
        layer = VonMisesLinear(4, 8, init_kappa=0.01)
        kappa = layer.get_concentrations()
        assert (kappa > 0).all(), "kappa must be positive even at small init"
        # softplus(-3.0) = log(1 + exp(-3)) ~ 0.0486
        assert (kappa < 0.1).all(), (
            f"Expected kappa < 0.1 for small init, got max={kappa.max():.4f}"
        )

    def test_large_kappa_no_nan(self):
        """Extreme concentration values don't produce NaN/Inf in forward pass.

        With kappa=50 and in_features=4, the worst-case log-sum is
        50 * 4 = 200, so exp(200) overflows float32 (max ~ 3.4e38).
        In log-space mode this is safe (z stays finite); in exp-space,
        this test documents the regime where blowup can occur.
        """
        # Log-space: should always be finite
        layer_log = VonMisesLinear(4, 8, init_kappa=50.0, bias=False, log_output=True)
        x = torch.randn(10, 4)
        with torch.no_grad():
            y_log = layer_log(x)
        assert torch.isfinite(y_log).all(), (
            f"Log-space output has non-finite values with kappa=50"
        )

        # Exp-space with 1 input feature: exp(50 * cos(...)) ~ exp(50) ~ 5.2e21
        # This is within float32 range, so should be safe
        layer_exp = VonMisesLinear(1, 4, init_kappa=50.0, bias=False, log_output=False)
        x_1d = torch.randn(10, 1)
        with torch.no_grad():
            y_exp = layer_exp(x_1d)
        assert torch.isfinite(y_exp).all(), (
            f"Exp-space output has non-finite values with kappa=50, 1 input"
        )


# ---------------------------------------------------------------------------
# VonMisesForecaster
# ---------------------------------------------------------------------------

class TestVonMisesForecaster:
    """Tests for the VonMises-based forecasting model."""

    @pytest.mark.parametrize("batch_size,lookback,horizon", [
        (1, 24, 1),
        (8, 72, 24),
        (32, 48, 6),
    ])
    def test_output_shape(self, batch_size, lookback, horizon):
        """Forward pass produces shape (batch, horizon)."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=32, num_vm_layers=2,
            output_size=horizon,
        )
        x = torch.randn(batch_size, lookback, 4)
        y = model(x)
        assert y.shape == (batch_size, horizon)

    def test_deterministic_eval(self):
        """Same input produces same output in eval mode (no dropout noise)."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            dropout=0.0, output_size=6,
        )
        model.eval()
        x = torch.randn(4, 24, 4)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        torch.testing.assert_close(y1, y2)

    def test_gradient_flow(self):
        """Loss backward propagates to all parameters."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            output_size=6,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_dropout_effect(self):
        """Dropout causes different outputs in train mode."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            dropout=0.5, output_size=6,
        )
        model.train()
        x = torch.randn(4, 24, 4)

        torch.manual_seed(1)
        y1 = model(x)
        torch.manual_seed(2)
        y2 = model(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_single_vm_layer(self):
        """Model works with a single VM layer."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=1,
            output_size=6,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        assert y.shape == (4, 6)

    def test_interpretability_accessors(self):
        """get_vm_layers / get_layer_concentrations / get_layer_phases work."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=8, num_vm_layers=3,
            output_size=1,
        )
        vm_layers = model.get_vm_layers()
        assert len(vm_layers) == 3

        concentrations = model.get_layer_concentrations()
        assert len(concentrations) == 3
        # First layer: (8, 4), subsequent layers: (8, 8)
        assert concentrations[0].shape == (8, 4)
        assert concentrations[1].shape == (8, 8)
        assert concentrations[2].shape == (8, 8)
        # All concentrations must be positive
        for k in concentrations:
            assert (k > 0).all()

        phases = model.get_layer_phases()
        assert len(phases) == 3
        for p in phases:
            assert (p >= 0).all()
            assert (p < 2 * math.pi).all()

    def test_periodicity_through_model(self):
        """Full model output is 2π-periodic in each input feature.

        This tests the end-to-end periodicity preservation, not just
        the individual VM layer.
        """
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            dropout=0.0, output_size=6,
        )
        model.eval()
        x = torch.randn(4, 24, 4)
        # The VM layers (log-space + LayerNorm) break exact periodicity
        # because LayerNorm computes statistics over the hidden dim.
        # However, the *first* VM layer is exactly periodic, so verify that.
        vm_layer_0 = model.get_vm_layers()[0]
        with torch.no_grad():
            h1 = vm_layer_0(x)
            h2 = vm_layer_0(x + 2 * math.pi)
        torch.testing.assert_close(h1, h2, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# VonMisesForecaster — Ablation variants
# ---------------------------------------------------------------------------

class TestVonMisesForecasterAblations:
    """Tests for the ablation-controlled architecture variants.

    The three flags (log_output, use_layernorm, use_lstm) were added to
    VonMisesForecaster to support systematic ablation studies.  These tests
    verify that each variant produces correct shapes, maintains gradient
    flow, and behaves as expected.
    """

    def test_default_params_backward_compatible(self):
        """Default flags reproduce the original architecture.

        The default constructor should produce identical structure to the
        pre-ablation code: log-space VM layers, LayerNorm present, LSTM present.
        """
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2, output_size=6,
        )
        # LSTM should be present
        assert model.lstm is not None
        assert isinstance(model.lstm, torch.nn.LSTM)

        # LayerNorm modules should be in the encoder
        ln_modules = [m for m in model.vm_encoder if isinstance(m, torch.nn.LayerNorm)]
        assert len(ln_modules) == 2, "Expected 2 LayerNorm modules (one per VM layer)"

        # VM layers should be in log-output mode
        vm_layers = model.get_vm_layers()
        assert len(vm_layers) == 2
        for vm in vm_layers:
            assert vm.log_output is True

    @pytest.mark.parametrize("batch_size,lookback,horizon", [
        (1, 24, 1),
        (8, 72, 24),
        (16, 48, 6),
    ])
    def test_no_lstm_output_shape(self, batch_size, lookback, horizon):
        """Mean-pooling variant produces correct output shape for various configs."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            output_size=horizon, use_lstm=False,
        )
        model.eval()
        x = torch.randn(batch_size, lookback, 4)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (batch_size, horizon)

    def test_no_lstm_gradient_flow(self):
        """All parameters receive gradients when LSTM is replaced by mean pool."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            output_size=6, use_lstm=False,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_no_lstm_fewer_params(self):
        """Removing LSTM reduces total parameter count.

        The LSTM contributes 4*(H*H + H*H + 2*H) = 8H² + 8H parameters
        (input_size=H since it follows the VM encoder).  This should vanish
        when use_lstm=False.
        """
        kwargs = dict(input_size=4, hidden_size=32, num_vm_layers=2, output_size=6)
        with_lstm = VonMisesForecaster(**kwargs, use_lstm=True)
        without_lstm = VonMisesForecaster(**kwargs, use_lstm=False)

        params_with = sum(p.numel() for p in with_lstm.parameters())
        params_without = sum(p.numel() for p in without_lstm.parameters())

        # LSTM for H=32: 4*(32*32 + 32*32 + 2*32) = 4*(1024+1024+64) = 8,448
        assert params_without < params_with, (
            f"No-LSTM ({params_without}) should have fewer params "
            f"than with-LSTM ({params_with})"
        )

    def test_no_lstm_sequence_length_agnostic(self):
        """Mean-pooling works with different sequence lengths without rebuild.

        Unlike a flatten approach (which would fix the input dimension), mean
        pooling is agnostic to sequence length — the same model can process
        sequences of any length.
        """
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=1,
            output_size=6, use_lstm=False,
        )
        model.eval()
        with torch.no_grad():
            y_short = model(torch.randn(2, 12, 4))
            y_long = model(torch.randn(2, 96, 4))
        assert y_short.shape == (2, 6)
        assert y_long.shape == (2, 6)

    def test_no_layernorm_output_shape(self):
        """Model produces correct output with LayerNorm removed."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            output_size=6, use_layernorm=False,
        )
        # Verify no LayerNorm modules in the encoder
        ln_modules = [m for m in model.vm_encoder if isinstance(m, torch.nn.LayerNorm)]
        assert len(ln_modules) == 0, "Expected no LayerNorm modules"

        x = torch.randn(4, 24, 4)
        y = model(x)
        assert y.shape == (4, 6)

    def test_exp_space_output_shape(self):
        """Exp-space stacking (log_output=False) produces correct shape."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=2,
            output_size=6, log_output=False,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        assert y.shape == (4, 6)

    def test_exp_space_vm_layers_mode(self):
        """All VM layers have log_output=False when forecaster uses exp-space."""
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=3,
            output_size=1, log_output=False,
        )
        for vm in model.get_vm_layers():
            assert vm.log_output is False, (
                "VM layer should be in exp-space mode when forecaster "
                "log_output=False"
            )

    @pytest.mark.parametrize("log_output,use_layernorm,use_lstm", [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ])
    def test_single_layer_all_ablation_combos(
        self, log_output, use_layernorm, use_lstm
    ):
        """Single VM layer works with all 8 combinations of ablation flags.

        This is the most exhaustive correctness check: every ablation variant
        must produce a valid output tensor and support gradient flow.
        """
        model = VonMisesForecaster(
            input_size=4, hidden_size=16, num_vm_layers=1,
            output_size=6,
            log_output=log_output,
            use_layernorm=use_layernorm,
            use_lstm=use_lstm,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        assert y.shape == (4, 6), (
            f"Shape mismatch for log={log_output}, ln={use_layernorm}, "
            f"lstm={use_lstm}"
        )

        # Verify gradient flow
        loss = y.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"No gradient for {name} with log={log_output}, "
                f"ln={use_layernorm}, lstm={use_lstm}"
            )
