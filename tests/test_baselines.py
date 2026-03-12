"""Tests for src/baselines.py -- LSTMForecaster, TransformerForecaster,
HandCraftedLinearForecaster, and utility functions."""

import pytest
import torch
import torch.nn as nn

from src.baselines import (
    LSTMForecaster,
    TransformerForecaster,
    HandCraftedLinearForecaster,
    SinusoidalPositionalEncoding,
    compute_param_matched_hidden_size,
)
from src.von_mises_neuron import VonMisesForecaster


# ---------------------------------------------------------------------------
# LSTMForecaster
# ---------------------------------------------------------------------------

class TestLSTMForecaster:
    """Tests for the LSTM-only baseline (VM-replacement)."""

    @pytest.mark.parametrize("batch_size,lookback,horizon", [
        (1, 24, 1),
        (8, 72, 24),
        (32, 48, 6),
    ])
    def test_output_shape(self, batch_size, lookback, horizon):
        """Forward pass produces shape (batch, horizon)."""
        model = LSTMForecaster(
            input_size=4, hidden_size=32, num_layers=2,
            output_size=horizon,
        )
        x = torch.randn(batch_size, lookback, 4)
        y = model(x)
        assert y.shape == (batch_size, horizon)

    def test_deterministic_eval(self):
        """Same input produces same output in eval mode."""
        model = LSTMForecaster(
            input_size=4, hidden_size=16, num_layers=2,
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
        model = LSTMForecaster(
            input_size=4, hidden_size=16, num_layers=2,
            output_size=6,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_parameter_count_matching(self):
        """Parameter count is within 2% of VonMisesForecaster(hidden=64)."""
        vm_model = VonMisesForecaster(
            input_size=4, hidden_size=64, num_vm_layers=2,
            dropout=0.1, output_size=12, init_kappa=1.0,
        )
        vm_params = sum(p.numel() for p in vm_model.parameters())

        lstm_model = LSTMForecaster(
            input_size=4, hidden_size=68, num_layers=2,
            dropout=0.1, output_size=12,
        )
        lstm_params = sum(p.numel() for p in lstm_model.parameters())

        ratio = abs(lstm_params - vm_params) / vm_params
        assert ratio < 0.02, (
            f"Parameter mismatch: LSTM={lstm_params}, VM={vm_params}, "
            f"ratio={ratio:.3f} (> 2%)"
        )

    def test_dropout_effect(self):
        """Dropout causes different outputs in train mode."""
        model = LSTMForecaster(
            input_size=4, hidden_size=16, num_layers=2,
            dropout=0.5, output_size=6,
        )
        model.train()
        x = torch.randn(4, 24, 4)

        torch.manual_seed(1)
        y1 = model(x)
        torch.manual_seed(2)
        y2 = model(x)

        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_single_layer(self):
        """Model works with a single encoder layer."""
        model = LSTMForecaster(
            input_size=4, hidden_size=16, num_layers=1,
            output_size=6,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        assert y.shape == (4, 6)


# ---------------------------------------------------------------------------
# TransformerForecaster
# ---------------------------------------------------------------------------

class TestTransformerForecaster:
    """Tests for the Transformer baseline."""

    @pytest.mark.parametrize("batch_size,lookback,horizon", [
        (1, 24, 1),
        (8, 72, 24),
        (32, 48, 6),
    ])
    def test_output_shape(self, batch_size, lookback, horizon):
        """Forward pass produces shape (batch, horizon)."""
        model = TransformerForecaster(
            input_size=4, d_model=32, nhead=4, num_layers=2,
            dim_feedforward=64, output_size=horizon,
        )
        x = torch.randn(batch_size, lookback, 4)
        y = model(x)
        assert y.shape == (batch_size, horizon)

    def test_deterministic_eval(self):
        """Same input produces same output in eval mode."""
        model = TransformerForecaster(
            input_size=4, d_model=16, nhead=4, num_layers=1,
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
        model = TransformerForecaster(
            input_size=4, d_model=16, nhead=4, num_layers=1,
            output_size=6,
        )
        x = torch.randn(4, 24, 4)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_variable_sequence_length(self):
        """Model handles different sequence lengths without error."""
        model = TransformerForecaster(
            input_size=4, d_model=16, nhead=4, num_layers=1,
            output_size=6, max_len=512,
        )
        model.eval()
        for seq_len in [24, 48, 72, 96]:
            x = torch.randn(2, seq_len, 4)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2, 6)

    def test_positional_encoding_no_params(self):
        """SinusoidalPositionalEncoding adds zero learnable parameters."""
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
        # Only the dropout layer is a module; PE is a buffer
        learnable_params = sum(p.numel() for p in pe.parameters())
        assert learnable_params == 0


# ---------------------------------------------------------------------------
# HandCraftedLinearForecaster
# ---------------------------------------------------------------------------

class TestHandCraftedLinearForecaster:
    """Tests for the linear + hand-crafted features baseline."""

    @pytest.mark.parametrize("batch_size,lookback,horizon,input_size", [
        (1, 24, 1, 8),
        (8, 72, 12, 8),
        (32, 48, 6, 10),
    ])
    def test_output_shape(self, batch_size, lookback, horizon, input_size):
        """Forward pass produces shape (batch, horizon)."""
        model = HandCraftedLinearForecaster(
            input_size=input_size, lookback=lookback,
            output_size=horizon,
        )
        x = torch.randn(batch_size, lookback, input_size)
        y = model(x)
        assert y.shape == (batch_size, horizon)

    def test_gradient_flow(self):
        """Loss backward propagates to linear weights."""
        model = HandCraftedLinearForecaster(
            input_size=8, lookback=72, output_size=12,
        )
        x = torch.randn(4, 72, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert model.linear.weight.grad is not None
        assert model.linear.weight.grad.abs().sum() > 0
        assert model.linear.bias.grad is not None

    def test_is_truly_linear(self):
        """Output is an affine function of the input (superposition holds up to bias)."""
        model = HandCraftedLinearForecaster(
            input_size=8, lookback=24, output_size=6,
        )
        model.eval()

        x1 = torch.randn(1, 24, 8)
        x2 = torch.randn(1, 24, 8)
        a, b = 0.7, 0.3

        with torch.no_grad():
            y1 = model(x1)
            y2 = model(x2)
            y_combo = model(a * x1 + b * x2)

        # affine: f(ax+by) = a*f(x) + b*f(y) + (1-a-b)*bias
        # Since a + b = 1.0, this simplifies to f(ax+by) = a*f(x) + b*f(y)
        expected = a * y1 + b * y2
        torch.testing.assert_close(y_combo, expected, atol=1e-5, rtol=1e-5)

    def test_no_hidden_layers(self):
        """Model has exactly one Linear layer (no hidden layers)."""
        model = HandCraftedLinearForecaster(
            input_size=8, lookback=72, output_size=12,
        )
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 1


# ---------------------------------------------------------------------------
# compute_param_matched_hidden_size
# ---------------------------------------------------------------------------

class TestComputeParamMatchedHiddenSize:
    """Tests for the hidden_size solver utility."""

    def test_default_target(self):
        """Returns h=68 for the default VM(hidden=64) target of ~43148."""
        vm_model = VonMisesForecaster(
            input_size=4, hidden_size=64, num_vm_layers=2,
            dropout=0.1, output_size=12, init_kappa=1.0,
        )
        target = sum(p.numel() for p in vm_model.parameters())
        h = compute_param_matched_hidden_size(target, input_size=4, output_size=12)
        assert h == 68

    def test_round_trip(self):
        """Computed hidden_size yields param count within 2% of target."""
        target = 43148
        h = compute_param_matched_hidden_size(target, input_size=4, output_size=12)
        model = LSTMForecaster(
            input_size=4, hidden_size=h, num_layers=2,
            output_size=12,
        )
        actual = sum(p.numel() for p in model.parameters())
        ratio = abs(actual - target) / target
        assert ratio < 0.02, f"Mismatch: target={target}, actual={actual}, ratio={ratio:.3f}"
