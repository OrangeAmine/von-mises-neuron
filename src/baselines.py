"""Comparative baselines for the VonMisesForecaster.

Three controlled baselines isolate the VM neuron's contribution:

1. **LSTMForecaster** — Same encoder→LSTM→head architecture but replaces
   VonMisesLinear with standard Linear+ReLU.  Parameter-matched so any
   performance difference directly measures the value of periodicity
   preservation.

2. **TransformerForecaster** — Encoder-only Transformer with sinusoidal
   positional encoding.  Tests whether learned self-attention can implicitly
   capture periodicity as well as the VM neuron's explicit cos() encoding.

3. **HandCraftedLinearForecaster** — Pure linear regression on a flattened
   lookback window augmented with sin/cos time features.  Tests whether
   the VM neuron merely automates what manual feature engineering achieves.
"""

import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility: parameter-matched hidden size
# ---------------------------------------------------------------------------
def compute_param_matched_hidden_size(
    target_params: int,
    input_size: int = 4,
    output_size: int = 12,
    num_layers: int = 2,
) -> int:
    """Solve for the hidden_size that approximately matches a target parameter count.

    For an LSTMForecaster with `num_layers` encoder blocks, each containing
    Linear + ReLU + LayerNorm, followed by a single-layer LSTM and a linear
    output head, the total parameter count is a quadratic in h (hidden_size).

    The closed-form is derived by summing:
        - Encoder layer 0:  (input_size + 1)*h + 2*h
        - Encoder layers 1..L-1:  (h + 1)*h + 2*h  each
        - LSTM(h→h, 1 layer):  8*h^2 + 8*h
        - Output head:  (h + 1)*output_size

    Args:
        target_params: Target total parameter count to match.
        input_size: Number of input features (default 4).
        output_size: Forecast horizon (default 12).
        num_layers: Number of encoder layers (default 2).

    Returns:
        The integer hidden_size closest to the target parameter count.
    """
    # Coefficient derivation for the quadratic a*h^2 + b*h + c = target:
    #   Encoder layer 0: (input_size)*h + h + 2h = (input_size + 3)*h
    #   Each subsequent layer: h^2 + h + 2h = h^2 + 3h
    #   LSTM: 8h^2 + 8h
    #   Output: output_size*h + output_size
    a = (num_layers - 1) + 8  # h^2 coefficient: (num_layers - 1) from encoder + 8 from LSTM
    b = (input_size + 3) + 3 * (num_layers - 1) + 8 + output_size
    c = output_size - target_params

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError(f"No real solution: target_params={target_params} is too small")

    h_float = (-b + math.sqrt(discriminant)) / (2 * a)
    return round(h_float)


# ---------------------------------------------------------------------------
# Baseline 1: LSTM-only (parameter-matched)
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """LSTM forecaster with standard Linear+ReLU encoder (VM-replacement baseline).

    Mirrors the VonMisesForecaster architecture exactly, but replaces each
    VonMisesLinear layer with a standard Linear → ReLU block.  Used for a
    controlled comparison: any performance difference isolates the effect
    of the VM neuron's periodicity preservation vs. standard nonlinearities.

    Architecture::

        Input (B, T, F)
          → [Linear → ReLU → LayerNorm → Dropout] × num_layers  (encoder)
          → LSTM temporal backbone                                (sequence model)
          → Linear head from final hidden state                   (projection)

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Hidden dimension for encoder layers and LSTM.
        num_layers: Number of stacked Linear+ReLU+LayerNorm blocks (default 2).
        dropout: Dropout rate between encoder layers (default 0.0).
        output_size: Forecast horizon (number of output steps).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # --- Standard feedforward encoder ---
        # Replaces VM encoder: Linear+ReLU instead of VonMisesLinear.
        # ReLU is the simplest nonlinearity that does NOT preserve periodicity,
        # making it the ideal control for the VM neuron's cos-based activation.
        layers: List[nn.Module] = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.encoder = nn.Sequential(*layers)

        # --- LSTM temporal backbone (identical to VonMisesForecaster) ---
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # --- Output head (identical to VonMisesForecaster) ---
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        # Encode: apply Linear+ReLU per timestep (Sequential broadcasts over T)
        h = self.encoder(x)  # (B, T, hidden_size)

        # LSTM: capture temporal dependencies
        _, (h_n, _) = self.lstm(h)  # h_n: (1, B, hidden_size)
        h_final = h_n.squeeze(0)  # (B, hidden_size)

        return self.output_head(h_final)  # (B, output_size)


# ---------------------------------------------------------------------------
# Baseline 2: Transformer with sinusoidal positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Registered as a buffer (not a parameter) so it contributes zero
    learnable parameters and moves to the correct device automatically.

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length to precompute (default 512).
        dropout: Dropout applied after adding PE (default 0.1).
    """

    def __init__(
        self, d_model: int, max_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute PE matrix: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = 10000^(2i/d_model), computed in log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding and apply dropout.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer forecaster baseline.

    Uses sinusoidal positional encoding and multi-head self-attention to
    capture temporal dependencies.  Tests whether learned attention patterns
    can implicitly discover periodicity as effectively as the VM neuron's
    explicit periodic encoding.

    Architecture::

        Input (B, T, F)
          → Linear(F, d_model)                           (input projection)
          → + SinusoidalPositionalEncoding(d_model)      (position info)
          → TransformerEncoderLayer × num_layers          (self-attention)
          → Mean pooling over T                           (sequence aggregation)
          → Linear(d_model, horizon)                      (output head)

    Args:
        input_size: Number of input features per timestep.
        d_model: Transformer model dimension (default 64).
        nhead: Number of attention heads (default 4). Must divide d_model.
        num_layers: Number of TransformerEncoderLayer blocks (default 2).
        dim_feedforward: FFN inner dimension (default 128).
        dropout: Dropout rate (default 0.1).
        output_size: Forecast horizon (number of output steps).
        max_len: Maximum sequence length for positional encoding (default 512).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        # Project raw features to d_model dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Sinusoidal positional encoding (no learnable parameters)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_len, dropout=dropout
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # expects (B, T, d_model) — requires PyTorch >= 1.11
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head: mean-pooled representation → forecast horizon
        self.output_head = nn.Linear(d_model, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        # Project to d_model and add positional encoding
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_encoder(h)  # (B, T, d_model)

        # Self-attention across timesteps
        h = self.transformer_encoder(h)  # (B, T, d_model)

        # Mean pool over the sequence dimension — aggregates all timesteps
        # (alternative: use last timestep or CLS token; mean is simplest)
        h = h.mean(dim=1)  # (B, d_model)

        return self.output_head(h)  # (B, output_size)


# ---------------------------------------------------------------------------
# Baseline 3: Linear + hand-crafted features
# ---------------------------------------------------------------------------
class HandCraftedLinearForecaster(nn.Module):
    """Pure linear forecaster with hand-crafted periodic features.

    Flattens the full lookback window (which should include pre-computed
    sin/cos time features) and applies a single linear layer.  No hidden
    layers, no nonlinearities — this is the "manual engineering" baseline
    that the VM neuron should automate.

    The time features (sin(2*pi*t/24), cos(2*pi*t/24), etc.) must be
    pre-computed and concatenated to the input tensor before calling
    forward().  This class receives the augmented tensor directly.

    Architecture::

        Input (B, T, F)     F = original_features + crafted_features
          → Flatten          (B, T*F)
          → Linear(T*F, H)  (single linear regression)

    Args:
        input_size: Number of features per timestep (original + crafted).
        lookback: Lookback window length (needed for flattening dimension).
        output_size: Forecast horizon (number of output steps).
    """

    def __init__(
        self,
        input_size: int,
        lookback: int,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.output_size = output_size

        # Single linear layer — the entire model is a linear regression
        self.linear = nn.Linear(lookback * input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, lookback, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        # Flatten the lookback window into a single feature vector
        h = x.reshape(x.size(0), -1)  # (B, lookback * input_size)
        return self.linear(h)  # (B, output_size)
