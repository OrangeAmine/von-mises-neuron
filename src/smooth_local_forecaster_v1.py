"""Small forecasting adapter for VM-N01 engineering integration only.

The scalar basis is shared across sensors. A learned projection mixes its
per-sensor features before an LSTM; this model is not a scientific study arm.
Standardized sensor values are treated as angles for the legacy benchmark,
which does not establish physical circularity of those measurements.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.smooth_local_v1 import SmoothLocalBasisV1


class SmoothLocalForecasterV1(nn.Module):
    """Encode each sensor with M3/K10 features, then forecast with one LSTM.

    Args:
        input_size: Number of normalized sensor columns.
        output_size: Forecast horizon.
        projection_size: Number of projected per-timestep features.
        hidden_size: LSTM hidden size.
        dropout: Dropout probability after layer normalization.

    The adapter preserves a separate ``2*pi`` shift in each input sensor in
    evaluation mode; there are no affine angle projections or raw residuals.
    The integration protocol fixes the constructor defaults for its fit.
    """

    version = "SmoothLocalForecasterV1"

    def __init__(self, input_size: int = 4, output_size: int = 12,
                 projection_size: int = 32, hidden_size: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        for name, value in (("input_size", input_size), ("output_size", output_size),
                            ("projection_size", projection_size), ("hidden_size", hidden_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.input_size = input_size
        self.output_size = output_size
        self.basis = SmoothLocalBasisV1(
            harmonics=3, kernels=10, adaptive=True, rotation=0.0,
            concentration=4.0, dtype=torch.float32)
        self.projection = nn.Linear(input_size * 16, projection_size)
        self.normalization = nn.LayerNorm(projection_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(projection_size, hidden_size, num_layers=1,
                            batch_first=True)
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(batch, lookback, sensors)`` to ``(batch, horizon)``."""
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] == 0:
            raise ValueError("input must have shape (batch, positive lookback, input_size)")
        features = self.basis(x.unsqueeze(-1)).flatten(start_dim=-2)
        encoded = self.dropout(self.normalization(self.projection(features)))
        _, (hidden, _) = self.lstm(encoded)
        return self.output_head(hidden[0])
