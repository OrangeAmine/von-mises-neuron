"""Temporal Von Mises components: time-aware encoding and forecasting.

Extends the Von Mises neuron with explicit temporal features:
  1. Angular time extraction: timestamps → raw angles in [0, 2π)
  2. Frequency bank: frozen-κ multi-scale VM layer
  3. Phase fusion: time-conditioned Δμ shifts for sensor VM neurons
  4. Temporal conv bridges: depthwise Conv1d for local temporal mixing

See docs/superpowers/specs/2026-03-12-temporal-vm-encoder-design.md
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.von_mises_neuron import VonMisesLinear


def extract_angular_time_features(
    index: pd.DatetimeIndex,
    periods: Optional[List[int]] = None,
) -> np.ndarray:
    """Convert a DatetimeIndex into angular time features in [0, 2π).

    Each period maps a cyclic time component to an angle using a unified
    formula: θ = 2π · (fractional_position_within_period).

      - 24   → θ = 2π · hour / 24               (daily cycle)
      - 168  → θ = 2π · (dow·24 + hour) / 168    (weekly cycle, continuous)
      - 8766 → θ = 2π · day_of_year / 365.25     (annual cycle)

    The weekly formula uses continuous hours within the week (not discrete
    dayofweek) so the angle interpolates smoothly between days.

    Args:
        index: DatetimeIndex from a sensor DataFrame.
        periods: List of periods in hours. Default [24, 168, 8766].

    Returns:
        np.ndarray of shape (T, len(periods)), dtype float64, values in [0, 2π).
    """
    if periods is None:
        periods = [24, 168, 8766]

    # Fractional hour within the day (0.0 to ~23.99)
    hours = index.hour + index.minute / 60.0 + index.second / 3600.0

    features = []
    for p in periods:
        if p == 24:
            # Daily: fractional hour → angle
            theta = 2.0 * np.pi * hours / 24.0
        elif p == 168:
            # Weekly: continuous hours within the week → angle
            # dayofweek * 24 + fractional_hour gives position in [0, 168)
            weekly_hours = index.dayofweek * 24.0 + hours
            theta = 2.0 * np.pi * weekly_hours / 168.0
        elif p == 8766:
            # Annual: day-of-year → angle (day_of_year starts at 1)
            theta = 2.0 * np.pi * index.dayofyear / 365.25
        else:
            # Generic: compute hours since start, mod by period
            t_hours = (index - index[0]).total_seconds() / 3600.0
            theta = 2.0 * np.pi * (t_hours % p) / p

        # Wrap to [0, 2π) for safety
        theta = np.asarray(theta, dtype=np.float64) % (2.0 * np.pi)
        features.append(theta)

    return np.column_stack(features)


class FrequencyBankVonMises(nn.Module):
    """Von Mises layer with frozen κ organized into frequency bands.

    Neurons are divided equally among bands. Each band has a frozen
    concentration κ (stored as a buffer) that controls its temporal
    resolution: high κ → sharp/fast, low κ → broad/slow. Only the
    phase shifts μ are learned.

    This forces multi-scale decomposition: the network *must* use all
    timescales rather than collapsing to a single dominant one.

    Eq:  z_j = κ_band(j) · cos(θᵢ + μⱼᵢ)   (summed over i)

    Args:
        in_features: Number of angular input features (= len(time_periods)).
        out_features: Total number of neurons across all bands.
        freq_bands: List of κ values, one per band. Default [0.5, 2.0, 8.0].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        freq_bands: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if freq_bands is None:
            freq_bands = [0.5, 2.0, 8.0]

        self.in_features = in_features
        self.out_features = out_features
        self.freq_bands = freq_bands
        n_bands = len(freq_bands)

        # Allocate neurons to bands: equal split, remainder to first bands
        base_count = out_features // n_bands
        remainder = out_features % n_bands
        band_sizes = [base_count + (1 if i < remainder else 0) for i in range(n_bands)]

        # Build frozen κ buffer: (out_features,)
        kappa_values = []
        for band_kappa, size in zip(freq_bands, band_sizes):
            kappa_values.extend([band_kappa] * size)
        self.register_buffer("kappa", torch.tensor(kappa_values, dtype=torch.float32))

        # Learnable μ: phase shifts, (out_features, in_features)
        self.mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.mu, 0, 2 * math.pi)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the frequency bank.

        Args:
            x: Angular input, shape (..., in_features).

        Returns:
            Log-space VM response, shape (..., out_features).
        """
        # cos(θᵢ + μⱼᵢ) for all (neuron j, input i) pairs
        cos_term = torch.cos(x.unsqueeze(-2) + self.mu)  # (..., out, in)

        # κ: (out,) → (out, 1) for broadcasting over in_features
        kappa_expanded = self.kappa.unsqueeze(-1)  # (out, 1)

        # z_j = Σᵢ κ_band(j) · cos(θᵢ + μⱼᵢ)
        z = (kappa_expanded * cos_term).sum(dim=-1)  # (..., out)

        # No exp() here — the Δμ projection expects unbounded reals, not positive values
        return z + self.bias

    def get_band_indices(self) -> List[List[int]]:
        """Return neuron indices for each frequency band."""
        indices = []
        offset = 0
        n_bands = len(self.freq_bands)
        base_count = self.out_features // n_bands
        remainder = self.out_features % n_bands
        for i in range(n_bands):
            size = base_count + (1 if i < remainder else 0)
            indices.append(list(range(offset, offset + size)))
            offset += size
        return indices


class TemporalVonMisesEncoder(nn.Module):
    """Time-conditioned Von Mises encoder with optional temporal conv bridges.

    Architecture per layer::

        FrequencyBank(time_features)
          → Linear → Δμ (per-neuron phase shift)
        VonMisesLinear(sensor_features, phase_shift=Δμ)
          → LayerNorm
          → [DepthwiseConv1d → Residual]  (if use_temporal_conv)
          → Dropout

    Args:
        sensor_size: Number of sensor input features.
        time_size: Number of angular time features (= len(time_periods)).
        hidden_size: Working dimension for VM layers.
        num_vm_layers: Number of stacked VM layers.
        freq_bands: κ values for frequency bank bands.
        dropout: Dropout rate.
        use_temporal_conv: If True, add depthwise Conv1d + residual per layer.
        conv_kernel_size: Kernel size for depthwise Conv1d.
    """

    def __init__(
        self,
        sensor_size: int,
        time_size: int,
        hidden_size: int = 64,
        num_vm_layers: int = 2,
        freq_bands: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_temporal_conv: bool = True,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.sensor_size = sensor_size
        self.time_size = time_size
        self.hidden_size = hidden_size
        self.num_vm_layers = num_vm_layers
        self._use_temporal_conv = use_temporal_conv

        if freq_bands is None:
            freq_bands = [0.5, 2.0, 8.0]

        # Frequency bank: processes angular time features once
        self.freq_bank = FrequencyBankVonMises(
            in_features=time_size,
            out_features=hidden_size,
            freq_bands=freq_bands,
        )

        # Per-layer Δμ projections: map freq_bank output to phase shifts
        self.delta_mu_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_vm_layers)
        ])

        # VM layers for sensor features
        self.vm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_dim = sensor_size
        for _ in range(num_vm_layers):
            self.vm_layers.append(
                VonMisesLinear(
                    in_dim, hidden_size,
                    bias=True,
                    log_output=True,
                    init_kappa=1.0,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size

        # Optional temporal conv bridges
        if use_temporal_conv:
            self.convs = nn.ModuleList([
                nn.Conv1d(
                    hidden_size, hidden_size,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    groups=hidden_size,  # depthwise
                )
                for _ in range(num_vm_layers)
            ])
            # First-layer residual projection (sensor_size → hidden_size)
            if sensor_size != hidden_size:
                self.residual_proj = nn.Linear(sensor_size, hidden_size)
            else:
                self.residual_proj = None
        else:
            self.convs = None
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, sensor_size + time_size).

        Returns:
            Encoded tensor, shape (B, T, hidden_size).
        """
        # Split sensor and time features
        x_sensor = x[..., :self.sensor_size]
        x_time = x[..., self.sensor_size:]

        # Run frequency bank once on angular time features
        freq_out = self.freq_bank(x_time)  # (B, T, hidden_size)

        # Process sensor features through VM layers with time-conditioned phases
        h = x_sensor
        for i in range(self.num_vm_layers):
            residual = h

            # Compute time-conditioned phase shift for this layer
            # (B, T, hidden_size) → unsqueeze → (B, T, hidden_size, 1)
            delta_mu = self.delta_mu_projections[i](freq_out).unsqueeze(-1)

            # VM layer with phase conditioning
            h = self.vm_layers[i](h, phase_shift=delta_mu)
            h = self.layer_norms[i](h)

            # Optional temporal conv + residual
            if self._use_temporal_conv and self.convs is not None:
                # Conv1d expects (B, C, T)
                h_conv = h.transpose(1, 2)
                h_conv = F.gelu(self.convs[i](h_conv))
                h = h_conv.transpose(1, 2)

                # Residual: project first layer if dimensions mismatch
                if i == 0 and self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                h = h + residual

            h = self.dropout(h)

        return h
