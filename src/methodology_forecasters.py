"""Explicit engineering adapters that leave frozen forecasting sources intact.

These adapters establish integration and conversion parity. They are not stage
A/B experimental arms and are not validated by numerical parity alone.
"""

from __future__ import annotations

import copy

from torch import Tensor, nn
from torch.nn import functional as F

from src.circular_layers_v2 import DirectCalendarPhaseV1, PolarCircularLinearV2
from src.temporal import TemporalVonMisesEncoder, TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster, VonMisesLinear


class PolarForecasterV2(nn.Module):
    """Base legacy architecture with explicitly converted numerical VM layers.

    Construct with ``from_legacy``; no legacy checkpoint is implicitly migrated.
    The converter preserves every non-VM state and avoids RNG draws. Use the
    same training RNG after conversion to pair dropout with the source model.
    """

    version = "PolarForecasterV2"

    @classmethod
    def from_legacy(cls, reference: VonMisesForecaster, *, implementation: str = "factored") -> PolarForecasterV2:
        """Clone a base forecaster and replace only its numerical circular layers."""
        if not isinstance(reference, VonMisesForecaster):
            raise TypeError("reference must be VonMisesForecaster")
        result = cls()
        result.input_size, result.hidden_size = reference.input_size, reference.hidden_size
        result.num_vm_layers, result.output_size = reference.num_vm_layers, reference.output_size
        result.vm_encoder = nn.Sequential(*[
            PolarCircularLinearV2.from_legacy(layer, implementation=implementation)
            if isinstance(layer, VonMisesLinear) else copy.deepcopy(layer)
            for layer in reference.vm_encoder
        ])
        result.vm_encoder.training = reference.vm_encoder.training
        result.lstm = copy.deepcopy(reference.lstm)
        if result.lstm is not None:
            result.lstm.flatten_parameters()
        result.output_head = copy.deepcopy(reference.output_head)
        result.training = reference.training
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Map (batch, lookback, features) to (batch, horizon)."""
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError("input must have shape (batch, lookback, input_size)")
        hidden = self.vm_encoder(x)
        if self.lstm is None:
            final = hidden.mean(dim=1)
        else:
            _, (state, _) = self.lstm(hidden)
            final = state.squeeze(0)
        return self.output_head(final)

    def get_vm_layers(self) -> list[PolarCircularLinearV2]:
        """Return converted layers for numerical diagnostics."""
        return [layer for layer in self.vm_encoder if isinstance(layer, PolarCircularLinearV2)]


class DirectCalendarEncoderV1(nn.Module):
    """Legacy sensor/backbone branches with direct calendar phase projections.

    Time-angle periodicity survives conversion. A raw-sensor residual retains
    its original absence of sensor periodicity; the converter does not remove it.
    """

    version = "DirectCalendarEncoderV1"

    @classmethod
    def from_legacy(cls, reference: TemporalVonMisesEncoder, *, implementation: str = "factored") -> DirectCalendarEncoderV1:
        """Clone all non-bank branches and collapse the bank plus projections."""
        if not isinstance(reference, TemporalVonMisesEncoder):
            raise TypeError("reference must be TemporalVonMisesEncoder")
        result = cls()
        for name in ("sensor_size", "time_size", "hidden_size", "num_vm_layers", "_use_temporal_conv", "_use_time_conditioning"):
            setattr(result, name, getattr(reference, name))
        result.calendar = DirectCalendarPhaseV1.from_legacy_encoder(reference) if reference.freq_bank is not None else None
        result.vm_layers = nn.ModuleList([PolarCircularLinearV2.from_legacy(layer, implementation=implementation) for layer in reference.vm_layers])
        result.vm_layers.training = reference.vm_layers.training
        for name in ("layer_norms", "convs", "residual_proj", "dropout"):
            setattr(result, name, copy.deepcopy(getattr(reference, name)))
        result.training = reference.training
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Encode sensors followed by raw angular calendar inputs."""
        if x.ndim != 3 or x.shape[-1] != self.sensor_size + self.time_size:
            raise ValueError("input must have shape (batch, lookback, sensor_size + time_size)")
        hidden = x[..., :self.sensor_size]
        shifts = self.calendar(x[..., self.sensor_size:]) if self.calendar is not None else None
        for index, layer in enumerate(self.vm_layers):
            residual = hidden
            phase_shift = None if shifts is None else shifts[index].unsqueeze(-1)
            hidden = self.layer_norms[index](layer(hidden, phase_shift=phase_shift))
            if self._use_temporal_conv and self.convs is not None:
                hidden = F.gelu(self.convs[index](hidden.transpose(1, 2))).transpose(1, 2)
                if index == 0 and self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                hidden = hidden + residual
            hidden = self.dropout(hidden)
        return hidden


class DirectCalendarForecasterV1(nn.Module):
    """Explicit complete-forecaster conversion for engineering checks."""

    version = "DirectCalendarForecasterV1"

    @classmethod
    def from_legacy(cls, reference: TemporalVonMisesForecaster, *, implementation: str = "factored") -> DirectCalendarForecasterV1:
        """Copy independent source states without RNG draws or hidden reinit."""
        if not isinstance(reference, TemporalVonMisesForecaster):
            raise TypeError("reference must be TemporalVonMisesForecaster")
        result = cls()
        for name in ("sensor_size", "time_periods", "hidden_size", "output_size", "_use_lstm"):
            setattr(result, name, copy.deepcopy(getattr(reference, name)))
        result.encoder = DirectCalendarEncoderV1.from_legacy(reference.encoder, implementation=implementation)
        result.lstm = copy.deepcopy(reference.lstm)
        if result.lstm is not None:
            result.lstm.flatten_parameters()
        result.output_head = copy.deepcopy(reference.output_head)
        result.training = reference.training
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Map (batch, lookback, sensors+angles) to (batch, horizon)."""
        hidden = self.encoder(x)
        if self.lstm is None:
            final = hidden.mean(dim=1)
        else:
            _, (state, _) = self.lstm(hidden)
            final = state.squeeze(0)
        return self.output_head(final)

    def get_vm_layers(self) -> list[PolarCircularLinearV2]:
        """Return converted sensor layers in order."""
        return list(self.encoder.vm_layers)
