"""Matched calendar branches for the prospective VM-N04 dynamics study.

The sensor branch is an ordinary MLP; its real-valued inputs are not angular.
Only the known future daily clock coordinates have a 2*pi periodic guarantee.
Peak-normalized circular kernels reuse the frozen, versioned kernel API.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Literal

import torch
from torch import Tensor, nn

from src.circular_layers_v2 import PeakCircularKernelBankV1

CalendarMode = Literal["none", "fourier", "fixed", "learned"]
CALENDAR_MODES = ("none", "fourier", "fixed", "learned")


def _positive_integer(value: int, name: str) -> None:
    """Reject ambiguous/nonpositive dimensions before allocating parameters."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class CalendarBasisV1(nn.Module):
    """Return 2H Fourier or K peak-normalized scalar daily-clock features.

    Args:
        mode: Fourier, uniformly fixed kernels, or learned kernel geometry.
        num_features: Positive even feature count; H is half this count.
        init_kappa: Positive initial kernel concentration (not a frequency).
        device: Optional initial device.
        dtype: Floating initial dtype; defaults to PyTorch's default dtype.

    The kernel centers start uniformly over [0,2*pi). Learning permits centers
    and independent softplus concentrations to adapt; it adds 2K parameters.
    All modes use exactly K feature coordinates, which does not imply equal
    fitted parameter counts. Construction consumes no random numbers.
    """

    def __init__(
        self,
        mode: Literal["fourier", "fixed", "learned"],
        num_features: int = 32,
        init_kappa: float = 32.0,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mode not in CALENDAR_MODES[1:]:
            raise ValueError("basis mode must be fourier, fixed, or learned")
        _positive_integer(num_features, "num_features")
        if num_features % 2:
            raise ValueError("num_features must be even to match the Fourier count")
        if not math.isfinite(init_kappa) or init_kappa <= 0:
            raise ValueError("init_kappa must be positive and finite")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating")
        self.mode, self.num_features = mode, int(num_features)
        self.init_kappa = float(init_kappa)
        self.bank: PeakCircularKernelBankV1 | None = None
        if mode == "fourier":
            self.register_buffer("frequencies", torch.arange(
                1, num_features // 2 + 1, dtype=dtype, device=device))
        else:
            centers = (torch.arange(num_features, dtype=dtype, device=device)
                       * (2 * math.pi / num_features))[:, None]
            concentrations = torch.full((num_features,), init_kappa,
                                        dtype=dtype, device=device)
            self.bank = PeakCircularKernelBankV1(
                1, num_features, centers=centers, concentrations=concentrations,
                learn_centers=mode == "learned",
                learn_concentrations=mode == "learned", width_sharing="per_kernel")

    def forward(self, angles: Tensor) -> Tensor:
        """Map floating ``(...,)`` raw angles to ``(..., num_features)``."""
        if not isinstance(angles, Tensor) or not angles.is_floating_point():
            raise TypeError("angles must be a floating tensor")
        reference = self.frequencies if self.bank is None else self.bank.centers
        if angles.dtype != reference.dtype or angles.device != reference.device:
            raise ValueError("angles must share the basis dtype and device")
        if self.bank is not None:
            return self.bank(angles[..., None])
        phase = angles[..., None] * self.frequencies
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


class DynamicsForecasterV1(nn.Module):
    """Direct persistence-residual MLP with one optional future-clock branch.

    Args:
        calendar: Which calendar basis to use; ``none`` ignores future angles.
        lookback: Number of historical rows, fixed by construction.
        input_size: Number of historical channels. The study uses normalized
            causally filled sensor, observed mask, and observation age.
        horizon: Number of future predictions.
        hidden_size: Width of both sensor MLP hidden layers.
        target_index: Historical channel whose last available value anchors
            the residual forecast. Input and output target units must match.
        num_features: Calendar basis width; 32 means 16 Fourier harmonics.
        init_kappa: Fixed or initial learned kernel concentration.
        device: Optional initial device.
        dtype: Optional initial floating dtype.

    Sensor flattening -> Linear -> GELU -> Linear -> GELU -> Linear(horizon)
    is identical across arms. A shared scalar linear calendar readout is
    evaluated separately at every known future timestamp. There are no future
    measured sensors or realized attendance arguments. Seed immediately before
    constructing each arm to pair every sensor weight and calendar readout:
    deterministic basis construction consumes no random numbers. Both output
    weight matrices begin small but nonzero, allowing initial gradient flow.
    """

    version = "DynamicsForecasterV1"

    def __init__(
        self,
        calendar: CalendarMode = "none",
        lookback: int = 24,
        input_size: int = 3,
        horizon: int = 16,
        hidden_size: int = 32,
        target_index: int = 0,
        num_features: int = 32,
        init_kappa: float = 32.0,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if calendar not in CALENDAR_MODES:
            raise ValueError(f"calendar must be one of {CALENDAR_MODES}")
        for name, value in (("lookback", lookback), ("input_size", input_size),
                            ("horizon", horizon), ("hidden_size", hidden_size),
                            ("num_features", num_features)):
            _positive_integer(value, name)
        if (isinstance(target_index, bool) or not isinstance(target_index, Integral)
                or not 0 <= target_index < input_size):
            raise ValueError("target_index must index a historical input channel")
        if num_features % 2:
            raise ValueError("num_features must be even")
        if not math.isfinite(init_kappa) or init_kappa <= 0:
            raise ValueError("init_kappa must be positive and finite")
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError("dtype must be floating")
        self.calendar = calendar
        self.lookback, self.input_size = int(lookback), int(input_size)
        self.horizon = self.output_size = int(horizon)
        self.hidden_size, self.target_index = int(hidden_size), int(target_index)
        self.num_features, self.init_kappa = int(num_features), float(init_kappa)
        kwargs = {"device": device, "dtype": dtype}
        self.sensor_backbone = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(lookback * input_size, hidden_size, **kwargs),
            nn.GELU(), nn.Linear(hidden_size, hidden_size, **kwargs), nn.GELU())
        self.sensor_head = nn.Linear(hidden_size, horizon, **kwargs)
        with torch.no_grad():
            self.sensor_head.weight.mul_(0.01)
            self.sensor_head.bias.zero_()
        self.calendar_basis: CalendarBasisV1 | None = None
        self.calendar_readout: nn.Linear | None = None
        if calendar != "none":
            self.calendar_basis = CalendarBasisV1(
                calendar, num_features, init_kappa, **kwargs)
            self.calendar_readout = nn.Linear(num_features, 1, **kwargs)
            with torch.no_grad():
                self.calendar_readout.weight.mul_(0.01)
                self.calendar_readout.bias.zero_()

    def forward(self, history: Tensor, future_angles: Tensor | None = None) -> Tensor:
        """Return ``(batch, horizon)`` forecasts from causal history and clocks.

        Args:
            history: Floating ``(batch, lookback, input_size)`` tensor.
            future_angles: Raw daily angles shaped ``(batch, horizon)`` or
                ``(batch, horizon, 1)``. Required for calendar-enabled arms.

        Returns:
            Direct multi-step predictions in the normalized target units.
        """
        if not isinstance(history, Tensor) or not history.is_floating_point():
            raise TypeError("history must be a floating tensor")
        if history.ndim != 3 or history.shape[1:] != (self.lookback, self.input_size):
            raise ValueError("history must have shape (batch, lookback, input_size)")
        if history.shape[0] < 1:
            raise ValueError("history batch must be nonempty")
        if (history.dtype != self.sensor_head.weight.dtype
                or history.device != self.sensor_head.weight.device):
            raise ValueError("history must share the model dtype and device")
        prediction = (history[:, -1, self.target_index:self.target_index + 1]
                      + self.sensor_head(self.sensor_backbone(history)))
        if self.calendar_basis is not None:
            if not isinstance(future_angles, Tensor) or not future_angles.is_floating_point():
                raise TypeError("future_angles must be a floating tensor")
            if future_angles.ndim == 3 and future_angles.shape[-1] == 1:
                future_angles = future_angles[..., 0]
            if future_angles.shape != (history.shape[0], self.horizon):
                raise ValueError("future_angles must have shape (batch, horizon) or (batch, horizon, 1)")
            prediction = prediction + self.calendar_readout(
                self.calendar_basis(future_angles)).squeeze(-1)
        return prediction

    def get_config(self) -> dict[str, Any]:
        """Return reconstructable arguments and honest capacity counts."""
        sensor_parameters = sum(p.numel() for module in
                                (self.sensor_backbone, self.sensor_head)
                                for p in module.parameters())
        basis_parameters = (0 if self.calendar_basis is None else
                            sum(p.numel() for p in self.calendar_basis.parameters()))
        readout_parameters = (0 if self.calendar_readout is None else
                              sum(p.numel() for p in self.calendar_readout.parameters()))
        return {"class": self.version, "calendar": self.calendar,
                "lookback": self.lookback, "input_size": self.input_size,
                "horizon": self.horizon, "hidden_size": self.hidden_size,
                "target_index": self.target_index, "num_features": self.num_features,
                "init_kappa": self.init_kappa,
                "calendar_feature_count": 0 if self.calendar == "none" else self.num_features,
                "fourier_harmonics": self.num_features // 2 if self.calendar == "fourier" else 0,
                "sensor_parameters": sensor_parameters,
                "basis_parameters": basis_parameters,
                "calendar_readout_parameters": readout_parameters,
                "trainable_parameters": sensor_parameters + basis_parameters + readout_parameters}


class StandardDynamicsAdapterV1(nn.Module):
    """Adapt the unchanged hourly benchmark's sensor/angle tensor to this model.

    The standard pipeline provides four z-scored sensors then daily/weekly raw
    angles. All observations are present: a common observed flag of one and age
    of zero are appended to the four historical sensor channels. Only the last
    known daily angle determines future clock coordinates at hourly increments;
    weekly history is unused. This cadence extension does not infer DST jumps:
    it preserves the legacy local-clock condition as a distinct engineering
    limitation. The VM-N04 study supplies explicit UTC future timestamps.
    """

    def __init__(self, calendar: CalendarMode, *, lookback: int = 72,
                 horizon: int = 12, sensor_size: int = 4, target_index: int = 2,
                 cadence_hours: float = 1.0, hidden_size: int = 32) -> None:
        super().__init__()
        _positive_integer(sensor_size, "sensor_size")
        if not math.isfinite(cadence_hours) or cadence_hours <= 0:
            raise ValueError("cadence_hours must be positive and finite")
        if isinstance(target_index, bool) or not isinstance(target_index, Integral) or not 0 <= target_index < sensor_size:
            raise ValueError("target_index must index a sensor")
        self.sensor_size, self.cadence_hours = int(sensor_size), float(cadence_hours)
        self.body = DynamicsForecasterV1(
            calendar, lookback=lookback, input_size=sensor_size + 2,
            horizon=horizon, hidden_size=hidden_size, target_index=target_index)
        self.register_buffer("future_phase_increments", torch.arange(
            1, horizon + 1, dtype=self.body.sensor_head.weight.dtype)
            * (2 * math.pi * cadence_hours / 24))

    def forward(self, x: Tensor) -> Tensor:
        """Map standard normalized sensors + daily/weekly raw angles to forecasts."""
        if not isinstance(x, Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating tensor")
        if x.ndim != 3 or x.shape[-1] != self.sensor_size + 2:
            raise ValueError("x must contain sensors followed by daily and weekly raw angles")
        sensors = x[..., :self.sensor_size]
        observed = torch.ones_like(sensors[..., :1])
        age = torch.zeros_like(observed)
        history = torch.cat((sensors, observed, age), dim=-1)
        future = x[:, -1, self.sensor_size:self.sensor_size + 1] + self.future_phase_increments
        return self.body(history, future)
