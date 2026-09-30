"""Versioned Fourier backbones and peak-normalized circular corrections.

These features implement the VM-N01 equation exactly. They reuse the frozen
kernel and ridge APIs without modifying the v1/v2 scientific implementations.
Raw features have no fitted scaling; component roles are not identifiable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.circular_layers_v2 import PeakCircularKernelBankV1
from src.locality_models import solve_readout


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _scalar_angles(theta: Tensor, reference: Tensor) -> Tensor:
    """Accept scalar, N-vector, or arbitrary leading dimensions ending in one."""
    if not isinstance(theta, Tensor) or theta.dtype != reference.dtype or theta.device != reference.device:
        raise ValueError("angles and basis state must share floating dtype and device")
    if theta.ndim >= 2:
        if theta.shape[-1] != 1:
            raise ValueError("scalar angle tensors must end in a singleton dimension")
        return theta[..., 0]
    return theta


class SmoothLocalBasisV1(nn.Module):
    """Raw integer Fourier features followed by localized circular kernels.

    Args:
        harmonics: Number M of positive integer harmonics, beginning at one.
        kernels: Number K of kernels. At least one feature must be present.
        adaptive: Whether both kernel centers and softplus widths are learned.
        rotation: Shared additive rotation of the initial evenly spaced centers.
        concentration: Positive effective initial concentration for a single band.
        concentration_bands: Optional fixed multiscale concentrations; K must be
            divisible by the number of bands. Centers are repeated per band.
        dtype: Float32 for engineering adapters or float64 for static fitting.
        device: Destination device, with CPU default.

    Inputs shaped (..., 1) yield (..., 2M+K). Scalar and N-vector inputs also
    work. The stable kernel is exp(-2*kappa*sin((theta-center)/2)**2), equal to
    exp(kappa*(cos(theta-center)-1)). No random numbers are consumed.
    """

    version = "SmoothLocalBasisV1"

    def __init__(self, harmonics: int, kernels: int, adaptive: bool,
                 rotation: float = 0., concentration: float = 4.,
                 concentration_bands: Sequence[float] | None = None,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        self.harmonics = _count(harmonics, "harmonics")
        self.num_kernels = _count(kernels, "kernels")
        if not isinstance(adaptive, bool):
            raise ValueError("adaptive must be boolean")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if not math.isfinite(float(rotation)):
            raise ValueError("rotation must be finite")
        if self.harmonics * 2 + self.num_kernels == 0:
            raise ValueError("at least one Fourier or kernel feature is required")
        self.adaptive = adaptive
        self.basis_count = 2 * self.harmonics + self.num_kernels
        self.register_buffer("frequencies", torch.arange(1, harmonics + 1, dtype=dtype, device=device))
        self.bank: PeakCircularKernelBankV1 | None = None
        if kernels:
            bands = [concentration] if concentration_bands is None else list(concentration_bands)
            if not bands or kernels % len(bands) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in bands):
                raise ValueError("positive finite concentration bands must divide the kernel count")
            if adaptive and concentration_bands is not None:
                raise ValueError("multiscale bands are reserved for fixed bases")
            n_centers = kernels // len(bands)
            centers = -math.pi + 2 * math.pi * torch.arange(n_centers, dtype=dtype, device=device) / n_centers
            centers = (centers + float(rotation)).repeat_interleave(len(bands))[:, None]
            widths = torch.tensor(bands, dtype=dtype, device=device).repeat(n_centers)
            self.bank = PeakCircularKernelBankV1(
                1, kernels, centers=centers, concentrations=widths,
                learn_centers=adaptive, learn_concentrations=adaptive,
                width_sharing="per_kernel")
        elif concentration_bands is not None:
            raise ValueError("concentration bands require kernels")

    def forward(self, theta: Tensor) -> Tensor:
        """Evaluate raw features in cosine, sine, then kernel column order."""
        angles = _scalar_angles(theta, self.frequencies)
        parts = []
        if self.harmonics:
            phase = angles[..., None] * self.frequencies
            parts.extend((phase.cos(), phase.sin()))
        if self.bank is not None:
            parts.append(self.bank(angles[..., None]))
        return torch.cat(parts, dim=-1)

    def get_config(self) -> dict[str, Any]:
        """Return architecture and feature-scaling policy without fit state."""
        return {"class": self.version, "harmonics": self.harmonics,
                "kernels": self.num_kernels, "adaptive": self.adaptive,
                "features": self.basis_count, "feature_scaling": "raw",
                "width_parameterization": "softplus"}


class LineRBFBasisV1(nn.Module):
    """Unrestricted theta/pi plus fixed ordinary line Gaussian RBFs."""

    def __init__(self, count: int = 15, width: float = .45, *,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        if _count(count, "count") < 1 or not math.isfinite(width) or width <= 0:
            raise ValueError("line RBF count and width must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.basis_count = count + 1
        self.register_buffer("centers", torch.linspace(-math.pi, math.pi, count, dtype=dtype, device=device))
        self.register_buffer("width", torch.tensor(width, dtype=dtype, device=device))

    def forward(self, theta: Tensor) -> Tensor:
        """Evaluate line features without wrapping the angle."""
        angles = _scalar_angles(theta, self.centers)
        kernels = (-.5 * ((angles[..., None] - self.centers) / self.width).square()).exp()
        return torch.cat((angles[..., None] / math.pi, kernels), dim=-1)


class SmoothLocalRegressorV1(nn.Module):
    """Scalar regressor with explicit jointly optimized or solved readouts."""

    version = "SmoothLocalRegressorV1"

    def __init__(self, specification: Mapping[str, Any], rotation: float = 0.,
                 concentration: float = 4., *, dtype: torch.dtype = torch.float64,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        self.specification = dict(specification)
        self.model_id = str(specification["id"])
        self.kind = str(specification["kind"])
        if self.kind not in ("ridge", "joint", "vp"):
            raise ValueError("kind must be ridge, joint, or vp")
        if specification.get("unrestricted", False):
            if self.kind != "ridge":
                raise ValueError("line RBF comparator must use ridge")
            self.basis = LineRBFBasisV1(specification.get("line_rbf_count", 15),
                                        specification.get("line_rbf_width", .45), dtype=dtype, device=device)
        else:
            self.basis = SmoothLocalBasisV1(
                specification["harmonics"], specification["kernels"], self.kind != "ridge",
                rotation=rotation, concentration=concentration,
                concentration_bands=specification.get("concentration_bands"), dtype=dtype, device=device)
        self.basis_count = self.basis.basis_count
        geometry_count = sum(p.numel() for p in self.basis.parameters())
        if self.kind != "ridge" and not geometry_count:
            raise ValueError("adaptive fitting requires trainable kernel geometry")
        self.fitted_parameter_count = self.basis_count + 1 + geometry_count
        if self.basis_count != specification["features"] or self.fitted_parameter_count != specification["parameters"]:
            raise ValueError("registered feature/parameter counts do not match the constructed model")
        weight = torch.zeros((self.basis_count, 1), dtype=dtype, device=device)
        bias = torch.zeros(1, dtype=dtype, device=device)
        if self.kind == "joint":
            self.readout_weight, self.readout_bias = nn.Parameter(weight), nn.Parameter(bias)
        else:
            self.register_buffer("readout_weight", weight)
            self.register_buffer("readout_bias", bias)

    def features(self, theta: Tensor) -> Tensor:
        """Return the unscaled design matrix."""
        return self.basis(theta)

    def forward(self, theta: Tensor) -> Tensor:
        """Apply the signed linear readout and unrestricted intercept."""
        return self.features(theta) @ self.readout_weight + self.readout_bias

    @torch.no_grad()
    def refresh_readout(self, features: Tensor, target: Tensor, penalty: float) -> None:
        """Set the exact float64 ridge readout without replacing leaf tensors."""
        weight, bias = solve_readout(features, target, penalty)
        self.readout_weight.copy_(weight)
        self.readout_bias.copy_(bias)

    def snapshot(self) -> dict[str, Any]:
        """Capture independent CPU tensors sufficient for immutable replay."""
        return {"version": self.version, "specification": dict(self.specification),
                "basis_count": self.basis_count, "fitted_parameter_count": self.fitted_parameter_count,
                "optimized_parameter_count": sum(p.numel() for p in self.parameters()),
                "state_dict": {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}}


def make_model(spec: Mapping[str, Any], rotation: float = 0., concentration: float = 4., *,
               dtype: torch.dtype = torch.float64,
               device: torch.device | str | None = None) -> SmoothLocalRegressorV1:
    """Construct one registered static model without consuming RNG state."""
    return SmoothLocalRegressorV1(spec, rotation, concentration, dtype=dtype, device=device)


@torch.no_grad()
def predict_smooth_local_state(state: Mapping[str, Any], x: Any, *,
                               device: torch.device | str = "cpu") -> Tensor:
    """Replay a saved state, without optimization, refitting, or random draws."""
    weights = state["state_dict"]["readout_weight"]
    dtype = weights.dtype
    model = make_model(state["specification"], dtype=dtype, device=device)
    model.load_state_dict(state["state_dict"])
    result = model(torch.as_tensor(x, dtype=dtype, device=device))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("saved-state predictions are nonfinite")
    return result
