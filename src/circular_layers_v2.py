"""Versioned circular representations with explicit numerical semantics.

Polar and Cartesian sums describe the same sine/cosine function class (except
finite polar coordinates cannot attain exactly zero amplitude). Peak kernels
are Gaussian RBFs on circle embeddings, not probability densities. Exponential
responses may overflow; localized features may underflow. Neither is clipped.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def inverse_softplus(value: Tensor) -> Tensor:
    """Invert softplus for representable positive finite floating tensors.

    Args:
        value: Requested effective concentrations, already in the target dtype.

    Returns:
        Finite raw coordinates whose softplus remains positive and finite.

    Raises:
        ValueError: If a value is invalid or cannot be represented in its dtype.
    """
    if not value.is_floating_point():
        raise ValueError("concentrations must use a floating dtype")
    if not bool(torch.isfinite(value).all() and (value > 0).all()):
        raise ValueError("concentrations must be positive finite representable values")
    raw = value + torch.log(-torch.expm1(-value))
    recovered = F.softplus(raw)
    if not bool(torch.isfinite(raw).all() and torch.isfinite(recovered).all()
                and (recovered > 0).all()):
        raise ValueError("concentrations are not representable in the selected dtype")
    return raw


def _validate_input(x: Tensor, in_features: int) -> None:
    if x.ndim < 1 or x.shape[-1] != in_features:
        raise ValueError(f"input must have shape (..., {in_features})")
    if not x.is_floating_point():
        raise ValueError("input must use a floating dtype")


def _validate_shift(x: Tensor, shift: Tensor | None, out_features: int) -> None:
    if shift is None:
        return
    if shift.ndim < 2 or shift.shape[-2:] != (out_features, 1):
        raise ValueError("phase_shift must have shape (..., out_features, 1); per-edge shifts are unsupported")
    try:
        leading = torch.broadcast_shapes(x.shape[:-1], shift.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("phase_shift leading dimensions do not broadcast to input") from exc
    if leading != x.shape[:-1]:
        raise ValueError("phase_shift must not expand the input leading dimensions")
    if shift.dtype != x.dtype or shift.device != x.device:
        raise ValueError("phase_shift must match input dtype and device")


def _response(z: Tensor, response: str, bias: Tensor | None) -> Tensor:
    result = z if response == "sum" else z.exp()
    return result if bias is None else result + bias


class PolarCircularLinearV2(nn.Module):
    """Compute sums or exponentials of ``kappa*cos(x+phase)``.

    ``kappa`` is softplus(raw_kappa), not exp(raw_kappa). Uniform initial phase
    draws make the pre-bias variance sum(kappa**2)/2. ``unit_variance`` scales
    every edge by sqrt(2/in_features); it does not constrain trained variance.

    Args:
        in_features: Number of angular inputs.
        out_features: Number of output sums.
        bias: Add a bias after the response.
        response: Trigonometric sum or its unnormalized exponential.
        init_kappa: Positive finite requested concentration.
        init_policy: Constant amplitudes or fan-in variance scaling.
        implementation: Factored matrix products or broadcast reference.
        device: Parameter device.
        dtype: Floating parameter dtype.
    """

    version = "PolarCircularLinearV2"

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 response: Literal["sum", "exp"] = "sum", init_kappa: float = 1.0,
                 init_policy: Literal["constant", "unit_variance"] = "constant",
                 implementation: Literal["factored", "reference"] = "factored",
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        _positive_dimension(in_features, "in_features")
        _positive_dimension(out_features, "out_features")
        if response not in ("sum", "exp"):
            raise ValueError("response must be 'sum' or 'exp'")
        if init_policy not in ("constant", "unit_variance"):
            raise ValueError("init_policy must be 'constant' or 'unit_variance'")
        if implementation not in ("factored", "reference"):
            raise ValueError("implementation must be 'factored' or 'reference'")
        if not math.isfinite(init_kappa) or init_kappa <= 0:
            raise ValueError("init_kappa must be positive and finite")
        self.in_features, self.out_features = in_features, out_features
        self.response, self.init_policy = response, init_policy
        self.implementation = implementation
        self.requested_init_kappa = float(init_kappa)
        target = init_kappa * (math.sqrt(2 / in_features) if init_policy == "unit_variance" else 1)
        concentration = torch.tensor(target, device=device, dtype=dtype or torch.get_default_dtype())
        raw = inverse_softplus(concentration)
        self.effective_init_kappa: float | None = float(F.softplus(raw).detach().cpu())
        self.phase = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=concentration.dtype))
        self.raw_kappa = nn.Parameter(raw.expand(out_features, in_features).clone())
        nn.init.uniform_(self.phase, 0, 2 * math.pi)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=concentration.dtype)) if bias else None

    @property
    def kappa(self) -> Tensor:
        """Return differentiable positive effective amplitudes."""
        return F.softplus(self.raw_kappa)

    def preferred_centers(self) -> Tensor:
        """Return preferred input angles, ``-phase modulo 2*pi``."""
        return torch.remainder(-self.phase, 2 * math.pi)

    def coefficients(self) -> tuple[Tensor, Tensor]:
        """Return differentiable Cartesian cosine and sine coefficients."""
        kappa = self.kappa
        return kappa * self.phase.cos(), -kappa * self.phase.sin()

    def forward(self, x: Tensor, phase_shift: Tensor | None = None) -> Tensor:
        """Map ``(..., D)`` to ``(..., O)``; offsets end in ``(O, 1)``.

        Offset leading dimensions may broadcast within the input's leading
        shape, but cannot add or expand input axes. General per-edge offsets
        are rejected. Bias follows exponentiation when response is ``exp``.
        """
        _validate_input(x, self.in_features)
        _validate_shift(x, phase_shift, self.out_features)
        if self.implementation == "reference":
            phase = self.phase if phase_shift is None else self.phase + phase_shift
            z = (self.kappa * (x.unsqueeze(-2) + phase).cos()).sum(-1)
        elif self.implementation == "factored":
            a, b = self.coefficients()
            cosine, sine = x.cos(), x.sin()
            z = F.linear(cosine, a) + F.linear(sine, b)
            if phase_shift is not None:
                q = F.linear(sine, a) - F.linear(cosine, b)
                delta = phase_shift.squeeze(-1)
                z = delta.cos() * z - delta.sin() * q
        else:
            raise ValueError("unknown implementation")
        return _response(z, self.response, self.bias)

    @classmethod
    def from_legacy(cls, reference: nn.Module, *, implementation: str = "factored") -> PolarCircularLinearV2:
        """Clone actual legacy parameters without reinitializing or drawing RNG.

        Legacy initialization defects are preserved by copying raw coordinates.
        The source, its mode and its gradient flags are unchanged. Old state keys
        are deliberately accepted only here, never by ordinary load_state_dict.
        """
        from src.von_mises_neuron import VonMisesLinear

        if not isinstance(reference, VonMisesLinear):
            raise TypeError("reference must be a legacy VonMisesLinear")
        if implementation not in ("factored", "reference"):
            raise ValueError("unknown implementation")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = reference.in_features, reference.out_features
        result.response = "sum" if reference.log_output else "exp"
        result.implementation, result.init_policy = implementation, "legacy_copy"
        result.requested_init_kappa = float(reference._init_kappa_target)
        # The source may already be trained: no invented uniform initial state.
        result.effective_init_kappa = None
        result.phase = nn.Parameter(reference.mu.detach().clone(), reference.mu.requires_grad)
        result.raw_kappa = nn.Parameter(reference.log_kappa.detach().clone(), reference.log_kappa.requires_grad)
        result.bias = None if reference.bias is None else nn.Parameter(reference.bias.detach().clone(), reference.bias.requires_grad)
        result.training = reference.training
        return result

    def get_config(self) -> dict[str, object]:
        """Return versioned architecture and explicit initialization metadata."""
        return {"class": self.version, "in_features": self.in_features,
                "out_features": self.out_features, "bias": self.bias is not None,
                "response": self.response, "implementation": self.implementation,
                "init_policy": self.init_policy, "requested_init_kappa": self.requested_init_kappa,
                "effective_init_kappa": self.effective_init_kappa}


class CartesianCircularLinearV1(nn.Module):
    """Cartesian coordinates for the same per-input sine/cosine function class.

    Construction uses the declared polar initialization followed by conversion;
    explicit conversion of an existing layer avoids consuming random numbers.
    Coordinate gradients and optimizer trajectories need not equal polar ones.
    """

    version = "CartesianCircularLinearV1"

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True,
                 response: Literal["sum", "exp"] = "sum", init_kappa: float = 1.0,
                 init_policy: Literal["constant", "unit_variance"] = "constant",
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        reference = PolarCircularLinearV2(in_features, out_features, bias=bias,
            response=response, init_kappa=init_kappa, init_policy=init_policy,
            device=device, dtype=dtype)
        self._copy(reference)

    def _copy(self, reference: PolarCircularLinearV2) -> None:
        self.in_features, self.out_features = reference.in_features, reference.out_features
        self.response = reference.response
        a, b = reference.coefficients()
        learn = reference.phase.requires_grad or reference.raw_kappa.requires_grad
        self.weight_cos = nn.Parameter(a.detach().clone(), learn)
        self.weight_sin = nn.Parameter(b.detach().clone(), learn)
        self.bias = None if reference.bias is None else nn.Parameter(reference.bias.detach().clone(), reference.bias.requires_grad)
        self.training = reference.training

    @classmethod
    def from_polar(cls, reference: PolarCircularLinearV2) -> CartesianCircularLinearV1:
        """Convert function/dtype/device/mode without shared state or RNG draws."""
        if not isinstance(reference, PolarCircularLinearV2):
            raise TypeError("reference must be PolarCircularLinearV2")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._copy(reference)
        return result

    @classmethod
    def from_legacy(cls, reference: nn.Module) -> CartesianCircularLinearV1:
        """Explicitly convert a legacy polar layer."""
        return cls.from_polar(PolarCircularLinearV2.from_legacy(reference))

    def forward(self, x: Tensor, phase_shift: Tensor | None = None) -> Tensor:
        """Evaluate Cartesian coefficients, optionally under a shared rotation."""
        _validate_input(x, self.in_features)
        _validate_shift(x, phase_shift, self.out_features)
        cosine, sine = x.cos(), x.sin()
        z = F.linear(cosine, self.weight_cos) + F.linear(sine, self.weight_sin)
        if phase_shift is not None:
            q = F.linear(sine, self.weight_cos) - F.linear(cosine, self.weight_sin)
            delta = phase_shift.squeeze(-1)
            z = delta.cos() * z - delta.sin() * q
        return _response(z, self.response, self.bias)

    def get_config(self) -> dict[str, object]:
        """Return the versioned architecture."""
        return {"class": self.version, "in_features": self.in_features,
                "out_features": self.out_features, "bias": self.bias is not None,
                "response": self.response}


def amplitude_regularizer(layers: nn.Module | Iterable[nn.Module], coefficient: float) -> Tensor:
    """Return coefficient/2 times the sum of per-layer mean squared amplitudes.

    Phase, raw coordinates and biases are not directly penalized. This equals
    coefficient*kappa**2/2 for the single-edge stage A models. Layers contribute
    equal weight regardless of their edge count; no optimizer decay is implied.
    """
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    candidates = [layers] if isinstance(layers, (PolarCircularLinearV2, CartesianCircularLinearV1)) else list(layers)
    if not candidates:
        raise ValueError("at least one circular layer is required")
    terms = []
    for layer in candidates:
        if isinstance(layer, PolarCircularLinearV2):
            terms.append(layer.kappa.square().mean())
        elif isinstance(layer, CartesianCircularLinearV1):
            terms.append((layer.weight_cos.square() + layer.weight_sin.square()).mean())
        else:
            raise TypeError("regularizer accepts only polar or Cartesian circular layers")
    return torch.stack(terms).sum() * (coefficient / 2)


class PeakCircularKernelBankV1(nn.Module):
    """Peak-normalized local circular features, without an internal bias.

    Args:
        in_features: Angular input count.
        num_kernels: Feature count.
        centers: Explicit floating tensor shaped (num_kernels, in_features).
        concentrations: Positive floating tensor shaped (M,D), (M,), or ()
            for per_edge, per_kernel, or global sharing, respectively.
        learn_centers: Store centers as trainable parameters instead of buffers.
        learn_concentrations: Learn softplus widths instead of fixed buffers.
        width_sharing: Explicit width shape policy.
    """

    version = "PeakCircularKernelBankV1"

    def __init__(self, in_features: int, num_kernels: int, *, centers: Tensor,
                 concentrations: Tensor, learn_centers: bool,
                 learn_concentrations: bool,
                 width_sharing: Literal["per_edge", "per_kernel", "global"] = "per_edge") -> None:
        super().__init__()
        _positive_dimension(in_features, "in_features")
        _positive_dimension(num_kernels, "num_kernels")
        shapes = {"per_edge": (num_kernels, in_features), "per_kernel": (num_kernels,), "global": ()}
        if width_sharing not in shapes:
            raise ValueError("unknown width_sharing")
        if centers.shape != (num_kernels, in_features) or not centers.is_floating_point() or not bool(torch.isfinite(centers).all()):
            raise ValueError("centers must be finite floating values shaped (num_kernels, in_features)")
        if concentrations.shape != shapes[width_sharing]:
            raise ValueError(f"concentrations shape must be {shapes[width_sharing]} for {width_sharing}")
        if concentrations.dtype != centers.dtype or concentrations.device != centers.device:
            raise ValueError("centers and concentrations must match dtype and device")
        raw = inverse_softplus(concentrations)
        self.in_features, self.num_kernels = in_features, num_kernels
        self.width_sharing = width_sharing
        self.learn_centers, self.learn_concentrations = learn_centers, learn_concentrations
        if learn_centers:
            self.centers = nn.Parameter(centers.detach().clone())
        else:
            self.register_buffer("centers", centers.detach().clone())
        if learn_concentrations:
            self.raw_kappa = nn.Parameter(raw.detach().clone())
        else:
            self.register_buffer("fixed_kappa", concentrations.detach().clone())

    @property
    def concentrations(self) -> Tensor:
        """Return effective concentrations in their declared sharing shape."""
        return F.softplus(self.raw_kappa) if self.learn_concentrations else self.fixed_kappa

    @property
    def kappa(self) -> Tensor:
        """Alias for effective concentrations used in recorded diagnostics."""
        return self.concentrations

    def _edge_concentrations(self) -> Tensor:
        kappa = self.concentrations
        return kappa.unsqueeze(-1) if self.width_sharing == "per_kernel" else kappa

    def log_features(self, x: Tensor) -> Tensor:
        """Return a nonpositive log feature with stable differences near a peak."""
        _validate_input(x, self.in_features)
        difference = x.unsqueeze(-2) - self.centers
        return (-2 * self._edge_concentrations() * (difference / 2).sin().square()).sum(-1)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(..., num_kernels)`` features in [0,1] for finite states."""
        return self.log_features(x).exp()

    def chordal_features(self, x: Tensor) -> Tensor:
        """Evaluate the equivalent Gaussian RBF on unit-circle embeddings."""
        _validate_input(x, self.in_features)
        distance = ((x.cos().unsqueeze(-2) - self.centers.cos()).square()
                    + (x.sin().unsqueeze(-2) - self.centers.sin()).square())
        return (-0.5 * self._edge_concentrations() * distance).sum(-1).exp()

    def get_config(self) -> dict[str, object]:
        """Return feature capacity and the versioned learning/sharing policy."""
        return {"class": self.version, "in_features": self.in_features,
                "num_kernels": self.num_kernels, "width_sharing": self.width_sharing,
                "learn_centers": self.learn_centers, "learn_concentrations": self.learn_concentrations}


class DirectCalendarPhaseV1(nn.Module):
    """Independent affine maps of calendar sine/cosine features.

    Each map outputs a sensor layer's shared per-neuron phase shift. Conversion
    collapses a legacy bank plus projection, preserving the initial function but
    changing parameter count and optimization geometry.
    """

    version = "DirectCalendarPhaseV1"

    def __init__(self, time_size: int, hidden_size: int, num_layers: int, *,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        for name, value in (("time_size", time_size), ("hidden_size", hidden_size), ("num_layers", num_layers)):
            _positive_dimension(value, name)
        self.time_size, self.hidden_size, self.num_layers = time_size, hidden_size, num_layers
        self.phase_maps = nn.ModuleList([nn.Linear(2 * time_size, hidden_size, device=device, dtype=dtype) for _ in range(num_layers)])

    def forward(self, angles: Tensor) -> tuple[Tensor, ...]:
        """Return one ``(..., hidden_size)`` phase tensor for each sensor layer."""
        _validate_input(angles, self.time_size)
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return tuple(projection(features) for projection in self.phase_maps)

    @classmethod
    def from_legacy_encoder(cls, reference: nn.Module) -> DirectCalendarPhaseV1:
        """Collapse actual bank/projection states without RNG draws or aliases."""
        from src.temporal import TemporalVonMisesEncoder

        if not isinstance(reference, TemporalVonMisesEncoder) or reference.freq_bank is None:
            raise ValueError("reference must be a time-conditioned legacy encoder")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.time_size, result.hidden_size = reference.time_size, reference.hidden_size
        result.num_layers = reference.num_vm_layers
        result.phase_maps = nn.ModuleList()
        bank = reference.freq_bank
        with torch.no_grad():
            amplitude = bank.get_kappa().unsqueeze(-1)
            a, b = amplitude * bank.mu.cos(), -amplitude * bank.mu.sin()
            for source in reference.delta_mu_projections:
                projection = nn.Linear.__new__(nn.Linear)
                nn.Module.__init__(projection)
                projection.in_features, projection.out_features = 2 * reference.time_size, reference.hidden_size
                learn_weight = source.weight.requires_grad or bank.mu.requires_grad or (bank.learnable_kappa and bank.raw_kappa.requires_grad)
                learn_bias = source.bias.requires_grad or source.weight.requires_grad or bank.bias.requires_grad
                projection.weight = nn.Parameter(torch.cat((source.weight @ a, source.weight @ b), dim=-1).detach().clone(), learn_weight)
                projection.bias = nn.Parameter((source.weight @ bank.bias + source.bias).detach().clone(), learn_bias)
                projection.training = source.training
                result.phase_maps.append(projection)
        result.training = reference.training
        return result

    def get_config(self) -> dict[str, object]:
        """Return the direct map architecture with a version identifier."""
        return {"class": self.version, "time_size": self.time_size,
                "hidden_size": self.hidden_size, "num_layers": self.num_layers}
