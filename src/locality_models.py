"""Float64 circular bases and differentiable, regularized variable projection.

The readout minimizes ``0.5 * mean((Phi @ w + b - y)**2) +
0.5 * penalty * sum(w**2)``. Its intercept is unpenalized. The circular
features are the existing peak-normalized VM/chordal Gaussian function class;
this module changes neither v1 implementations nor their saved artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from src.circular_layers_v2 import PeakCircularKernelBankV1

BasisMode = Literal["fixed", "centers", "widths", "both"]


def _vector(value: Tensor | Sequence[float], name: str,
            device: torch.device | str | None = None) -> Tensor:
    """Convert an explicit nonempty vector to float64 without random draws."""
    if torch.is_complex(torch.as_tensor(value)):
        raise ValueError(f"{name} must be real")
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must have shape (K,) or (K, 1), K > 0")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _angles(theta: Tensor) -> Tensor:
    """Normalize scalar, (N,), or (N,1) angle inputs without copying."""
    if not isinstance(theta, Tensor) or theta.dtype != torch.float64:
        raise ValueError("angles must be a float64 tensor")
    if theta.ndim == 2 and theta.shape[1] == 1:
        return theta[:, 0]
    if theta.ndim not in (0, 1):
        raise ValueError("angles must have scalar, (N,), or (N, 1) shape")
    return theta


def _functional_inputs(theta: Tensor, centers: Tensor,
                       concentrations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Validate functional geometry while retaining its autograd graph."""
    angles = _angles(theta)
    for name, value in (("centers", centers), ("concentrations", concentrations)):
        if not isinstance(value, Tensor) or value.dtype != torch.float64:
            raise ValueError(f"{name} must be a float64 tensor")
        if value.device != theta.device:
            raise ValueError("angles, centers and concentrations must share a device")
    locations = _vector(centers, "centers")
    widths = _vector(concentrations, "concentrations")
    if locations.shape != widths.shape or not bool((widths > 0).all()):
        raise ValueError("concentrations must be positive and match centers")
    return angles, locations, widths


def circular_features(theta: Tensor, centers: Tensor,
                      concentrations: Tensor) -> Tensor:
    """Return differentiable ``exp(-2*kappa*sin((theta-center)/2)**2)``.

    Geometry tensors have K or K-by-1 entries. Outputs have shape (N,K), or
    (K,) for a scalar angle. Finite positive widths imply values in [0,1];
    very narrow features can underflow to zero and are not clipped.
    """
    angles, locations, widths = _functional_inputs(theta, centers, concentrations)
    return (-2 * widths * ((angles[..., None] - locations) / 2).sin().square()).exp()


def chordal_features(theta: Tensor, centers: Tensor,
                     concentrations: Tensor) -> Tensor:
    """Return the same kernels through independent unit-circle distances."""
    angles, locations, widths = _functional_inputs(theta, centers, concentrations)
    distance = ((angles.cos()[..., None] - locations.cos()).square()
                + (angles.sin()[..., None] - locations.sin()).square())
    return (-0.5 * widths * distance).exp()


def fourier_features(theta: Tensor, harmonics: int | Sequence[int]) -> Tensor:
    """Return cosines followed by sines for positive integer harmonics.

    An integer H means harmonics 1 through H, producing 2H features. An
    explicit sequence must contain distinct positive integers. The constant
    feature is omitted because the readout already has an intercept.
    """
    angles = _angles(theta)
    if isinstance(harmonics, int) and not isinstance(harmonics, bool):
        values = list(range(1, harmonics + 1))
    else:
        try:
            values = list(harmonics)
        except TypeError as error:
            raise ValueError("harmonics must be a positive integer or integer sequence") from error
    if (not values or any(isinstance(value, bool) or not isinstance(value, int)
                          or value <= 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("harmonics must be distinct positive integers")
    frequencies = torch.tensor(values, dtype=angles.dtype, device=angles.device)
    phase = angles[..., None] * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class CircularBasisV2(nn.Module):
    """An explicit scalar-angle basis with independently learnable geometry.

    Args:
        centers: K explicit centers, supplied as a vector or column vector.
        concentrations: K explicit strictly positive concentrations.
        mode: Which geometry coordinates become parameters. Widths use the
            unchanged v1 softplus parameterization; fixed values are buffers.
        device: Optional destination; otherwise preserve centers' device.

    Construction detaches and copies supplied geometry, consumes no random
    numbers, and creates float64 state. ``centers`` and ``concentrations``
    expose length-K tensors. Optimizers should use ``parameters()``; trainable
    leaves are ``bank.centers`` and/or ``bank.raw_kappa``.
    """

    version = "CircularBasisV2"

    def __init__(self, centers: Tensor | Sequence[float],
                 concentrations: Tensor | Sequence[float], mode: BasisMode = "fixed",
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        if mode not in ("fixed", "centers", "widths", "both"):
            raise ValueError("mode must be fixed, centers, widths, or both")
        locations = _vector(centers, "centers", device)
        widths = _vector(concentrations, "concentrations", locations.device)
        if locations.shape != widths.shape:
            raise ValueError("centers and concentrations must have the same length")
        self.mode = mode
        self.num_kernels = locations.numel()
        self.bank = PeakCircularKernelBankV1(
            1, self.num_kernels, centers=locations[:, None], concentrations=widths,
            learn_centers=mode in ("centers", "both"),
            learn_concentrations=mode in ("widths", "both"), width_sharing="per_kernel")

    @property
    def centers(self) -> Tensor:
        """Return length-K centers, retaining parameter gradients."""
        return self.bank.centers[:, 0]

    @property
    def concentrations(self) -> Tensor:
        """Return length-K positive effective widths (softplus if trained)."""
        return self.bank.concentrations

    @property
    def kappa(self) -> Tensor:
        """Alias for effective concentrations."""
        return self.concentrations

    def forward(self, theta: Tensor) -> Tensor:
        """Return a float64 feature vector or matrix on the basis device."""
        angles = _angles(theta)
        if angles.device != self.bank.centers.device or self.bank.centers.dtype != torch.float64:
            raise ValueError("basis state and float64 angles must share dtype and device")
        return self.bank(angles[..., None])

    def chordal_features(self, theta: Tensor) -> Tensor:
        """Evaluate the same state through its unit-circle Gaussian formula."""
        return chordal_features(theta, self.centers, self.concentrations)

    def get_config(self) -> dict[str, object]:
        """Return architecture and coordinate-learning policy, without state."""
        return {"class": self.version, "mode": self.mode, "num_kernels": self.num_kernels,
                "dtype": "float64", "width_parameterization": "softplus",
                "learn_centers": self.bank.learn_centers,
                "learn_concentrations": self.bank.learn_concentrations}


def make_basis(centers: Tensor | Sequence[float],
               concentrations: Tensor | Sequence[float], mode: BasisMode = "fixed",
               device: torch.device | str | None = None) -> CircularBasisV2:
    """Construct an explicit float64 circular basis without RNG or aliases."""
    return CircularBasisV2(centers, concentrations, mode=mode, device=device)


def _readout_inputs(features: Tensor, y: Tensor,
                    penalty: float | Tensor) -> tuple[Tensor, Tensor]:
    """Check the scalar-target readout contract and keep differentiable inputs."""
    if (not isinstance(features, Tensor) or features.dtype != torch.float64
            or features.ndim != 2 or min(features.shape) <= 0):
        raise ValueError("features must be a nonempty float64 (N,K) tensor")
    if (not isinstance(y, Tensor) or y.dtype != torch.float64 or y.device != features.device
            or y.shape not in ((features.shape[0],), (features.shape[0], 1))):
        raise ValueError("y must have shape (N,) or (N,1), matching feature dtype/device")
    if isinstance(penalty, bool) or torch.is_complex(torch.as_tensor(penalty)):
        raise ValueError("penalty must be a finite strictly positive real scalar")
    value = torch.as_tensor(penalty, dtype=features.dtype, device=features.device)
    if value.ndim != 0 or not bool(torch.isfinite(value) & (value > 0)):
        raise ValueError("penalty must be a finite strictly positive scalar")
    if not bool(torch.isfinite(features).all() and torch.isfinite(y).all()):
        raise ValueError("features and targets must be finite")
    return y[:, None] if y.ndim == 1 else y, value


def _centered_system(features: Tensor, target: Tensor,
                     penalty: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the ridge normal system without a penalized intercept."""
    feature_mean, target_mean = features.mean(0), target.mean(0)
    centered = features - feature_mean
    centered_y = target - target_mean
    identity = torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    matrix = centered.T @ centered + features.shape[0] * penalty * identity
    return matrix, centered.T @ centered_y, feature_mean, target_mean


def solve_readout(features: Tensor, y: Tensor,
                  penalty: float | Tensor) -> tuple[Tensor, Tensor]:
    """Solve the differentiable regularized readout in float64 by Cholesky.

    Args:
        features: Finite (N,K) float64 design on CPU or CUDA.
        y: Finite scalar targets shaped (N,) or (N,1), on the same device.
        penalty: Strictly positive lambda in the mean-squared objective.

    Returns:
        Weights and unpenalized intercept. Vector targets yield (K,) and a
        scalar; column targets yield (K,1) and (1,). Gradients pass through the
        solve. No jitter, clipping, fallback, or lambda change is applied.

    Raises:
        ValueError: For invalid inputs. Numerical factorization failures raise
            the original PyTorch error, rather than silently changing a fit.
    """
    target, value = _readout_inputs(features, y, penalty)
    matrix, rhs, feature_mean, target_mean = _centered_system(features, target, value)
    if not bool(torch.isfinite(matrix).all() and torch.isfinite(rhs).all()):
        raise FloatingPointError("regularized readout system is nonfinite")
    factor = torch.linalg.cholesky(matrix)
    weight = torch.cholesky_solve(rhs, factor)
    bias = target_mean - feature_mean @ weight
    if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        raise FloatingPointError("regularized readout coefficients are nonfinite")
    if y.ndim == 1:
        return weight[:, 0], bias[0]
    return weight, bias


def predict_readout(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply compatible float64 readout tensors without changing shapes."""
    if (not isinstance(features, Tensor) or features.ndim != 2
            or features.dtype != torch.float64):
        raise ValueError("features must be a float64 (N,K) tensor")
    if (not isinstance(weight, Tensor) or not isinstance(bias, Tensor)
            or weight.shape not in ((features.shape[1],), (features.shape[1], 1))
            or bias.shape != (() if weight.ndim == 1 else (1,))):
        raise ValueError("weight and bias must follow the solve_readout shape convention")
    if any(value.dtype != features.dtype or value.device != features.device for value in (weight, bias)):
        raise ValueError("features, weight and bias must share dtype and device")
    return features @ weight + bias


def reduced_objective(features: Tensor, y: Tensor, penalty: float | Tensor) -> Tensor:
    """Return the scalar training objective after exact readout elimination."""
    weight, bias = solve_readout(features, y, penalty)
    residual = predict_readout(features, weight, bias) - y
    value = torch.as_tensor(penalty, dtype=features.dtype, device=features.device)
    return 0.5 * residual.square().mean() + 0.5 * value * weight.square().sum()


@torch.no_grad()
def readout_diagnostics(features: Tensor, y: Tensor, penalty: float | Tensor,
                        weight: Tensor | None = None,
                        bias: Tensor | None = None) -> dict[str, float | int]:
    """Return detached conditioning, backward error and stationarity checks.

    The relative backward error is ``||A*w-rhs|| / (||A||_2*||w||+||rhs||)``
    for the centered regularized normal system. It is zero for an exactly zero
    numerator/denominator. Diagnostics use no autograd graph and no random
    sampling. A supplied readout is checked without silently replacing it.
    """
    target, value = _readout_inputs(features, y, penalty)
    if (weight is None) != (bias is None):
        raise ValueError("supply both weight and bias or neither")
    if weight is None:
        weight, bias = solve_readout(features, y, value)
    prediction = predict_readout(features, weight, bias)
    if prediction.shape != y.shape:
        raise ValueError("supplied readout must preserve the target shape convention")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        raise ValueError("readout coefficients must be finite")
    matrix, rhs, _, _ = _centered_system(features, target, value)
    column_weight = weight[:, None] if weight.ndim == 1 else weight
    residual = prediction - y
    normal_residual = matrix @ column_weight - rhs
    singular = torch.linalg.svdvals(matrix)
    numerator = torch.linalg.vector_norm(normal_residual)
    denominator = singular[0] * torch.linalg.vector_norm(column_weight) + torch.linalg.vector_norm(rhs)
    backward_error = 0.0 if denominator.item() == 0 else (numerator / denominator).item()
    weight_gradient = features.T @ residual / features.shape[0] + value * weight
    return {"n_samples": features.shape[0], "n_features": features.shape[1],
            "penalty": value.item(), "normal_matrix_condition_2": (singular[0] / singular[-1]).item(),
            "relative_backward_error": backward_error,
            "readout_stationarity_l2": torch.linalg.vector_norm(weight_gradient).item(),
            "intercept_stationarity_abs": residual.mean().abs().item(),
            "data_mse": residual.square().mean().item(),
            "readout_l2": torch.linalg.vector_norm(weight).item(),
            "regularized_objective": (0.5 * residual.square().mean()
                                       + 0.5 * value * weight.square().sum()).item()}
