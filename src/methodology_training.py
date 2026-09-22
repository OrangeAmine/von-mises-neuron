"""Versioned, bounded fitting for the VM methodology A/B experiment contract.

Training remains float32; metrics and linear solves use float64.  This module
does not generate task functions, choose pilot rates, or access confirmation
registries.  Each fit returns JSON-compatible evidence and separate tensor
artifacts, including failed and unfinished fits, for the orchestration runner.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AffineV1(nn.Module):
    """An explicitly initialized affine map that consumes no global RNG draws."""

    def __init__(self, weight: Any, bias: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weight, dtype=torch.float32).clone())
        self.bias = nn.Parameter(torch.as_tensor(bias, dtype=torch.float32).clone())
        if self.weight.ndim != 2 or self.bias.shape != (self.weight.shape[0],):
            raise ValueError("Affine weight/bias shapes must be (out,in) and (out,)")
        _finite(self.weight, "initial affine weight")
        _finite(self.bias, "initial affine bias")

    def forward(self, x: Tensor) -> Tensor:
        """Apply the declared weight and bias."""
        return F.linear(x, self.weight, self.bias)


class LearnedGeodesicBankV1(nn.Module):
    """Learned geodesic features, distinct from the VM/chordal kernel."""

    def __init__(self, centers: Any, widths: Any) -> None:
        super().__init__()
        c = torch.as_tensor(centers, dtype=torch.float32).clone()
        k = torch.as_tensor(widths, dtype=torch.float32).clone()
        if c.ndim != 2 or c.shape[1] != 1 or k.shape != c.shape:
            raise ValueError("Geodesic centers/widths must have equal (kernels,1) shapes")
        if not torch.isfinite(c).all() or not torch.isfinite(k).all() or not (k > 0).all():
            raise ValueError("Centers must be finite and widths positive and finite")
        self.centers = nn.Parameter(c)
        self.raw_kappa = nn.Parameter(k + torch.log(-torch.expm1(-k)))

    @property
    def concentrations(self) -> Tensor:
        """Return strictly positive effective concentrations."""
        return F.softplus(self.raw_kappa)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate exp(-kappa*wrapped_distance**2/2)."""
        d = torch.remainder(x.unsqueeze(-2) - self.centers + math.pi, 2 * math.pi) - math.pi
        return torch.exp(-0.5 * (self.concentrations * d.square()).sum(-1))


class KernelRegressorV1(nn.Module):
    """Circular feature bank with a separate unrestricted signed readout."""

    def __init__(self, bank: nn.Module, weight: Any, bias: Any, model_id: str) -> None:
        super().__init__()
        self.bank = bank
        self.readout = AffineV1(weight, bias)
        self.model_id = model_id

    def features(self, x: Tensor) -> Tensor:
        """Return circular kernel features before the signed readout."""
        return self.bank(x)

    def forward(self, x: Tensor) -> Tensor:
        """Map one-dimensional angles to scalar responses."""
        return self.readout(self.features(x))


class AngularMLPV1(nn.Module):
    """The declared 49-parameter raw-angle or sine/cosine tanh control."""

    def __init__(self, model_id: str, state: Mapping[str, Any]) -> None:
        super().__init__()
        if model_id not in ("sincos_mlp", "raw_angle_mlp"):
            raise ValueError("Unknown MLP model")
        self.model_id = model_id
        self.hidden = AffineV1(state["hidden.weight"], state["hidden.bias"])
        self.readout = AffineV1(state["readout.weight"], state["readout.bias"])

    def features(self, x: Tensor) -> Tensor:
        """Return tanh activations; raw inputs are explicitly theta/pi."""
        encoded = torch.cat((x.cos(), x.sin()), -1) if self.model_id == "sincos_mlp" else x / math.pi
        return self.hidden(encoded).tanh()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the scalar signed readout."""
        return self.readout(self.features(x))


class FixedFeaturesV1(nn.Module):
    """The two prescribed 48-feature banks for float64 ridge fitting."""

    def __init__(self, model_id: str) -> None:
        super().__init__()
        if model_id not in ("fourier_ridge", "fixed_vm_ridge"):
            raise ValueError("Unknown fixed-feature model")
        self.model_id = model_id
        if model_id == "fourier_ridge":
            self.register_buffer("harmonics", torch.arange(1, 25, dtype=torch.float64))
        else:
            centers = -math.pi + 2 * math.pi * torch.arange(16, dtype=torch.float64) / 16
            self.register_buffer("centers", centers.repeat_interleave(3).reshape(48, 1))
            self.register_buffer("concentrations", torch.tensor([1., 4., 16.], dtype=torch.float64).repeat(16).reshape(48, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Return 48 features in the recorded deterministic column order."""
        if self.model_id == "fourier_ridge":
            angles = x * self.harmonics
            return torch.cat((angles.cos(), angles.sin()), dim=-1)
        d = x.unsqueeze(-2) - self.centers
        return torch.exp((-2 * self.concentrations * (d / 2).sin().square()).sum(-1))


def make_a_model(
    coordinate: str, amplitude_before: float, phase_before: float, *, device: Any = None,
) -> nn.Module:
    """Construct the known pre-change function and its exact Cartesian conversion."""
    from src.circular_layers_v2 import CartesianCircularLinearV1, PolarCircularLinearV2

    with torch.random.fork_rng(devices=[]):
        model = PolarCircularLinearV2(1, 1, init_kappa=amplitude_before, init_policy="constant", response="sum")
    with torch.no_grad():
        model.phase.fill_(phase_before)
        model.bias.zero_()
    if coordinate == "cartesian":
        model = CartesianCircularLinearV1.from_polar(model)
    elif coordinate != "polar":
        raise ValueError("A coordinate must be polar or cartesian")
    model.model_id = coordinate
    if parameter_count(model) != 3:
        raise ValueError("A models must have exactly three trainable scalars")
    return model.to(_device(device), dtype=torch.float32)


def make_b_model(model_id: str, initialization: Mapping[str, Any], *, device: Any = None) -> nn.Module:
    """Build a gradient arm from the task package's dedicated initialization bundle.

    Args:
        model_id: One of learned_vm, learned_geodesic, sincos_mlp, raw_angle_mlp.
        initialization: Explicit centers, widths, readout, bias and mlp states.
        device: CUDA when available by default; tests explicitly select CPU.
    """
    if model_id in ("learned_vm", "learned_geodesic"):
        centers, widths = initialization["centers"], initialization["widths"]
        if model_id == "learned_vm":
            from src.circular_layers_v2 import PeakCircularKernelBankV1
            bank = PeakCircularKernelBankV1(1, 16, centers=torch.as_tensor(centers), concentrations=torch.as_tensor(widths),
                                          learn_centers=True, learn_concentrations=True, width_sharing="per_edge")
        else:
            bank = LearnedGeodesicBankV1(centers, widths)
        model = KernelRegressorV1(bank, initialization["readout"], initialization["bias"], model_id)
    elif model_id in ("sincos_mlp", "raw_angle_mlp"):
        model = AngularMLPV1(model_id, initialization[model_id])
    else:
        raise ValueError(f"Unknown B gradient model: {model_id}")
    if parameter_count(model) != 49:
        raise ValueError("B gradient models must have exactly 49 trainable scalars")
    return model.to(_device(device), dtype=torch.float32)


def parameter_count(model: nn.Module) -> int:
    """Count learned scalars independently of fixed feature capacity."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_manifest(model: nn.Module, stage: str) -> list[dict[str, Any]]:
    """List auditable groups, with zero native decay and explicit penalty roles."""
    rows = []
    seen: set[int] = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        if not p.requires_grad:
            continue
        if id(p) in seen:
            raise ValueError(f"Duplicate optimizer parameter membership: {name}")
        seen.add(id(p))
        role = ("amplitude_function_space" if stage == "A" and name in ("raw_kappa", "weight_cos", "weight_sin")
                else "readout_l2" if stage == "B" and name == "readout.weight" else "none")
        rows.append({"name": name, "shape": list(p.shape), "count": p.numel(), "native_weight_decay": 0., "explicit_penalty": role})
    if sum(row["count"] for row in rows) != parameter_count(model):
        raise ValueError("Missing optimizer parameter membership")
    return rows


def explicit_penalty(model: nn.Module, stage: str, coefficient: float) -> Tensor:
    """Return lambda/2 times amplitude norm (A) or readout weight norm (B)."""
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Penalty coefficient must be finite and nonnegative")
    if stage == "A":
        if hasattr(model, "raw_kappa"):
            squared = F.softplus(model.raw_kappa).square().sum()
        else:
            squared = model.weight_cos.square().sum() + model.weight_sin.square().sum()
    elif stage == "B":
        squared = model.readout.weight.square().sum()
    else:
        raise ValueError("Stage must be A or B")
    return coefficient * squared / 2


def _device(device: Any) -> torch.device:
    return torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))


def _finite(value: Tensor, context: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {context}")


def _state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _check_state(model: nn.Module) -> None:
    values = list(model.state_dict().values())
    if values and not torch.stack([torch.isfinite(value).all() for value in values]).all():
        raise FloatingPointError("Nonfinite model state")


def _arrays(data: Any) -> tuple[dict[str, Tensor], dict[str, Any]]:
    mapping = data.as_arrays() if hasattr(data, "as_arrays") else data
    metadata = copy.deepcopy(getattr(getattr(data, "function", None), "parameters", mapping.get("task_parameters", {})))
    arrays = {}
    for key in ("train_x", "train_y", "validation_x", "validation_y", "grid_x", "grid_clean", "grid_noisy"):
        value = torch.as_tensor(mapping[key], dtype=torch.float64).detach().cpu().clone()
        if value.ndim != 2 or value.shape[1] != 1 or len(value) == 0:
            raise ValueError(f"{key} must have nonempty (N,1) shape")
        _finite(value, key)
        arrays[key] = value
    for x, y in (("train_x", "train_y"), ("validation_x", "validation_y"), ("grid_x", "grid_clean"), ("grid_x", "grid_noisy")):
        if arrays[x].shape != arrays[y].shape:
            raise ValueError(f"Shape mismatch: {x}/{y}")
    if arrays["grid_clean"].var(unbiased=False).item() <= 0:
        raise ValueError("Clean grid variance must be positive")
    return arrays, metadata


def _predict(model: nn.Module, x: Tensor) -> Tensor:
    parameter = next(model.parameters())
    with torch.no_grad():
        output = model(x.to(device=parameter.device, dtype=parameter.dtype))
    if output.shape != x.shape:
        raise ValueError("Scalar model predictions must exactly match target (N,1) shape")
    _finite(output, "predictions")
    return output.detach().cpu().double()


def regression_metrics(prediction: Tensor, clean: Tensor, noisy: Tensor, x: Tensor) -> dict[str, Any]:
    """Accumulate clean/noisy errors and a declared outer-10%-domain boundary band."""
    prediction, clean, noisy, x = [v.detach().cpu().double() for v in (prediction, clean, noisy, x)]
    error = prediction - clean
    variance = clean.var(unbiased=False).item()
    boundary = x.abs() >= 0.9 * math.pi
    values = {
        "clean_mse": error.square().mean().item(), "clean_variance": variance,
        "normalized_clean_mse": error.square().mean().item() / variance,
        "noisy_rmse": (prediction - noisy).square().mean().sqrt().item(),
        "mean_absolute_error": error.abs().mean().item(), "max_absolute_error": error.abs().max().item(),
        "boundary_mse": error[boundary].square().mean().item() if boundary.any() else None,
        "boundary_mean_absolute_error": error[boundary].abs().mean().item() if boundary.any() else None,
        "boundary_band": "abs(theta)>=0.9*pi; outer 10% of domain", "boundary_points": int(boundary.sum()),
    }
    if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
        raise FloatingPointError("Nonfinite evaluation metric")
    return values


def _coefficient_metrics(model: nn.Module, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(model, "raw_kappa"):
        amplitude = F.softplus(model.raw_kappa.detach().double()).item()
        phase = model.phase.detach().double().item()
        a, b = amplitude * math.cos(phase), -amplitude * math.sin(phase)
    else:
        a, b = model.weight_cos.detach().double().item(), model.weight_sin.detach().double().item()
        amplitude, phase = math.hypot(a, b), math.atan2(-b, a)
    values: dict[str, Any] = {"coefficient_cos": a, "coefficient_sin": b, "amplitude": amplitude,
                               "phase": phase, "bias": model.bias.detach().double().item()}
    if "amplitude_after" in parameters and "phase_after" in parameters:
        true_amp, true_phase = parameters["amplitude_after"], parameters["phase_after"]
        threshold = max(0.001, 0.05 * true_amp)
        phase_defined = amplitude >= threshold
        values.update(amplitude_error=abs(amplitude - true_amp),
                      coefficient_l2_error=math.hypot(a - true_amp * math.cos(true_phase), b + true_amp * math.sin(true_phase)),
                      wrapped_phase_error=abs((phase - true_phase + math.pi) % (2 * math.pi) - math.pi) if phase_defined else None,
                      phase_error_undefined_reason=None if phase_defined else f"predicted amplitude below {threshold}",
                      phase_error_defined=phase_defined)
    return values


def _optimizer(model: nn.Module, stage: str, name: str, learning_rate: float) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    manifest = parameter_manifest(model, stage)
    named = dict(model.named_parameters())
    groups = [{"params": [named[row["name"]]], "weight_decay": 0., "name": row["name"]} for row in manifest]
    if name == "Adam":
        optimizer = torch.optim.Adam(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=0)
    elif name == "SGD":
        optimizer = torch.optim.SGD(groups, lr=learning_rate, momentum=0, weight_decay=0)
    else:
        raise ValueError("Optimizer must be Adam or SGD")
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("Optimizer membership is missing or duplicated")
    return optimizer, manifest


def _gradient_step(model: nn.Module, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor,
                   stage: str, penalty: float, clip: float | None) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    if prediction.shape != y.shape:
        raise ValueError("Predictions must exactly match target shape")
    data_loss = 0.5 * (prediction - y).square().mean()
    regularizer = explicit_penalty(model, stage, penalty)
    objective = data_loss + regularizer
    _finite(objective, "training objective")
    objective.backward()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if any(p.grad is None for _, p in named):
        raise FloatingPointError("Missing gradient for a learned parameter")
    norm = torch.stack([p.grad.norm() for _, p in named]).norm()
    _finite(norm, "gradient norm")
    norms = {name: p.grad.norm().item() for name, p in named}
    if clip is not None:
        nn.utils.clip_grad_norm_([p for _, p in named], clip, error_if_nonfinite=True)
    optimizer.step()
    _check_state(model)
    return {"training_half_mse": data_loss.item(), "explicit_penalty": regularizer.item(), "objective": objective.item(),
            "gradient_norm_before_clip": norm.item(), "parameter_gradient_norms": norms}


def _timing_start(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _finish(bundle: dict[str, Any], model: nn.Module, start: float, device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bundle["summary"]["runtime_seconds"] = time.perf_counter() - start
    bundle["summary"]["time_seconds"] = bundle["summary"]["runtime_seconds"]
    bundle["summary"]["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    bundle["summary"]["device"] = str(device)
    return bundle


def _failure(bundle: dict[str, Any], error: BaseException, step: int) -> None:
    bundle["summary"].update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "timeout" if isinstance(error, TimeoutError) else "numerical_failure" if isinstance(error, FloatingPointError) else "failed",
                             failure=f"{type(error).__name__}: {error}", failed_step=step)


def train_a(
    model: nn.Module, data: Any, *, optimizer_name: str, learning_rate: float, penalty: float,
    updates: int = 200, record_every: int = 10, device: Any = None, deadline: float | None = None,
) -> dict[str, Any]:
    """Run full-batch adaptation and retain every prescribed state and curve point.

    Args:
        model: Three-parameter known-prechange polar or Cartesian function.
        data: TaskDataV1 or mapping with train/validation/grid arrays.
        optimizer_name: Adam or momentum-free SGD, reset for this fit.
        learning_rate: The prescribed pilot candidate or locked optimizer rate.
        penalty: Explicit amplitude coefficient; native decay is always zero.
        updates: Exact update budget, with no early stopping.
        record_every: Record step zero and each interval (also the final step).
        device: Training device, CUDA by default when available.
        deadline: Absolute time.perf_counter deadline for the invocation.

    Returns:
        A summary/artifacts bundle. Failures retain partial evidence and are not
        converted to successful selected-checkpoint fits.
    """
    device = _device(device)
    model.to(device=device, dtype=torch.float32)
    bundle: dict[str, Any] = {"summary": {"status": "running", "stage": "A", "model_id": getattr(model, "model_id", type(model).__name__),
                                             "model_class": type(model).__name__, "class_version": getattr(model, "version", type(model).__name__), "parameter_count": parameter_count(model),
                                             "optimizer": optimizer_name, "learning_rate": learning_rate, "penalty": penalty, "native_weight_decay": 0,
                                             "updates_requested": updates, "updates_completed": 0, "record_every": record_every,
                                             "history": [], "update_history": [], "training_dtype": "float32", "metric_dtype": "float64"},
                              "artifacts": {"initial_state": _state(model), "recorded_states": {}, "recorded_grid_predictions": {}, "recorded_predictions": {}}}
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    start, step, optimizer = _timing_start(device), 0, None
    try:
        if updates < 0 or record_every < 1 or parameter_count(model) != 3:
            raise ValueError("A needs nonnegative updates, positive recording interval and 3 parameters")
        arrays, parameters = _arrays(data)
        optimizer, manifest = _optimizer(model, "A", optimizer_name, learning_rate)
        summary["optimizer_parameters"] = manifest
        x, y = arrays["train_x"].to(device, dtype=torch.float32), arrays["train_y"].to(device, dtype=torch.float32)
        _check_state(model)

        def record() -> None:
            predictions = {key: _predict(model, arrays[f"{key}_x"]) for key in ("train", "validation", "grid")}
            row = {"step": step, "train_mse": (predictions["train"] - arrays["train_y"]).square().mean().item(),
                   "validation_mse": (predictions["validation"] - arrays["validation_y"]).square().mean().item(),
                   **regression_metrics(predictions["grid"], arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"]),
                   **_coefficient_metrics(model, parameters)}
            summary["history"].append(row)
            artifacts["recorded_states"][str(step)] = _state(model)
            artifacts["recorded_grid_predictions"][str(step)] = predictions["grid"]
            artifacts["recorded_predictions"][str(step)] = predictions

        record()
        for step in range(1, updates + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Invocation ceiling reached before next A update")
            row = _gradient_step(model, optimizer, x, y, "A", penalty, None)
            summary["update_history"].append({"step": step, **row})
            summary["updates_completed"] = step
            if step % record_every == 0 or step == updates:
                record()
        artifacts["final_state"] = _state(model)
        artifacts["selected_state"] = _state(model)
        artifacts["predictions"] = {key: _predict(model, arrays[f"{key}_x"]) for key in ("train", "validation", "grid")}
        history = summary["history"]
        variance = arrays["grid_clean"].var(unbiased=False).item()
        steps = np.array([row["step"] for row in history])
        integrate = getattr(np, "trapezoid", np.trapz)
        summary["normalized_clean_curve_area"] = float(integrate([row["clean_mse"] for row in history], steps) / (updates * variance)) if updates else None
        summary["normalized_validation_curve_area"] = float(integrate([row["validation_mse"] for row in history], steps) / (updates * variance)) if updates else None
        summary["primary_error"] = summary["normalized_clean_curve_area"]
        summary["validation_curve_area_normalized"] = summary["normalized_validation_curve_area"]
        summary["final_metrics"] = history[-1]
        summary["undefined_phase_count"] = sum(row.get("phase_error_defined") is False for row in history)
        summary["diagnostics"] = {label: a_coordinate_diagnostics(model, state, arrays, penalty)
                                  for label, state in (("initial", artifacts["initial_state"]), ("final", artifacts["final_state"]))}
        summary.update(status="completed", selected_step=updates, stop_reason="fixed_update_budget")
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        _failure(bundle, error, step)
    finally:
        artifacts.setdefault("final_state", _state(model))
        if optimizer is not None:
            artifacts["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
    return _finish(bundle, model, start, device)


def train_b(
    model: nn.Module, data: Any, *, learning_rate: float, batch_seed: int | None = None,
    batch_indices: Any = None,
    max_updates: int = 600, batch_size: int = 64, validation_every: int = 20,
    patience_evaluations: int = 5, min_delta: float = 1e-6, readout_penalty: float = 1e-4,
    gradient_clip: float = 1., device: Any = None, deadline: float | None = None,
) -> dict[str, Any]:
    """Train a B arm with shared batch order and validation-only state selection.

    Initialization is eligible. The lowest validation MSE selects the checkpoint;
    min_delta controls the separate patience anchor. Final state is saved before restoring
    selection. No clean/test target participates in the stopping decision.
    """
    from src.methodology_tasks import cyclic_batch_indices

    penalty = readout_penalty
    device = _device(device)
    model.to(device=device, dtype=torch.float32)
    bundle: dict[str, Any] = {"summary": {"status": "running", "stage": "B", "model_id": getattr(model, "model_id", type(model).__name__),
                                             "model_class": type(model).__name__, "class_version": 1, "parameter_count": parameter_count(model),
                                             "optimizer": "Adam", "learning_rate": learning_rate, "penalty": penalty, "native_weight_decay": 0,
                                             "max_updates": max_updates, "updates_completed": 0, "batch_seed": int(batch_seed) if batch_seed is not None else None, "batch_size": batch_size,
                                             "validation_every": validation_every, "patience_evaluations": patience_evaluations,
                                             "min_delta": min_delta, "gradient_clip": gradient_clip, "history": [], "update_history": [],
                                             "training_dtype": "float32", "metric_dtype": "float64"},
                              "artifacts": {"initial_state": _state(model), "checkpoint_states": {}, "validation_predictions": {}}}
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    start, step, optimizer = _timing_start(device), 0, None
    try:
        if max_updates < 0 or min(batch_size, validation_every, patience_evaluations) < 1:
            raise ValueError("Invalid B update, batch or selection budget")
        if not math.isfinite(min_delta) or min_delta < 0 or not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("Invalid B min_delta or gradient clip")
        if parameter_count(model) != 49:
            raise ValueError("B gradient arms require exactly 49 learned scalars")
        arrays, _ = _arrays(data)
        optimizer, manifest = _optimizer(model, "B", "Adam", learning_rate)
        summary["optimizer_parameters"] = manifest
        x, y = arrays["train_x"].to(device, dtype=torch.float32), arrays["train_y"].to(device, dtype=torch.float32)
        if batch_indices is None:
            if batch_seed is None:
                raise ValueError("Provide the recorded batch seed or explicit shared batch indices")
            indices = cyclic_batch_indices(len(x), batch_size, max_updates, batch_seed) if max_updates else np.empty((0, batch_size), dtype=np.int64)
        else:
            indices = np.asarray(batch_indices)
            if indices.shape != (max_updates, batch_size) or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Shared batch indices must have integer (max_updates,batch_size) shape")
            if (indices < 0).any() or (indices >= len(x)).any():
                raise ValueError("Shared batch indices exceed training sample range")
        artifacts["batch_indices"] = torch.as_tensor(indices).clone()
        _check_state(model)
        initial_prediction = _predict(model, arrays["validation_x"])
        initial_val = (initial_prediction - arrays["validation_y"]).square().mean().item()
        best, significant_best, selected_step, stale = initial_val, initial_val, 0, 0
        artifacts["selected_state"] = _state(model)
        artifacts["checkpoint_states"]["0"] = _state(model)
        artifacts["validation_predictions"]["0"] = initial_prediction
        summary["history"].append({"step": 0, "validation_mse": best, "selected": True, "significant_improvement": True})
        summary["initial_metrics"] = regression_metrics(_predict(model, arrays["grid_x"]), arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"])
        summary["initial_boundary_audit"] = b_geometry_diagnostics(model, arrays["grid_x"])
        for step in range(1, max_updates + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Invocation ceiling reached before next B update")
            batch = torch.as_tensor(indices[step - 1], dtype=torch.long, device=device)
            row = _gradient_step(model, optimizer, x[batch], y[batch], "B", penalty, gradient_clip)
            summary["update_history"].append({"step": step, **row})
            summary["updates_completed"] = step
            if step % validation_every == 0 or step == max_updates:
                validation_prediction = _predict(model, arrays["validation_x"])
                validation = (validation_prediction - arrays["validation_y"]).square().mean().item()
                artifacts["checkpoint_states"][str(step)] = _state(model)
                artifacts["validation_predictions"][str(step)] = validation_prediction
                selected = validation < best
                if selected:
                    best, selected_step = validation, step
                    artifacts["selected_state"] = _state(model)
                significant = validation < significant_best - min_delta
                if significant:
                    significant_best, stale = validation, 0
                else:
                    stale += 1
                summary["history"].append({"step": step, "validation_mse": validation, "selected": selected,
                                          "significant_improvement": significant})
                if stale >= patience_evaluations:
                    break
        artifacts["final_state"] = _state(model)
        model.load_state_dict(artifacts["selected_state"])
        model.eval()
        artifacts["predictions"] = {key: _predict(model, arrays[f"{key}_x"]) for key in ("train", "validation", "grid")}
        summary["metrics"] = regression_metrics(artifacts["predictions"]["grid"], arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"])
        summary["primary_error"] = summary["metrics"]["normalized_clean_mse"]
        summary["feature_diagnostics"] = b_feature_diagnostics(model, arrays)
        summary["boundary_audit"] = b_geometry_diagnostics(model, arrays["grid_x"])
        summary.update(status="completed", selected_step=selected_step, best_validation_mse=best,
                       stop_reason="patience" if stale >= patience_evaluations else "maximum_updates")
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        _failure(bundle, error, step)
    finally:
        artifacts.setdefault("final_state", _state(model))
        if optimizer is not None:
            artifacts["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
    return _finish(bundle, model, start, device)


def _linear_solve(features: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    """Float64 minimum-norm solve; augment only non-bias rows for ridge."""
    design = np.column_stack((np.asarray(features, dtype=np.float64), np.ones(len(features))))
    target = np.asarray(target, dtype=np.float64)
    if penalty:
        regularizer = np.zeros((design.shape[1] - 1, design.shape[1]), dtype=np.float64)
        regularizer[:, :-1] = np.eye(design.shape[1] - 1) * math.sqrt(len(features) * penalty)
        design = np.concatenate((design, regularizer))
        target = np.concatenate((target, np.zeros((len(regularizer), target.shape[1]))))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Nonfinite float64 least-squares coefficients")
    return coefficients


def closed_form_a(data: Any, penalty: float) -> dict[str, Any]:
    """Solve one unique A diagnostic endpoint, independently of optimizer arms."""
    if penalty < 0 or not math.isfinite(penalty):
        raise ValueError("Penalty must be finite and nonnegative")
    arrays, _ = _arrays(data)
    features = torch.cat((arrays["train_x"].cos(), arrays["train_x"].sin()), -1).numpy()
    coefficients = _linear_solve(features, arrays["train_y"].numpy(), penalty)
    predictions = {}
    for split in ("train", "validation", "grid"):
        theta = arrays[f"{split}_x"].numpy()
        predictions[split] = torch.from_numpy(coefficients[0] * np.cos(theta) + coefficients[1] * np.sin(theta) + coefficients[2])
    objective = 0.5 * (predictions["train"] - arrays["train_y"]).square().mean().item() + 0.5 * penalty * float(np.square(coefficients[:2]).sum())
    return {"summary": {"status": "completed", "penalty": penalty, "alpha": len(features) * penalty,
                         "parameter_count": 3, "model_class": "CartesianClosedFormReferenceV1", "class_version": 1,
                         "scope": "diagnostic optimization endpoint; no gradient training arm",
                         "solver": "numpy.linalg.lstsq SVD augmented design minimum norm; unpenalized bias",
                         "training_objective": objective,
                         "metrics": regression_metrics(predictions["grid"], arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"])},
            "artifacts": {"coefficients": torch.from_numpy(coefficients), "predictions": predictions}}


def fit_ridge(model_id: str, data: Any, *, lambdas: Sequence[float] = (0., 1e-4, .01, 1.),
              tie_tolerance: float = 1e-12, deadline: float | None = None) -> dict[str, Any]:
    """Fit all four prescribed ridge candidates and select on noisy validation."""
    bundle: dict[str, Any] = {"summary": {"status": "running", "stage": "B", "model_id": model_id,
                                             "parameter_count": 49, "feature_count": 48, "history": [], "solves_completed": 0,
                                             "solver": "numpy.linalg.lstsq SVD augmented design minimum norm; unpenalized bias",
                                             "training_dtype": "float64_closed_form", "metric_dtype": "float64"},
                              "artifacts": {"candidate_states": {}}}
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    start, step = time.perf_counter(), 0
    try:
        if not lambdas or any(v < 0 or not math.isfinite(v) for v in lambdas) or len(set(lambdas)) != len(lambdas):
            raise ValueError("Ridge candidates must be distinct finite nonnegative lambdas")
        if tie_tolerance < 0 or not math.isfinite(tie_tolerance):
            raise ValueError("Invalid ridge tie tolerance")
        arrays, _ = _arrays(data)
        bank = FixedFeaturesV1(model_id)
        features = {split: bank(arrays[f"{split}_x"]).numpy() for split in ("train", "validation", "grid")}
        for step, penalty in enumerate(sorted(lambdas), start=1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Invocation ceiling reached before next ridge solve")
            coefficients = _linear_solve(features["train"], arrays["train_y"].numpy(), penalty)
            validation = features["validation"] @ coefficients[:-1] + coefficients[-1]
            score = float(np.square(validation - arrays["validation_y"].numpy()).mean())
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite ridge validation score")
            artifacts["candidate_states"][str(penalty)] = torch.from_numpy(coefficients.copy())
            summary["history"].append({"lambda": penalty, "alpha": len(features["train"]) * penalty, "validation_mse": score})
            summary["solves_completed"] = step
        minimum = min(row["validation_mse"] for row in summary["history"])
        selected = min((row for row in summary["history"] if row["validation_mse"] <= minimum + tie_tolerance),
                       key=lambda row: row["lambda"])
        best, selected_lambda = selected["validation_mse"], selected["lambda"]
        coefficients = artifacts["candidate_states"][str(selected_lambda)].numpy()
        artifacts["selected_state"] = {"coefficients": torch.from_numpy(coefficients.copy()), **_state(bank)}
        artifacts["final_state"] = artifacts["selected_state"]
        artifacts["initial_state"] = {"coefficients": torch.zeros_like(torch.from_numpy(coefficients)), **_state(bank)}
        predictions = {split: torch.from_numpy(values @ coefficients[:-1] + coefficients[-1]) for split, values in features.items()}
        artifacts["predictions"] = predictions
        summary["metrics"] = regression_metrics(predictions["grid"], arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"])
        summary["primary_error"] = summary["metrics"]["normalized_clean_mse"]
        predictor = KernelRegressorV1(bank, coefficients[:-1].T, coefficients[-1], model_id).double()
        # AffineV1 deliberately initializes float32 for gradient models; restore
        # solve coefficients exactly so diagnostic rounding never changes ridge.
        with torch.no_grad():
            predictor.readout.weight.copy_(torch.from_numpy(coefficients[:-1].T))
            predictor.readout.bias.copy_(torch.from_numpy(coefficients[-1]))
        summary["feature_diagnostics"] = b_feature_diagnostics(predictor, arrays)
        summary["boundary_audit"] = b_geometry_diagnostics(predictor, arrays["grid_x"])
        summary.update(status="completed", selected_lambda=selected_lambda, best_validation_mse=best)
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        _failure(bundle, error, step)
    summary.update(runtime_seconds=time.perf_counter() - start, device="cpu", peak_cuda_allocated_bytes=None)
    summary["time_seconds"] = summary["runtime_seconds"]
    return bundle


def a_coordinate_diagnostics(model: nn.Module, state: Mapping[str, Tensor], arrays: Mapping[str, Tensor], penalty: float) -> dict[str, Any]:
    """Check stored-state function/objective/gradient identities, without refits.

    Historical phase/raw-concentration L2 is reported only as a coordinate
    dependence diagnostic. It never enters the adaptation objective.
    """
    reference = copy.deepcopy(model).cpu().double()
    reference.load_state_dict(state)
    reference.double()
    x, y = arrays["train_x"], arrays["train_y"]
    rotation = math.pi / 3

    def objective_and_grad(candidate: nn.Module, angles: Tensor, historical: bool = False) -> tuple[float, dict[str, Tensor]]:
        loss = 0.5 * (candidate(angles) - y).square().mean()
        if historical:
            loss = loss + 0.5 * 0.001 * (candidate.phase.square().sum() + candidate.raw_kappa.square().sum())
        else:
            loss = loss + explicit_penalty(candidate, "A", penalty)
        names, params = zip(*candidate.named_parameters())
        grads = torch.autograd.grad(loss, params)
        return loss.item(), dict(zip(names, [g.detach() for g in grads]))

    base_objective, base_grads = objective_and_grad(reference, x)
    result: dict[str, Any] = {"dtype": "float64 stored-state checks", "additional_fits": 0, "rotation": rotation,
                              "scope": "functional/objective/gradient identities only; no optimizer-origin robustness claim"}
    rotated = copy.deepcopy(reference)
    with torch.no_grad():
        if hasattr(reference, "phase"):
            rotated.phase.sub_(rotation)
        else:
            a, b = reference.weight_cos, reference.weight_sin
            rotated.weight_cos.copy_(a * math.cos(rotation) - b * math.sin(rotation))
            rotated.weight_sin.copy_(a * math.sin(rotation) + b * math.cos(rotation))
    obj, grads = objective_and_grad(rotated, x + rotation)
    if not hasattr(reference, "phase"):
        ga, gb = grads["weight_cos"].clone(), grads["weight_sin"].clone()
        grads["weight_cos"] = ga * math.cos(rotation) + gb * math.sin(rotation)
        grads["weight_sin"] = -ga * math.sin(rotation) + gb * math.cos(rotation)
    result.update(rotation_output_max_abs=(rotated(x + rotation) - reference(x)).abs().max().item(),
                  rotation_objective_abs=abs(obj - base_objective),
                  rotation_gradient_mapped_max_abs=max((grads[name] - base_grads[name]).abs().max().item() for name in grads))
    if hasattr(reference, "phase"):
        shifted = copy.deepcopy(reference)
        with torch.no_grad():
            shifted.phase.add_(2 * math.pi)
        obj, grads = objective_and_grad(shifted, x)
        legacy, legacy_grads = objective_and_grad(reference, x, True)
        legacy_shifted, legacy_shifted_grads = objective_and_grad(shifted, x, True)
        legacy_rotated, legacy_rotated_grads = objective_and_grad(rotated, x + rotation, True)
        result.update(parameter_phase_shift=2 * math.pi,
                      phase_shift_output_max_abs=(shifted(x) - reference(x)).abs().max().item(),
                      phase_shift_objective_abs=abs(obj - base_objective),
                      phase_shift_gradient_max_abs=max((grads[name] - base_grads[name]).abs().max().item() for name in grads),
                      historical_decay_coefficient=0.001,
                      historical_decay_phase_shift_objective_difference=legacy_shifted - legacy,
                      historical_decay_phase_shift_gradient_difference=(legacy_shifted_grads["phase"] - legacy_grads["phase"]).item(),
                      historical_decay_rotation_objective_difference=legacy_rotated - legacy,
                      historical_decay_rotation_gradient_difference=(legacy_rotated_grads["phase"] - legacy_grads["phase"]).item())
    else:
        result["phase_shift_diagnostic"] = "not applicable: Cartesian has no trainable phase coordinate"
    return result


def b_feature_diagnostics(model: nn.Module, arrays: Mapping[str, Tensor]) -> dict[str, Any]:
    """Report kernel geometry, vanishing features and fixed feature capacity."""
    parameter = next(model.parameters())
    values: dict[str, Any] = {}
    with torch.no_grad():
        for split in ("train", "grid"):
            features = model.features(arrays[f"{split}_x"].to(parameter.device, dtype=parameter.dtype))
            _finite(features, "features")
            values[f"{split}_feature_absolute_below_1e_minus12_fraction"] = (features.abs() < 1e-12).double().mean().item()
            values["feature_count"] = features.shape[-1]
    bank = getattr(model, "bank", None)
    if bank is not None and hasattr(bank, "centers"):
        centers = bank.centers.detach().cpu().double()
        concentrations = bank.concentrations
        if callable(concentrations):
            concentrations = concentrations()
        widths = concentrations.detach().cpu().double()
        values.update(centers_raw=centers.flatten().tolist(), centers_wrapped=((centers + math.pi) % (2 * math.pi) - math.pi).flatten().tolist(),
                      concentrations=widths.flatten().tolist(), concentration_min=widths.min().item(),
                      concentration_median=widths.median().item(), concentration_max=widths.max().item())
    return values


def b_geometry_diagnostics(model: nn.Module, grid_x: Tensor) -> dict[str, Any]:
    """Audit periodic boundary and mapped rotations at stored states only."""
    reference = copy.deepcopy(model).cpu().double()
    rotated = copy.deepcopy(reference)
    angle = math.pi / 3
    model_id = reference.model_id
    with torch.no_grad():
        if model_id in ("learned_vm", "learned_geodesic", "fixed_vm_ridge"):
            rotated.bank.centers.add_(angle)
        elif model_id == "fourier_ridge":
            h = reference.bank.harmonics
            a, b = reference.readout.weight[:, :24], reference.readout.weight[:, 24:]
            rotated.readout.weight[:, :24].copy_(a * (h * angle).cos() - b * (h * angle).sin())
            rotated.readout.weight[:, 24:].copy_(a * (h * angle).sin() + b * (h * angle).cos())
        elif model_id == "sincos_mlp":
            a, b = reference.hidden.weight[:, 0], reference.hidden.weight[:, 1]
            rotated.hidden.weight[:, 0].copy_(a * math.cos(angle) - b * math.sin(angle))
            rotated.hidden.weight[:, 1].copy_(a * math.sin(angle) + b * math.cos(angle))
        elif model_id == "raw_angle_mlp":
            rotated.hidden.bias.sub_(reference.hidden.weight[:, 0] * angle / math.pi)
        else:
            raise ValueError("Unknown model for mapped geometry diagnostic")
        base = reference(grid_x)
        difference = reference(grid_x + 2 * math.pi) - base
        endpoint = reference(torch.tensor([[-math.pi], [math.pi]], dtype=torch.float64))
        shifted_x = grid_x + angle
        if model_id != "raw_angle_mlp":
            shifted_x = torch.remainder(shifted_x + math.pi, 2 * math.pi) - math.pi
        return {"dtype": "float64 stored-state checks", "rotation": angle, "additional_fits": 0,
                "scope": "mapped geometry only; no optimizer robustness or independent task claim",
                "rotation_coordinate_map": "unwrapped theta+pi/3" if model_id == "raw_angle_mlp" else "wrap(theta+pi/3)",
                "rotation_output_max_abs": (rotated(shifted_x) - base).abs().max().item(),
                "two_pi_output_max_abs": difference.abs().max().item(),
                "periodicity_expected": model_id != "raw_angle_mlp",
                "endpoint_prediction_abs_difference": (endpoint[0] - endpoint[1]).abs().item()}
