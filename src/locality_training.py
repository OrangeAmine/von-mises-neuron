"""Timed float64 locality fitting, with immutable checkpoint evidence.

Every allowance selects only checkpoints whose validation, diagnostics and CPU
state copies completed within that allowance. Clean grid targets are first read
after optimization and all checkpoint decisions have finished. A fit is never
retried and numerical failures never become successful early stopping.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.locality_models import (
    circular_features, fourier_features, make_basis, predict_readout,
    readout_diagnostics, solve_readout,
)


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}")


def _finite_scalars(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite {name}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_scalars(item, name + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_scalars(item, name + "." + str(index))


def _column(value: Any, name: str, device: torch.device) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float64, device=device).detach()
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[1] != 1 or not len(result):
        raise ValueError(f"{name} must have nonempty (N,1) shape")
    _finite(result, name)
    return result


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def allowance_key(seconds: float) -> str:
    """Return the stable dictionary key used for a registered time allowance."""
    return str(float(seconds))


class _LocalityFit(nn.Module):
    """Separate geometry, fitted readout coefficients and Adam membership."""

    def __init__(self, specification: Mapping[str, Any], initialization: Mapping[str, Any],
                 device: torch.device) -> None:
        super().__init__()
        self.model_id = str(specification["id"])
        self.kind = str(specification["kind"])
        self.basis_count = int(specification["features"])
        self.fitted_parameter_count = int(specification["parameters"])
        self.is_fourier = self.model_id.startswith("fourier")
        if self.is_fourier:
            if self.basis_count % 2:
                raise ValueError("Fourier feature count must be even")
            self.basis = None
        else:
            geometry = initialization[self.model_id] if self.model_id in ("fixed32", "fixed48") else initialization
            self.basis = make_basis(geometry["centers"], geometry["kappas"],
                                    mode=specification.get("geometry", "fixed"), device=device)
            if self.basis.num_kernels != self.basis_count:
                raise ValueError("Initialization does not match registered feature count")
        weight = torch.zeros((self.basis_count, 1), dtype=torch.float64, device=device)
        bias = torch.zeros(1, dtype=torch.float64, device=device)
        if self.kind == "joint":
            self.readout_weight = nn.Parameter(weight)
            self.readout_bias = nn.Parameter(bias)
        else:
            self.register_buffer("readout_weight", weight)
            self.register_buffer("readout_bias", bias)
        if self.kind not in ("ridge", "vp", "joint"):
            raise ValueError("Unknown fit kind")
        self.readout_current = False

    def features(self, x: Tensor) -> Tensor:
        return fourier_features(x, self.basis_count // 2) if self.is_fourier else self.basis(x)

    @torch.no_grad()
    def refresh_readout(self, features: Tensor, y: Tensor, penalty: float) -> None:
        weight, bias = solve_readout(features, y, penalty)
        self.readout_weight.copy_(weight)
        self.readout_bias.copy_(bias)
        self.readout_current = True

    def snapshot(self) -> dict[str, Any]:
        state = {"model_id": self.model_id, "kind": self.kind,
                 "basis_type": "fourier" if self.is_fourier else "circular",
                 "basis_count": self.basis_count, "fitted_parameter_count": self.fitted_parameter_count,
                 "optimized_parameter_count": sum(p.numel() for p in self.parameters()),
                 "readout_matches_geometry": self.readout_current,
                 "readout_weight": self.readout_weight, "readout_bias": self.readout_bias,
                 "basis_state": {} if self.basis is None else self.basis.state_dict()}
        if self.basis is not None:
            state.update(centers=self.basis.centers, kappas=self.basis.kappa)
        return _cpu_copy(state)


def predict_locality_state(state: Mapping[str, Any], x: Any,
                           *, device: Any = "cpu") -> Tensor:
    """Replay a saved checkpoint without constructing an optimizer or refitting."""
    destination = torch.device(device)
    angles = _column(x, "angles", destination)
    if state["basis_type"] == "fourier":
        features = fourier_features(angles, int(state["basis_count"]) // 2)
    else:
        features = circular_features(angles, torch.as_tensor(state["centers"], device=destination),
                                     torch.as_tensor(state["kappas"], device=destination))
    weight = torch.as_tensor(state["readout_weight"], dtype=torch.float64, device=destination)
    bias = torch.as_tensor(state["readout_bias"], dtype=torch.float64, device=destination)
    result = predict_readout(features, weight, bias)
    _finite(result, "saved-state predictions")
    return result


def _objective(prediction: Tensor, target: Tensor, weight: Tensor,
               penalty: float) -> Tensor:
    return .5 * (prediction - target).square().mean() + .5 * penalty * weight.square().sum()


def _update(model: _LocalityFit, optimizer: torch.optim.Optimizer,
            x: Tensor, y: Tensor, penalty: float, clip: float | None) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    features = model.features(x)
    if model.kind == "vp":
        # The differentiable solve implements the same reduced objective as the
        # envelope gradient; only registered geometry enters the optimizer.
        weight, bias = solve_readout(features, y, penalty)
    else:
        weight, bias = model.readout_weight, model.readout_bias
    prediction = predict_readout(features, weight, bias)
    loss = _objective(prediction, y, weight, penalty)
    _finite(loss, "training objective")
    loss.backward()
    named = list(model.named_parameters())
    if any(p.grad is None for _, p in named):
        raise FloatingPointError("Missing optimized-parameter gradient")
    norms = {name: torch.linalg.vector_norm(p.grad) for name, p in named}
    total = torch.linalg.vector_norm(torch.stack(list(norms.values())))
    _finite(total, "gradient norm")
    geometry_norms = [v for name, v in norms.items() if name.startswith("basis.")]
    geometry_total = torch.linalg.vector_norm(torch.stack(geometry_norms))
    row = {"objective": loss.item(), "training_mse": (prediction-y).square().mean().item(),
           "readout_penalty": .5 * penalty * weight.square().sum().item(),
           "gradient_norm_before_clip": total.item(), "geometry_gradient_norm": geometry_total.item(),
           "parameter_gradient_norms": {name: value.item() for name, value in norms.items()}}
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), clip, error_if_nonfinite=True)
    optimizer.step()
    for name, value in model.state_dict().items():
        _finite(value, "updated state " + name)
    model.readout_current = model.kind == "joint"
    return row


def _metrics(prediction: Tensor, arrays: Mapping[str, Any]) -> dict[str, Any]:
    clean = _column(arrays["grid_clean"], "grid_clean", torch.device("cpu"))
    x = _column(arrays["grid_x"], "grid_x", torch.device("cpu"))
    if prediction.shape != clean.shape or x.shape != clean.shape:
        raise ValueError("Grid arrays must have identical shapes")
    variance = clean.var(unbiased=False).item()
    if variance <= 0:
        raise ValueError("Clean grid variance must be positive")
    error = prediction - clean
    band = x.abs() >= .9 * math.pi
    result = {"clean_mse": error.square().mean().item(), "clean_variance": variance,
              "normalized_clean_mse": error.square().mean().item()/variance,
              "mean_absolute_error": error.abs().mean().item(), "max_absolute_error": error.abs().max().item(),
              "boundary_mse": error[band].square().mean().item() if bool(band.any()) else None,
              "boundary_points": int(band.sum()), "boundary_band": "abs(theta)>=0.9*pi"}
    if "grid_noisy" in arrays:
        noisy = _column(arrays["grid_noisy"], "grid_noisy", torch.device("cpu"))
        if noisy.shape != prediction.shape:
            raise ValueError("Noisy grid shape differs")
        result["noisy_rmse"] = (prediction-noisy).square().mean().sqrt().item()
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite postfit grid metric")
    return result


def fit_locality(model_id: str, arrays: Mapping[str, Any], initialization: Mapping[str, Any],
                 config: Mapping[str, Any], *, learning_rate: float | None = None,
                 device: Any = None, clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """Fit one registered model and select checkpoints under every time cap.

    Args:
        model_id: One of the ten explicitly registered v2 model IDs.
        arrays: Shared train/validation/grid arrays. Clean targets never select.
        initialization: Explicit shared circular centers and effective kappas,
            plus ``fixed32``/``fixed48`` geometry dictionaries.
        config: Registered protocol. Small CPU unit tests may supply tiny caps.
        learning_rate: Pilot-selected Adam rate; unused for a fixed model.
        device: CUDA when available unless explicitly supplied.
        clock: Optional monotonic callable for deterministic timing tests.

    Returns:
        JSON-finite summary and tensor/primitive artifacts. Unavailable caps and
        failed/interrupted fits remain explicit; no exception causes a retry.
        Each normal stop occurs after an atomic checkpoint block, retaining any
        late checkpoint but never admitting it to an earlier allowance.
    """
    summary: dict[str, Any] = {"model_id": model_id, "status": "running", "history": [],
                               "update_history": [], "selections": {}, "updates_completed": 0}
    artifacts: dict[str, Any] = {"checkpoint_states": {}, "checkpoint_predictions": {},
                                 "selected_states": {}, "selected_predictions": {}}
    bundle = {"summary": summary, "artifacts": artifacts}
    model = optimizer = None
    started = None
    last_elapsed = 0.
    attempted_step = 0
    now = clock or time.perf_counter
    destination = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def elapsed() -> float:
        nonlocal last_elapsed
        if destination.type == "cuda":
            torch.cuda.synchronize(destination)
        value = float(now()) - started
        if not math.isfinite(value) or value < last_elapsed:
            raise ValueError("Clock must be finite and monotonic")
        last_elapsed = value
        return value

    try:
        specification = next(row for row in config["models"] if row["id"] == model_id)
        training = config["training"]
        caps = [float(value) for value in training["time_budgets_seconds"]]
        if not caps or caps != sorted(set(caps)) or any(not math.isfinite(v) or v <= 0 for v in caps):
            raise ValueError("Time allowances must be strictly increasing finite positive values")
        primary = float(training["primary_time_seconds"])
        if primary not in caps:
            raise ValueError("Primary allowance must be registered")
        penalty = float(training["penalty"])
        max_updates, every = int(training["max_updates"]), int(training["validation_every"])
        if max_updates < 0 or every < 1 or not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("Invalid update, validation or regularization configuration")
        if (training["dtype"] != "float64" or training["native_weight_decay"] != 0
                or training["bias_penalized"] or training["early_stopping"]):
            raise ValueError("Unsupported departure from the registered objective/stopping policy")
        if training["optimizer"] != "Adam" or training["width_parameterization"] != "softplus_unconstrained_raw_kappa":
            raise ValueError("Unsupported optimizer or width coordinate")
        clip = training["clip_gradient_norm"]
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError("Gradient clipping must be disabled or finite positive")
        if specification["kind"] != "ridge" and (learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("Adaptive models require a positive finite learning rate")
        if destination.type == "cuda":
            torch.cuda.synchronize(destination)
            torch.cuda.reset_peak_memory_stats(destination)
        started = float(now())
        if not math.isfinite(started):
            raise ValueError("Clock start must be finite")
        tensors = {name: _column(arrays[name], name, destination)
                   for name in ("train_x", "train_y", "validation_x", "validation_y")}
        for x, y in [("train_x","train_y"),("validation_x","validation_y")]:
            if tensors[x].shape != tensors[y].shape:
                raise ValueError("Input and target shapes differ")
        model = _LocalityFit(specification, initialization, destination)
        model.refresh_readout(model.features(tensors["train_x"]), tensors["train_y"], penalty)
        named = list(model.named_parameters())
        if specification["kind"] != "ridge":
            optimizer = torch.optim.Adam([p for _, p in named], lr=learning_rate,
                betas=tuple(training["adam_betas"]), eps=training["adam_epsilon"],
                weight_decay=0, amsgrad=False)
        optimized = sum(p.numel() for _, p in named)
        expected_fitted = model.basis_count + 1 + sum(p.numel() for name,p in named if name.startswith("basis."))
        if expected_fitted != model.fitted_parameter_count:
            raise ValueError("Registered fitted parameter count does not match constructed model")
        summary.update(kind=model.kind, basis_count=model.basis_count, fitted_parameter_count=expected_fitted,
                       optimized_parameter_count=optimized, parameter_count=expected_fitted,
                       optimizer_parameters=[{"name":name,"shape":list(p.shape),"count":p.numel()} for name,p in named],
                       learning_rate=learning_rate if optimizer else None, penalty=penalty,
                       ridge_alpha=len(tensors["train_x"])*penalty, dtype="float64", device=str(destination),
                       max_updates=max_updates, validation_every=every, time_budgets_seconds=caps,
                       primary_time_seconds=primary, initialization_readout="exact_regularized_solve",
                       timing_scope=training["timing"], n_train=len(tensors["train_x"]),
                       n_validation=len(tensors["validation_x"]))
        selected_rows: dict[str, dict[str, Any]] = {}

        def checkpoint(step: int) -> None:
            checkpoint_started = elapsed()
            with torch.no_grad():
                features = model.features(tensors["train_x"])
                if model.kind == "vp" and step:
                    model.refresh_readout(features, tensors["train_y"], penalty)
                train_prediction = predict_readout(features, model.readout_weight, model.readout_bias)
                validation_prediction = predict_readout(model.features(tensors["validation_x"]),
                                                        model.readout_weight, model.readout_bias)
                _finite(train_prediction, "training predictions")
                _finite(validation_prediction, "validation predictions")
                row = {"step":step, "checkpoint_start_elapsed_seconds":checkpoint_started,
                       "train_mse":(train_prediction-tensors["train_y"]).square().mean().item(),
                       "validation_mse":(validation_prediction-tensors["validation_y"]).square().mean().item(),
                       "objective":_objective(train_prediction,tensors["train_y"],model.readout_weight,penalty).item(),
                       "readout_diagnostics":readout_diagnostics(features,tensors["train_y"],penalty,
                                                                 model.readout_weight,model.readout_bias)}
                _finite_scalars(row, "checkpoint diagnostics")
                state = model.snapshot()
                predictions = {"train":_cpu_copy(train_prediction),"validation":_cpu_copy(validation_prediction)}
                artifacts["checkpoint_states"][str(step)] = state
                artifacts["checkpoint_predictions"][str(step)] = predictions
                if step == 0:
                    artifacts["initial_state"] = state
                artifacts["final_state"] = state
                # Compute checkpoint comparisons inside the timed region. The
                # subsequent timestamp determines whether this candidate may be
                # committed for a cap; a late better candidate never displaces
                # the previously admissible one.
                improve = {allowance_key(cap): allowance_key(cap) not in selected_rows or
                           row["validation_mse"] < selected_rows[allowance_key(cap)]["validation_mse"] for cap in caps}
                # No state/array copying remains after this admission timestamp.
                row["elapsed_seconds"] = elapsed()
                row["admissible_time_budgets"] = [cap for cap in caps if row["elapsed_seconds"] <= cap]
                for cap in row["admissible_time_budgets"]:
                    key = allowance_key(cap)
                    if improve[key]:
                        selected_rows[key] = row
                summary["history"].append(row)

        checkpoint(0)
        if model.kind == "ridge":
            stop_reason = "fixed_closed_form"
        elif last_elapsed >= caps[-1]:
            stop_reason = "wall_time"
        else:
            stop_reason = "maximum_updates"
            for step in range(1,max_updates+1):
                attempted_step = step
                row = _update(model,optimizer,tensors["train_x"],tensors["train_y"],penalty,clip)
                row.update(step=step, elapsed_seconds=elapsed())
                summary["update_history"].append(row)
                summary["updates_completed"] = step
                if step % every == 0 or step == max_updates:
                    checkpoint(step)
                    if last_elapsed >= caps[-1]:
                        stop_reason = "wall_time"
                        break
        summary.update(stop_reason=stop_reason,fit_seconds=last_elapsed,fit_elapsed_seconds=last_elapsed,time_seconds=last_elapsed,
                       overshoot_seconds=max(0.,last_elapsed-caps[-1]),
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(destination) if destination.type=="cuda" else None)
        for cap in caps:
            key = allowance_key(cap)
            eligible = [row for row in summary["history"] if row["elapsed_seconds"] <= cap]
            selection: dict[str, Any] = {"allowance_seconds":cap,"eligible_checkpoints":len(eligible),
                                        "actual_trajectory_fit_seconds":last_elapsed,
                                        "completed_updates_within_allowance":sum(r["elapsed_seconds"]<=cap for r in summary["update_history"])}
            if not eligible:
                selection.update(status="budget_unavailable",selected_step=None,best_validation_mse=None)
            else:
                selected = selected_rows[key]
                selection.update(status="completed",selected_step=selected["step"],
                                 best_validation_mse=selected["validation_mse"],
                                 checkpoint_elapsed_seconds=selected["elapsed_seconds"],
                                 last_admissible_checkpoint_step=eligible[-1]["step"],
                                 last_admissible_checkpoint_seconds=eligible[-1]["elapsed_seconds"],
                                 checkpoint_budget_utilization=eligible[-1]["elapsed_seconds"]/cap)
                artifacts["selected_states"][key] = artifacts["checkpoint_states"][str(selected["step"])]
                artifacts["selected_predictions"][key] = dict(artifacts["checkpoint_predictions"][str(selected["step"])])
            summary["selections"][key] = selection
        primary_selection = summary["selections"][allowance_key(primary)]
        summary.update(status=primary_selection["status"],selected_step=primary_selection["selected_step"],
                       best_validation_mse=primary_selection["best_validation_mse"])
        # The fit and all model-selection decisions are over before reading any
        # clean grid target. Selected CPU snapshots are replayed without refits.
        score_start = time.perf_counter()
        for key,state in artifacts["selected_states"].items():
            grid_prediction = predict_locality_state(state,arrays["grid_x"])
            metrics = _metrics(grid_prediction,arrays)
            artifacts["selected_predictions"][key]["grid"] = grid_prediction
            summary["selections"][key].update(metrics=metrics,primary_error=metrics["normalized_clean_mse"])
        if "metrics" in primary_selection:
            summary.update(metrics=primary_selection["metrics"],primary_error=primary_selection["primary_error"])
        summary["postfit_scoring_seconds"] = time.perf_counter()-score_start
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed",
                       failure=f"{type(error).__name__}: {error}",failure_kind=type(error).__name__,failed_step=attempted_step)
        summary.pop("primary_error",None)
        if model is not None:
            try:
                artifacts["final_state"] = model.snapshot()
            except BaseException as snapshot_error:
                summary["final_snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
        if started is not None:
            try:
                elapsed()
            except BaseException as clock_error:
                summary["failure_timing_error"] = f"{type(clock_error).__name__}: {clock_error}"
        summary.update(fit_seconds=last_elapsed,fit_elapsed_seconds=last_elapsed,time_seconds=last_elapsed)
    return bundle


def warmup_locality(device: Any = None) -> dict[str, Any]:
    """Warm lazy Adam/float64 solve paths on a fixed disposable fixture.

    Call once before formal fit timing. This fixture is not a registered task,
    consumes no RNG and never scores a clean target or chooses a hyperparameter.
    """
    destination = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.perf_counter()
    x = torch.linspace(-math.pi,math.pi,8,dtype=torch.float64,device=destination)[:,None]
    y = x.sin()
    initialization = {"centers":torch.linspace(-math.pi,math.pi,4,dtype=torch.float64),
                      "kappas":torch.full((4,),4.,dtype=torch.float64)}
    for kind in ("vp","joint"):
        model = _LocalityFit({"id":kind+"_both","kind":kind,"geometry":"both","features":4,"parameters":13},
                              initialization,destination)
        model.refresh_readout(model.features(x),y,.001)
        optimizer = torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
        _update(model,optimizer,x,y,.001,None)
        features = model.features(x)
        model.refresh_readout(features,y,.001)
        readout_diagnostics(features,y,.001,model.readout_weight,model.readout_bias)
        model.snapshot()
    if destination.type == "cuda":
        torch.cuda.synchronize(destination)
    return {"device":str(destination),"seconds":time.perf_counter()-start,
            "scope":"disposable 8-angle/4-kernel VP and joint optimizer warmup; no RNG or registered task"}
