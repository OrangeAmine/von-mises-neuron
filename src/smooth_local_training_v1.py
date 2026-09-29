"""Bounded VM-N01 fits with timed, immutable checkpoint eligibility.

Initialization, transfer, optimization, noisy validation, and full CPU state
capture count against every allowance. Clean grid truth is accessed only after
checkpoint selection. Failed fits and late states remain in the result bundle.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from src.locality_models import predict_readout, solve_readout
from src.smooth_local_v1 import make_model, predict_smooth_local_state


def budget_key(seconds: float) -> str:
    """Return canonical allowance keys, e.g. '0.25' and '1'."""
    return format(float(seconds), "g")


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")


def _column(value: Any, name: str, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device).detach()
    if tensor.ndim != 2 or tensor.shape[1] != 1 or tensor.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty (N,1)")
    _finite(tensor, name)
    return tensor


def _objective(prediction: Tensor, target: Tensor, weight: Tensor, penalty: float) -> Tensor:
    return .5 * (prediction - target).square().mean() + .5 * penalty * weight.square().sum()


def _check_model(model: Any) -> None:
    for name, value in model.state_dict().items():
        _finite(value, "model state " + name)
    bank = getattr(model.basis, "bank", None)
    if bank is not None and not bool((bank.concentrations > 0).all()):
        # Softplus can underflow at extreme coordinates. This is a failed fit,
        # never an implicit concentration floor or altered objective.
        raise FloatingPointError("effective concentrations must remain positive")


def fit_smooth_local(spec: Mapping[str, Any], arrays: Mapping[str, Any],
                     rotation: float, learning_rate: float | None,
                     training: Mapping[str, Any], device: Any = None, *,
                     clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """Fit a registered model and preserve every fully captured checkpoint.

    Args:
        spec: Registered model specification including exact capacity counts.
        arrays: Train/validation x,y and postfit grid_x/grid_clean, all (N,1).
        rotation: Explicit shared initial center rotation.
        learning_rate: Adam rate for adaptive models; unused for ridge fits.
        training: Frozen training settings and time allowances in seconds.
        device: CUDA when available unless an explicit device is supplied.
        clock: Optional monotone clock for deterministic timing unit tests.

    Returns:
        A tensor-bearing evidence bundle. ``selected`` maps canonical budget
        keys to checkpoint indices or None. ``metrics`` has postfit clean scores
        per selected budget. Non-completed statuses never become success by
        retaining an earlier checkpoint. No fit is retried here.
    """
    time_clock = time.perf_counter if clock is None else clock
    total_started = time_clock()
    destination = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def now() -> float:
        if destination.type == "cuda":
            torch.cuda.synchronize(destination)
        return time_clock()

    budgets = sorted(float(value) for value in training["time_budgets_seconds"])
    if not budgets or len(set(budgets)) != len(budgets) or any(not math.isfinite(v) or v <= 0 for v in budgets):
        raise ValueError("time allowances must be distinct finite positive values")
    penalty = float(training["penalty"])
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    max_updates = int(training["max_updates"])
    every = int(training["validation_every"])
    if max_updates < 0 or every < 1:
        raise ValueError("update limit must be nonnegative and validation interval positive")
    if training.get("dtype", "float64") != "float64":
        raise ValueError("static fits require float64")
    if training.get("weight_decay", 0.) != 0.:
        raise ValueError("native weight decay must be zero; the readout penalty is explicit")
    result: dict[str, Any] = {
        "model_id": spec["id"], "status": "running", "device": str(destination),
        "dtype": "float64", "rotation": float(rotation), "learning_rate": learning_rate,
        "penalty": penalty, "basis_count": spec["features"],
        "fitted_parameter_count": spec["parameters"], "checkpoints": [],
        "selected": {budget_key(v): None for v in budgets}, "metrics": {},
        "failures": [], "updates": 0, "max_block_seconds": 0.,
        "fit_seconds": None, "total_seconds": None, "nonfit_seconds": None,
        "overshoot_seconds": None, "partial_state": None,
    }
    # The first CUDA synchronize is charged too, as are all following setup
    # operations; a caller may not exclude expensive initialization implicitly.
    started = total_started
    model = None
    interrupted: BaseException | None = None
    try:
        model = make_model(spec, rotation, device=destination)
        x = _column(arrays["train_x"], "train_x", destination)
        y = _column(arrays["train_y"], "train_y", destination)
        vx = _column(arrays["validation_x"], "validation_x", destination)
        vy = _column(arrays["validation_y"], "validation_y", destination)
        if x.shape != y.shape or vx.shape != vy.shape:
            raise ValueError("paired input and target arrays must have identical shapes")
        model.refresh_readout(model.features(x), y, penalty)
        optimizer = None
        if model.kind != "ridge":
            if learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("adaptive fits require a finite positive learning rate")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate,
                betas=tuple(training.get("adam_betas", (.9, .999))),
                eps=float(training.get("adam_epsilon", 1e-8)), weight_decay=0.)
        previous_elapsed = 0.

        @torch.no_grad()
        def checkpoint(step: int) -> None:
            nonlocal previous_elapsed
            if model.kind == "vp" and step > 0:
                model.refresh_readout(model.features(x), y, penalty)
            _check_model(model)
            validation_mse = (model(vx) - vy).square().mean()
            train_mse = (model(x) - y).square().mean()
            _finite(validation_mse, "validation MSE")
            _finite(train_mse, "training MSE")
            row = {"step": step, "validation_mse": validation_mse.item(),
                   "training_mse": train_mse.item(), "state": model.snapshot()}
            # This synchronized timestamp follows all scalar transfers and
            # every CPU clone, so a late snapshot can never enter an allowance.
            row["elapsed"] = now() - started
            if row["elapsed"] < previous_elapsed:
                raise ValueError("clock must be monotone")
            result["max_block_seconds"] = max(result["max_block_seconds"], row["elapsed"] - previous_elapsed)
            previous_elapsed = row["elapsed"]
            result["checkpoints"].append(row)

        checkpoint(0)
        step = 0
        while optimizer is not None and step < max_updates and previous_elapsed < budgets[-1]:
            block_end = min(step + every, max_updates)
            while step < block_end:
                optimizer.zero_grad(set_to_none=True)
                features = model.features(x)
                if model.kind == "vp":
                    weight, bias = solve_readout(features, y, penalty)
                else:
                    weight, bias = model.readout_weight, model.readout_bias
                prediction = predict_readout(features, weight, bias)
                loss = _objective(prediction, y, weight, penalty)
                _finite(loss, "training objective")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise FloatingPointError(f"missing gradient for {name}")
                    _finite(parameter.grad, "gradient " + name)
                if training.get("clip_gradient_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training["clip_gradient_norm"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                result["updates"] = step
                _check_model(model)
            checkpoint(step)
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "numerical_failure" if isinstance(error, (FloatingPointError, torch.linalg.LinAlgError)) else "failed"
        result["failures"].append(f"{type(error).__name__}: {error}")
    except BaseException as error:
        # Return interruption evidence as well; the orchestration layer decides
        # whether to abort the phase after persisting the bundle.
        interrupted = error
        result["status"] = "interrupted"
        result["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        if model is not None and result["status"] != "completed":
            try:
                result["partial_state"] = model.snapshot()
            except Exception as error:
                result["failures"].append(f"partial-state capture: {type(error).__name__}: {error}")
        result["fit_seconds"] = now() - started
        result["overshoot_seconds"] = max(0., result["fit_seconds"] - budgets[-1])

    for seconds in budgets:
        eligible = [index for index, row in enumerate(result["checkpoints"]) if row["elapsed"] <= seconds]
        if eligible:
            result["selected"][budget_key(seconds)] = min(eligible, key=lambda index: (result["checkpoints"][index]["validation_mse"], index))
    if result["status"] == "completed" and any(index is None for index in result["selected"].values()):
        result["status"] = "incomplete_budget"
        result["failures"].append("no fully captured checkpoint within one or more declared allowances")

    # No grid_x or grid_clean access occurs until every selection is fixed.
    if interrupted is None and any(index is not None for index in result["selected"].values()):
        try:
            grid_x = _column(arrays["grid_x"], "grid_x", torch.device("cpu"))
            clean = _column(arrays["grid_clean"], "grid_clean", torch.device("cpu"))
            if grid_x.shape != clean.shape:
                raise ValueError("grid input and target arrays must have identical shapes")
            variance = clean.var(unbiased=False).item()
            if variance <= 0:
                raise ValueError("clean grid variance must be positive")
            cache = {}
            for key, index in result["selected"].items():
                if index is None:
                    continue
                if index not in cache:
                    prediction = predict_smooth_local_state(result["checkpoints"][index]["state"], grid_x)
                    error = prediction - clean
                    mse = error.square().mean().item()
                    boundary = grid_x.abs() >= .9 * math.pi
                    cache[index] = {"normalized_mse": mse / variance, "clean_mse": mse,
                                    "clean_variance": variance, "mae": error.abs().mean().item(),
                                    "boundary_mse": error[boundary].square().mean().item() if bool(boundary.any()) else None}
                    if any(v is not None and not math.isfinite(v) for v in cache[index].values()):
                        raise FloatingPointError("nonfinite clean metric")
                result["metrics"][key] = dict(cache[index])
        except Exception as error:
            result["status"] = "postfit_failure"
            result["failures"].append(f"{type(error).__name__}: {error}")
    result["total_seconds"] = time_clock() - total_started
    result["nonfit_seconds"] = result["total_seconds"] - result["fit_seconds"]
    return result
