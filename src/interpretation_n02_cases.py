"""Fresh VM-N02 mechanisms and bounded fits of existing circular models.

The runner owns prospective source locks and stage-wide budgets. This module
never uses clean targets, evaluation labels, or generator parameters to fit or
select a checkpoint. Every random stream has explicit SeedSequence entropy;
the ``fixture`` namespace is disjoint from all scientific study stages.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.locality_models import CircularBasisV2, predict_readout, solve_readout


_CASE_IDS = {"isolated": 1, "redundant": 2, "proxy": 3}
_FIXTURE_POOL = {"seed_offset": 40000, "units": 2}


def _entropy(config: Mapping[str, Any], stage: str, index: int,
             case: str) -> tuple[int, ...]:
    """Validate a unit address and return its explicit RNG namespace."""
    if stage == "fixture":
        pool = _FIXTURE_POOL
    elif stage in ("pilot", "development", "evaluation"):
        pool = config["pools"][stage]
    else:
        raise ValueError("unknown VM-N02 stage")
    if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
            or not 0 <= index < int(pool["units"])):
        raise ValueError("unit index is outside the registered stage")
    if case not in _CASE_IDS:
        raise ValueError("unknown VM-N02 case")
    return int(config["seed_namespace"]), int(pool["seed_offset"]), _CASE_IDS[case], int(index)


def _rng(entropy: tuple[int, ...], stream: int,
         init_seed: int | None = None) -> np.random.Generator:
    """Use independent named streams, retaining reproducibility across devices."""
    extra = (stream,) if init_seed is None else (stream, init_seed)
    return np.random.default_rng(np.random.SeedSequence((*entropy, *extra)))


def _tensor(value: Any, device: torch.device | str) -> Tensor:
    """Make a fresh float64 tensor without touching global random state."""
    return torch.tensor(value, dtype=torch.float64, device=device)


def _grid(size: int) -> np.ndarray:
    """Return a full-circle half-open grid without a duplicated endpoint."""
    return -math.pi + 2 * math.pi * np.arange(size, dtype=np.float64) / size


def _peak_truth(x: np.ndarray, centers: np.ndarray, kappa: np.ndarray,
                weights: np.ndarray, bias: float) -> np.ndarray:
    """Evaluate the stated generator independently of the fitted model API."""
    return np.exp(kappa * (np.cos(x[:, None] - centers) - 1)) @ weights + bias


def make_peak_unit(config: Mapping[str, Any], stage: str, index: int,
                   case: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Generate one isolated or redundant function with independent observations.

    Args:
        config: Frozen VM-N02 machine-readable protocol.
        stage: Pilot, development, evaluation, or disposable fixture namespace.
        index: Zero-based independent function index in the registered stage.
        case: ``isolated`` or ``redundant``.
        device: Destination of all truth and observation tensors.

    Returns:
        Address metadata, truth geometry, train/validation/evaluation splits,
        and a noise-free dense curve; inputs and targets are vectors.
    """
    if case not in ("isolated", "redundant"):
        raise ValueError("peak case must be isolated or redundant")
    entropy = _entropy(config, stage, index, case)
    options, samples = config["peak"], config["samples"]
    truth_rng = _rng(entropy, 0)
    count = 1 if case == "isolated" else 2
    first = truth_rng.uniform(-math.pi, math.pi)
    centers = np.array([first] if count == 1 else [
        first, first + truth_rng.uniform(*options["separation_range"])])
    centers = (centers + math.pi) % (2 * math.pi) - math.pi
    kappa = truth_rng.uniform(*options["kappa_range"], count)
    weights = truth_rng.uniform(*options["amplitude_range"], count)
    bias = float(truth_rng.uniform(*options["bias_range"]))
    result: dict[str, Any] = {
        "stage": stage, "index": int(index), "case": case,
        "seed_components": entropy,
        "truth": {name: _tensor(value, device) for name, value in (
            ("centers", centers), ("kappa", kappa), ("weights", weights), ("bias", bias))},
    }
    for split, input_stream, noise_stream in (("train", 1, 2), ("validation", 3, 4),
                                               ("evaluation", 5, 6), ("curve", 7, 8)):
        size = int(samples[split])
        x = (_grid(size) if split in ("evaluation", "curve") else
             _rng(entropy, input_stream).uniform(-math.pi, math.pi, size))
        clean = _peak_truth(x, centers, kappa, weights, bias)
        result[split] = {"x": _tensor(x, device), "clean": _tensor(clean, device)}
        if split != "curve":
            noise = _rng(entropy, noise_stream).normal(0, options["noise_sd"], size)
            result[split]["y"] = _tensor(clean + noise, device)
    return result


def _finite(value: Tensor, name: str) -> None:
    """Reject nonfinite evidence without clipping or retrying a failed fit."""
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")


def _deadline(deadline: float | None, device: torch.device) -> None:
    """Check an absolute perf_counter deadline, including queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("VM-N02 fit exceeded the stage deadline")


def _copy_state(basis: CircularBasisV2, weight: Tensor, bias: Tensor) -> dict[str, Tensor]:
    """Detach the complete effective state, including signed readout weights."""
    return {name: value.detach().cpu().clone() for name, value in (
        ("centers", basis.centers), ("kappa", basis.concentrations),
        ("weights", weight), ("bias", bias))}


def _check_split(split: Mapping[str, Tensor], dimensions: int) -> None:
    """Check fitting input shape, float64, shared device, and finite labels."""
    x, y = split["x"], split["y"]
    if (x.dtype != torch.float64 or y.dtype != torch.float64 or x.device != y.device
            or x.ndim != dimensions or y.ndim != 1 or len(x) != len(y) or not len(y)):
        raise ValueError("fit inputs and labels must share float64 dtype, device and sample count")
    if dimensions == 2 and x.shape[1] != 2:
        raise ValueError("proxy inputs must have sensor first and clock last")
    _finite(x, "fit inputs")
    _finite(y, "fit labels")


def fit_peak(unit: Mapping[str, Any], config: Mapping[str, Any], init_seed: int,
             deadline: float | None = None) -> dict[str, Any]:
    """Fit existing CircularBasisV2 geometry and readout with fixed joint Adam.

    Selection uses the earliest strict minimum noisy validation MSE among step
    zero and the registered 25-step checkpoints. All 600 updates run even when
    an earlier checkpoint is selected. Exceptions propagate; there is no retry.

    Args:
        unit: A peak unit produced by ``make_peak_unit``.
        config: Frozen protocol (fixture-only overrides are useful in tests).
        init_seed: One declared initialization seed, independent of observations.
        deadline: Optional absolute ``time.perf_counter()`` stage deadline.

    Returns:
        Selected detached CPU state, checkpoint scores, initial geometry,
        parameter count, initialization address, and synchronized elapsed time.
    """
    started = time.perf_counter()
    train, validation = unit["train"], unit["validation"]
    _check_split(train, 1)
    _check_split(validation, 1)
    device = train["x"].device
    if validation["x"].device != device:
        raise ValueError("train and validation must share a device")
    _deadline(deadline, device)
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or init_seed < 0:
        raise ValueError("init_seed must be a nonnegative integer")
    options = config["peak"]
    count = int(options["kernels"][unit["case"]])
    init_rng = _rng(tuple(unit["seed_components"]), 100, init_seed)
    if unit["case"] == "isolated":
        center = float(train["x"][train["y"].argmax()].detach().cpu())
        centers = np.array([center + init_rng.uniform(
            -options["isolated_center_jitter"], options["isolated_center_jitter"])])
    else:
        centers = _grid(count) + init_rng.uniform(-math.pi, math.pi)
    concentrations = init_rng.uniform(*options["initial_kappa_range"], count)
    basis = CircularBasisV2(centers, concentrations, mode="both", device=device)
    with torch.no_grad():
        initial_weight, initial_bias = solve_readout(basis(train["x"]), train["y"], options["ridge"])
    weight, bias = nn.Parameter(initial_weight.clone()), nn.Parameter(initial_bias.clone())
    parameters = [*basis.parameters(), weight, bias]
    optimizer = torch.optim.Adam(parameters, lr=options["learning_rate"])
    initial_state = _copy_state(basis, weight, bias)
    best_mse = math.inf
    selected: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []

    def checkpoint(step: int) -> None:
        nonlocal best_mse, selected
        with torch.no_grad():
            prediction = predict_readout(basis(validation["x"]), weight, bias)
            _finite(prediction, "validation predictions")
            mse_tensor = (prediction - validation["y"]).square().mean()
            _finite(mse_tensor, "validation MSE")
            mse = float(mse_tensor.cpu())
        history.append({"step": step, "validation_mse": mse})
        if mse < best_mse:
            best_mse = mse
            selected = {**_copy_state(basis, weight, bias), "selected_step": step}

    checkpoint(0)
    steps, validate_every = int(options["steps"]), int(options["validate_every"])
    if steps <= 0 or validate_every <= 0 or steps % validate_every:
        raise ValueError("steps must be positive and divisible by validate_every")
    for step in range(1, steps + 1):
        _deadline(deadline, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_readout(basis(train["x"]), weight, bias)
        residual = prediction - train["y"]
        loss = 0.5 * residual.square().mean() + 0.5 * options["ridge"] * weight.square().sum()
        _finite(loss, "training objective")
        loss.backward()
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("missing gradient in registered joint fit")
            _finite(parameter.grad, "training gradient")
        optimizer.step()
        for parameter in parameters:
            _finite(parameter, "updated parameter")
        if step % validate_every == 0:
            checkpoint(step)
    _deadline(deadline, device)
    assert selected is not None
    return {**selected, "validation_mse": best_mse, "validation_history": history,
            "initial_state": initial_state, "init_seed": init_seed,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "seconds": time.perf_counter() - started}


def make_proxy_unit(config: Mapping[str, Any], stage: str, index: int,
                    device: torch.device | str = "cpu") -> dict[str, Any]:
    """Generate independent calendar trajectories and paired shifted evaluation.

    Recorded clock is a predictive proxy for the latent direction. Clean truth
    depends on ``latent`` only. Each split has a separate random clock origin,
    latent jitter, sensor noise, and target noise; aligned/shifted evaluation
    share these noises to define paired effects.

    Args:
        config: Frozen machine-readable protocol.
        stage: Scientific stage or the disjoint ``fixture`` namespace.
        index: Independent trajectory unit index.
        device: Destination for all numerical tensors.

    Returns:
        Truth amplitude/center/bias and train/validation splits with N-by-2
        sensor/clock inputs. Evaluation contains ``aligned`` and ``shifted``.
        Every split retains its latent direction for generator counterfactuals.
    """
    entropy = _entropy(config, stage, index, "proxy")
    options, peak = config["proxy"], config["peak"]
    truth_rng = _rng(entropy, 0)
    center = float(truth_rng.uniform(-math.pi, math.pi))
    amplitude = float(truth_rng.uniform(*peak["amplitude_range"]))
    bias = float(truth_rng.uniform(*peak["bias_range"]))
    result: dict[str, Any] = {
        "stage": stage, "index": int(index), "case": "proxy", "seed_components": entropy,
        "truth": {name: _tensor(value, device) for name, value in (
            ("center", center), ("amplitude", amplitude), ("bias", bias))},
    }
    for split, base_stream in (("train", 10), ("validation", 20), ("evaluation", 30)):
        size = int(config["samples"][split])
        clock = (_rng(entropy, base_stream).uniform(-math.pi, math.pi)
                 + np.arange(size, dtype=np.float64) * (2 * math.pi * options["cycles"] / size))
        latent_noise = _rng(entropy, base_stream + 1).normal(0, options["latent_jitter_sd"], size)
        sensor_noise = _rng(entropy, base_stream + 2).normal(0, options["sensor_noise_sd"], size)
        target_noise = _rng(entropy, base_stream + 3).normal(0, options["target_noise_sd"], size)

        def trajectory(shift: float) -> dict[str, Tensor]:
            latent = clock + shift + latent_noise
            clean = bias + amplitude * np.cos(latent - center)
            return {"x": _tensor(np.column_stack((latent + sensor_noise, clock)), device),
                    "y": _tensor(clean + target_noise, device), "clean": _tensor(clean, device),
                    "latent": _tensor(latent, device)}

        result[split] = ({"aligned": trajectory(0.0),
                          "shifted": trajectory(options["shift_radians"])}
                         if split == "evaluation" else trajectory(0.0))
    return result


def fit_proxy(unit: Mapping[str, Any], config: Mapping[str, Any],
              deadline: float | None = None) -> dict[str, Any]:
    """Fit the existing Cartesian function class and retrain sensor-only ridge.

    Args:
        unit: Proxy trajectories; only noisy training inputs/labels are read.
        config: Frozen protocol containing a strictly positive ridge penalty.
        deadline: Optional absolute stage deadline, checked around both solves.

    Returns:
        Detached CPU weights ordered cosine sensor/clock then sine sensor/clock,
        intercept and separately retrained cosine/sine sensor-only coefficients.
        The full model has five fitted parameters; sensor-only has three.
    """
    started = time.perf_counter()
    train = unit["train"]
    _check_split(train, 2)
    device = train["x"].device
    _deadline(deadline, device)
    with torch.no_grad():
        x = train["x"]
        features = torch.cat((x.cos(), x.sin()), dim=1)
        weight, bias = solve_readout(features, train["y"], config["proxy"]["ridge"])
        sensor_weight, sensor_bias = solve_readout(features[:, [0, 2]], train["y"], config["proxy"]["ridge"])
        _finite(predict_readout(features, weight, bias), "proxy predictions")
    _deadline(deadline, device)
    return {"weights": weight.detach().cpu().clone(), "bias": bias.detach().cpu().clone(),
            "sensor_weights": sensor_weight.detach().cpu().clone(),
            "sensor_bias": sensor_bias.detach().cpu().clone(),
            "parameter_count": 5, "parameters_count": 5, "sensor_parameter_count": 3,
            "seconds": time.perf_counter() - started}
