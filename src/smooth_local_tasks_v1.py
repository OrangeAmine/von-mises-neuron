"""Fresh, paired target pools for the prospective VM-N01 comparison.

The normalization grid is part of the synthetic function definition, not fitted
preprocessing. Training and validation have separate extensible input/noise
streams. No v1/v2 task generator or historical target is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


VERSION = "smooth_local_tasks_v1"
PHASE_IDS = {"pilot": 0, "development": 1, "confirmation": 2}
STREAM_IDS = {"function": 0, "train_inputs": 1, "train_noise": 2,
              "validation_inputs": 3, "validation_noise": 4}
FAMILIES = ("smooth", "narrow", "multimodal", "mixed", "nonperiodic")
_RESERVED_ROOT = 31092203


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _context(config: Mapping[str, Any], phase: str, family: str, replicate: int) -> tuple[int, int]:
    if phase not in PHASE_IDS or family not in FAMILIES:
        raise ValueError("Unknown VM-N01 phase or family")
    if list(config["tasks"]["families"]) != list(FAMILIES):
        raise ValueError("VM-N01 family order changed")
    replicate = _integer(replicate, "replicate")
    if replicate >= _integer(config["tasks"]["replicates"][phase], "replicate count", 1):
        raise ValueError("Replicate is outside the declared phase")
    roots = [_integer(config["seeds"][name], f"{name} seed") for name in PHASE_IDS]
    if len(set(roots)) != len(roots):
        raise ValueError("Phase seeds must be distinct")
    root = _integer(config["seeds"][phase], "phase seed")
    if phase != "confirmation" and root in {_RESERVED_ROOT, config["seeds"]["confirmation"]}:
        raise ValueError("Reserved root cannot be relabeled as a development or pilot root")
    return root, FAMILIES.index(family)


def midpoint_grid(points: int) -> np.ndarray:
    """Return deterministic float64 midpoint angles in [-pi, pi)."""
    points = _integer(points, "grid points", 1)
    return (-math.pi + (np.arange(points, dtype=np.float64) + .5) * (2 * math.pi / points))[:, None]


def stream_components(config: Mapping[str, Any], phase: str, family: str,
                      replicate: int, stream: str) -> tuple[int, ...]:
    """Return the declared entropy tuple without instantiating a generator."""
    root, family_index = _context(config, phase, family, replicate)
    if stream not in STREAM_IDS:
        raise ValueError("Unknown VM-N01 stream")
    return root, 101, family_index, int(replicate), STREAM_IDS[stream]


def _rng(components: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(components))


def _smooth_parameters(rng: np.random.Generator) -> dict[str, Any]:
    scale = 1 / np.arange(1, 4, dtype=np.float64) ** 2
    return {"cos_coefficients": rng.normal(0, scale).tolist(),
            "sin_coefficients": rng.normal(0, scale).tolist()}


def _peaks(rng: np.random.Generator, count: int, low: float,
           high: float, signed: bool) -> dict[str, Any]:
    centers = rng.uniform(-math.pi, math.pi, count)
    widths = rng.uniform(low, high, count)
    amplitudes = rng.uniform(.5, 1.5, count)
    if signed:
        amplitudes *= rng.choice([-1., 1.], size=count)
    return {"count": count, "centers": centers.tolist(), "widths": widths.tolist(),
            "amplitudes": amplitudes.tolist()}


def _parameters(family: str, rng: np.random.Generator) -> dict[str, Any]:
    if family == "smooth":
        return _smooth_parameters(rng)
    if family == "narrow":
        return _peaks(rng, int(rng.integers(1, 3)), .15, .45, False)
    if family == "multimodal":
        return _peaks(rng, int(rng.integers(3, 6)), .2, .8, True)
    if family == "mixed":
        return {"smooth": _smooth_parameters(rng), "local": _peaks(rng, 2, .15, .45, True)}
    if family == "nonperiodic":
        return {"phase": float(rng.uniform(0, 2 * math.pi))}
    raise ValueError("Unknown target family")


def evaluate_raw(family: str, parameters: Mapping[str, Any], theta: Any) -> np.ndarray:
    """Evaluate a recorded target equation, with no sampling or preprocessing.

    Args:
        family: One of the five prospective families.
        parameters: Recorded generator coefficients and geometry.
        theta: Scalar or arbitrary-shaped angle array.

    Returns:
        Raw target values with the same shape as ``theta``.
    """
    x = np.asarray(theta, dtype=np.float64)
    if family == "smooth":
        harmonics = np.arange(1, 4, dtype=np.float64)
        return (np.cos(x[..., None] * harmonics) @ np.asarray(parameters["cos_coefficients"])
                + np.sin(x[..., None] * harmonics) @ np.asarray(parameters["sin_coefficients"]))
    if family in {"narrow", "multimodal"}:
        distance = (x[..., None] - np.asarray(parameters["centers"]) + math.pi) % (2 * math.pi) - math.pi
        scaled = np.abs(distance) / np.asarray(parameters["widths"])
        profiles = np.maximum(1 - scaled, 0) if family == "multimodal" else np.maximum(1 - scaled ** 2, 0) ** 3
        return profiles @ np.asarray(parameters["amplitudes"])
    if family == "mixed":
        return evaluate_raw("smooth", parameters["smooth"], x) + evaluate_raw("narrow", parameters["local"], x)
    if family == "nonperiodic":
        return x / math.pi + .2 * np.sin(x + parameters["phase"])
    raise ValueError("Unknown target family")


def evaluate_function(metadata: Mapping[str, Any], theta: Any) -> np.ndarray:
    """Evaluate a previously recorded standardized function without random draws."""
    normalization = metadata["normalization"]
    return ((evaluate_raw(metadata["family"], metadata["parameters"], theta)
             - normalization["mean"]) / normalization["scale"])


def generate_task(config: Mapping[str, Any], phase: str, family: str,
                  replicate: int = 0, confirmation_lock: str | Path | None = None) -> dict[str, Any]:
    """Generate one largest paired task pool, verifying reserved access first.

    Args:
        config: Frozen prospective configuration.
        phase: Pilot, development, or confirmation.
        family: Registered target family.
        replicate: Zero-based function index within the family.
        confirmation_lock: Verified evidence lock required for confirmation.

    Returns:
        Metadata and independent float64 ``(N,1)`` arrays; all budgets use
        prefixes within their original training or validation pools.
    """
    _context(config, phase, family, replicate)
    lock_verification: Any = None
    if phase == "confirmation":
        if confirmation_lock is None:
            raise ValueError("Confirmation targets require a verified source/config/evidence lock")
        from .smooth_local_protocol_v1 import verify_lock

        # Deliberately precedes *every* generator construction and target call.
        lock_verification = verify_lock(Path(confirmation_lock), config)
    spec = config["tasks"]
    budgets = [_integer(value, "label budget", 4) for value in spec["label_budgets"]]
    if not budgets or budgets != sorted(set(budgets)) or any(value % 4 for value in budgets):
        raise ValueError("Label budgets must be increasing distinct multiples of four")
    if spec["train_fraction"] != .75:
        raise ValueError("This version requires 75% training labels")
    noise_std = float(spec["noise_std"])
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Noise standard deviation must be finite and nonnegative")
    entropy = {name: stream_components(config, phase, family, replicate, name) for name in STREAM_IDS}
    parameters = _parameters(family, _rng(entropy["function"]))
    normalization_grid = midpoint_grid(spec["normalization_points"])
    raw = evaluate_raw(family, parameters, normalization_grid)
    normalization = {"mean": float(raw.mean()), "scale": float(raw.std(ddof=0)),
                     "points": int(spec["normalization_points"]), "rule": "deterministic_midpoints"}
    if not math.isfinite(normalization["scale"]) or normalization["scale"] <= 0:
        raise ValueError("Degenerate target function")
    metadata = {"version": VERSION, "phase": phase, "family": family, "replicate": int(replicate),
                "parameters": parameters, "normalization": normalization,
                "label_budgets": budgets, "noise_std": noise_std,
                "stream_components": {name: list(value) for name, value in entropy.items()},
                "config_sha256": _hash(config), "dtype": "float64",
                "sample_identity": "(split,index); each budget is an independent split prefix"}
    metadata["function_sha256"] = _hash({key: metadata[key] for key in ("version", "family", "parameters", "normalization")})
    if phase == "confirmation":
        metadata["confirmation_lock_sha256"] = hashlib.sha256(Path(confirmation_lock).read_bytes()).hexdigest()
        # Preserve a verifier's compact hash when it provides one; no lock
        # payload is required to regenerate an already authorized target.
        if isinstance(lock_verification, Mapping) and "lock_sha256" in lock_verification:
            if metadata["confirmation_lock_sha256"] != lock_verification["lock_sha256"]:
                raise ValueError("Confirmation lock changed during verification")
    arrays: dict[str, np.ndarray] = {}
    for split, count in (("train", 3 * max(budgets) // 4), ("validation", max(budgets) // 4)):
        x = _rng(entropy[f"{split}_inputs"]).uniform(-math.pi, math.pi, (count, 1))
        noise = _rng(entropy[f"{split}_noise"]).standard_normal((count, 1))
        clean = evaluate_function(metadata, x)
        arrays.update({f"{split}_x": x, f"{split}_clean": clean, f"{split}_unit_noise": noise,
                       f"{split}_y": clean + noise_std * noise})
    arrays["grid_x"] = midpoint_grid(spec["grid_points"])
    if np.intersect1d(normalization_grid.ravel(), arrays["grid_x"].ravel()).size:
        raise ValueError("Scoring and normalization midpoint grids must be distinct")
    arrays["grid_clean"] = evaluate_function(metadata, arrays["grid_x"])
    for name, value in arrays.items():
        if value.dtype != np.float64 or value.ndim != 2 or value.shape[1] != 1 or not np.isfinite(value).all():
            raise ValueError(f"Invalid task array: {name}")
    metadata["array_sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    return {"metadata": metadata, "arrays": arrays}


def cell_arrays(task: Mapping[str, Any], budget: int) -> dict[str, np.ndarray]:
    """Return independent copies for a declared total training/validation budget."""
    budget = _integer(budget, "label budget", 4)
    if budget not in task["metadata"]["label_budgets"]:
        raise ValueError("Unknown label budget")
    arrays = task["arrays"]
    counts = {"train": 3 * budget // 4, "validation": budget // 4}
    result = {f"{split}_{name}": arrays[f"{split}_{name}"][:count].copy()
              for split, count in counts.items() for name in ("x", "y")}
    result.update({name: arrays[name].copy() for name in ("grid_x", "grid_clean")})
    return result


def initializer_rotation(config: Mapping[str, Any], phase: str, family: str,
                         replicate: int, init_index: int) -> float:
    """Draw one shared geometry rotation independently of all target truth."""
    _, family_index = _context(config, phase, family, replicate)
    init_index = _integer(init_index, "initialization index")
    if init_index >= _integer(config["tasks"]["initializations"][phase], "initialization count", 1):
        raise ValueError("Initialization index is outside the declared phase")
    entropy = (_integer(config["seeds"]["initialization"], "initialization seed"),
               101, PHASE_IDS[phase], family_index, int(replicate), init_index)
    return float(_rng(entropy).uniform(-math.pi, math.pi))
