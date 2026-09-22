"""Fresh, budget-accounted circular-locality tasks for protocol v2.

Only the mathematical scalar target evaluator is reused from v1. This module
owns its seed scheme, task sampling, label budgets, serialization and holdout
guard; the frozen v1 generator and configuration are never called or modified.
Train and validation pools have separate input and unit-noise streams. Larger
budgets extend each pool's prefix, so a validation label never becomes training
data when the total label budget grows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .methodology_tasks import TaskFunctionV1


VERSION = "circular_locality_tasks_v2"
FAMILY_IDS = {"smooth": 1, "narrow": 2, "multimodal": 3, "nonperiodic": 4}
PHASE_IDS = {"pilot": 1, "development": 2, "confirmation": 3}
REPLICATES = {"pilot": 1, "development": 3, "confirmation": 8}
STREAM_IDS = {"function": 0, "train_inputs": 1, "train_noise": 2,
              "validation_inputs": 3, "validation_noise": 4, "grid_noise": 5}
_RESERVED_ROOTS = frozenset({2026092501, 2026092601})
_CONSUMED_V1_ROOTS = frozenset({*range(5101, 5109), *range(6101, 6109), 7101, 7102})
_ARRAY_NAMES = ("train_x", "train_clean", "train_unit_noise", "validation_x",
                "validation_clean", "validation_unit_noise", "grid_x",
                "grid_clean", "grid_unit_noise")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _grid(points: int) -> np.ndarray:
    points = _integer(points, "grid points", minimum=1)
    return (-math.pi + (np.arange(points, dtype=np.float64) + .5) * (2 * math.pi / points))[:, None]


def _context(config: Mapping[str, Any], phase: str, family: str,
              replicate: int) -> tuple[int, int]:
    if phase not in PHASE_IDS or family not in FAMILY_IDS:
        raise ValueError("Unknown locality phase or family")
    replicate = _integer(replicate, "replicate")
    count = config["tasks"].get("replicates", REPLICATES)[phase]
    if replicate >= _integer(count, "replicate count", minimum=1):
        raise ValueError("Replicate is outside the registered phase")
    roots = config["seeds"][phase]
    task_root = _integer(roots["task"], "task root")
    observation_root = _integer(roots["observation"], "observation root")
    if task_root in _CONSUMED_V1_ROOTS or observation_root in _CONSUMED_V1_ROOTS:
        raise ValueError("Consumed v1 confirmation roots cannot be reused by the locality protocol")
    if task_root == observation_root:
        raise ValueError("Task and observation root seeds must be distinct")
    reserved = _RESERVED_ROOTS | frozenset(config["seeds"]["confirmation"][name] for name in ("task", "observation"))
    if phase != "confirmation" and (task_root in reserved or observation_root in reserved):
        raise ValueError("Reserved confirmation roots cannot be relabeled as disposable tasks")
    return task_root, observation_root


def _task_options(config: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    specification = config["tasks"]
    budgets = tuple(_integer(value, "total label budget", minimum=4) for value in specification["label_budgets"])
    if not budgets or list(budgets) != sorted(set(budgets)) or any(value % 4 for value in budgets):
        raise ValueError("Label budgets must be distinct increasing positive multiples of four")
    if specification.get("train_fraction", .75) != .75:
        raise ValueError("This version requires a 75% training / 25% validation split")
    noise = tuple(float(value) for value in specification.get("noise_levels", (.05, .2)))
    if not noise or len(noise) != len(set(noise)) or any(not np.isfinite(value) or value < 0 for value in noise):
        raise ValueError("Noise levels must be distinct finite nonnegative values")
    return budgets, noise


def stream_components(config: Mapping[str, Any], phase: str, family: str,
                       replicate: int, stream: str) -> tuple[int, ...]:
    """Return explicit entropy for one independent v2 target/observation stream."""
    task_root, observation_root = _context(config, phase, family, replicate)
    if stream not in STREAM_IDS:
        raise ValueError("Unknown locality stream")
    seed_root = task_root if stream == "function" else observation_root
    return seed_root, 2, PHASE_IDS[phase], FAMILY_IDS[family], int(replicate), STREAM_IDS[stream]


def _rng(components: tuple[int, ...] | list[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(components))


def _guard(config: Mapping[str, Any], phase: str, confirmation_lock: str | Path | None,
            root: str | Path | None) -> str | None:
    if phase != "confirmation":
        return None
    if confirmation_lock is None:
        raise ValueError("Confirmation generation or restoration requires a verified locality lock")
    from .locality_protocol import verify_confirmation_lock

    repository = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    verified = verify_confirmation_lock(Path(confirmation_lock), config, root=repository)
    return verified["lock_sha256"]


@dataclass(frozen=True)
class LocalityFunctionV2:
    """Serializable clean target using the already audited scalar equations."""

    family: str
    parameters: dict[str, Any]
    normalization: dict[str, Any]

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        """Evaluate an existing target without drawing seeds or touching a lock."""
        equation = TaskFunctionV1("B", self.family, self.parameters, self.normalization, {})
        return equation.evaluate(theta)

    def to_dict(self) -> dict[str, Any]:
        """Describe target parameters with v2 provenance rather than a v1 label."""
        return {"version": VERSION, "family": self.family,
                "parameters": json.loads(_json(self.parameters)),
                "normalization": json.loads(_json(self.normalization))}

    @property
    def sha256(self) -> str:
        """Return a deterministic hash of the represented clean function."""
        return _hash(self.to_dict())


@dataclass(frozen=True)
class LocalityTaskV2:
    """Immutable source pools for every crossed budget/noise cell of one task."""

    function: LocalityFunctionV2
    label_budgets: tuple[int, ...]
    noise_levels: tuple[float, ...]
    train_x: np.ndarray
    train_clean: np.ndarray
    train_unit_noise: np.ndarray
    validation_x: np.ndarray
    validation_clean: np.ndarray
    validation_unit_noise: np.ndarray
    grid_x: np.ndarray
    grid_clean: np.ndarray
    grid_unit_noise: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for name in _ARRAY_NAMES:
            value = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if value.ndim != 2 or value.shape[1] != 1 or not len(value) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite float64 (N,1) values")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for prefix in ("train", "validation", "grid"):
            if getattr(self, f"{prefix}_x").shape != getattr(self, f"{prefix}_clean").shape or getattr(self, f"{prefix}_x").shape != getattr(self, f"{prefix}_unit_noise").shape:
                raise ValueError(f"{prefix} input, clean and noise pools must have matching shapes")

    def budget_metadata(self, total_labels: int, noise_std: float) -> dict[str, Any]:
        """Report every used label, including validation, and paired-noise policy."""
        total_labels = _integer(total_labels, "total label budget", minimum=4)
        if total_labels not in self.label_budgets:
            raise ValueError("Total labels must be one of the registered budgets")
        noise_std = float(noise_std)
        if not np.isfinite(noise_std) or noise_std not in self.noise_levels:
            raise ValueError("Noise standard deviation must be a registered crossed level")
        return {"total_labels": total_labels, "train_labels": 3 * total_labels // 4,
                "validation_labels": total_labels // 4, "noise_std": noise_std,
                "function_sha256": self.function.sha256,
                "sample_identity": "(split,index); prefixes extend within each split only",
                "noise_pairing": "The same standard-normal draws are scaled by each crossed noise level",
                "grid_label_accounting": "Clean integration grid is evaluation only; never training or selection labels"}

    def for_budget(self, total_labels: int, noise_std: float) -> dict[str, np.ndarray]:
        """Return fresh float64 arrays for a fixed total-label/noise cell.

        Args:
            total_labels: Training plus validation labels, from the fixed registry.
            noise_std: A crossed response-noise level; no phase/parity confounding.

        Returns:
            Independent array copies with scalar columns. Sampling pools, target
            scaling and other cells remain unchanged if a caller modifies them.
        """
        budget = self.budget_metadata(total_labels, noise_std)
        n, v = budget["train_labels"], budget["validation_labels"]
        return {"train_x": self.train_x[:n].copy(),
                "train_y": self.train_clean[:n] + noise_std * self.train_unit_noise[:n],
                "train_clean": self.train_clean[:n].copy(),
                "validation_x": self.validation_x[:v].copy(),
                "validation_y": self.validation_clean[:v] + noise_std * self.validation_unit_noise[:v],
                "validation_clean": self.validation_clean[:v].copy(),
                "grid_x": self.grid_x.copy(), "grid_clean": self.grid_clean.copy(),
                "grid_noisy": self.grid_clean + noise_std * self.grid_unit_noise}

    def to_dict(self) -> dict[str, Any]:
        """Serialize exact source pools, their hashes, parameters and seed records."""
        arrays = {name: getattr(self, name).tolist() for name in _ARRAY_NAMES}
        hashes = {name: hashlib.sha256(getattr(self, name).tobytes()).hexdigest() for name in _ARRAY_NAMES}
        value = {"version": VERSION, "function": self.function.to_dict(),
                 "function_sha256": self.function.sha256, "label_budgets": list(self.label_budgets),
                 "noise_levels": list(self.noise_levels), "dtype": "float64",
                 "arrays": arrays, "array_sha256": hashes, "metadata": json.loads(_json(self.metadata))}
        value["sha256"] = _hash(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], config: Mapping[str, Any], *,
                  confirmation_lock: str | Path | None = None,
                  root: str | Path | None = None) -> LocalityTaskV2:
        """Restore exact recorded pools, verifying confirmation access first."""
        metadata = value["metadata"]
        task_root, observation_root = _context(config, metadata["phase"], metadata["family"], metadata["replicate"])
        if metadata["task_root"] != task_root or metadata["observation_root"] != observation_root:
            raise ValueError("Serialized seed registry does not match its declared phase")
        expected_streams = {name: list(stream_components(config, metadata["phase"], metadata["family"], metadata["replicate"], name)) for name in STREAM_IDS}
        if metadata["stream_components"] != expected_streams:
            raise ValueError("Serialized locality stream components changed")
        lock_hash = _guard(config, metadata["phase"], confirmation_lock, root)
        if lock_hash != metadata.get("confirmation_lock_sha256"):
            raise ValueError("Restored locality task belongs to a different confirmation lock")
        if value.get("version") != VERSION or value["function"].get("version") != VERSION:
            raise ValueError("Unsupported locality task version")
        if metadata["config_sha256"] != _hash(config):
            raise ValueError("Restored locality configuration changed")
        if _hash({key: item for key, item in value.items() if key != "sha256"}) != value.get("sha256"):
            raise ValueError("Serialized locality task content changed")
        budgets, levels = _task_options(config)
        if list(budgets) != value["label_budgets"] or list(levels) != value["noise_levels"]:
            raise ValueError("Serialized label/noise registry changed")
        function_data = value["function"]
        function = LocalityFunctionV2(function_data["family"], function_data["parameters"], function_data["normalization"])
        if function.sha256 != value["function_sha256"] or function.family != metadata["family"]:
            raise ValueError("Serialized function parameters changed")
        if set(value["arrays"]) != set(_ARRAY_NAMES) or value["dtype"] != "float64":
            raise ValueError("Serialized task arrays are missing or have the wrong dtype")
        arrays = {name: np.asarray(value["arrays"][name], dtype=np.float64) for name in _ARRAY_NAMES}
        for name, array in arrays.items():
            if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != value["array_sha256"][name]:
                raise ValueError(f"Changed locality array: {name}")
        result = cls(function, budgets, levels, **arrays, metadata=json.loads(_json(metadata)))
        _verify_pool_sizes(result, config)
        return result


def _verify_pool_sizes(task: LocalityTaskV2, config: Mapping[str, Any]) -> None:
    largest = max(task.label_budgets)
    if len(task.train_x) != 3 * largest // 4 or len(task.validation_x) != largest // 4 or len(task.grid_x) != config["tasks"]["grid_points"]:
        raise ValueError("Stored task pools do not match the registered split sizes")


def _parameters(specification: Mapping[str, Any], family: str,
                  rng: np.random.Generator) -> dict[str, Any]:
    if family == "smooth":
        harmonics = np.asarray(specification["harmonics"], dtype=np.int64)
        scale = 1 / harmonics.astype(np.float64) ** 2
        return {"harmonics": harmonics.tolist(), "cos_coefficients": rng.normal(0, scale).tolist(),
                "sin_coefficients": rng.normal(0, scale).tolist()}
    if family == "nonperiodic":
        return {"phase": float(rng.uniform(*specification["phase_uniform"]))}
    low, high = specification["count_inclusive"]
    count = int(rng.integers(low, high + 1))
    centers = rng.uniform(-math.pi, math.pi, count)
    widths = rng.uniform(*specification["width_uniform"], count)
    key = "amplitude_uniform" if family == "narrow" else "amplitude_absolute_uniform"
    amplitudes = rng.uniform(*specification[key], count)
    if family == "multimodal":
        amplitudes *= rng.choice([-1, 1], count)
    return {"count": count, "centers": centers.tolist(), "widths": widths.tolist(),
            "amplitudes": amplitudes.tolist()}


def generate_locality_task(
    config: Mapping[str, Any], phase: str, family: str, replicate: int = 0, *,
    confirmation_lock: str | Path | None = None, root: str | Path | None = None,
) -> LocalityTaskV2:
    """Generate a fresh task with separate, nested training/validation pools.

    Confirmation validation occurs before constructing any RNG or target.
    Function normalization is a declared generator property evaluated on the
    dense grid, never a statistic selected from training/validation outcomes.
    """
    task_root, observation_root = _context(config, phase, family, replicate)
    budgets, levels = _task_options(config)
    lock_hash = _guard(config, phase, confirmation_lock, root)
    streams = {name: stream_components(config, phase, family, replicate, name) for name in STREAM_IDS}
    parameters = _parameters(config["tasks"]["generators"][family], family, _rng(streams["function"]))
    raw_function = LocalityFunctionV2(family, parameters, {"mean": 0., "scale": 1.})
    generator_grid = _grid(config["tasks"]["normalization_points"])
    raw = raw_function.evaluate(generator_grid)
    mean, scale = float(raw.mean()), float(raw.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot normalize a degenerate or nonfinite locality function")
    function = LocalityFunctionV2(family, parameters,
        {"mean": mean, "scale": scale, "grid_points": len(generator_grid), "grid_placement": "midpoints",
         "scope": "Declared zero-mean/unit-variance generator normalization; no fitted data preprocessing"})
    largest = max(budgets)
    arrays = {}
    for split, count in (("train", 3 * largest // 4), ("validation", largest // 4)):
        x = _rng(streams[f"{split}_inputs"]).uniform(-math.pi, math.pi, (count, 1))
        arrays[f"{split}_x"] = x
        arrays[f"{split}_clean"] = function.evaluate(x)
        arrays[f"{split}_unit_noise"] = _rng(streams[f"{split}_noise"]).standard_normal((count, 1))
    arrays["grid_x"] = _grid(config["tasks"]["grid_points"])
    arrays["grid_clean"] = function.evaluate(arrays["grid_x"])
    arrays["grid_unit_noise"] = _rng(streams["grid_noise"]).standard_normal(arrays["grid_x"].shape)
    metadata = {"protocol_id": config["protocol_id"], "version": VERSION, "phase": phase, "family": family,
                "replicate": int(replicate), "phase_id": PHASE_IDS[phase], "family_id": FAMILY_IDS[family],
                "task_root": task_root, "observation_root": observation_root, "config_sha256": _hash(config),
                "stream_components": {name: list(value) for name, value in streams.items()},
                "seed_scheme": "SeedSequence([role_root,2,phase_id,family_id,replicate,stream_id])",
                "numpy_version": np.__version__, "rng_bit_generator": "PCG64",
                "confirmation_lock_sha256": lock_hash, "angle_units": "radians",
                "target_units": "Generator-standardized synthetic scalar", "noise_crossed_with_every_task": True,
                "nested_sampling": "Independent train and validation streams; prefixes grow only within their original split",
                "noise_pairing": "Shared standard-normal draws scaled across noise levels; no additional independent task",
                "test_grid_role": "Integration only; not training/validation labels"}
    task = LocalityTaskV2(function, budgets, levels, **arrays, metadata=metadata)
    _verify_pool_sizes(task, config)
    return task


def initializer(config: Mapping[str, Any], phase: str, family: str,
                 replicate: int, init_index: int) -> dict[str, Any]:
    """Create explicit shared kernel geometry without observing target truth.

    The sixteen centers and rotation are shared by the learned bank and both
    fixed banks. The declared fixed32/48 width grids come from configuration.
    Noise level and label budget do not enter the initialization entropy.
    """
    _context(config, phase, family, replicate)
    init_index = _integer(init_index, "initialization index")
    repeats = config["tasks"].get("initializations", {"pilot": 1, "development": 2, "confirmation": 2})[phase]
    if init_index >= repeats:
        raise ValueError("Initialization index is outside the registered phase")
    seed_root = _integer(config["seeds"]["initialization"], "initialization root")
    if seed_root in _CONSUMED_V1_ROOTS:
        raise ValueError("Consumed v1 confirmation initialization roots cannot be reused")
    data_roots = {config["seeds"][name][role] for name in PHASE_IDS for role in ("task", "observation")}
    if seed_root in data_roots:
        raise ValueError("Initialization root must be independent of every target/observation root")
    components = (seed_root, 2, PHASE_IDS[phase], FAMILY_IDS[family], int(replicate), init_index)
    rotation = float(_rng(components).uniform(0, 2 * math.pi))
    specification = config["initialization"]
    kernels = _integer(specification["kernels"], "kernel count", minimum=1)
    concentration = float(specification["concentration"])
    widths32 = [float(value) for value in specification["fixed32_widths"]]
    widths48 = [float(value) for value in specification["fixed48_widths"]]
    if kernels != 16 or len(widths32) != 2 or len(widths48) != 3:
        raise ValueError("This version requires 16 centers and fixed banks with 32/48 features")
    if any(not np.isfinite(value) or value <= 0 for value in [concentration, *widths32, *widths48]):
        raise ValueError("All initial kernel concentrations must be positive and finite")
    centers = ((-math.pi + 2 * math.pi * np.arange(kernels) / kernels + rotation + math.pi) % (2 * math.pi) - math.pi)[:, None]

    def fixed(widths: list[float]) -> dict[str, np.ndarray]:
        return {"centers": np.repeat(centers, len(widths), axis=0),
                "kappas": np.tile(widths, kernels).astype(np.float64)[:, None]}

    return {"centers": centers, "kappas": np.full((kernels, 1), concentration, dtype=np.float64),
            "fixed32": fixed(widths32), "fixed48": fixed(widths48),
            "metadata": {"version": VERSION, "stream_components": list(components),
                         "seed_scheme": "SeedSequence([initialization_root,2,phase_id,family_id,replicate,initialization_index])",
                         "global_grid_rotation": rotation, "uses_target_truth": False,
                         "shared_across_budgets_noise_and_banks": True,
                         "widths16": [concentration], "widths32": widths32, "widths48": widths48}}
