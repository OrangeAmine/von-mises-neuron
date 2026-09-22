"""Seeded scalar tasks for the versioned VM methodology contract.

This module never touches NumPy's process-global RNG. Random stream components
are ``[role_root, stage_id, stochastic_family_id, replicate_id, split_id]``.
Within initialization, child 0 supplies model draws and child 1 supplies shared
minibatch order. All stored observations and numerical-integration targets are
float64; the trainer is responsible for its declared float32 conversion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GENERATOR_VERSION = "vm_methodology_tasks_v1"
_RESERVED_TASK_ROOTS = frozenset(range(5101, 5109))
_RESERVED_OBSERVATION_ROOTS = frozenset(range(6101, 6109))


def canonical_json(value: Any) -> str:
    """Serialize finite JSON deterministically for manifests and hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    """Return a SHA256 of a JSON-serializable object."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def wrap_angle(theta: np.ndarray | float) -> np.ndarray:
    """Wrap radians to the half-open interval [-pi, pi)."""
    angles = np.asarray(theta, dtype=np.float64)
    return (angles + math.pi) % (2 * math.pi) - math.pi


def midpoint_grid(points: int) -> np.ndarray:
    """Return a float64 column of equally spaced circle midpoints."""
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 1:
        raise ValueError("points must be a positive integer")
    return (-math.pi + (np.arange(points, dtype=np.float64) + 0.5)
            * (2 * math.pi / points))[:, None]


def seed_components(
    config: Mapping[str, Any], root_seed: int, stage: str, family_id: int,
    replicate_id: int, split: str,
) -> tuple[int, int, int, int, int]:
    """Build the contract's recorded SeedSequence entropy components."""
    derivation = config["seed_derivation"]
    components = (root_seed, derivation["stage_ids"][stage], family_id,
                  replicate_id, derivation["split_ids"][split])
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
           for x in components):
        raise ValueError("seed components must be nonnegative integers")
    return tuple(int(x) for x in components)


def _rng(components: Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(components))


@dataclass(frozen=True)
class TaskFunctionV1:
    """A serializable target function, including its declared generator scale."""

    stage: str
    family: str
    parameters: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]

    def _raw(self, theta: np.ndarray | float) -> np.ndarray:
        angles = np.asarray(theta, dtype=np.float64)
        p = self.parameters
        if self.stage == "A":
            return p["amplitude_after"] * np.cos(angles + p["phase_after"]) + p["bias"]
        if self.family == "smooth":
            result = np.zeros_like(angles)
            for h, a, b in zip(p["harmonics"], p["cos_coefficients"], p["sin_coefficients"]):
                result += a * np.cos(h * angles) + b * np.sin(h * angles)
            return result
        if self.family in {"narrow", "multimodal"}:
            result = np.zeros_like(angles)
            for center, width, amplitude in zip(p["centers"], p["widths"], p["amplitudes"]):
                distance = wrap_angle(angles - center)
                if self.family == "narrow":
                    # Evaluate only inside support: no overflowing off-support power.
                    profile = np.zeros_like(angles)
                    inside = np.abs(distance) < width
                    profile[inside] = (1 - (distance[inside] / width) ** 2) ** 3
                else:
                    profile = np.maximum(1 - np.abs(distance) / width, 0)
                result += amplitude * profile
            return result
        if self.family == "nonperiodic":
            return angles / math.pi + 0.2 * np.sin(angles + p["phase"])
        raise ValueError(f"Unsupported target family: {self.stage}/{self.family}")

    def evaluate(self, theta: np.ndarray | float) -> np.ndarray:
        """Evaluate the clean target, preserving the supplied angle shape."""
        return (self._raw(theta) - self.normalization["mean"]) / self.normalization["scale"]

    def prechange(self, theta: np.ndarray | float) -> np.ndarray:
        """Evaluate A's known initial function without drawing or fitting it."""
        if self.stage != "A":
            raise ValueError("Only stage A has a prechange target")
        p = self.parameters
        return p["amplitude_before"] * np.cos(np.asarray(theta) + p["phase_before"]) + p["bias"]

    def rotate(self, delta: float) -> TaskFunctionV1:
        """Map a periodic function so f_rot(wrap(theta+delta)) equals f(theta).

        Generator normalization remains unchanged under this coordinate map;
        it is never refitted on the rotated evaluation grid. The nonperiodic
        negative control does not have this circular transformation law.
        """
        if not np.isfinite(delta):
            raise ValueError("rotation must be finite")
        if self.family == "nonperiodic":
            raise ValueError("A circular rotation is undefined for the nonperiodic control")
        p = json.loads(canonical_json(self.parameters))
        if self.stage == "A":
            p["phase_before"] -= delta
            p["phase_after"] -= delta
        elif self.family == "smooth":
            h = np.asarray(p["harmonics"])
            a, b = np.asarray(p["cos_coefficients"]), np.asarray(p["sin_coefficients"])
            p["cos_coefficients"] = (a * np.cos(h * delta) - b * np.sin(h * delta)).tolist()
            p["sin_coefficients"] = (a * np.sin(h * delta) + b * np.cos(h * delta)).tolist()
        else:
            p["centers"] = wrap_angle(np.asarray(p["centers"]) + delta).tolist()
        metadata = dict(self.metadata, mapped_rotation_radians=float(delta),
                        mapped_from_function_hash=self.sha256)
        return replace(self, parameters=p, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return complete finite generator parameters and provenance."""
        return {"version": GENERATOR_VERSION, "stage": self.stage, "family": self.family,
                "parameters": self.parameters, "normalization": self.normalization,
                "metadata": self.metadata}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, config: Mapping[str, Any] | None = None,
        confirmation_lock: str | Path | None = None, root: str | Path | None = None,
    ) -> TaskFunctionV1:
        """Restore parameters, requiring the original lock for reserved tasks."""
        if value.get("version") != GENERATOR_VERSION:
            raise ValueError("Unsupported generator version")
        if value["metadata"].get("phase") == "confirmation":
            if config is None or confirmation_lock is None:
                raise ValueError("Restoring a confirmation function requires its verified lock")
            from .methodology_protocol import verify_confirmation_lock

            repository = Path(root) if root is not None else Path(__file__).resolve().parents[1]
            lock = verify_confirmation_lock(Path(confirmation_lock), config, repo_root=repository)
            if lock["lock_sha256"] != value["metadata"].get("confirmation_lock_sha256"):
                raise ValueError("Restored function belongs to a different confirmation lock")
        clean = json.loads(canonical_json(value))
        return cls(**{key: clean[key] for key in
                      ("stage", "family", "parameters", "normalization", "metadata")})

    @property
    def sha256(self) -> str:
        """Return the function's complete parameter/provenance hash."""
        return content_hash(self.to_dict())


@dataclass(frozen=True)
class TaskDataV1:
    """Shared observations and integration targets for one function bundle."""

    function: TaskFunctionV1
    x_train: np.ndarray
    y_train: np.ndarray
    y_train_clean: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    y_validation_clean: np.ndarray
    x_grid: np.ndarray
    y_grid_clean: np.ndarray
    y_grid_noisy: np.ndarray
    noise_std: float
    metadata: dict[str, Any]

    def train_prefix(self, observations: int) -> TaskDataV1:
        """Select a nested prefix, leaving validation and clean support shared."""
        if (isinstance(observations, bool) or not isinstance(observations, (int, np.integer))
                or not 1 <= observations <= len(self.x_train)):
            raise ValueError("Requested prefix must fit the stored training pool")
        return replace(self, x_train=self.x_train[:observations], y_train=self.y_train[:observations],
                       y_train_clean=self.y_train_clean[:observations],
                       metadata=dict(self.metadata, train_observations=int(observations)))

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Expose trainer-facing names without copying shared float64 arrays."""
        return {"train_x": self.x_train, "train_y": self.y_train,
                "train_clean": self.y_train_clean, "validation_x": self.x_validation,
                "validation_y": self.y_validation, "validation_clean": self.y_validation_clean,
                "grid_x": self.x_grid, "grid_clean": self.y_grid_clean,
                "grid_noisy": self.y_grid_noisy}

    def to_manifest(self) -> dict[str, Any]:
        """Return parameters, seeds, sizes and reproducible array-content hashes."""
        arrays = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                  for name, value in self.as_arrays().items()}
        return {"version": GENERATOR_VERSION, "function": self.function.to_dict(),
                "function_sha256": self.function.sha256, "noise_std": self.noise_std,
                "metadata": self.metadata, "arrays": arrays}

    def save(self, path: str | Path) -> None:
        """Save an NPZ with an embedded manifest, refusing to replace evidence."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents accidentally replacing an incomplete file.
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **self.as_arrays(),
                                manifest=np.array(canonical_json(self.to_manifest())))


def _sample_parameters(config: Mapping[str, Any], stage: str, family: str,
                       rng: np.random.Generator) -> dict[str, Any]:
    if stage == "A":
        condition = next(item for item in config["A"]["conditions"] if item["id"] == family)
        phase = float(rng.uniform(0, 2 * math.pi))
        return {"amplitude_before": condition["amplitude_before"],
                "amplitude_after": condition["amplitude_after"], "phase_before": phase,
                "phase_after": phase + condition["phase_change"], "bias": 0.0}
    specification = config["B"]["generators"][family]
    if family == "smooth":
        harmonics = np.asarray(specification["harmonics"])
        scale = 1 / harmonics.astype(np.float64) ** 2
        return {"harmonics": harmonics.tolist(),
                "cos_coefficients": rng.normal(0, scale).tolist(),
                "sin_coefficients": rng.normal(0, scale).tolist()}
    if family == "nonperiodic":
        return {"phase": float(rng.uniform(*specification["phase_uniform"]))}
    lower, upper = specification["count_inclusive"]
    count = int(rng.integers(lower, upper + 1))
    centers = rng.uniform(-math.pi, math.pi, count)
    widths = rng.uniform(*specification["width_uniform"], count)
    amplitude_key = "amplitude_uniform" if family == "narrow" else "amplitude_absolute_uniform"
    amplitudes = rng.uniform(*specification[amplitude_key], count)
    if family == "multimodal":
        amplitudes *= rng.choice([-1, 1], count)
    return {"count": count, "centers": centers.tolist(), "widths": widths.tolist(),
            "amplitudes": amplitudes.tolist()}


def generate_task(
    config: Mapping[str, Any], stage: str, phase: str, family: str, replicate_id: int = 0,
    *, confirmation_lock: str | Path | None = None, root: str | Path | None = None,
) -> TaskDataV1:
    """Generate one registered paired task, guarding confirmation before RNG use.

    Args:
        config: Canonical protocol configuration, including registered role seeds.
        stage: ``A`` (coordinate adaptation) or ``B`` (locality).
        phase: ``pilot``, ``development`` or guarded ``confirmation``.
        family: A condition or B target-family ID.
        replicate_id: Zero-based index pairing task and observation roots.
        confirmation_lock: Required verified lock for reserved access.
        root: Repository root used by the lock verifier.

    Returns:
        One float64 shared bundle; B sample sizes use its ``train_prefix`` method.
    """
    if stage not in {"A", "B"} or phase not in {"pilot", "development", "confirmation"}:
        raise ValueError("Unknown stage or execution phase")
    if family not in config[stage]["family_ids"]:
        raise ValueError(f"Unknown {stage} family: {family}")
    allowed = config[stage].get(phase, {}).get("conditions")
    if allowed is not None and family not in allowed:
        raise ValueError(f"{family} is outside the {stage} {phase} task matrix")
    pairs = config[stage][phase]["task_pairs"]
    if (isinstance(replicate_id, bool) or not isinstance(replicate_id, int)
            or not 0 <= replicate_id < pairs):
        raise ValueError("replicate_id is outside this stage's task-pair registry")
    roots = config["seeds"][phase]
    task_root, observation_root = roots["task"][replicate_id], roots["observation"][replicate_id]
    lock_metadata: dict[str, Any] = {}
    if phase == "confirmation":
        if confirmation_lock is None:
            raise ValueError("Confirmation task generation requires a verified lock")
        from .methodology_protocol import verify_confirmation_lock

        repository = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        lock = verify_confirmation_lock(Path(confirmation_lock), config, repo_root=repository)
        lock_metadata = {"confirmation_lock_sha256": lock["lock_sha256"]}
    elif task_root in _RESERVED_TASK_ROOTS or observation_root in _RESERVED_OBSERVATION_ROOTS:
        raise ValueError("Reserved confirmation roots cannot be relabeled as disposable tasks")

    family_id = config["A"]["stochastic_family_id"] if stage == "A" else config["B"]["family_ids"][family]
    streams = {"function": seed_components(config, task_root, stage, family_id, replicate_id, "function")}
    streams.update({split: seed_components(config, observation_root, stage, family_id, replicate_id, split)
                    for split in ("train", "validation", "test_noise")})
    metadata = {"protocol_id": config["protocol_id"], "stage": stage, "phase": phase,
                "family": family, "replicate_id": replicate_id, "task_root": task_root,
                "observation_root": observation_root, "stochastic_family_id": family_id,
                "stream_components": {key: list(value) for key, value in streams.items()},
                "rng_bit_generator": "PCG64", "numpy_version": np.__version__,
                "angle_units": "radians", "target_units": "synthetic scalar",
                **lock_metadata}
    function = TaskFunctionV1(stage, family, _sample_parameters(config, stage, family, _rng(streams["function"])),
                              {"mean": 0.0, "scale": 1.0, "grid_points": 0}, dict(metadata))
    if stage == "B":
        normalization = config["B"]["generator_normalization"]
        generator_grid = midpoint_grid(normalization["grid_points"])
        raw = function._raw(generator_grid)
        mean, scale = float(np.mean(raw)), float(np.std(raw, ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Degenerate or nonfinite target normalization")
        function = replace(function, normalization={"mean": mean, "scale": scale,
                            "grid_points": len(generator_grid), "grid_placement": "midpoints",
                            "scope": "declared generator normalization; no model-fitted preprocessing"})
        count = (config["B"]["pilot"]["train_observations"] if phase == "pilot"
                 else max(config["B"]["train_observations"]))
        noise_std = float(config["B"]["pilot_noise_std"] if phase == "pilot" else
                          config["B"]["noise_std_by_replicate_parity"][replicate_id % 2])
    else:
        count = config["A"]["train_observations"]
        noise_std = (config["A"]["noise_std_fraction_of_signal_std"]
                     * function.parameters["amplitude_after"] / math.sqrt(2))

    def observations(split: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = _rng(streams[split])
        angles = rng.uniform(-math.pi, math.pi, (n, 1))
        clean = function.evaluate(angles)
        noisy = clean + noise_std * rng.standard_normal((n, 1))
        return angles, noisy, clean

    train_x, train_y, train_clean = observations("train", count)
    validation_x, validation_y, validation_clean = observations("validation", config["data"]["validation_observations"])
    grid_x = midpoint_grid(config["data"]["clean_evaluation_grid"]["points"])
    grid_clean = function.evaluate(grid_x)
    grid_noisy = grid_clean + noise_std * _rng(streams["test_noise"]).standard_normal(grid_x.shape)
    metadata.update(train_observations=count, train_pool_observations=count,
                    observation_draw_order="all angles followed by all standard-normal draws",
                    normalization_grid_placement="midpoints" if stage == "B" else None)
    return TaskDataV1(function, train_x, train_y, train_clean, validation_x, validation_y,
                      validation_clean, grid_x, grid_clean, grid_noisy, float(noise_std), metadata)


def _initialization_stream(config: Mapping[str, Any], data: TaskDataV1,
                           initialization_index: int, child: int) -> tuple[int, dict[str, Any]]:
    roots = config["seeds"][data.metadata["phase"]]["initialization"]
    if not isinstance(initialization_index, int) or not 0 <= initialization_index < len(roots):
        raise ValueError("initialization_index is outside its registry")
    components = seed_components(config, roots[initialization_index], data.metadata["stage"],
                                 data.metadata["stochastic_family_id"], data.metadata["replicate_id"], "initialization")
    sequence = np.random.SeedSequence(components, spawn_key=(child,))
    # A signed-63-bit seed is usable unchanged by NumPy and Torch generators.
    seed = int(sequence.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)
    return seed, {"components": list(components), "spawn_key": [child], "derived_seed": seed}


def initialization_seed(config: Mapping[str, Any], data: TaskDataV1,
                         initialization_index: int = 0) -> int:
    """Return the recorded model-initialization child-stream seed."""
    return _initialization_stream(config, data, initialization_index, 0)[0]


def batch_seed(config: Mapping[str, Any], data: TaskDataV1,
               initialization_index: int = 0) -> int:
    """Return a model-independent, initialization-specific batch stream seed."""
    return _initialization_stream(config, data, initialization_index, 1)[0]


def initialization_bundle(config: Mapping[str, Any], data: TaskDataV1,
                           initialization_index: int = 0) -> dict[str, Any]:
    """Draw all B starts in a fixed order, shared across kernels and sample sizes.

    MLP state arrays use PyTorch Linear names/shapes and the specified independent
    fan-in uniform rule. Drawing both model families at once prevents model
    construction order from changing either initialization.
    """
    if data.metadata["stage"] != "B":
        raise ValueError("Stage A starts from known prechange coefficients and has no initialization repeats")
    seed, stream = _initialization_stream(config, data, initialization_index, 0)
    batch_value, batch_stream = _initialization_stream(config, data, initialization_index, 1)
    rng = np.random.default_rng(seed)
    spec = config["B"]["kernel_initialization"]
    models = {model["id"]: model for model in config["B"]["models"]}
    kernels = models["learned_vm"]["kernels"]
    rotation = float(rng.uniform(0, 2 * math.pi))
    centers = wrap_angle(-math.pi + 2 * math.pi * np.arange(kernels) / kernels + rotation)[:, None]
    readout = rng.normal(0, spec["readout_std"], (1, kernels))

    def mlp(inputs: int, hidden: int) -> dict[str, np.ndarray]:
        first_bound, second_bound = 1 / math.sqrt(inputs), 1 / math.sqrt(hidden)
        return {"hidden.weight": rng.uniform(-first_bound, first_bound, (hidden, inputs)),
                "hidden.bias": rng.uniform(-first_bound, first_bound, hidden),
                "readout.weight": rng.uniform(-second_bound, second_bound, (1, hidden)),
                "readout.bias": rng.uniform(-second_bound, second_bound, 1)}

    return {"centers": centers, "widths": np.full((kernels, 1), spec["width"], dtype=np.float64),
            "readout": readout, "bias": np.array([spec["bias"]], dtype=np.float64),
            "sincos_mlp": mlp(models["sincos_mlp"]["inputs"], models["sincos_mlp"]["hidden"]),
            "raw_angle_mlp": mlp(models["raw_angle_mlp"]["inputs"], models["raw_angle_mlp"]["hidden"]),
            "metadata": {"initialization": stream, "batch": batch_stream,
                         "global_grid_rotation": rotation, "batch_seed": batch_value,
                         "draw_order": "rotation, kernel readout, sincos MLP, raw-angle MLP"}}


def cyclic_batch_indices(n: int, batch_size: int, updates: int, seed: int) -> np.ndarray:
    """Return fixed-size batches from consecutive independently shuffled epochs.

    A batch may cross an epoch boundary; for n < batch_size it may contain the
    same example more than once. Every complete epoch visits each row once.
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (n, batch_size, updates)):
        raise ValueError("n, batch_size and updates must be positive integers")
    rng = np.random.default_rng(seed)
    needed = batch_size * updates
    permutations = [rng.permutation(n) for _ in range((needed + n - 1) // n)]
    return np.concatenate(permutations)[:needed].reshape(updates, batch_size)
