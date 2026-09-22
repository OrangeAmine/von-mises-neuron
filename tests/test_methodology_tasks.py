"""Task-contract tests use disposable roots; no reserved functions are generated."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.methodology_tasks import (
    TaskFunctionV1,
    batch_seed,
    content_hash,
    cyclic_batch_indices,
    generate_task,
    initialization_bundle,
    initialization_seed,
    midpoint_grid,
    seed_components,
    wrap_angle,
)


@pytest.fixture
def config():
    """Load the contract without executing any confirmation task."""
    return json.loads((Path(__file__).resolve().parents[1] / "configs/vm_methodology_v1.json").read_text())


def test_stream_components_match_declared_numpy_seedsequence(config):
    components = seed_components(config, 101, "A", 0, 0, "function")
    assert components == (101, 1, 0, 0, 0)
    task = generate_task(config, "A", "pilot", "phase_only")
    expected = np.random.default_rng(np.random.SeedSequence(components)).uniform(0, 2 * math.pi)
    assert task.function.parameters["phase_before"] == expected


def test_a_condition_bundle_shares_phase_angles_and_noise(config):
    tasks = [generate_task(config, "A", "development", family) for family in config["A"]["family_ids"]]
    reference = tasks[0]
    for task in tasks:
        assert task.function.parameters["phase_before"] == reference.function.parameters["phase_before"]
        assert task.metadata["stochastic_family_id"] == 0
        assert task.metadata["stream_components"] == reference.metadata["stream_components"]
        for x in ("x_train", "x_validation", "x_grid"):
            np.testing.assert_array_equal(getattr(task, x), getattr(reference, x))
        for noisy, clean in (("y_train", "y_train_clean"), ("y_validation", "y_validation_clean"),
                             ("y_grid_noisy", "y_grid_clean")):
            np.testing.assert_allclose((getattr(task, noisy) - getattr(task, clean)) / task.noise_std,
                                       (getattr(reference, noisy) - getattr(reference, clean)) / reference.noise_std,
                                       atol=4e-14, rtol=4e-14)
        p = task.function.parameters
        assert task.noise_std == pytest.approx(0.1 * p["amplitude_after"] / math.sqrt(2))
        np.testing.assert_allclose(task.function.prechange(task.x_grid),
                                   p["amplitude_before"] * np.cos(task.x_grid + p["phase_before"]))
        assert float(np.var(task.y_grid_clean)) == pytest.approx(p["amplitude_after"] ** 2 / 2)
        assert task.function.normalization["mean"] == 0
        assert task.function.normalization["scale"] == 1


@pytest.mark.parametrize("family", ["smooth", "narrow", "multimodal", "nonperiodic"])
def test_b_nested_prefixes_generator_normalization_and_noise(config, family):
    task = generate_task(config, "B", "development", family)
    assert task.x_train.shape == (512, 1)
    assert task.x_validation.shape == (256, 1)
    assert task.x_grid.shape == (4096, 1)
    for n in (32, 128, 512):
        part = task.train_prefix(n)
        np.testing.assert_array_equal(part.y_train, task.y_train[:n])
        np.testing.assert_array_equal(part.x_train, task.x_train[:n])
        assert part.x_validation is task.x_validation
        assert part.x_grid is task.x_grid
    generated = task.function.evaluate(midpoint_grid(65536))
    assert abs(float(np.mean(generated))) < 2e-14
    assert float(np.var(generated)) == pytest.approx(1, abs=2e-14)
    assert task.noise_std == 0.05
    assert generate_task(config, "B", "development", family, 1).noise_std == 0.2
    pilot = generate_task(config, "B", "pilot", family)
    assert pilot.noise_std == 0.1
    assert len(pilot.x_train) == 128
    assert all(array.dtype == np.float64 and np.isfinite(array).all() for array in task.as_arrays().values())


def test_independent_observation_streams_and_no_global_rng_change(config):
    np.random.seed(81234)
    before = np.random.get_state()
    task = generate_task(config, "B", "development", "narrow")
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert task.metadata["task_root"] == 1101
    assert task.metadata["observation_root"] == 2101
    assert not np.array_equal(task.x_train[:256], task.x_validation)
    components = task.metadata["stream_components"]["test_noise"]
    expected_noise = np.random.default_rng(np.random.SeedSequence(components)).standard_normal((4096, 1))
    np.testing.assert_allclose(task.y_grid_noisy - task.y_grid_clean, task.noise_std * expected_noise, atol=3e-16)
    second = generate_task(config, "B", "development", "narrow", 1)
    assert second.metadata["task_root"] == 1102
    assert second.metadata["observation_root"] == 2102
    assert task.function.sha256 != second.function.sha256


@pytest.mark.parametrize("family", ["smooth", "narrow", "multimodal"])
def test_periodic_function_rotation_crosses_branch_cut(config, family):
    task = generate_task(config, "B", "pilot", family)
    delta = math.pi / 3
    rotated = task.function.rotate(delta)
    x = np.array([-math.pi, -3.0, -0.5, 2.0, math.pi - 1e-12])[:, None]
    np.testing.assert_allclose(rotated.evaluate(wrap_angle(x + delta)), task.function.evaluate(x), atol=8e-14)
    np.testing.assert_allclose(task.function.evaluate(x + 2 * math.pi), task.function.evaluate(x), atol=8e-14)
    assert rotated.normalization == task.function.normalization


def test_a_rotation_maps_prechange_and_postchange(config):
    task = generate_task(config, "A", "pilot", "phase_only")
    rotated = task.function.rotate(math.pi / 3)
    shifted_x = wrap_angle(task.x_grid + math.pi / 3)
    np.testing.assert_allclose(rotated.evaluate(shifted_x), task.y_grid_clean, atol=3e-15)
    np.testing.assert_allclose(rotated.prechange(shifted_x), task.function.prechange(task.x_grid), atol=3e-15)


def test_negative_control_keeps_endpoint_mismatch(config):
    task = generate_task(config, "B", "pilot", "nonperiodic")
    expected = 2 / task.function.normalization["scale"]
    assert float(task.function.evaluate(math.pi) - task.function.evaluate(-math.pi)) == pytest.approx(expected)
    with pytest.raises(ValueError, match="nonperiodic"):
        task.function.rotate(math.pi / 3)


def test_target_formulas_and_compact_support():
    normalization = {"mean": 0.0, "scale": 1.0}
    narrow = TaskFunctionV1("B", "narrow", {"centers": [3.1], "widths": [0.4], "amplitudes": [1.2]}, normalization, {})
    x = np.array([3.1, 3.3, 3.5, 2.7, 0.0])
    expected = np.array([1.2, 1.2 * (1 - 0.5 ** 2) ** 3, 0, 0, 0])
    np.testing.assert_allclose(narrow.evaluate(x), expected, atol=2e-15)
    np.testing.assert_allclose(narrow.evaluate(wrap_angle(x)), expected, atol=2e-15)
    assert float(narrow.evaluate(3.1)) == pytest.approx(1.2)
    triangle = TaskFunctionV1("B", "multimodal", {"centers": [0], "widths": [0.5], "amplitudes": [-1.2]}, normalization, {})
    np.testing.assert_allclose(triangle.evaluate(np.array([0, 0.25, 0.5, 1])), [-1.2, -0.6, 0, 0])
    smooth = TaskFunctionV1("B", "smooth", {"harmonics": [1, 2, 3], "cos_coefficients": [0.2, 0.3, 0.4],
                            "sin_coefficients": [-0.1, 0.5, 0.9]}, normalization, {})
    expected = sum(a * np.cos(h * x) + b * np.sin(h * x) for h, a, b in
                   zip([1, 2, 3], [0.2, 0.3, 0.4], [-0.1, 0.5, 0.9]))
    np.testing.assert_array_equal(smooth.evaluate(x), expected)


def test_b_generator_parameter_distributions_follow_recorded_stream(config):
    for family in ("smooth", "narrow", "multimodal"):
        task = generate_task(config, "B", "pilot", family)
        p = task.function.parameters
        rng = np.random.default_rng(np.random.SeedSequence(task.metadata["stream_components"]["function"]))
        if family == "smooth":
            scales = 1 / np.arange(1, 4, dtype=np.float64) ** 2
            np.testing.assert_array_equal(p["cos_coefficients"], rng.normal(0, scales))
            np.testing.assert_array_equal(p["sin_coefficients"], rng.normal(0, scales))
        else:
            spec = config["B"]["generators"][family]
            count = rng.integers(spec["count_inclusive"][0], spec["count_inclusive"][1] + 1)
            assert p["count"] == count
            np.testing.assert_array_equal(p["centers"], rng.uniform(-math.pi, math.pi, count))
            assert all(spec["width_uniform"][0] <= width <= spec["width_uniform"][1] for width in p["widths"])
            assert all(0.5 <= abs(amplitude) <= 1.5 for amplitude in p["amplitudes"])


def test_initialization_is_shared_across_sample_sizes_and_reproducible(config):
    task = generate_task(config, "B", "development", "smooth")
    start = initialization_bundle(config, task)
    again = initialization_bundle(config, task.train_prefix(32))
    for key in ("centers", "widths", "readout", "bias"):
        np.testing.assert_array_equal(start[key], again[key])
    assert start["centers"].shape == (16, 1)
    assert start["readout"].shape == (1, 16)
    np.testing.assert_array_equal(start["widths"], np.full((16, 1), 4))
    np.testing.assert_array_equal(start["bias"], [0])
    sorted_centers = np.sort(start["centers"].ravel())
    np.testing.assert_allclose(np.diff(sorted_centers), 2 * math.pi / 16, atol=2e-15)
    assert initialization_seed(config, task) != batch_seed(config, task)
    assert initialization_seed(config, task) != initialization_seed(config, task, 1)
    assert batch_seed(config, task) == start["metadata"]["batch_seed"]
    for name, inputs, hidden in (("sincos_mlp", 2, 12), ("raw_angle_mlp", 1, 16)):
        state = start[name]
        assert sum(x.size for x in state.values()) == 49
        for key, value in state.items():
            bound = 1 / math.sqrt(inputs if key.startswith("hidden") else hidden)
            assert np.max(np.abs(value)) <= bound
        np.testing.assert_array_equal(state["hidden.weight"], again[name]["hidden.weight"])
    with pytest.raises(ValueError, match="no initialization repeats"):
        initialization_bundle(config, generate_task(config, "A", "pilot", "phase_only"))


@pytest.mark.parametrize("n,batch_size", [(32, 64), (128, 64), (133, 64)])
def test_shuffled_cyclic_batches_cover_complete_epochs(n, batch_size):
    batches = cyclic_batch_indices(n, batch_size, 20, 12345)
    assert batches.shape == (20, batch_size)
    np.testing.assert_array_equal(batches, cyclic_batch_indices(n, batch_size, 20, 12345))
    flat = batches.ravel()
    for start in range(0, len(flat) - n + 1, n):
        np.testing.assert_array_equal(np.sort(flat[start:start + n]), np.arange(n))
    assert not np.array_equal(batches, cyclic_batch_indices(n, batch_size, 20, 54321))


def test_manifest_serialization_and_refuse_overwrite(config, tmp_path):
    task = generate_task(config, "B", "pilot", "narrow")
    restored = TaskFunctionV1.from_dict(json.loads(json.dumps(task.function.to_dict())))
    assert restored.sha256 == task.function.sha256
    np.testing.assert_array_equal(restored.evaluate(task.x_grid), task.y_grid_clean)
    manifest = task.to_manifest()
    assert content_hash(manifest) == content_hash(generate_task(config, "B", "pilot", "narrow").to_manifest())
    target = tmp_path / "task.npz"
    task.save(target)
    with np.load(target, allow_pickle=False) as archive:
        assert json.loads(str(archive["manifest"])) == manifest
        np.testing.assert_array_equal(archive["train_x"], task.x_train)
    with pytest.raises(FileExistsError):
        task.save(target)
    with pytest.raises(ValueError):
        content_hash({"bad": float("nan")})


def test_confirmation_missing_lock_rejected_before_rng(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No RNG may be constructed before confirming a valid lock")
    monkeypatch.setattr("src.methodology_tasks._rng", forbidden)
    with pytest.raises(ValueError, match="requires a verified lock"):
        generate_task(config, "A", "confirmation", "phase_only")


def test_confirmation_changed_lock_rejected_before_rng(config, monkeypatch, tmp_path):
    disposable = copy.deepcopy(config)
    disposable["seeds"]["confirmation"]["task"] = list(range(80101, 80109))
    disposable["seeds"]["confirmation"]["observation"] = list(range(80201, 80209))
    def reject(*args, **kwargs):
        raise ValueError("changed lock")
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected locks must not initialize an RNG")
    monkeypatch.setitem(sys.modules, "src.methodology_protocol", SimpleNamespace(verify_confirmation_lock=reject))
    monkeypatch.setattr("src.methodology_tasks._rng", forbidden)
    with pytest.raises(ValueError, match="changed lock"):
        generate_task(disposable, "B", "confirmation", "smooth", confirmation_lock=tmp_path / "changed.json")


def test_guarded_path_with_disposable_confirmation_roots(config, monkeypatch, tmp_path):
    disposable = copy.deepcopy(config)
    disposable["seeds"]["confirmation"]["task"] = list(range(80101, 80109))
    disposable["seeds"]["confirmation"]["observation"] = list(range(80201, 80209))
    called = []
    def verify(path, received_config, *, repo_root):
        assert received_config is disposable
        assert repo_root == tmp_path
        called.append(path)
        return {"lock_sha256": "disposable-tested-lock"}
    monkeypatch.setitem(sys.modules, "src.methodology_protocol", SimpleNamespace(verify_confirmation_lock=verify))
    task = generate_task(disposable, "B", "confirmation", "smooth", confirmation_lock=tmp_path / "lock.json", root=tmp_path)
    assert called == [tmp_path / "lock.json"]
    assert task.metadata["confirmation_lock_sha256"] == "disposable-tested-lock"
    assert task.metadata["task_root"] == 80101
    with pytest.raises(ValueError, match="Restoring a confirmation"):
        TaskFunctionV1.from_dict(task.function.to_dict())
    restored = TaskFunctionV1.from_dict(task.function.to_dict(), config=disposable,
                    confirmation_lock=tmp_path / "lock.json", root=tmp_path)
    assert restored.sha256 == task.function.sha256


def test_reserved_roots_cannot_be_relabeled_pilot(config):
    changed = copy.deepcopy(config)
    changed["seeds"]["pilot"]["task"] = [5101]
    with pytest.raises(ValueError, match="cannot be relabeled"):
        generate_task(changed, "B", "pilot", "smooth")


@pytest.mark.parametrize("stage,phase,family,replicate", [("C", "pilot", "smooth", 0),
    ("B", "test", "smooth", 0), ("A", "pilot", "weak_phase", 0), ("B", "pilot", "bogus", 0),
    ("B", "development", "smooth", 2), ("B", "confirmation", "smooth", 6)])
def test_invalid_matrix_cells_are_rejected(config, stage, phase, family, replicate):
    with pytest.raises(ValueError):
        generate_task(config, stage, phase, family, replicate)


def test_grid_prefix_and_batch_argument_validation(config):
    grid = midpoint_grid(4096)
    assert grid[0, 0] > -math.pi and grid[-1, 0] < math.pi
    assert grid[0, 0] == pytest.approx(-grid[-1, 0], abs=1e-15)
    np.testing.assert_array_equal(wrap_angle([-math.pi, math.pi, 3 * math.pi]), [-math.pi] * 3)
    task = generate_task(config, "B", "pilot", "smooth")
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            midpoint_grid(invalid)
        with pytest.raises(ValueError):
            task.train_prefix(invalid)
        with pytest.raises(ValueError):
            cyclic_batch_indices(invalid, 64, 10, 1)
    with pytest.raises(ValueError):
        task.train_prefix(129)
