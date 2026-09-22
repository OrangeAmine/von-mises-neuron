"""Fresh-stream and total-label contracts; valid holdout tests use fake roots."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.locality_tasks import (
    FAMILY_IDS, STREAM_IDS, VERSION, LocalityTaskV2,
    generate_locality_task, initializer, stream_components,
)


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/vm_locality_v2.json").read_text())


def grid(points: int) -> np.ndarray:
    return (-math.pi + (np.arange(points, dtype=np.float64) + .5) * (2 * math.pi / points))[:, None]


@pytest.mark.parametrize("family", FAMILY_IDS)
def test_total_label_budgets_nest_within_disjoint_split_pools(config, family):
    task = generate_locality_task(config, "pilot", family)
    assert task.train_x.shape == (384, 1)
    assert task.validation_x.shape == (128, 1)
    assert task.grid_x.shape == (4096, 1)
    assert not np.intersect1d(task.train_x.ravel(), task.validation_x.ravel()).size
    largest = task.for_budget(512, .05)
    for total, train_count, validation_count in ((64, 48, 16), (128, 96, 32), (512, 384, 128)):
        arrays = task.for_budget(total, .05)
        meta = task.budget_metadata(total, .05)
        assert len(arrays["train_x"]) == train_count
        assert len(arrays["validation_x"]) == validation_count
        assert meta["train_labels"] + meta["validation_labels"] == meta["total_labels"] == total
        for name in ("train_x", "train_y", "train_clean"):
            np.testing.assert_array_equal(arrays[name], largest[name][:train_count])
        for name in ("validation_x", "validation_y", "validation_clean"):
            np.testing.assert_array_equal(arrays[name], largest[name][:validation_count])
        assert all(value.dtype == np.float64 and value.ndim == 2 and value.shape[1] == 1 and np.isfinite(value).all()
                   for value in arrays.values())
    # Small-budget held-out labels remain held out at the larger training budget.
    small_validation = task.for_budget(64, .05)["validation_x"]
    assert not np.intersect1d(small_validation.ravel(), largest["train_x"].ravel()).size


@pytest.mark.parametrize("family", FAMILY_IDS)
def test_noise_is_crossed_and_paired_without_task_or_split_confounds(config, family):
    task = generate_locality_task(config, "development", family, 2)
    low, high = task.for_budget(128, .05), task.for_budget(128, .2)
    for name in ("train_x", "validation_x", "grid_x", "train_clean", "validation_clean", "grid_clean"):
        np.testing.assert_array_equal(low[name], high[name])
    for noisy, clean in (("train_y", "train_clean"), ("validation_y", "validation_clean"), ("grid_noisy", "grid_clean")):
        np.testing.assert_allclose((low[noisy] - low[clean]) / .05,
                                   (high[noisy] - high[clean]) / .2, atol=4e-14, rtol=4e-14)
    assert not np.array_equal(task.train_unit_noise[:128], task.validation_unit_noise)
    assert not np.array_equal(task.validation_unit_noise, task.grid_unit_noise[:128])
    assert task.metadata["noise_crossed_with_every_task"]


def test_stream_derivation_and_separate_input_noise_generators(config):
    task = generate_locality_task(config, "development", "narrow", 1)
    for stream, split_id in STREAM_IDS.items():
        components = stream_components(config, "development", "narrow", 1, stream)
        expected_root = 2026092301 if stream == "function" else 2026092401
        assert components == (expected_root, 2, 2, 2, 1, split_id)
        assert task.metadata["stream_components"][stream] == list(components)
    for split in ("train", "validation"):
        x = getattr(task, f"{split}_x")
        rng = np.random.default_rng(np.random.SeedSequence(task.metadata["stream_components"][f"{split}_inputs"]))
        np.testing.assert_array_equal(x, rng.uniform(-math.pi, math.pi, x.shape))
        rng = np.random.default_rng(np.random.SeedSequence(task.metadata["stream_components"][f"{split}_noise"]))
        np.testing.assert_array_equal(getattr(task, f"{split}_unit_noise"), rng.standard_normal(x.shape))
    # Independent noise streams make prefixes stable even if maximum pool size
    # changes in a separate disposable configuration.
    small = copy.deepcopy(config)
    small["tasks"]["label_budgets"] = [64, 128]
    shorter = generate_locality_task(small, "development", "narrow", 1)
    np.testing.assert_array_equal(shorter.train_x, task.train_x[:96])
    np.testing.assert_array_equal(shorter.train_unit_noise, task.train_unit_noise[:96])
    np.testing.assert_array_equal(shorter.validation_unit_noise, task.validation_unit_noise[:32])


@pytest.mark.parametrize("family", FAMILY_IDS)
def test_declared_normalization_periodicity_and_negative_control(config, family):
    task = generate_locality_task(config, "pilot", family)
    normalized = task.function.evaluate(grid(65536))
    assert abs(normalized.mean()) < 2e-14
    assert normalized.var() == pytest.approx(1., abs=2e-14)
    np.testing.assert_array_equal(task.grid_x, grid(4096))
    assert task.grid_x.min() > -math.pi and task.grid_x.max() < math.pi
    np.testing.assert_array_equal(task.function.evaluate(task.train_x), task.train_clean)
    x = np.array([[-math.pi], [-1.3], [.14], [math.pi-1e-8]])
    if family != "nonperiodic":
        np.testing.assert_allclose(task.function.evaluate(x), task.function.evaluate(x + 2 * math.pi), atol=2e-13, rtol=1e-13)
    else:
        difference = task.function.evaluate(math.pi) - task.function.evaluate(-math.pi)
        assert float(difference) == pytest.approx(2 / task.function.normalization["scale"])


def test_task_generation_reproducible_distinct_and_process_rng_unchanged(config):
    np.random.seed(83401)
    before = np.random.get_state()
    first = generate_locality_task(config, "pilot", "smooth")
    initialization = initializer(config, "pilot", "smooth", 0, 0)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert first.to_dict() == generate_locality_task(config, "pilot", "smooth").to_dict()
    development = [generate_locality_task(config, "development", "smooth", r) for r in range(3)]
    assert len({task.function.sha256 for task in [first, *development]}) == 4
    assert len({tuple(task.metadata["stream_components"]["train_inputs"]) for task in [first, *development]}) == 4
    assert initialization["metadata"]["uses_target_truth"] is False


def test_fresh_function_parameter_draws_follow_v2_entropy(config):
    task = generate_locality_task(config, "pilot", "smooth")
    rng = np.random.default_rng(np.random.SeedSequence([2026092101, 2, 1, 1, 0, 0]))
    scale = 1 / np.arange(1, 4, dtype=np.float64) ** 2
    np.testing.assert_array_equal(task.function.parameters["cos_coefficients"], rng.normal(0, scale))
    np.testing.assert_array_equal(task.function.parameters["sin_coefficients"], rng.normal(0, scale))
    for family in ("narrow", "multimodal"):
        task = generate_locality_task(config, "pilot", family)
        spec, parameters = config["tasks"]["generators"][family], task.function.parameters
        assert spec["count_inclusive"][0] <= parameters["count"] <= spec["count_inclusive"][1]
        assert all(spec["width_uniform"][0] <= x <= spec["width_uniform"][1] for x in parameters["widths"])
        assert all(.5 <= abs(x) <= 1.5 for x in parameters["amplitudes"])


def test_initialization_shared_geometry_and_nested_capacity_controls(config):
    first = initializer(config, "development", "narrow", 1, 0)
    second = initializer(config, "development", "narrow", 1, 1)
    assert first["centers"].shape == first["kappas"].shape == (16, 1)
    np.testing.assert_array_equal(first["kappas"], np.full((16, 1), 4.))
    assert first["metadata"]["stream_components"] == [2026092701, 2, 2, 2, 1, 0]
    assert not np.array_equal(first["centers"], second["centers"])
    assert first["metadata"]["shared_across_budgets_noise_and_banks"]
    np.testing.assert_allclose(np.diff(np.sort(first["centers"].ravel())), 2 * math.pi / 16, atol=2e-15)
    for name, widths in (("fixed32", [4., 16.]), ("fixed48", [1., 4., 16.])):
        bank = first[name]
        np.testing.assert_array_equal(bank["centers"], np.repeat(first["centers"], len(widths), axis=0))
        np.testing.assert_array_equal(bank["kappas"], np.tile(widths, 16)[:, None])
        width4 = bank["kappas"].ravel() == 4
        np.testing.assert_array_equal(bank["centers"][width4], first["centers"])
    changed = copy.deepcopy(config)
    changed["initialization"]["concentration"] = 3.
    changed["initialization"]["fixed32_widths"] = [3., 7.]
    value = initializer(changed, "development", "narrow", 1, 0)
    np.testing.assert_array_equal(value["kappas"], np.full((16, 1), 3.))
    np.testing.assert_array_equal(value["fixed32"]["kappas"], np.tile([3., 7.], 16)[:, None])


def test_initialization_does_not_consult_function_generation_or_observations(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Initialization must not evaluate or sample a target")
    monkeypatch.setattr("src.locality_tasks._parameters", forbidden)
    monkeypatch.setattr("src.locality_tasks.LocalityFunctionV2.evaluate", forbidden)
    result = initializer(config, "confirmation", "narrow", 7, 1)
    assert result["centers"].shape == (16, 1)


def test_serialization_roundtrip_content_hash_and_no_caller_aliases(config):
    task = generate_locality_task(config, "pilot", "multimodal")
    value = json.loads(json.dumps(task.to_dict(), allow_nan=False))
    assert value["version"] == VERSION
    digest = hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "sha256"},
                            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert value["sha256"] == digest
    restored = LocalityTaskV2.from_dict(value, config)
    assert restored.to_dict() == value
    for key, array in task.for_budget(64, .2).items():
        np.testing.assert_array_equal(restored.for_budget(64, .2)[key], array)
    output = task.for_budget(64, .05)
    output["train_x"][0] = 12345
    assert task.train_x[0, 0] != 12345
    value["metadata"]["replicate"] = 999
    assert task.metadata["replicate"] == 0


def test_modified_serialized_parameters_or_arrays_rejected(config):
    task = generate_locality_task(config, "pilot", "narrow")
    value = task.to_dict()
    value["arrays"]["train_x"][0][0] += .1
    with pytest.raises(ValueError, match="content changed"):
        LocalityTaskV2.from_dict(value, config)
    changed = copy.deepcopy(config)
    changed["tasks"]["noise_levels"] = [.05]
    with pytest.raises(ValueError, match="configuration changed"):
        LocalityTaskV2.from_dict(task.to_dict(), changed)


def test_missing_and_changed_confirmation_lock_rejected_before_rng(config, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("No target RNG may run before successful holdout validation")
    monkeypatch.setattr("src.locality_tasks._rng", forbidden)
    with pytest.raises(ValueError, match="requires a verified locality lock"):
        generate_locality_task(config, "confirmation", "smooth")
    def reject(*args, **kwargs):
        raise ValueError("changed protocol lock")
    monkeypatch.setitem(sys.modules, "src.locality_protocol", SimpleNamespace(verify_confirmation_lock=reject))
    with pytest.raises(ValueError, match="changed protocol lock"):
        generate_locality_task(config, "confirmation", "smooth", confirmation_lock=tmp_path / "lock.json")


def test_valid_confirmation_generation_and_restoration_only_use_disposable_test_roots(config, monkeypatch, tmp_path):
    disposable = copy.deepcopy(config)
    disposable["seeds"]["confirmation"] = {"task": 800021, "observation": 800022}
    calls = []
    def verify(path, received, *, root):
        assert received is disposable and root == tmp_path
        calls.append(path)
        return {"lock_sha256": "disposable-test-lock"}
    monkeypatch.setitem(sys.modules, "src.locality_protocol", SimpleNamespace(verify_confirmation_lock=verify))
    task = generate_locality_task(disposable, "confirmation", "narrow", 1,
                                  confirmation_lock=tmp_path / "lock.json", root=tmp_path)
    assert task.metadata["task_root"] == 800021
    assert task.metadata["confirmation_lock_sha256"] == "disposable-test-lock"
    value = task.to_dict()
    with pytest.raises(ValueError, match="requires a verified locality lock"):
        LocalityTaskV2.from_dict(value, disposable)
    restored = LocalityTaskV2.from_dict(value, disposable, confirmation_lock=tmp_path / "lock.json", root=tmp_path)
    assert restored.to_dict() == value and len(calls) == 2
    monkeypatch.setitem(sys.modules, "src.locality_protocol", SimpleNamespace(
        verify_confirmation_lock=lambda *args, **kwargs: {"lock_sha256": "different-lock"}))
    with pytest.raises(ValueError, match="different confirmation lock"):
        LocalityTaskV2.from_dict(value, disposable, confirmation_lock=tmp_path / "lock.json", root=tmp_path)
    value["metadata"]["phase"] = "development"
    with pytest.raises(ValueError, match="seed registry"):
        LocalityTaskV2.from_dict(value, disposable)


@pytest.mark.parametrize("root_seed", [2026092501, 2026092601, 5101, 5108, 6101, 6108, 7101])
def test_reserved_or_consumed_seeds_cannot_be_relabeled(config, root_seed):
    config["seeds"]["pilot"]["task"] = root_seed
    with pytest.raises(ValueError, match="Reserved|Consumed"):
        generate_locality_task(config, "pilot", "smooth")


def test_invalid_phase_budget_noise_and_initialization_arguments(config):
    task = generate_locality_task(config, "pilot", "smooth")
    for budget in (0, 48, 96, 256, True, 64.5):
        with pytest.raises(ValueError):
            task.for_budget(budget, .05)
    for noise in (.1, -.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            task.for_budget(64, noise)
    for phase, family, replicate in (("bad", "smooth", 0), ("pilot", "bad", 0), ("pilot", "smooth", 1),
                                     ("development", "smooth", 3), ("confirmation", "smooth", 8), ("pilot", "smooth", True)):
        with pytest.raises(ValueError):
            generate_locality_task(config, phase, family, replicate)
    for phase, index in (("pilot", 1), ("development", 2), ("development", -1), ("development", True)):
        with pytest.raises(ValueError):
            initializer(config, phase, "smooth", 0, index)


def test_initialization_cannot_share_target_seed_or_change_declared_feature_counts(config):
    changed = copy.deepcopy(config)
    changed["seeds"]["initialization"] = changed["seeds"]["development"]["task"]
    with pytest.raises(ValueError, match="independent"):
        initializer(changed, "development", "smooth", 0, 0)
    changed = copy.deepcopy(config)
    changed["initialization"]["fixed32_widths"] = [4.]
    with pytest.raises(ValueError, match="32/48"):
        initializer(changed, "development", "smooth", 0, 0)
