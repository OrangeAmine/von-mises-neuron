"""VM-N01 generator contracts; no real reserved target is generated here."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.smooth_local_tasks_v1 import (
    FAMILIES, STREAM_IDS, cell_arrays, evaluate_function, evaluate_raw,
    generate_task, initializer_rotation, midpoint_grid, stream_components,
)


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/vm_n01_v1.json").read_text())


@pytest.mark.parametrize("family", FAMILIES)
def test_shapes_normalization_nested_labels_and_noise(config, family):
    task = generate_task(config, "pilot", family)
    arrays, meta = task["arrays"], task["metadata"]
    assert arrays["train_x"].shape == (192, 1)
    assert arrays["validation_x"].shape == (64, 1)
    assert arrays["grid_x"].shape == (2048, 1)
    assert all(value.dtype == np.float64 and value.ndim == 2 and value.shape[1] == 1
               and np.isfinite(value).all() for value in arrays.values())
    normalized = evaluate_function(meta, midpoint_grid(8192))
    assert normalized.mean() == pytest.approx(0., abs=2e-14)
    assert normalized.var() == pytest.approx(1., abs=2e-14)
    assert not np.intersect1d(midpoint_grid(8192), arrays["grid_x"]).size
    assert not np.intersect1d(arrays["train_x"], arrays["validation_x"]).size
    small, large = cell_arrays(task, 64), cell_arrays(task, 256)
    for split, count in (("train", 48), ("validation", 16)):
        for field in ("x", "y"):
            np.testing.assert_array_equal(small[f"{split}_{field}"], large[f"{split}_{field}"][:count])
        np.testing.assert_allclose(arrays[f"{split}_y"] - arrays[f"{split}_clean"],
                                   .1*arrays[f"{split}_unit_noise"], atol=1e-15)
    small["train_y"][:] = 999
    assert not np.any(arrays["train_y"] == 999)
    for name, value in arrays.items():
        assert hashlib.sha256(value.tobytes()).hexdigest() == meta["array_sha256"][name]


@pytest.mark.parametrize("family", FAMILIES)
def test_periodic_equations_and_nonperiodic_diagnostic(config, family):
    meta = generate_task(config, "pilot", family)["metadata"]
    x = np.asarray([[-math.pi], [-.91], [0.], [math.pi], [2.43]])
    if family != "nonperiodic":
        np.testing.assert_allclose(evaluate_function(meta, x), evaluate_function(meta, x+2*math.pi), atol=8e-14, rtol=2e-14)
    else:
        np.testing.assert_allclose(evaluate_function(meta, x+2*math.pi)-evaluate_function(meta, x),
                                   2/meta["normalization"]["scale"], atol=2e-14)


def test_exact_piecewise_equations_and_mixed_sum():
    peaks = {"centers": [0.], "widths": [.4], "amplitudes": [2.], "count": 1}
    x = np.asarray([0., .2, .4, .5, 2*math.pi])
    np.testing.assert_allclose(evaluate_raw("narrow", peaks, x), [2., 2*.75**3, 0., 0., 2.], atol=1e-14)
    np.testing.assert_allclose(evaluate_raw("multimodal", peaks, x), [2., 1., 0., 0., 2.], atol=1e-14)
    smooth = {"cos_coefficients": [1., 0., 0.], "sin_coefficients": [0., 2., 0.]}
    mixed = {"smooth": smooth, "local": peaks}
    np.testing.assert_allclose(evaluate_raw("mixed", mixed, x), np.cos(x)+2*np.sin(2*x)+evaluate_raw("narrow", peaks, x))


def test_streams_deterministic_separate_and_prefix_stable(config):
    np.random.seed(111)
    before = np.random.get_state()
    task = generate_task(config, "development", "mixed", 1)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    repeated = generate_task(config, "development", "mixed", 1)
    assert task["metadata"] == repeated["metadata"]
    for name in STREAM_IDS:
        assert stream_components(config, "development", "mixed", 1, name) == (31092202, 101, 3, 1, STREAM_IDS[name])
    for split in ("train", "validation"):
        entropy = task["metadata"]["stream_components"][f"{split}_inputs"]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        np.testing.assert_array_equal(task["arrays"][f"{split}_x"], rng.uniform(-math.pi, math.pi, task["arrays"][f"{split}_x"].shape))
    shorter_config = copy.deepcopy(config)
    shorter_config["tasks"]["label_budgets"] = [64]
    shorter = generate_task(shorter_config, "development", "mixed", 1)
    for split, count in (("train", 48), ("validation", 16)):
        for field in ("x", "y", "unit_noise", "clean"):
            np.testing.assert_array_equal(shorter["arrays"][f"{split}_{field}"], task["arrays"][f"{split}_{field}"][:count])
    pilot = generate_task(config, "pilot", "mixed")
    other = generate_task(config, "development", "mixed", 0)
    assert len({row["metadata"]["function_sha256"] for row in (task, pilot, other)}) == 3


def test_normal_coefficients_and_peak_ranges_follow_definition(config):
    smooth = generate_task(config, "pilot", "smooth")["metadata"]["parameters"]
    rng = np.random.default_rng(np.random.SeedSequence([31092201, 101, 0, 0, 0]))
    scale = 1 / np.arange(1, 4)**2
    np.testing.assert_array_equal(smooth["cos_coefficients"], rng.normal(0, scale))
    np.testing.assert_array_equal(smooth["sin_coefficients"], rng.normal(0, scale))
    for family, count_range, widths in (("narrow", (1, 2), (.15, .45)), ("multimodal", (3, 5), (.2, .8))):
        parameters = generate_task(config, "pilot", family)["metadata"]["parameters"]
        assert count_range[0] <= parameters["count"] <= count_range[1]
        assert all(widths[0] <= value <= widths[1] for value in parameters["widths"])
        assert all(.5 <= abs(value) <= 1.5 for value in parameters["amplitudes"])


def test_confirmation_guard_precedes_every_rng_and_target(config, monkeypatch, tmp_path):
    import src.smooth_local_tasks_v1 as tasks
    import src.smooth_local_protocol_v1 as protocol

    def forbidden(*args, **kwargs):
        pytest.fail("Random generator or target evaluation before lock verification")

    def reject(*args, **kwargs):
        raise ValueError("lock was checked")

    monkeypatch.setattr(tasks, "_rng", forbidden)
    monkeypatch.setattr(tasks, "evaluate_raw", forbidden)
    monkeypatch.setattr(protocol, "verify_lock", reject)
    with pytest.raises(ValueError, match="require a verified"):
        generate_task(config, "confirmation", "smooth")
    with pytest.raises(ValueError, match="lock was checked"):
        generate_task(config, "confirmation", "smooth", confirmation_lock=tmp_path/"absent.json")


def test_disposable_confirmation_calls_verifier(config, monkeypatch, tmp_path):
    # The only successful confirmation generation in this module uses roots
    # outside the scientific registry, so it consumes no reserved task.
    disposable = copy.deepcopy(config)
    disposable["seeds"].update(pilot=9001, development=9002, confirmation=9003, initialization=9004)
    lock = tmp_path/"disposable_lock.json"
    lock.write_text("{}")
    calls = []

    def verify(path, cfg):
        calls.append((path, cfg["seeds"]["confirmation"]))
        return {"lock_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    monkeypatch.setattr("src.smooth_local_protocol_v1.verify_lock", verify)
    task = generate_task(disposable, "confirmation", "smooth", confirmation_lock=lock)
    assert calls == [(lock, 9003)]
    assert task["metadata"]["confirmation_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()


def test_initialization_is_shared_and_target_independent(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Initializer inspected target truth")

    monkeypatch.setattr("src.smooth_local_tasks_v1._parameters", forbidden)
    monkeypatch.setattr("src.smooth_local_tasks_v1.evaluate_raw", forbidden)
    first = initializer_rotation(config, "confirmation", "mixed", 5, 0)
    second = initializer_rotation(config, "confirmation", "mixed", 5, 1)
    assert first != second
    assert -math.pi <= first < math.pi
    rng = np.random.default_rng(np.random.SeedSequence([31092204, 101, 2, 3, 5, 0]))
    assert first == rng.uniform(-math.pi, math.pi)


@pytest.mark.parametrize("phase,family,replicate", [("bad", "smooth", 0), ("pilot", "bad", 0), ("pilot", "smooth", 1), ("pilot", "smooth", -1), ("pilot", "smooth", True)])
def test_bad_context_rejected(config, phase, family, replicate):
    with pytest.raises(ValueError):
        generate_task(config, phase, family, replicate)


def test_reserved_relabel_and_bad_budget_rejected(config):
    config["seeds"]["pilot"] = config["seeds"]["confirmation"]
    with pytest.raises(ValueError):
        generate_task(config, "pilot", "smooth")
    config["seeds"]["confirmation"] = 123
    with pytest.raises(ValueError, match="Reserved"):
        generate_task(config, "pilot", "smooth")


def test_initializer_and_budget_bounds(config):
    with pytest.raises(ValueError):
        initializer_rotation(config, "pilot", "smooth", 0, 1)
    task = generate_task(config, "pilot", "smooth")
    with pytest.raises(ValueError):
        cell_arrays(task, 65)
    config["tasks"]["label_budgets"] = [256, 64]
    with pytest.raises(ValueError):
        generate_task(config, "pilot", "smooth")
