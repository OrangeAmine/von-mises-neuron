"""Independent integration checks using disposable data and mocked fit calls.

No gradient updates or reserved confirmation functions are executed here.
"""

import io
import math
from collections import Counter

import numpy as np
import pytest
import torch

from scripts import analyze_vm_methodology as analysis
from scripts import run_vm_methodology as runner
from src import methodology_protocol as protocol
from src import methodology_tasks as tasks
from src import methodology_training as training


@pytest.fixture
def config():
    return protocol.read_json(protocol.CONFIG_PATH)


def choices(config):
    return {"A": {"Adam": .01, "SGD": .03}, "B": {
        item["id"]: .001 for item in config["B"]["models"] if item["kind"] == "gradient"}}


@pytest.mark.parametrize("stage,phase", [(stage, phase) for stage in ("A", "B")
                                       for phase in ("pilot", "development", "confirmation")])
def test_independent_matrix_enumeration_agrees_without_generating_tasks(config, stage, phase):
    settings = choices(config)
    actual = Counter(analysis._identity(row) for row in protocol.planned_runs(config, stage, phase, settings))
    expected = Counter(analysis._expected_cells(config, stage, phase, settings))
    assert actual == expected
    if stage == "A":
        assert all(key[-1] is None for key in actual)


def test_runner_confirmation_guard_precedes_output_and_generation(config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reserved generation must not occur without a verified lock")

    monkeypatch.setattr(tasks, "generate_task", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="lock"):
        runner.execute_stage(config, "B", "confirmation", output)
    assert not output.exists()


@pytest.mark.parametrize("stage,family", [("A", "phase_only"), ("B", "smooth")])
def test_generator_lock_rejection_precedes_all_random_draws(config, monkeypatch, stage, family):
    def forbidden(*args, **kwargs):
        pytest.fail("No function or observation RNG before lock validation")

    monkeypatch.setattr(tasks, "_rng", forbidden)
    monkeypatch.setattr(tasks, "_sample_parameters", forbidden)
    with pytest.raises(ValueError, match="lock"):
        tasks.generate_task(config, stage, "confirmation", family)


def test_runner_all_b_arms_receive_same_nested_data_and_batch_order(config, monkeypatch):
    task = tasks.generate_task(config, "B", "development", "smooth")
    rows = [row for row in protocol.planned_runs(config, "B", "development", choices(config))
            if row["family"] == "smooth" and row["replicate"] == 0 and row["kind"] == "gradient"]
    seen = {}

    def fake_train(model, data, **options):
        n = len(data.x_train)
        np.testing.assert_array_equal(data.x_train, task.x_train[:n])
        np.testing.assert_array_equal(data.y_train, task.y_train[:n])
        np.testing.assert_array_equal(data.x_validation, task.x_validation)
        np.testing.assert_array_equal(data.y_validation, task.y_validation)
        np.testing.assert_array_equal(data.y_grid_noisy, task.y_grid_noisy)
        key = (n, options["batch_seed"])
        batches = options["batch_indices"]
        if key in seen:
            np.testing.assert_array_equal(batches, seen[key])
        else:
            seen[key] = batches.copy()
        return {"summary": {"status": "mock_only"}, "artifacts": {"initial_state": model.state_dict()}}

    monkeypatch.setattr(training, "train_b", fake_train)
    for row in rows:
        bundle = runner._fit(row, task, config, math.inf, torch.device("cpu"))
        assert bundle["summary"]["status"] == "mock_only"
        assert bundle["summary"]["initialization_metadata"]["batch_seed"] in [key[1] for key in seen]
        # Actual root serialization must remain safe-loadable with initialization
        # tensors, nested RNG metadata and state dictionaries together.
        buffer = io.BytesIO()
        torch.save(bundle["artifacts"], buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=True)
        assert isinstance(restored["initialization_bundle"]["centers"], torch.Tensor)
    assert len(rows) == 3 * 2 * 4
    assert len(seen) == 3 * 2


def test_runner_a_coordinates_receive_same_task_and_full_batch_budget(config, monkeypatch):
    task = tasks.generate_task(config, "A", "pilot", "phase_only")
    rows = [row for row in protocol.planned_runs(config, "A", "pilot")
            if row["family"] == "phase_only" and row["kind"] == "gradient"]
    seen = []

    def fake_train(model, data, **options):
        assert data is task
        assert options["updates"] == 40 and options["record_every"] == 10
        assert set(options) == {"optimizer_name", "learning_rate", "penalty", "updates", "record_every", "device", "deadline"}
        seen.append(model.model_id)
        return {"summary": {"status": "mock_only"}, "artifacts": {}}

    monkeypatch.setattr(training, "train_a", fake_train)
    for row in rows:
        runner._fit(row, task, config, math.inf, torch.device("cpu"))
    assert Counter(seen) == {"polar": 12, "cartesian": 12}


def test_a_bootstrap_resamples_whole_paired_condition_bundle(config):
    rows = []
    for family, values in (("phase_only", [0., 2.]), ("both", [2., 0.])):
        for replicate in range(2):
            for model in ("polar", "cartesian"):
                rows.append({"stage": "A", "kind": "gradient", "audited": True,
                             "family": family, "replicate": replicate, "model": model,
                             "optimizer": "Adam", "penalty": 0,
                             "primary_error": values[replicate] if model == "polar" else 2.})
    result = analysis.analyze_a(rows, config, complete=True)
    assert result["relative_improvement"] == .5
    # Averaging both conditions within task leaves exactly the same paired
    # difference for each task. Resampling observations/conditions would not.
    assert result["bootstrap"]["lower"] == result["bootstrap"]["upper"] == .5
    assert result["paired_task_differences"] == [-1., -1.]


def test_b_balances_sample_sizes_tasks_families_and_excludes_negative_control(config):
    rows = []
    constants = {"learned_vm": 2., "fourier_ridge": 3., "fixed_vm_ridge": 4.,
                 "learned_geodesic": 5., "sincos_mlp": 6., "raw_angle_mlp": .001}
    for model, base in constants.items():
        for family in config["B"]["family_ids"]:
            for replicate in range(2):
                for n, factor in zip((32, 128, 512), (3., 2., 1.)):
                    for initialization in range(1 if "ridge" in model else 2):
                        rows.append({"stage": "B", "audited": True, "model": model,
                                     "family": family, "replicate": replicate, "n_train": n,
                                     "initialization_index": initialization,
                                     "primary_error": base * factor if family != "nonperiodic" else 1000 / base})
    result = analysis.analyze_b(rows, config, complete=True)
    assert result["primary_comparator"] == "fourier_ridge"
    assert result["models"]["learned_vm"]["balanced_periodic_error"] == 4
    assert result["relative_improvement"] == pytest.approx(1 - 4 / 6)
    assert result["bootstrap"]["lower"] == pytest.approx(1 - 4 / 6)
    assert result["bootstrap"]["upper"] == pytest.approx(1 - 4 / 6)
    assert result["models"]["learned_vm"]["family_errors"]["nonperiodic"] == 500
    assert result["gate_passed"]


def test_incomplete_a_or_b_cannot_select_success_from_favorable_survivors(config):
    assert not analysis.analyze_a([], config, complete=False)["gate_passed"]
    result = analysis.analyze_b([], config, complete=False)
    assert not result["gate_passed"]
    assert result["primary_comparator"] is None
