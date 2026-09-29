"""Disposable CPU integration and reserved-access guards for VM-N04.

The miniature stages use only 81001/81002/81003, never any prospective room
seed. Source roots and all artifacts are isolated under pytest's temporary path.
"""

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.run_vm_n04_v1 as runner
from src.dynamics_data_v1 import DynamicsConfig, generate_trajectory, split_bounds


REAL_CONFIG_PATH = runner.ROOT / "configs/vm_n04_v1.json"


@pytest.fixture
def config():
    """Load the prospective matrix without generating its trajectories."""
    return json.loads(REAL_CONFIG_PATH.read_text())


@pytest.fixture
def mini_execution(config, tmp_path, monkeypatch):
    """Prepare a tiny real CPU fit with synthetic source-lock test files."""
    tiny = copy.deepcopy(config)
    tiny["data"].update(split_days=[2, 1, 1], lookback=4, horizon=4, stride=4,
                        outage_test_days=[0])
    tiny["tau_ratios"] = [1.0]
    tiny["attendance"] = [False]
    tiny["test_shift"] = [False, True]
    tiny["test_outage"] = [False, True]
    tiny["model"].update(hidden_size=4, calendar_features=4, fourier_harmonics=2)
    tiny["training"].update(epochs=1, batch_size=128, validation_every=1)
    tiny["analysis"].update(endpoint_steps=[1, 4], primary_step=4, bootstrap_draws=20)
    for index, stage in enumerate(("pilot", "development", "confirmation")):
        tiny["stages"][stage] = dict(room_seeds=[81001 + index],
                                     initialization_seeds=[17], cap_seconds=120)
    project = tmp_path / "project"
    project.mkdir()
    for name in runner.SOURCES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Disposable source fixture: {name}\n", encoding="utf-8")
    runner.write_json(project / "configs/vm_n04_v1.json", tiny)
    output = tmp_path / "artifacts"
    runner.write_json(output / "source_lock.json", {
        "created_utc": "2026-01-01T00:00:00+00:00",
        "baseline_revision": tiny["baseline_revision"],
        "sources": {name: runner.digest(project / name) for name in runner.SOURCES},
        "config": tiny,
    })
    monkeypatch.setattr(runner, "ROOT", project)
    monkeypatch.setattr(runner, "environment", lambda: {
        "test_fixture": True, "gpu": None, "threads": 1, "revision": "disposable",
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tiny, output, project


def read_json(path: Path) -> dict:
    """Read JSON while rejecting nonstandard NaN/Infinity tokens."""
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=reject_constant)


def test_prospective_fit_counts_and_disjoint_rooms(config):
    rates = config["training"]["learning_rates"]
    plans = {
        "pilot": runner.stage_plan(config, "pilot", rates),
        "development": runner.stage_plan(config, "development", [rates[0]]),
        "confirmation": runner.stage_plan(config, "confirmation", [rates[0]]),
    }
    assert {stage: len(plan) for stage, plan in plans.items()} == {
        "pilot": 48, "development": 96, "confirmation": 288,
    }
    seen = set()
    for stage, plan in plans.items():
        rooms = set(config["stages"][stage]["room_seeds"])
        assert not rooms & seen
        seen |= rooms
        identities = [tuple(sorted(row.items())) for row in plan]
        assert len(identities) == len(set(identities))
        assert all("shifted" not in row and "outage" not in row for row in plan)


def test_real_cpu_miniature_all_stages_save_replayable_artifacts(mini_execution):
    config, output, _ = mini_execution
    for stage in ("pilot", "development", "confirmation"):
        if stage == "confirmation":
            runner.write_json(output / "engineering.json", {
                "full_suite_completed": True, "standard_benchmarks_completed": True,
                "scope": "disposable test fixture only; no scientific acceptance assertion",
            })
        runner.run_stage(config, stage, output)
        directory = output / stage
        summary = read_json(directory / "summary.json")
        terminal = read_json(directory / "terminal.json")
        plan = json.loads((directory / "plan.json").read_text())
        assert summary["status"] == terminal["status"] == "complete"
        assert summary["planned_fits"] == summary["completed_fits"] == len(plan)
        assert len(summary["records"]) == 4 * len(plan)
        assert len(summary["references"]) == 4 * len(config["references"])
        assert len(summary["datasets"]) == 4
        assert terminal["summary_sha256"] == runner.digest(directory / "summary.json")
        assert summary["cuda_peak_allocated_bytes"] == 0
        checkpoints = list(directory.glob("*.pt"))
        assert len(checkpoints) == len(plan)
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
        assert set(checkpoint) >= {"state_dict", "identity", "normalizer", "history"}
        for case in directory.glob("*_cases.npz"):
            with np.load(case, allow_pickle=False) as bundle:
                assert bundle["sensor"].shape[1:] == (4, 3)
                assert bundle["y_ppm"].shape[1] == 4
                assert bundle["target_times"].shape == bundle["y_ppm"].shape
                assert np.isfinite(bundle["y_ppm"]).all()
        # Every JSON result uses strict serializable native values, including
        # explicitly empty event/outage diagnostic strata.
        for path in directory.glob("*.json"):
            json.loads(path.read_text(), parse_constant=lambda x: pytest.fail(x))
        if stage == "pilot":
            scores = summary["pilot_rate_scores"]
            expected = min(config["training"]["learning_rates"], key=lambda r: (scores[str(r)], r))
            assert summary["selected_rate"] == expected
            assert set(summary["projected_stage_seconds"]) == {"development", "confirmation"}
        else:
            access = read_json(directory / "access_lock.json")
            assert "pilot/summary.json" in access["dependencies"]
            if stage == "confirmation":
                assert "development/summary.json" in access["dependencies"]
                assert "engineering.json" in access["dependencies"]


def test_stage_cannot_be_overwritten(mini_execution, monkeypatch):
    config, output, _ = mini_execution
    runner.run_stage(config, "pilot", output)
    old_hash = runner.digest(output / "pilot/summary.json")
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Generation repeated"))
    with pytest.raises(FileExistsError):
        runner.run_stage(config, "pilot", output)
    assert runner.digest(output / "pilot/summary.json") == old_hash


def test_source_mismatch_precedes_generation(mini_execution, monkeypatch):
    config, output, project = mini_execution
    (project / "src/dynamics_data_v1.py").write_text("changed source", encoding="utf-8")
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Unverified generation"))
    with pytest.raises(ValueError, match="source mismatch"):
        runner.run_stage(config, "pilot", output)
    assert not (output / "pilot").exists()


def test_missing_required_source_lock_entry_is_rejected(mini_execution, monkeypatch):
    config, output, _ = mini_execution
    lock = read_json(output / "source_lock.json")
    del lock["sources"]["src/dynamics_data_v1.py"]
    runner.write_json(output / "source_lock.json", lock)
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Incomplete lock allowed generation"))
    with pytest.raises(ValueError):
        runner.run_stage(config, "pilot", output)


def test_passed_config_must_match_frozen_config(mini_execution, monkeypatch):
    config, output, _ = mini_execution
    altered = copy.deepcopy(config)
    altered["training"]["epochs"] += 1
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Changed config allowed generation"))
    with pytest.raises(ValueError):
        runner.run_stage(altered, "pilot", output)


def test_failure_receipt_retains_plan_and_refuses_retry(mini_execution, monkeypatch):
    config, output, _ = mini_execution
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise FloatingPointError("intentional disposable fit failure")
    monkeypatch.setattr(runner, "fit_neural", fail)
    with pytest.raises(FloatingPointError):
        runner.run_stage(config, "pilot", output)
    summary = read_json(output / "pilot/summary.json")
    plan = json.loads((output / "pilot/plan.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_fits"] == 0
    assert summary["planned_fits"] == len(plan) == 8
    assert summary["failed_fit"] == plan[0]
    assert "intentional disposable fit failure" in summary["error"]
    assert not (output / "pilot/terminal.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_stage(config, "pilot", output)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["terminal", "checkpoint", "summary", "counts"])
def test_incomplete_or_altered_prior_evidence_blocks_access(mini_execution, monkeypatch, damage):
    config, output, _ = mini_execution
    runner.run_stage(config, "pilot", output)
    if damage == "terminal":
        (output / "pilot/terminal.json").unlink()
    elif damage == "checkpoint":
        next((output / "pilot").glob("*.pt")).unlink()
    else:
        summary_path = output / "pilot/summary.json"
        summary = read_json(summary_path)
        if damage == "counts":
            summary["completed_fits"] -= 1
        else:
            summary["selected_rate"] = 123.0
        runner.write_json(summary_path, summary)
        if damage == "counts":
            # A rehashed but incomplete receipt must also fail semantic checks.
            terminal_path = output / "pilot/terminal.json"
            terminal = read_json(terminal_path)
            terminal["summary_sha256"] = runner.digest(summary_path)
            runner.write_json(terminal_path, terminal)
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Incomplete evidence allowed generation"))
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run_stage(config, "development", output)
    assert not (output / "development").exists()


def test_confirmation_requires_engineering_completion(mini_execution, monkeypatch):
    config, output, _ = mini_execution
    runner.run_stage(config, "pilot", output)
    runner.run_stage(config, "development", output)
    runner.write_json(output / "engineering.json", {
        "full_suite_completed": True, "standard_benchmarks_completed": False,
    })
    monkeypatch.setattr(runner, "generate_trajectory", lambda *a, **k: pytest.fail("Unvalidated access reached"))
    with pytest.raises(ValueError, match="engineering"):
        runner.run_stage(config, "confirmation", output)
    assert not (output / "confirmation").exists()


def test_seasonal_reference_excludes_last_packet_arriving_after_train_cutoff():
    trajectory = generate_trajectory(81551, DynamicsConfig(split_days=(2, 1, 1)))
    cutoff = split_bounds(trajectory, "train")[1]
    changed = replace(trajectory, observed_ppm=trajectory.observed_ppm.copy())
    changed.observed_ppm[cutoff - 1:] = 1e10
    np.testing.assert_array_equal(runner.seasonal_table(trajectory), runner.seasonal_table(changed))


def test_forecast_metrics_separate_endpoint_and_horizon_and_empty_strata(mini_execution):
    config, _, _ = mini_execution
    trajectory = generate_trajectory(81888, DynamicsConfig(**config["data"]))
    norm = runner.fit_normalizer(trajectory)
    batch = runner.make_windows(trajectory, "test", norm)
    prediction = batch.y_ppm.copy()
    prediction[:, 0] += 1.0
    prediction[:, -1] += 2.0
    metrics = runner.evaluate(prediction, batch, norm.std, [1, 4])
    assert metrics["endpoints"]["1"]["rmse"] == pytest.approx(1.0)
    assert metrics["endpoints"]["4"]["rmse"] == pytest.approx(2.0)
    assert metrics["all_horizon"]["rmse"] == pytest.approx(np.sqrt(5.0 / 4))
    assert metrics["endpoints"]["1"]["origin_outage"] == {"n": 0, "rmse": None, "mae": None}
