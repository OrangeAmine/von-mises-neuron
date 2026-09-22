"""Unchanged synthetic acceptance, reproducible construction and failure records."""

import json

import pytest
import torch

from scripts import validate_methodology_forecasters as runner
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster


def passing_rows():
    return [{"model": name, "execution_status": "completed", "metrics": {
        "rmse": 10., "mae": 8., "r2": 0.9, "mape": 2.}} for name in runner.MODELS]


def test_all_required_gates_pass():
    summary = runner.validation_summary(passing_rows())
    assert summary["all_passed"] and summary["all_runs_attempted"]
    assert all(summary["candidates"].values())


@pytest.mark.parametrize("name", runner.CANDIDATES)
@pytest.mark.parametrize("metric,value", [("rmse", 35.01), ("mae", 25.01), ("r2", 0.199), ("mape", 5.001)])
def test_each_unchanged_absolute_threshold_is_required(name, metric, value):
    rows = passing_rows()
    next(row for row in rows if row["model"] == name)["metrics"][metric] = value
    summary = runner.validation_summary(rows)
    assert not summary["candidates"][name] and not summary["all_passed"]


@pytest.mark.parametrize("metric,value", [("rmse", 11.01), ("mae", 8.801), ("mape", 2.201), ("r2", 0.799)])
def test_temporal_relative_gates_apply_inside_absolute_thresholds(metric, value):
    rows = passing_rows()
    rows[-1]["metrics"][metric] = value
    summary = runner.validation_summary(rows)
    assert summary["acceptance"]["DirectCalendarForecasterV1"]["absolute_passed"]
    assert not summary["candidates"]["DirectCalendarForecasterV1"]


@pytest.mark.parametrize("status", ["pending", "running", "failed", "numerical_failure", "interrupted"])
def test_incomplete_or_failed_reference_cannot_certify_temporal_regression(status):
    rows = passing_rows()
    rows[0]["execution_status"] = status
    assert not runner.validation_summary(rows)["candidates"]["DirectCalendarForecasterV1"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_missing_nonfinite_metrics_cannot_pass(value):
    rows = passing_rows()
    rows[1]["metrics"]["rmse"] = value
    assert not runner.validation_summary(rows)["candidates"]["PolarForecasterV2"]


def test_duplicate_and_unknown_model_rows_rejected():
    with pytest.raises(ValueError):
        runner.validation_summary(passing_rows() + passing_rows()[:1])
    with pytest.raises(ValueError):
        runner.validation_summary([{"model": "undeclared"}])


@pytest.mark.parametrize("name", runner.CANDIDATES)
def test_constructor_matches_exact_source_and_rng_without_training(name):
    torch.manual_seed(42)
    adapter = runner.build_model(name).double().eval()
    rng = torch.get_rng_state().clone()
    config = runner.model_config(name)["source"]
    torch.manual_seed(42)
    cls = VonMisesForecaster if name == "PolarForecasterV2" else TemporalVonMisesForecaster
    reference = cls(**config["kwargs"]).double().eval()
    assert torch.equal(rng, torch.get_rng_state())
    # Conversions happen in float32 before this test promotes the state.
    x = torch.randn(2, 8, 4 if name == "PolarForecasterV2" else 6, dtype=torch.float64)
    torch.testing.assert_close(adapter(x), reference(x), atol=1e-5, rtol=1e-4)
    assert runner.model_config(name)["conversion"] == "from_legacy"


def test_mocked_failure_retains_planned_rows_states_and_successes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "source_manifest", lambda: {"revision": "test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_fit(name, model, data, epochs, device):
        assert epochs == 50 and device.type == "cpu"
        model.training_history = {"validation_loss": [1.], "best_val_loss": 1., "best_epoch": 1}
        if name == "PolarForecasterV2":
            raise FloatingPointError("injected failure")
        return model

    monkeypatch.setattr(runner, "_fit", fake_fit)
    monkeypatch.setattr(runner, "predict", lambda model, data, device: torch.zeros_like(data["y_test"]))
    monkeypatch.setattr(runner, "compute_benchmark_metrics", lambda *args: passing_rows()[0]["metrics"])
    output = tmp_path / "new"
    summary = runner.run_validation(output)
    assert summary["candidates"] == {"PolarForecasterV2": False, "DirectCalendarForecasterV1": True}
    assert not summary["all_passed"] and summary["all_runs_attempted"]
    saved = json.loads((output / "results.json").read_text())
    assert len(saved["runs"]) == 3
    assert saved["runs"][1]["execution_status"] == "numerical_failure"
    assert (output / "PolarForecasterV2_42_initial_state.pt").exists()
    assert (output / "PolarForecasterV2_42_failed_state.pt").exists()
    assert not (output / "PolarForecasterV2_42_state.pt").exists()
    assert (output / "DirectCalendarForecasterV1_42_predictions.npz").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["validation_status"] == "failed"
    assert "src\\methodology_forecasters.py" in manifest["snapshot_sha256"] or "src/methodology_forecasters.py" in manifest["snapshot_sha256"]


def test_no_overwrite_or_budget_reduction(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run_validation(tmp_path)
    with pytest.raises(ValueError, match="exactly 50"):
        runner.run_validation(tmp_path / "new", 10)
    assert not (tmp_path / "new").exists()


def test_cli_exit_status_reflects_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_validation", lambda *args: {"all_passed": False, "candidates": {}})
    assert runner.main(["--output-dir", str(tmp_path / "new")]) == 1
