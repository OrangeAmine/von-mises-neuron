"""Synthetic readiness gates, failure retention, and candidate reconstruction."""

import json

import pytest
import torch

from scripts import validate_residual_forecasters as runner


def passing_rows():
    """Give every required run finite metrics inside the standard gates."""
    return [{"model": name, "execution_status": "completed",
             "metrics": {"rmse": 10., "mae": 8., "r2": 0.9, "mape": 2.}}
            for name in runner.MODELS]


def test_all_required_models_pass():
    validation = runner.validation_summary(passing_rows())
    assert validation["all_passed"]
    assert validation["all_runs_attempted"]
    assert all(validation["candidates"].values())


@pytest.mark.parametrize("model", runner.CANDIDATES)
def test_absolute_failure_revokes_only_affected_candidate(model):
    rows = passing_rows()
    next(row for row in rows if row["model"] == model)["metrics"]["rmse"] = 36.
    validation = runner.validation_summary(rows)
    assert not validation["all_passed"]
    assert not validation["candidates"][model]
    assert all(passed for name, passed in validation["candidates"].items() if name != model)


def test_temporal_regression_is_required_even_inside_absolute_limits():
    rows = passing_rows()
    rows[-1]["metrics"]["rmse"] = 11.01
    result = runner.validation_summary(rows)
    assert result["acceptance"]["ResidualTemporal"]["absolute_passed"]
    assert not result["candidates"]["ResidualTemporal"]
    assert not result["all_passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_nonfinite_metric_cannot_pass(value):
    rows = passing_rows()
    rows[1]["metrics"]["rmse"] = value
    assert not runner.validation_summary(rows)["candidates"]["ResidualLSTM"]


def test_failed_or_missing_base_prevents_temporal_certification():
    rows = passing_rows()
    rows[0]["execution_status"] = "failed"
    assert not runner.validation_summary(rows)["candidates"]["ResidualTemporal"]
    result = runner.validation_summary(rows[1:])
    assert not result["all_runs_attempted"]
    assert not result["candidates"]["ResidualTemporal"]
    assert result["candidates"]["ResidualSinCos"]


@pytest.mark.parametrize("name", runner.CANDIDATES)
def test_candidate_configuration_reconstructs_persistence(name):
    model = runner.build_model(name).eval()
    x = torch.randn(2, 72, 6)
    torch.testing.assert_close(model(x), x[:, -1, 2:3].expand(-1, 12), rtol=0, atol=0)
    assert model.sensor_size == 4 and model.target_index == 2
    assert model.body.hidden_size == (68 if name == "ResidualLSTM" else 64)
    assert runner.model_config(name)["body"]["dropout"] == 0.1


def test_mocked_run_retains_numerical_failure_and_other_candidates(tmp_path, monkeypatch):
    """A failed fit is archived and cannot be concealed by later successes."""
    monkeypatch.setattr(runner, "source_manifest", lambda: {"revision": "test"})
    def fake_fit(name, model, data, epochs, device):
        model.training_history = {"validation_loss": [1.], "best_val_loss": 1., "best_epoch": 0}
        if name == "ResidualLSTM":
            model.training_history["status"] = "failed"
            raise FloatingPointError("injected nonfinite gradient")
        return model
    monkeypatch.setattr(runner, "_fit", fake_fit)
    monkeypatch.setattr(runner, "batched_predict", lambda model, x: torch.zeros(len(x), 12))
    monkeypatch.setattr(runner, "compute_benchmark_metrics", lambda *args: passing_rows()[0]["metrics"])
    output = tmp_path / "run"
    result = runner.run_validation(output)
    assert not result["all_passed"]
    assert result["candidates"] == {"ResidualLSTM": False, "ResidualSinCos": True,
                                    "ResidualTemporal": True}
    saved = json.loads((output / "results.json").read_text())
    failed = next(row for row in saved["runs"] if row["model"] == "ResidualLSTM")
    assert failed["execution_status"] == "numerical_failure"
    assert "injected" in failed["failures"][0]
    assert (output / "ResidualLSTM_42_failed_state.pt").exists()
    assert not (output / "ResidualLSTM_42_state.pt").exists()
    assert (output / "ResidualSinCos_42_state.pt").exists()
    assert (output / "ResidualSinCos_42_predictions.npz").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["calendar_clock"] == "local"
    assert manifest["validation_status"] == "failed"
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 42 and manifest["data_seed"] == 42


def test_refuses_existing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run_validation(tmp_path)


def test_main_exit_status_reflects_aggregate_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_validation", lambda *args: {
        "all_passed": False, "candidates": {"ResidualLSTM": False}})
    assert runner.main(["--output-dir", str(tmp_path / "new")]) == 1


def test_nonpositive_epoch_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        runner.run_validation(tmp_path / "new", epochs=0)
    assert not (tmp_path / "new").exists()
