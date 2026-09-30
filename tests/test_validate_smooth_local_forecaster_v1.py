"""Standard gates, fixed budgets and retained failure evidence for VM-N01."""

import json

import pytest
import torch

from scripts import validate_smooth_local_forecaster_v1 as runner


def passing_rows():
    return [{"model": name, "execution_status": "completed", "metrics": {
        "rmse": 10.0, "mae": 8.0, "r2": 0.9, "mape": 2.0}} for name in runner.MODELS]


def test_all_declared_absolute_gates_pass():
    summary = runner.validation_summary(passing_rows())
    assert summary["all_passed"] and summary["all_runs_attempted"]
    assert summary["candidates"] == {"SmoothLocalForecasterV1": True}


@pytest.mark.parametrize("name", runner.MODELS)
@pytest.mark.parametrize("metric,value", [("rmse", 35.01), ("mae", 25.01),
    ("r2", 0.199), ("mape", 5.001)])
def test_each_unchanged_absolute_threshold_is_required(name, metric, value):
    rows = passing_rows()
    next(row for row in rows if row["model"] == name)["metrics"][metric] = value
    summary = runner.validation_summary(rows)
    assert not summary["acceptance"][name]["passed"] and not summary["all_passed"]


@pytest.mark.parametrize("status", ["pending", "running", "failed", "numerical_failure", "interrupted"])
def test_incomplete_and_failed_runs_cannot_pass(status):
    rows = passing_rows()
    rows[-1]["execution_status"] = status
    summary = runner.validation_summary(rows)
    assert not summary["candidates"]["SmoothLocalForecasterV1"]
    assert summary["all_runs_attempted"] == (status not in ("pending", "running"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_missing_nonfinite_metrics_cannot_pass(value):
    rows = passing_rows()
    rows[-1]["metrics"]["rmse"] = value
    assert not runner.validation_summary(rows)["all_passed"]


def test_unknown_duplicate_missing_rows_rejected_or_fail():
    with pytest.raises(ValueError):
        runner.validation_summary(passing_rows() + passing_rows()[:1])
    with pytest.raises(ValueError):
        runner.validation_summary([{"model": "undeclared"}])
    assert not runner.validation_summary(passing_rows()[:1])["all_passed"]


def _mock_fit_environment(monkeypatch):
    monkeypatch.setattr(runner, "source_manifest", lambda: {"revision": "test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "predict", lambda model, data, device: torch.zeros_like(data["y_test"]))
    monkeypatch.setattr(runner, "compute_benchmark_metrics", lambda *args: passing_rows()[0]["metrics"])


def test_mock_failure_retains_states_planned_rows_and_data(tmp_path, monkeypatch):
    _mock_fit_environment(monkeypatch)

    def fake_fit(model, data, device):
        model.training_history = {"validation_loss": [1.0], "best_val_loss": 1.0, "best_epoch": 1}
        if isinstance(model, runner.SmoothLocalForecasterV1):
            raise FloatingPointError("injected failure")
        return model

    monkeypatch.setattr(runner, "_fit", fake_fit)
    output = tmp_path / "new"
    summary = runner.run_validation(output)
    assert not summary["all_passed"] and summary["all_runs_attempted"]
    saved = json.loads((output / "results.json").read_text())
    assert len(saved["runs"]) == 2
    assert saved["runs"][-1]["execution_status"] == "numerical_failure"
    assert (output / "SmoothLocalForecasterV1_42_initial_state.pt").exists()
    assert (output / "SmoothLocalForecasterV1_42_failed_state.pt").exists()
    assert not (output / "SmoothLocalForecasterV1_42_state.pt").exists()
    assert (output / "VonMises_42_predictions.npz").exists()
    assert (output / "benchmark_data.npz").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["validation_status"] == "failed"
    assert "src/smooth_local_forecaster_v1.py" in manifest["snapshot_sha256"]


def test_interruption_preserves_partial_attempt_and_pending_candidate(tmp_path, monkeypatch):
    _mock_fit_environment(monkeypatch)

    def interrupt(*args):
        raise KeyboardInterrupt("injected interruption")

    monkeypatch.setattr(runner, "_fit", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.run_validation(output)
    saved = json.loads((output / "results.json").read_text())
    assert [row["execution_status"] for row in saved["runs"]] == ["interrupted", "pending"]
    assert not saved["all_passed"] and not saved["all_runs_attempted"]
    assert (output / "VonMises_42_failed_state.pt").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_no_overwrite_or_budget_reduction(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run_validation(tmp_path)
    with pytest.raises(ValueError, match="exactly 50"):
        runner.run_validation(tmp_path / "new", 10)
    assert not (tmp_path / "new").exists()


def test_fit_invokes_unchanged_training_defaults(monkeypatch):
    received = {}

    def fake_train(model, *tensors, **kwargs):
        received.update(kwargs=kwargs, tensors=tensors)
        return model

    monkeypatch.setattr(runner, "_train_model", fake_train)
    model = torch.nn.Linear(2, 1)
    data = {key: torch.zeros(2, 3) for key in ("X_train", "y_train", "X_val", "y_val")}
    assert runner._fit(model, data, torch.device("cpu")) is model
    assert received["kwargs"] == {"max_epochs": 50}
    assert len(received["tensors"]) == 4


def test_cli_exit_status_reflects_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_validation", lambda *args: {"all_passed": False, "candidates": {}})
    assert runner.main(["--output-dir", str(tmp_path / "new")]) == 1
