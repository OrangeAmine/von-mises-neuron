"""Protocol and runner tests using small CPU inputs and mocked model fitting."""

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.practical_forecasting as runner


def mock_splits():
    """Return distinct split tensors with actual 168/24 forecast dimensions."""
    splits = {}
    for i, name in enumerate(("train", "val", "test")):
        x = np.zeros((2, 168, 9), dtype=np.float32)
        x[..., 0] = np.arange(168) / 100 + i
        y = np.full((2, 24), i + 2, dtype=np.float32)
        origins = np.array(["2008-02-01", "2008-02-02"], dtype="datetime64[ns]")
        splits[name] = {"X": x, "y": y, "y_raw": y.astype(float) * 3 + 2,
                        "origins": origins,
                        "target_times": origins[:, None] + np.arange(1, 25).astype("timedelta64[h]")}
    meta = {"sensor_columns": [runner.TARGET, *[f"sensor{i}" for i in range(6)]],
            "target_idx": 0, "norm_stats": {runner.TARGET: [2, 3]}}
    return splits, meta


@pytest.mark.parametrize("start", [*runner.DEVELOPMENT_STARTS, runner.FINAL_START])
def test_fold_boundaries_match_frozen_protocol(start):
    boundaries = runner.fold_boundaries(start)
    assert boundaries["test_start"] == pd.Timestamp(start)
    assert boundaries["test_end"] - boundaries["test_start"] == pd.Timedelta(days=28)
    assert boundaries["test_start"] - boundaries["val_start"] == pd.Timedelta(days=28)
    assert boundaries["val_start"] - boundaries["train_start"] == pd.Timedelta(days=180)
    if start == runner.FINAL_START:
        assert boundaries["test_end"] == pd.Timestamp("2010-11-26")
    else:
        assert boundaries["test_end"] < pd.Timestamp(runner.FINAL_START)


@pytest.mark.parametrize("family", runner.NEURAL)
def test_model_input_contract_and_persistence_initialization(family):
    _, metadata = mock_splits()
    model = runner.make_neural(family, metadata).eval()
    x = torch.randn(2, 8, 9)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.shape == (2, 24)
    torch.testing.assert_close(prediction, x[:, -1, 0, None].expand(-1, 24), rtol=0, atol=0)
    assert model.sensor_size == 7
    assert model.input_size == 9


def test_reference_and_ensemble_prediction_archives_are_aligned(monkeypatch, tmp_path):
    splits, metadata = mock_splits()
    monkeypatch.setattr(runner, "prepare_forecast_windows", lambda *a, **kw: (splits, metadata))
    monkeypatch.setattr(runner, "seed_everything", lambda seed: None)
    monkeypatch.setattr(runner, "make_neural", lambda *a: torch.nn.Linear(1, 1))
    fits = []

    def train(model, x_train, y_train, x_val, y_val, **kwargs):
        assert torch.equal(x_train, torch.from_numpy(splits["train"]["X"]))
        assert torch.equal(y_val, torch.from_numpy(splits["val"]["y"]))
        assert torch.equal(x_val, torch.from_numpy(splits["val"]["X"]))
        fits.append(kwargs["source_config"]["seed"])
        model.forecast_value = len(fits)
        model.training_history = {"best_epoch": 0}
        return model

    monkeypatch.setattr(runner, "train_forecaster", train)
    monkeypatch.setattr(runner, "batched_predict", lambda model, x: torch.full((len(x), 24), model.forecast_value))
    directory = tmp_path / "fold"
    report = runner.evaluate_fold(pd.DataFrame(), "2008-02-01", directory,
                                  ["ResidualLSTM"], [42, 43, 44], 50)
    assert fits == [42, 43, 44]
    assert report["status"] == "completed"
    assert report["all_fits_finite"] is True
    assert set(report["families"]) == {*runner.SIMPLE, "ResidualLSTM"}
    paths = [directory / f"{name}.npz" for name in runner.SIMPLE]
    paths += [directory / "ResidualLSTM_ensemble.npz"]
    for path in paths:
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["target"], splits["test"]["y_raw"])
            np.testing.assert_array_equal(saved["origins"], splits["test"]["origins"])
            np.testing.assert_array_equal(saved["target_times"], splits["test"]["target_times"])
    with np.load(directory / "ResidualLSTM_ensemble.npz") as saved:
        np.testing.assert_array_equal(saved["prediction"], np.full((2, 24), 8))


def test_numerical_failure_excludes_entire_seed_family(monkeypatch, tmp_path):
    splits, metadata = mock_splits()
    monkeypatch.setattr(runner, "prepare_forecast_windows", lambda *a, **kw: (splits, metadata))
    monkeypatch.setattr(runner, "seed_everything", lambda seed: None)
    monkeypatch.setattr(runner, "make_neural", lambda *a: torch.nn.Linear(1, 1))

    def train(model, *a, source_config, **kwargs):
        model.training_history = {"best_epoch": 0}
        if source_config["seed"] == 43:
            raise FloatingPointError("Mock nonfinite gradient")
        return model

    monkeypatch.setattr(runner, "train_forecaster", train)
    monkeypatch.setattr(runner, "batched_predict", lambda model, x: torch.ones(len(x), 24))
    directory = tmp_path / "fold"
    report = runner.evaluate_fold(pd.DataFrame(), "2008-02-01", directory,
                                  ["ResidualLSTM"], [42, 43, 44], 50)
    assert "ResidualLSTM" not in report["families"]
    assert report["all_fits_finite"] is False
    assert [row["status"] for row in report["runs"]] == ["completed", "numerical_failure", "completed"]
    assert not (directory / "ResidualLSTM_ensemble.npz").exists()
    saved = json.loads((directory / "results.json").read_text())
    assert "Mock nonfinite gradient" in saved["runs"][1]["error"]


def test_prepare_only_never_scores_or_trains(monkeypatch, tmp_path):
    splits, metadata = mock_splits()
    monkeypatch.setattr(runner, "prepare_forecast_windows", lambda *a, **kw: (splits, metadata))

    def prohibited(*args, **kwargs):
        pytest.fail("Preparation must not inspect forecast scores or fit models")

    for function in ["forecast_metrics", "simple_predictions", "fit_ridge_residual", "train_forecaster"]:
        monkeypatch.setattr(runner, function, prohibited)
    report = runner.evaluate_fold(pd.DataFrame(), "2008-02-01", tmp_path / "fold",
                                  ["RidgeResidual", *runner.NEURAL], [42, 43, 44], 50,
                                  prepare_only=True)
    assert report["status"] == "prepared"
    assert report["runs"] == []
    assert report["families"] == {}


@pytest.mark.parametrize("arguments", [[], ["--phase", "final"], ["--epochs", "0"], ["--seeds", "42", "42"]])
def test_cli_rejects_missing_evidence_and_invalid_settings(monkeypatch, tmp_path, arguments):
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["practical_forecasting", "--output-dir", str(output), *arguments])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_final_rejects_changed_training_settings_before_fitting(monkeypatch, tmp_path):
    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "manifest.json").write_text(json.dumps({"status": "completed", "phase": "development",
                                                        "epochs": 49, "seeds": [42, 43, 44]}))
    (selection / "selection.json").write_text(json.dumps({"selected_family": "Persistence"}))
    root = tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    (root / "docs/practical_forecasting_protocol.md").write_text("Frozen test protocol")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "IMPLEMENTATION_PATHS", [])
    monkeypatch.setattr(runner, "source_manifest", lambda: {})
    monkeypatch.setattr(runner, "load_cached_hourly", lambda *a, **kw: (pd.DataFrame(), {}))

    def prohibited(*args, **kwargs):
        pytest.fail("Final settings mismatch must fail before any fold evaluation")

    monkeypatch.setattr(runner, "evaluate_fold", prohibited)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["practical_forecasting", "--output-dir", str(output),
                                     "--phase", "final", "--selection-dir", str(selection)])
    with pytest.raises(ValueError, match="settings"):
        runner.main()
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "failed"


def isolated_final_cli(monkeypatch, tmp_path):
    """Create self-consistent development evidence for final integrity checks."""
    root = tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    protocol = b"Frozen protocol fixture\n"
    (root / "docs/practical_forecasting_protocol.md").write_bytes(protocol)
    (root / "src").mkdir()
    implementation = b"# implementation fixture\n"
    (root / "src/fixture.py").write_bytes(implementation)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "IMPLEMENTATION_PATHS", ["src/fixture.py"])
    source = {"raw_files": [{"sha256": "cached-byte-fingerprint"}]}
    environment = {"packages": {"numpy": "fixture-version", "torch": "fixture-version"}}
    monkeypatch.setattr(runner, "source_manifest", lambda: environment)
    monkeypatch.setattr(runner, "load_cached_hourly", lambda *a, **kw: (pd.DataFrame(), source))
    directory = tmp_path / "development"
    directory.mkdir()
    previous = {
        "status": "completed", "phase": "development", "epochs": 50, "seeds": [42, 43, 44],
        "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
        "implementation_sha256": {"src/fixture.py": hashlib.sha256(implementation).hexdigest()},
        "data": source, "environment": environment,
    }
    (directory / "manifest.json").write_text(json.dumps(previous))
    metrics = {
        family: runner.forecast_metrics(np.full((2, 24), 0.5 if family == "RidgeResidual" else 0.),
                                         np.ones((2, 24)))
        for family in [*runner.SIMPLE, "RidgeResidual"]
    }
    for start in runner.DEVELOPMENT_STARTS:
        fold = directory / start
        fold.mkdir()
        (fold / "results.json").write_text(json.dumps({"status": "completed", "families": metrics}))
    selection = runner.select_candidate({start: metrics for start in runner.DEVELOPMENT_STARTS},
                                         family_order=[*runner.SIMPLE, "RidgeResidual", *runner.NEURAL])
    assert selection["selected_family"] == "RidgeResidual"
    (directory / "selection.json").write_text(json.dumps(selection))
    output = tmp_path / "final"
    monkeypatch.setattr(sys, "argv", ["practical_forecasting", "--output-dir", str(output),
                                     "--phase", "final", "--selection-dir", str(directory)])
    return directory, output, metrics


@pytest.mark.parametrize("mismatch,match", [
    ("protocol_sha256", "protocol_sha256"),
    ("implementation_sha256", "implementation_sha256"),
    ("data", "data"),
    ("packages", "package versions"),
    ("selection", "selection rule"),
])
def test_final_rejects_changed_evidence_before_scoring(monkeypatch, tmp_path, mismatch, match):
    directory, output, _ = isolated_final_cli(monkeypatch, tmp_path)
    if mismatch == "selection":
        path = directory / "selection.json"
        document = json.loads(path.read_text())
        document["selected_family"] = "ResidualLSTM"
    else:
        path = directory / "manifest.json"
        document = json.loads(path.read_text())
        if mismatch == "packages":
            document["environment"]["packages"]["numpy"] = "different-version"
        elif mismatch == "implementation_sha256":
            document[mismatch]["src/fixture.py"] = "different-implementation"
        elif mismatch == "data":
            document[mismatch]["raw_files"][0]["sha256"] = "different-data"
        else:
            document[mismatch] = "different-protocol"
    path.write_text(json.dumps(document))

    def prohibited(*args, **kwargs):
        pytest.fail("Changed final evidence must be rejected before fitting or scoring")

    monkeypatch.setattr(runner, "evaluate_fold", prohibited)
    with pytest.raises(ValueError, match=match):
        runner.main()
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "failed"
    assert not (output / "final_assessment.json").exists()


def test_valid_final_evaluates_only_selected_family_at_reserved_start(monkeypatch, tmp_path):
    _, output, metrics = isolated_final_cli(monkeypatch, tmp_path)
    calls = []

    def evaluate(frame, start, directory, families, seeds, epochs, **kwargs):
        calls.append((start, families, seeds, epochs))
        return {"status": "completed", "all_fits_finite": True, "families": metrics}

    monkeypatch.setattr(runner, "evaluate_fold", evaluate)
    runner.main()
    assert calls == [(runner.FINAL_START, ["RidgeResidual"], [42, 43, 44], 50)]
    assessment = json.loads((output / "final_assessment.json").read_text())
    assert assessment["selected_family"] == "RidgeResidual"
    assert assessment["learned_5pct_improvement_confirmed"] is True
    assert not (output / "selection.json").exists()


@pytest.mark.parametrize("arguments", [["--epochs", "49"], ["--epochs", "51"],
                                        ["--seeds", "42"], ["--seeds", "43", "42", "44"],
                                        ["--seeds", "42", "43", "45"]])
def test_frozen_protocol_rejects_custom_budget_or_seed_set(monkeypatch, tmp_path, capsys, arguments):
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["practical_forecasting", "--output-dir", str(output),
                                     "--prepare-only", *arguments])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert "Frozen v1 protocol" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("failure_stage", ["fit", "prediction"])
def test_ridge_numerical_failure_is_recorded_and_excluded(monkeypatch, tmp_path, failure_stage):
    splits, metadata = mock_splits()
    monkeypatch.setattr(runner, "prepare_forecast_windows", lambda *a, **kw: (splits, metadata))

    class FailedRidge:
        def predict(self, x):
            raise FloatingPointError("Mock nonfinite ridge prediction")

    def fit(*args, **kwargs):
        if failure_stage == "fit":
            raise FloatingPointError("Mock nonfinite ridge fit")
        return FailedRidge(), {"selected_alpha": 1.0}

    monkeypatch.setattr(runner, "fit_ridge_residual", fit)
    directory = tmp_path / "fold"
    report = runner.evaluate_fold(pd.DataFrame(), "2008-02-01", directory,
                                  ["RidgeResidual"], [42, 43, 44], 50)
    assert set(report["families"]) == set(runner.SIMPLE)
    assert report["status"] == "completed"
    assert report["all_fits_finite"] is False
    assert len(report["runs"]) == 1
    row = report["runs"][0]
    assert row["family"] == "RidgeResidual"
    assert row["status"] == "numerical_failure"
    assert "Mock nonfinite ridge" in row["error"]
    assert not (directory / "RidgeResidual.npz").exists()
    saved = json.loads((directory / "results.json").read_text())
    assert saved["runs"][0] == row
