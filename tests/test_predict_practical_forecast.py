"""Offline artifact reconstruction and causal CSV history contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import predict_practical_forecast as inference


def write(path, value):
    """Write fixture metadata without invoking any fitting code."""
    path.write_text(json.dumps(value), encoding="utf-8")


def final_fixture(tmp_path, family="Persistence"):
    """Construct consistent completed final artifacts with known train scales."""
    run = tmp_path / "final"
    fold = run / inference.FINAL_START
    fold.mkdir(parents=True)
    meta = {"lookback": 168, "horizon": 24, "time_periods": [24, 168],
            "target_col": inference.TARGET, "target_idx": 1,
            "sensor_columns": ["sensor_A", inference.TARGET],
            "norm_stats": {"sensor_A": [10., 2.], inference.TARGET: [3., 4.]}}
    manifest = {"status": "completed", "phase": "final", "all_fits_finite": True,
                "protocol": "practical_forecasting_v1", "lookback": 168, "horizon": 24,
                "target": inference.TARGET, "unit": "kW", "coverage": .75,
                "test_starts": [inference.FINAL_START], "families": [family],
                "selection": {"selected_family": family}, "seeds": [42, 43, 44],
                "implementation_sha256": {name: inference.sha256(inference.ROOT / name)
                                           for name in inference.IMPLEMENTATION_PATHS}}
    rows = []
    if family in inference.NEURAL:
        rows = [{"family": family, "seed": seed, "status": "completed"} for seed in [42, 43, 44]]
        for seed, bias in zip((42, 43, 44), (.1, .2, .3)):
            model = inference.make_neural(family, meta)
            with torch.no_grad():
                model.body.output_head.bias.fill_(bias)
            torch.save(model.state_dict(), fold / f"{family}_{seed}_state.pt")
    elif family == "RidgeResidual":
        rows = [{"family": family, "status": "completed", "training": {
            "persistence_selected": False, "selected_candidate": "RidgeResidual",
            "n_features": 168 * 2 + 4}}]
        coefficients = np.zeros((24, 168 * 2 + 4))
        coefficients[:, 0] = 2.  # Twice the first sensor's first centered lag.
        coefficients[:, -4] = 3.  # Three times sine of the origin's daily angle.
        np.savez(fold / "RidgeResidual_coefficients.npz", coefficients=coefficients,
                 intercept=np.arange(24, dtype=float) / 100)
    write(run / "manifest.json", manifest)
    write(fold / "data_manifest.json", meta)
    write(fold / "results.json", {"status": "completed", "all_fits_finite": True,
                                  "families": {family: {}}, "runs": rows})
    write(run / "final_assessment.json", {"selected_family": family, "selected_metrics": {},
                                         "learned_5pct_improvement_confirmed": False})
    # Deliberately reverse CSV sensor order; the archive defines tensor order.
    history = pd.DataFrame({"timestamp": pd.date_range("2012-01-01", periods=170, freq="1h"),
                            inference.TARGET: np.arange(170) / 10 + 3,
                            "sensor_A": np.arange(170) + 10.})
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)
    return run, path, meta, history


@pytest.mark.parametrize("family", inference.SIMPLE)
def test_simple_artifact_normalization_and_causal_timestamp_roundtrip(tmp_path, family):
    run, csv_path, meta, history = final_fixture(tmp_path, family)
    path = inference.run_prediction(run, csv_path, tmp_path / "prediction", device="cpu")
    prediction = pd.read_csv(path)
    target = history[inference.TARGET].to_numpy()[-168:]
    expected = {"Persistence": np.repeat(target[-1], 24),
                "Seasonal24": target[-24:], "Seasonal168": target[:24]}[family]
    np.testing.assert_allclose(prediction["Global_active_power_kW"], expected, atol=2e-6)
    times = pd.to_datetime(prediction["timestamp"])
    assert times.iloc[0] == history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    assert times.iloc[-1] == history["timestamp"].iloc[-1] + pd.Timedelta(hours=24)
    provenance = json.loads((path.parent / "provenance.json").read_text())
    assert provenance["sensor_columns"] == meta["sensor_columns"]
    assert provenance["input_sha256"] == inference.sha256(csv_path)
    assert "not independently checked" in provenance["coverage"]
    assert provenance["selected_family"] == family
    assert provenance["clipping"] == "none"
    with pytest.raises(FileExistsError):
        inference.run_prediction(run, csv_path, path.parent)


def test_ridge_coefficients_reconstruct_centered_lags_and_origin_angles(tmp_path):
    run, path, meta, history = final_fixture(tmp_path, "RidgeResidual")
    x, times = inference.prepare_history(path, meta)
    bundle = inference.load_final_run(run)
    prediction = inference.predict_selected(run, bundle, x, device="cpu")
    # Compute independently from the declared trained feature layout.
    centered = float(x[0, 0, 0]) - float(x[0, -1, 0])
    angle = float(x[0, -1, 2])
    residual = 2 * centered + 3 * np.sin(angle) + np.arange(24) / 100
    expected = (float(x[0, -1, 1]) + residual) * 4 + 3
    np.testing.assert_allclose(prediction, expected, atol=1e-10)
    assert f"{inference.FINAL_START}/RidgeResidual_coefficients.npz" in bundle["artifact_sha256"]


def test_ridge_explicit_persistence_fallback_needs_no_coefficient_file(tmp_path):
    run, path, meta, history = final_fixture(tmp_path, "RidgeResidual")
    results_path = run / inference.FINAL_START / "results.json"
    report = json.loads(results_path.read_text())
    report["runs"][0]["training"].update(persistence_selected=True, selected_candidate="Persistence")
    write(results_path, report)
    (run / inference.FINAL_START / "RidgeResidual_coefficients.npz").unlink()
    bundle = inference.load_final_run(run)
    x, _ = inference.prepare_history(path, meta)
    np.testing.assert_allclose(inference.predict_selected(run, bundle, x), history[inference.TARGET].iloc[-1], atol=2e-6)


@pytest.mark.parametrize("family", inference.NEURAL)
def test_neural_three_checkpoint_ensemble_roundtrip(tmp_path, family):
    run, path, meta, history = final_fixture(tmp_path, family)
    bundle = inference.load_final_run(run)
    x, _ = inference.prepare_history(path, meta)
    actual = inference.predict_selected(run, bundle, x, device="cpu")
    expected = history[inference.TARGET].iloc[-1] + .2 * meta["norm_stats"][inference.TARGET][1]
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert sum(name.endswith("_state.pt") for name in bundle["artifact_sha256"]) == 3


@pytest.mark.parametrize("defect", ["short", "gap", "duplicate", "descending", "half_hour",
                                  "missing_sensor", "nan", "infinity", "timezone"])
def test_rejects_invalid_history_before_forecasting(tmp_path, defect):
    run, path, meta, frame = final_fixture(tmp_path)
    if defect == "short":
        frame = frame.iloc[:167]
    elif defect == "gap":
        frame = frame.drop(index=50)
    elif defect == "duplicate":
        frame.loc[50, "timestamp"] = frame.loc[49, "timestamp"]
    elif defect == "descending":
        frame = frame.iloc[::-1]
    elif defect == "half_hour":
        frame["timestamp"] += pd.Timedelta(minutes=30)
    elif defect == "missing_sensor":
        frame = frame.drop(columns="sensor_A")
    elif defect in ("nan", "infinity"):
        frame.loc[50, "sensor_A"] = np.nan if defect == "nan" else np.inf
    elif defect == "timezone":
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        inference.prepare_history(path, meta)


@pytest.mark.parametrize("defect", ["not_final", "incomplete", "failed_fit", "wrong_family", "changed_source"])
def test_rejects_invalid_final_run(tmp_path, defect):
    run, path, meta, history = final_fixture(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if defect == "not_final":
        manifest["phase"] = "development"
    elif defect == "incomplete":
        manifest["status"] = "running"
    elif defect == "failed_fit":
        manifest["all_fits_finite"] = False
    elif defect == "wrong_family":
        manifest["families"] = ["ResidualLSTM"]
    else:
        manifest["implementation_sha256"]["src/forecasting.py"] = "changed"
    write(manifest_path, manifest)
    with pytest.raises(ValueError):
        inference.load_final_run(run)


def test_neural_missing_seed_never_forms_survivor_ensemble(tmp_path):
    run, path, meta, history = final_fixture(tmp_path, "ResidualLSTM")
    report_path = run / inference.FINAL_START / "results.json"
    report = json.loads(report_path.read_text())
    report["runs"].pop()
    write(report_path, report)
    with pytest.raises(ValueError, match="three completed"):
        inference.load_final_run(run)


def test_normalization_is_stored_and_no_angle_normalization_occurs(tmp_path):
    run, path, meta, history = final_fixture(tmp_path)
    x, times = inference.prepare_history(path, meta)
    np.testing.assert_allclose(x[0, :, 0], (history["sensor_A"].to_numpy()[-168:] - 10) / 2)
    np.testing.assert_allclose(x[0, :, 1], (history[inference.TARGET].to_numpy()[-168:] - 3) / 4)
    np.testing.assert_array_equal(x[0, :, 2:], inference.extract_angular_time_features(times, [24, 168]).astype(np.float32))
    meta["norm_stats"]["sensor_A"][1] = 0
    with pytest.raises(ValueError, match="positive scales"):
        inference.prepare_history(path, meta)
