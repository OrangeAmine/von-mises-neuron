"""Occupancy comparison tests without downloading data or training models."""

import numpy as np
import pandas as pd
import pytest
import torch
import json
import sys

import scripts.controlled_real_comparison as runner

from scripts.controlled_real_comparison import (
    checked_metrics, contiguous_sequences, prepare_occupancy_data,
    references, save_predictions, summarize_real,
)


def occupancy_frame(rows: int = 300) -> pd.DataFrame:
    """Build finite hourly observations whose target makes alignment visible."""
    index = pd.date_range("2024-01-01", periods=rows, freq="1h")
    return pd.DataFrame({"Temperature": np.arange(rows) / 100 + 20,
                         "CO2": np.arange(rows, dtype=float) + 400}, index=index)


@pytest.mark.parametrize("gap_position", [5, 24, 28])
def test_rejects_gap_anywhere_in_input_and_forecast(gap_position):
    frame = occupancy_frame(31)
    frame = frame.drop(frame.index[gap_position])
    x, y, origins, counts = contiguous_sequences(frame)
    assert counts == {"candidate_windows": 1, "finite_windows": 1,
                      "valid_windows": 0, "gap_rejected_finite_windows": 1}
    assert x.shape == (0, 24, 2)
    assert y.shape == (0, 6)
    assert origins.size == 0


def test_retains_unaffected_windows_after_gap():
    frame = occupancy_frame(70).drop(pd.Timestamp("2024-01-02 06:00"))
    x, y, origins, counts = contiguous_sequences(frame)
    assert counts["candidate_windows"] == 40
    assert counts["gap_rejected_finite_windows"] == 29
    assert counts["valid_windows"] == 11
    for sample, origin in enumerate(origins):
        all_times = frame.index[origin - 23:origin + 7]
        assert (all_times[1:] - all_times[:-1] == pd.Timedelta(hours=1)).all()
        np.testing.assert_array_equal(y[sample], frame["CO2"].iloc[origin + 1:origin + 7])


def test_rejects_nonfinite_windows():
    frame = occupancy_frame(31)
    frame.iloc[5, 0] = np.inf
    _, _, _, counts = contiguous_sequences(frame)
    assert counts["finite_windows"] == 0
    assert counts["valid_windows"] == 0


def test_train_only_normalization_and_identical_forecast_origins():
    frame = occupancy_frame()
    shifted = frame.copy()
    shifted.iloc[210:] += 10000
    original, metadata = prepare_occupancy_data(frame, torch.device("cpu"))
    changed, _ = prepare_occupancy_data(shifted, torch.device("cpu"))
    assert original["sensor"]["norm_stats"] == changed["sensor"]["norm_stats"]
    torch.testing.assert_close(original["sensor"]["X_train"], changed["sensor"]["X_train"])
    assert [metadata[s]["rows"] for s in ("train", "val", "test")] == [210, 45, 45]
    for split in ("train", "val", "test"):
        base = original["sensor"]
        for mode in ("temporal", "handcrafted"):
            other = original[mode]
            torch.testing.assert_close(other[f"X_{split}"][..., :2], base[f"X_{split}"])
            torch.testing.assert_close(other[f"y_{split}"], base[f"y_{split}"])
            np.testing.assert_array_equal(other[f"origins_{split}"], base[f"origins_{split}"])
            np.testing.assert_array_equal(other[f"target_times_{split}"], base[f"target_times_{split}"])
        angle = original["temporal"][f"X_{split}"][..., -1]
        assert (angle >= 0).all() and (angle < 2 * np.pi).all()
        torch.testing.assert_close(original["handcrafted"][f"X_{split}"][..., -2], angle.sin(), atol=1e-6, rtol=1e-6)


def test_reference_alignment_and_prediction_export(tmp_path):
    datasets, _ = prepare_occupancy_data(occupancy_frame(), torch.device("cpu"))
    data = datasets["sensor"]
    predictions = references(data)
    mean, std = data["norm_stats"]["CO2"]
    target = data["y_test"] * std + mean
    seasonal = predictions["SeasonalPersistence24h"] * std + mean
    torch.testing.assert_close(seasonal, target - 24)
    persistence = predictions["Persistence"] * std + mean
    torch.testing.assert_close(persistence, target[:, :1].expand_as(target) - 1)
    torch.testing.assert_close(predictions["TrainMean"] * std + mean, torch.full_like(target, mean))
    path = tmp_path / "prediction.npz"
    save_predictions(path, predictions["Persistence"], data)
    with np.load(path) as artifact:
        np.testing.assert_allclose(artifact["prediction_ppm"], persistence.numpy())
        np.testing.assert_array_equal(artifact["target_bin_start"][:, 0],
                                      artifact["origin_bin_start"] + np.timedelta64(1, "h"))


def test_real_metrics_have_no_synthetic_success_gate():
    data = {"y_test": torch.tensor([[0., 1.], [2., 3.]]), "norm_stats": {"CO2": (400., 100.)}}
    metrics = checked_metrics(torch.full((2, 2), 20.), data)
    assert metrics["rmse"] > 35
    assert metrics["r2"] < 0
    assert set(metrics) == {"rmse", "mae", "r2", "mape"}
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        checked_metrics(torch.full((2, 2), float("nan")), data)


def test_seed_summary_uses_sample_standard_deviation():
    rows = [{"model": "VM", "seed": seed, "params": 5,
             "metrics": {metric: float(seed) for metric in ("rmse", "mae", "r2", "mape")}}
            for seed in (1, 3)]
    summary = summarize_real(rows)["VM"]
    assert summary["n_seeds"] == 2
    assert summary["rmse"]["mean"] == 2
    assert summary["rmse"]["std"] == pytest.approx(np.sqrt(2))
    assert "absolute_passed_all" not in summary


def test_missing_cache_fails_before_loader_and_records_status(monkeypatch, tmp_path):
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(output), "--data-dir", str(tmp_path / "missing")])
    monkeypatch.setattr(runner, "load_occupancy", lambda *_: pytest.fail("Loader must not run without cache"))
    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        runner.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "FileNotFoundError" in manifest["error"]


@pytest.mark.parametrize("unexpected", [False, True, "interrupt"])
def test_failure_reporting_and_continuation_without_training(monkeypatch, tmp_path, unexpected):
    cache = tmp_path / "data" / "occupancy"
    (cache / "extracted").mkdir(parents=True)
    (cache / "occupancy.zip").write_bytes(b"cache guard fixture")
    (cache / "extracted" / "data.txt").write_text("cache guard fixture")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(output), "--data-dir", str(cache.parent),
                                      "--models", "VonMises", "SinCos", "--seeds", "42"])
    monkeypatch.setattr(runner, "load_occupancy", lambda *_: occupancy_frame())
    monkeypatch.setattr(runner, "source_manifest", lambda: {"revision": "test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.weight.expand(len(x), 6)

    monkeypatch.setattr(runner, "build_model", lambda *_: StubModel())
    attempts = []

    def fake_train(model, *args, **kwargs):
        attempts.append(model)
        if len(attempts) == 1:
            if unexpected == "interrupt":
                raise KeyboardInterrupt("interrupted fixture")
            if unexpected:
                raise RuntimeError("unexpected fixture failure")
            raise FloatingPointError("numerical fixture failure")
        model.training_history = {"validation_loss": [1.], "best_val_loss": 1.}
        return model

    monkeypatch.setattr(runner, "_train_model", fake_train)
    if unexpected == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="interrupted fixture"):
            runner.main()
    elif unexpected:
        with pytest.raises(RuntimeError, match="unexpected fixture"):
            runner.main()
    else:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 1
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    expected_status = "interrupted" if unexpected == "interrupt" else ("failed" if unexpected else "numerical_failure")
    assert result["runs"][0]["execution_status"] == expected_status
    assert result["summary"]["VonMises"]["rmse"]["n_finite"] == 0
    assert result["summary"]["VonMises"]["rmse"]["n_excluded"] == 1
    if unexpected:
        assert manifest["status"] == expected_status
        assert len(attempts) == 1
    else:
        assert manifest["status"] == "completed_with_numerical_failures"
        assert manifest["finite_passed"] is False
        assert len(attempts) == 2
        assert result["runs"][1]["execution_status"] == "completed"
        assert (output / "SinCos_42_state.pt").exists()
        assert (output / "SinCos_42_predictions.npz").exists()
