"""Checks for fairness and causal reference predictions in the comparison runner."""

import json
import sys

import numpy as np
import pytest
import torch

from scripts import controlled_temporal_comparison as comparison
from scripts.controlled_temporal_comparison import build_model, reference_predictions, summarize
from scripts.benchmark_standard_dataset import seed_everything
from src.temporal import extract_angular_time_features


def test_base_and_cartesian_share_function_and_training_rng():
    """Independent construction preserves weights and random training streams."""
    data = {"X_train": torch.zeros(2, 5, 4), "horizon": 3}
    seed_everything(42)
    base = build_model("VonMises", data).eval()
    seed_everything(42)
    cartesian = build_model("SinCos", data).eval()
    x = torch.randn(2, 5, 4)
    torch.testing.assert_close(base(x), cartesian(x), atol=2e-6, rtol=2e-5)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in cartesian.parameters())


def test_seasonal_persistence_repeats_observed_season_for_long_horizon():
    data = {"X_test": torch.arange(30.0).view(1, 30, 1), "horizon": 50}
    predictions = reference_predictions(data, 0)
    torch.testing.assert_close(predictions["Persistence"], torch.full((1, 50), 29.0))
    expected = torch.tensor([6.0 + step % 24 for step in range(50)]).view(1, 50)
    torch.testing.assert_close(predictions["SeasonalPersistence"], expected)


def test_season_requires_enough_past_observations():
    with pytest.raises(ValueError, match="full season"):
        reference_predictions({"X_test": torch.zeros(1, 5, 1), "horizon": 2}, 0)


def test_summary_uses_sample_sd_and_retains_failed_runs():
    rows = [{"model": "test", "params": 1, "absolute_passed": passed,
             "metrics": dict(rmse=v, mae=v, r2=v, mape=v)}
            for v, passed in [(1.0, True), (3.0, False)]]
    summary = summarize(rows)["test"]
    assert summary["n_seeds"] == 2
    assert summary["rmse"]["mean"] == 2.0
    assert summary["rmse"]["std"] == pytest.approx(2 ** 0.5)
    assert not summary["absolute_passed_all"]


@pytest.mark.parametrize("name", ["UnitScales", "LearnedScales"])
def test_amplitude_controls_match_initial_temporal_function(name):
    """Amplitude ablations start with the same phases, backbone and outputs."""
    data = {"horizon": 3, "sensor_size": 4, "time_periods": [24, 168]}
    seed_everything(42)
    reference = build_model("TemporalVM", data).double().eval()
    seed_everything(42)
    variant = build_model(name, data).double().eval()
    x = torch.randn(2, 5, 6, dtype=torch.float64)
    # Coefficient rescaling occurs in float32 at initialization.
    torch.testing.assert_close(reference(x), variant(x), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(reference.output_head.weight, variant.output_head.weight, atol=0, rtol=0)
    if name == "UnitScales":
        torch.testing.assert_close(variant.encoder.freq_bank.get_kappa(), torch.ones(64).double())
    else:
        assert sum(p.numel() for p in variant.parameters()) == sum(
            p.numel() for p in reference.parameters()) + 64


@pytest.mark.parametrize("name", ["NoTime", "NoConv", "Pooling"])
def test_component_removals_retain_every_shared_reference_tensor(name):
    data = {"horizon": 3, "sensor_size": 4, "time_periods": [24, 168]}
    seed_everything(42)
    reference = build_model("TemporalVM", data)
    seed_everything(42)
    variant = build_model(name, data)
    source = reference.state_dict()
    assert len(variant.state_dict()) < len(source)
    for key, tensor in variant.state_dict().items():
        torch.testing.assert_close(tensor, source[key], atol=0, rtol=0)
        assert tensor.data_ptr() != source[key].data_ptr()


def test_paired_differences_retain_failed_seed_exclusions():
    rows = []
    for name, values in [("VonMises", [1., 2., 3.]), ("SinCos", [2., 4., None])]:
        for seed, value in enumerate(values):
            rows.append({"model": name, "seed": seed,
                         "metrics": {metric: value for metric in comparison.THRESHOLDS}})
    stats = comparison.paired_differences(rows)["SinCos_minus_VonMises"]["metrics"]["rmse"]
    assert stats["by_seed"] == {"0": 1., "1": 2.}
    assert stats["excluded_seeds"] == [2]
    assert stats["n"] == 2 and stats["mean"] == 1.5
    assert stats["std"] == pytest.approx(2 ** -0.5)


def _mock_runner(monkeypatch, tmp_path, modes, models=None):
    """Use real protocol/reporting with tiny fake models and no training."""
    models = models or list(modes)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["comparison", "--output-dir", str(output),
                                     "--seeds", "42", "--models", *models])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(comparison, "source_manifest", lambda: {"revision": "test"})

    class FakeModel(torch.nn.Module):
        def __init__(self, name, horizon):
            super().__init__()
            self.name = name
            self.horizon = horizon
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            mode = modes.get(self.name, "good")
            value = float("nan") if mode == "prediction_nan" else (
                1. if mode == "absolute_failure" else 2. if mode == "regression_failure" else
                3. if mode == "metric_inf" else 0.)
            return torch.full((len(x), self.horizon), value)

    def fake_train(model, *args, **kwargs):
        mode = modes.get(model.name, "good")
        if mode == "exception":
            raise RuntimeError("injected training error")
        if mode == "floating_exception":
            raise FloatingPointError("injected numerical error")
        model.training_history = {
            "validation_loss": [float("nan") if mode == "history_nan" else 1.],
            "best_val_loss": 1., "epochs_completed": 1,
        }
        return model.eval()

    def fake_metrics(prediction, *args, **kwargs):
        value = float(prediction.flatten()[0])
        rmse = 40. if value == 1. else 12. if value == 2. else float("inf") if value == 3. else 10.
        return dict(rmse=rmse, mae=8., r2=0.8, mape=1.)

    monkeypatch.setattr(comparison, "build_model", lambda name, data: FakeModel(name, data["horizon"]))
    monkeypatch.setattr(comparison, "_train_model", fake_train)
    monkeypatch.setattr(comparison, "compute_benchmark_metrics", fake_metrics)
    return output


@pytest.mark.parametrize("mode", ["prediction_nan", "history_nan", "metric_inf", "floating_exception"])
def test_numerical_failures_survive_and_later_models_run(monkeypatch, tmp_path, mode):
    output = _mock_runner(monkeypatch, tmp_path, {"SinCos": mode, "VonMises": "good"})
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 1
    text = (output / "results.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    result = json.loads(text)
    assert len(result["runs"]) == 2
    failed, later = result["runs"]
    assert failed["execution_status"] == "numerical_failure"
    assert failed["absolute_failures"] and not failed["absolute_passed"]
    assert any(value is None for value in failed["metrics"].values())
    assert later["execution_status"] == "completed" and later["absolute_passed"]
    assert result["summary"]["SinCos"]["numerical_failures"] == 1
    assert not result["validation"]["core_passed"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_status"] == "completed"
    assert manifest["validation_status"] == "failed"
    if mode == "history_nan":
        assert failed["training"]["validation_loss"] == [None]


@pytest.mark.parametrize("mode", ["absolute_failure", "prediction_nan"])
def test_exploratory_failures_do_not_fail_core_execution(monkeypatch, tmp_path, mode):
    output = _mock_runner(monkeypatch, tmp_path, {"NoTime": mode, "VonMises": "good"})
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 0
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert result["validation"]["core_passed"]
    assert not result["validation"]["all_absolute_passed"]
    assert result["validation"]["exploratory_absolute_checks"] == {"NoTime": False}


@pytest.mark.parametrize("name,mode", [("SinCosTime", "absolute_failure"),
                                     ("VonMisesTime", "absolute_failure"),
                                     ("TemporalVM", "regression_failure")])
def test_requested_core_and_regression_failures_set_exit(monkeypatch, tmp_path, name, mode):
    output = _mock_runner(monkeypatch, tmp_path, {"VonMises": "good", name: mode})
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 1
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert not result["validation"]["core_passed"]
    if mode == "regression_failure":
        assert result["validation"]["core_absolute_checks"]["TemporalVM"]
        assert not result["temporal_regression"]["42"]["passed"]


def test_unexpected_exception_retains_partial_rows_and_failed_manifest(monkeypatch, tmp_path):
    output = _mock_runner(monkeypatch, tmp_path,
                          {"VonMises": "good", "SinCos": "exception", "TemporalVM": "good"})
    with pytest.raises(RuntimeError, match="injected training error"):
        comparison.main()
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert [row["model"] for row in result["runs"]] == ["VonMises", "SinCos"]
    assert result["runs"][-1]["execution_status"] == "failed"
    assert not result["validation"]["all_runs_attempted"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_status"] == "failed"
    assert "RuntimeError" in manifest["error"]


@pytest.mark.parametrize("clock", [None, "local", "fixed"])
def test_calendar_argument_reaches_pipeline_and_manifest(monkeypatch, tmp_path, clock):
    """The default and explicit modes are recorded even with mocked training."""
    output = _mock_runner(monkeypatch, tmp_path, {"VonMises": "good"})
    if clock is not None:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--calendar-clock", clock])
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = clock or "local"
    assert manifest["calendar_clock"] == expected
    assert manifest["calendar_timezone"] == ("Europe/Paris" if expected == "local" else "UTC+01:00")
    assert manifest["split"]["test"]["last"] == (
        "2021-04-01 00:00:00+02:00" if expected == "local" else "2021-03-31 23:00:00+01:00")


def test_calendar_choice_preserves_observations_instants_and_fitting_angles():
    frame = comparison.generate_synthetic_co2()
    assert comparison.apply_calendar_clock(frame) is frame
    assert comparison.apply_calendar_clock(frame, "local") is frame
    fixed = comparison.apply_calendar_clock(frame, "fixed")
    np.testing.assert_array_equal(frame.to_numpy(), fixed.to_numpy())
    np.testing.assert_array_equal(frame.index.asi8, fixed.index.asi8)
    original_angles = extract_angular_time_features(frame.index, [24, 168])
    fixed_angles = extract_angular_time_features(fixed.index, [24, 168])
    np.testing.assert_array_equal(original_angles[:1944], fixed_angles[:1944])
    assert not np.array_equal(original_angles[2066:], fixed_angles[2066:])
    np.testing.assert_array_equal(comparison.augment_with_time_features(frame).to_numpy(),
                                  comparison.augment_with_time_features(fixed).to_numpy())
    with pytest.raises(ValueError, match="local or fixed"):
        comparison.apply_calendar_clock(frame, "unknown")
