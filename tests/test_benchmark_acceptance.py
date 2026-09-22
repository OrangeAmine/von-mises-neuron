"""Failure-case coverage for synthetic benchmark acceptance and CLI reporting."""

import json
import math
import sys

import pytest
import torch

from scripts import benchmark_standard_dataset as benchmark


GOOD = {"rmse": 10.0, "mae": 8.0, "r2": 0.8, "mape": 1.0}


@pytest.mark.parametrize("metric", benchmark.THRESHOLDS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_thresholds_reject_nonfinite(metric, invalid):
    """Even favorable infinity cannot satisfy an absolute gate."""
    passed, failures = benchmark.check_thresholds({**GOOD, metric: invalid})
    assert not passed
    assert any(metric.upper() in failure and "finite" in failure for failure in failures)


def test_threshold_boundaries_are_inclusive():
    assert benchmark.check_thresholds(benchmark.THRESHOLDS.copy()) == (True, [])


@pytest.mark.parametrize("metric,value", [("rmse", 35.01), ("mae", 25.01),
                                        ("r2", 0.199), ("mape", 5.01)])
def test_each_absolute_threshold_is_enforced(metric, value):
    assert not benchmark.check_thresholds({**GOOD, metric: value})[0]


@pytest.mark.parametrize("side", ["base", "new"])
@pytest.mark.parametrize("metric", benchmark.THRESHOLDS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_regression_rejects_nonfinite_on_both_sides(side, metric, invalid):
    base, new = GOOD.copy(), GOOD.copy()
    (base if side == "base" else new)[metric] = invalid
    passed, failures = benchmark.check_regression(new, base)
    assert not passed
    assert any(side.capitalize() in failure for failure in failures)


def test_regression_boundaries_are_inclusive():
    new = {"rmse": 11.0, "mae": 8.8, "mape": 1.1, "r2": 0.7}
    assert benchmark.check_regression(new, GOOD) == (True, [])


@pytest.mark.parametrize("metric,value", [("rmse", 11.001), ("mae", 8.801),
                                        ("r2", 0.6999), ("mape", 1.1001)])
def test_each_regression_limit_is_enforced(metric, value):
    assert not benchmark.check_regression({**GOOD, metric: value}, GOOD)[0]


@pytest.mark.parametrize("metric", ["rmse", "mae", "mape"])
def test_zero_base_error_does_not_skip_regression(metric):
    base = {**GOOD, metric: 0.0}
    assert benchmark.check_regression(base, base) == (True, [])
    assert not benchmark.check_regression({**base, metric: 1e-15}, base)[0]


def test_r2_regression_is_absolute_even_for_negative_base_r2():
    base = {**GOOD, "r2": -1.0}
    assert benchmark.check_regression({**GOOD, "r2": -1.1}, base)[0]
    assert not benchmark.check_regression({**GOOD, "r2": -1.1001}, base)[0]


@pytest.mark.parametrize("skip_baselines", [False, True])
@pytest.mark.parametrize("scenario", ["pass", "base_absolute", "temporal_absolute",
                                      "temporal_regression", "temporal_nan"])
def test_cli_aggregate_matches_gates_and_exit(monkeypatch, tmp_path, capsys,
                                            skip_baselines, scenario):
    """Mock training/evaluation while retaining actual CLI, pipelines and JSON."""
    base, temporal = GOOD.copy(), GOOD.copy()
    if scenario == "base_absolute":
        base["rmse"] = 36.0
    elif scenario == "temporal_absolute":
        base["rmse"], temporal["rmse"] = 34.0, 36.0
    elif scenario == "temporal_regression":
        temporal["rmse"] = 12.0
    elif scenario == "temporal_nan":
        temporal["r2"] = float("nan")
    _mock_benchmarks(monkeypatch, base, temporal)
    argv = ["benchmark", "--n-days", "10", "--lookback", "2", "--horizon", "1",
            "--output-dir", str(tmp_path)]
    if skip_baselines:
        argv.append("--skip-baselines")
    monkeypatch.setattr(sys, "argv", argv)
    expected_pass = scenario == "pass"
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == (0 if expected_pass else 1)
    report_text = (tmp_path / "benchmark_results.json").read_text(encoding="utf-8")
    assert "NaN" not in report_text and "Infinity" not in report_text
    report = json.loads(report_text)
    assert report["all_passed"] is expected_pass
    for name, result in report["results"].items():
        assert result["passed"] is report["acceptance"][name]["passed"]
    assert report["all_passed"] == all(r["passed"] for r in report["results"].values())
    assert len(report["results"]) == (2 if skip_baselines else 6)
    assert ("SinCos" in report["results"]) is not skip_baselines
    if not skip_baselines:
        assert report["results"]["SinCos"]["params"] == report["results"]["VonMises"]["params"]
        assert report["acceptance"]["SinCos"]["checks"]["absolute_thresholds"]
    temporal_checks = report["acceptance"]["TemporalVM"]["checks"]
    if scenario == "temporal_absolute":
        assert not temporal_checks["absolute_thresholds"]
        assert temporal_checks["regression_vs_base"]
    elif scenario == "temporal_regression":
        assert temporal_checks["absolute_thresholds"]
        assert not temporal_checks["regression_vs_base"]
    elif scenario == "temporal_nan":
        assert report["results"]["TemporalVM"]["r2"] is None
    output = capsys.readouterr().out
    assert f"OVERALL RESULT: {'PASS' if expected_pass else 'FAIL'}" in output


def _mock_benchmarks(monkeypatch, base, temporal):
    """Keep CLI tests inexpensive and independent of GPU availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name, metrics in [("benchmark_von_mises", base),
                          ("benchmark_temporal_von_mises", temporal),
                          ("benchmark_sincos", GOOD),
                          ("benchmark_lstm", GOOD),
                          ("benchmark_transformer", GOOD),
                          ("benchmark_handcrafted", GOOD)]:
        monkeypatch.setattr(benchmark, name, lambda *args, metrics=metrics: metrics.copy())


def test_cli_nonfinite_baseline_fails_without_baseline_accuracy_gate(monkeypatch, tmp_path):
    _mock_benchmarks(monkeypatch, GOOD, GOOD)
    monkeypatch.setattr(benchmark, "benchmark_lstm", lambda *args: {**GOOD, "mae": math.inf})
    monkeypatch.setattr(benchmark, "benchmark_transformer",
                        lambda *args: {"rmse": 1000., "mae": 1000., "r2": -100., "mape": 1000.})
    monkeypatch.setattr(sys, "argv", ["benchmark", "--n-days", "10", "--lookback", "2",
                                      "--horizon", "1", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 1
    report = json.loads((tmp_path / "benchmark_results.json").read_text(encoding="utf-8"))
    assert not report["all_passed"]
    assert report["acceptance"]["LSTM"]["checks"] == {"finite_metrics": False}
    assert report["results"]["LSTM"]["mae"] is None
    assert report["results"]["Transformer"]["passed"]


def test_training_history_records_completed_epochs_and_best_checkpoint():
    """A tiny CPU fit checks metadata without launching a model benchmark."""
    model = torch.nn.Linear(2, 1)
    inputs = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    targets = inputs.sum(dim=1, keepdim=True)
    result = benchmark._train_model(model, inputs, targets, inputs, targets,
                                    max_epochs=2, batch_size=2, patience=3)
    history = result.training_history
    assert history["epochs_completed"] == 2
    assert len(history["validation_loss"]) == 2
    assert history["best_epoch"] in (1, 2)
    assert history["best_val_loss"] == history["validation_loss"][history["best_epoch"] - 1]
    assert history["max_epochs"] == 2 and history["batch_size"] == 2
    assert history["patience"] == 3
    assert history["lr"] == 0.001 and history["weight_decay"] == 0.0001
    assert history["gradient_clip"] == 1.0


@pytest.mark.parametrize("metric,value", [("rmse", 35.01), ("mae", 25.01),
                                        ("r2", 0.199), ("mape", 5.01),
                                        ("r2", float("nan"))])
def test_cli_sincos_failure_is_enforced(monkeypatch, tmp_path, metric, value):
    """The new baseline must pass the same absolute gates before validation."""
    _mock_benchmarks(monkeypatch, GOOD, GOOD)
    monkeypatch.setattr(benchmark, "benchmark_sincos", lambda *args: {**GOOD, metric: value})
    monkeypatch.setattr(sys, "argv", ["benchmark", "--n-days", "10", "--lookback", "2",
                                     "--horizon", "1", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 1
    report = json.loads((tmp_path / "benchmark_results.json").read_text(encoding="utf-8"))
    assert not report["all_passed"]
    assert not report["results"]["SinCos"]["passed"]
    assert not report["acceptance"]["SinCos"]["checks"]["absolute_thresholds"]
    assert "regression_vs_base" not in report["acceptance"]["SinCos"]["checks"]


def test_sincos_benchmark_matches_initial_function_rng_data_and_budget(monkeypatch):
    """Intercept fitting to verify pairing without doing a training run."""
    x = torch.linspace(-1, 1, 48).reshape(2, 6, 4)
    y = torch.zeros(2, 3)
    data = {
        "X_train": x, "y_train": y, "X_val": x, "y_val": y,
        "X_test": x, "y_test": y, "horizon": 3, "norm_stats": {"CO2": (600., 35.)},
    }
    captured = []

    def capture_model(model, *fit_data, **fit_options):
        captured.append((model, torch.random.get_rng_state().clone(), fit_data, fit_options))
        model.eval()
        return model

    monkeypatch.setattr(benchmark, "_train_model", capture_model)
    benchmark.seed_everything(42)
    benchmark.benchmark_von_mises(data, 50, torch.device("cpu"))
    benchmark.seed_everything(42)
    benchmark.benchmark_sincos(data, 50, torch.device("cpu"))
    (vm, vm_rng, vm_data, vm_options), (sc, sc_rng, sc_data, sc_options) = captured
    assert torch.equal(vm_rng, sc_rng)
    assert all(left is right for left, right in zip(vm_data, sc_data))
    assert vm_options == sc_options == {"max_epochs": 50}
    assert benchmark._count_params(vm) == benchmark._count_params(sc)
    torch.testing.assert_close(vm(x), sc(x), atol=2e-6, rtol=2e-5)
