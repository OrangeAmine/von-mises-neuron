"""Numerical, periodic-boundary and acceptance checks for VM-N04 adapters."""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import validate_dynamics_forecasters_v1 as validator
from src.dynamics_models_v1 import (
    CALENDAR_MODES, CalendarBasisV1, DynamicsForecasterV1, StandardDynamicsAdapterV1,
)


@pytest.mark.parametrize("calendar", CALENDAR_MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_dtype_finite_outputs_and_all_parameter_gradients(calendar, dtype):
    torch.manual_seed(71)
    model = DynamicsForecasterV1(calendar, dtype=dtype)
    history = torch.randn(5, 24, 3, dtype=dtype, requires_grad=True)
    future = torch.rand(5, 16, dtype=dtype, requires_grad=True) * (2 * math.pi)
    prediction = model(history, future)
    assert prediction.shape == (5, 16)
    assert prediction.dtype == dtype and prediction.device.type == "cpu"
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.any(parameter.grad != 0), name
    torch.testing.assert_close(model(history, future[..., None]), prediction)


@pytest.mark.parametrize("calendar", CALENDAR_MODES[1:])
def test_periodicity_at_future_clock_boundary_and_value_gradient_continuity(calendar):
    torch.manual_seed(29)
    model = DynamicsForecasterV1(calendar, dtype=torch.float64)
    history = torch.randn(2, 24, 3, dtype=torch.float64)
    angles = torch.linspace(-math.pi, math.pi, 32, dtype=torch.float64).reshape(2, 16).requires_grad_()
    shifted = (angles.detach() + 2 * math.pi).requires_grad_()
    baseline, wrapped = model(history, angles), model(history, shifted)
    torch.testing.assert_close(baseline, wrapped, rtol=1e-12, atol=1e-13)
    first = torch.autograd.grad(baseline.sum(), angles)[0]
    second = torch.autograd.grad(wrapped.sum(), shifted)[0]
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-13)
    # Independently wrap each target coordinate; periodicity is pointwise.
    windings = torch.arange(32, dtype=torch.float64).reshape(2, 16) % 3 - 1
    torch.testing.assert_close(model(history, angles + 2 * math.pi * windings), baseline,
                               rtol=1e-12, atol=1e-13)


def test_sensor_residual_does_not_impose_periodicity_on_sensor_values():
    model = DynamicsForecasterV1("fixed", dtype=torch.float64)
    with torch.no_grad():
        model.sensor_head.weight.zero_()
        model.sensor_head.bias.zero_()
    history = torch.zeros(2, 24, 3, dtype=torch.float64)
    future = torch.zeros(2, 16, dtype=torch.float64)
    shifted = history.clone()
    shifted[..., 0] += 2 * math.pi
    torch.testing.assert_close(model(shifted, future) - model(history, future),
                               torch.full((2, 16), 2 * math.pi, dtype=torch.float64))


def test_seeded_initialization_pairs_sensor_and_calendar_readout_weights():
    models = {}
    for calendar in CALENDAR_MODES:
        torch.manual_seed(701)
        models[calendar] = DynamicsForecasterV1(calendar)
    source = models["none"].state_dict()
    for model in models.values():
        for key, value in source.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    for name in ("fixed", "learned"):
        torch.testing.assert_close(models[name].calendar_readout.weight,
                                   models["fourier"].calendar_readout.weight, atol=0, rtol=0)
        torch.testing.assert_close(models[name].calendar_readout.bias,
                                   models["fourier"].calendar_readout.bias, atol=0, rtol=0)
    history, angles = torch.randn(2, 24, 3), torch.rand(2, 16)
    torch.testing.assert_close(models["fixed"](history, angles), models["learned"](history, angles), atol=0, rtol=0)


@pytest.mark.parametrize("calendar,count,basis_count", [
    ("none", 3920, 0), ("fourier", 3953, 0), ("fixed", 3953, 0), ("learned", 4017, 64)])
def test_declared_feature_and_fitted_parameter_counts(calendar, count, basis_count):
    model = DynamicsForecasterV1(calendar)
    config = model.get_config()
    assert sum(p.numel() for p in model.parameters()) == count == config["trainable_parameters"]
    assert config["basis_parameters"] == basis_count
    assert config["sensor_parameters"] == 3920
    assert config["calendar_feature_count"] == (0 if calendar == "none" else 32)
    assert config["fourier_harmonics"] == (16 if calendar == "fourier" else 0)


@pytest.mark.parametrize("mode", CALENDAR_MODES[1:])
def test_basis_matches_independent_equation_without_random_draws(mode):
    before = torch.get_rng_state().clone()
    basis = CalendarBasisV1(mode, dtype=torch.float64)
    assert torch.equal(before, torch.get_rng_state())
    angle = torch.tensor([0., 0.3, 2.1, 2 * math.pi], dtype=torch.float64)
    if mode == "fourier":
        phase = np.outer(angle.numpy(), np.arange(1, 17))
        expected = np.concatenate((np.cos(phase), np.sin(phase)), axis=-1)
    else:
        center = np.arange(32) * 2 * math.pi / 32
        expected = np.exp(32 * (np.cos(angle.numpy()[:, None] - center) - 1))
        assert torch.all(basis(angle) >= 0) and torch.all(basis(angle) <= 1)
        assert torch.isclose(basis(angle)[0, 0], torch.tensor(1., dtype=torch.float64))
    np.testing.assert_allclose(basis(angle).detach().numpy(), expected, rtol=1e-12, atol=1e-14)


def test_none_ignores_future_information_and_accepts_no_angle_argument():
    model = DynamicsForecasterV1("none")
    history = torch.randn(3, 24, 3)
    torch.testing.assert_close(model(history), model(history, torch.full((3, 16), float("nan"))), atol=0, rtol=0)


@pytest.mark.parametrize("calendar", CALENDAR_MODES)
def test_cpu_state_roundtrip_and_dtype_conversion(calendar):
    model = DynamicsForecasterV1(calendar).double()
    clone = DynamicsForecasterV1(calendar, dtype=torch.float64)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    history, future = torch.randn(3, 24, 3, dtype=torch.float64), torch.rand(3, 16, dtype=torch.float64)
    torch.testing.assert_close(model(history, future), clone(history, future), atol=0, rtol=0)
    assert all(value.dtype == torch.float64 for value in model.state_dict().values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("calendar", CALENDAR_MODES)
def test_cuda_device_buffers_outputs_and_gradient_parity(calendar):
    torch.manual_seed(818)
    cpu = DynamicsForecasterV1(calendar)
    gpu = copy.deepcopy(cpu).to("cuda")
    history, future = torch.randn(2, 24, 3), torch.rand(2, 16)
    expected = cpu(history, future)
    actual = gpu(history.cuda(), future.cuda())
    assert actual.device.type == "cuda"
    assert all(t.device.type == "cuda" for t in gpu.state_dict().values())
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=3e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(left.grad, right.grad.cpu(), atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize("kwargs", [
    {"calendar": "invalid"}, {"lookback": 0}, {"input_size": True}, {"horizon": -1},
    {"hidden_size": 1.5}, {"num_features": 31}, {"target_index": 3},
    {"target_index": True}, {"init_kappa": 0}, {"init_kappa": float("nan")}, {"dtype": torch.int64}])
def test_invalid_constructor_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicsForecasterV1(**kwargs)


@pytest.mark.parametrize("history", [torch.zeros(2, 23, 3), torch.zeros(2, 24, 4),
                                      torch.zeros(24, 3), torch.zeros(0, 24, 3),
                                      torch.zeros(2, 24, 3, dtype=torch.float64)])
def test_invalid_history_shape_and_dtype_rejected(history):
    with pytest.raises(ValueError):
        DynamicsForecasterV1("none")(history)


@pytest.mark.parametrize("angles", [torch.zeros(2, 15), torch.zeros(2, 16, 2),
                                     torch.zeros(2, 16, dtype=torch.float64)])
def test_invalid_future_shape_and_dtype_rejected(angles):
    with pytest.raises(ValueError):
        DynamicsForecasterV1("fixed")(torch.zeros(2, 24, 3), angles)


def test_missing_or_nonfloating_required_tensors_rejected():
    with pytest.raises(TypeError):
        DynamicsForecasterV1("fixed")(torch.zeros(2, 24, 3))
    with pytest.raises(TypeError):
        DynamicsForecasterV1("none")(torch.zeros(2, 24, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        CalendarBasisV1("fixed")(torch.ones(2, dtype=torch.int64))


@pytest.mark.parametrize("calendar", CALENDAR_MODES)
def test_standard_adapter_exact_available_information_and_midnight_crossing(calendar):
    model = StandardDynamicsAdapterV1(calendar).double()
    x = torch.randn(2, 72, 6, dtype=torch.float64)
    x[:, -1, 4] = 23 * 2 * math.pi / 24
    history = torch.cat((x[..., :4], torch.ones_like(x[..., :1]), torch.zeros_like(x[..., :1])), dim=-1)
    future = x[:, -1, 4, None] + torch.arange(1, 13, dtype=torch.float64) * 2 * math.pi / 24
    torch.testing.assert_close(model(x), model.body(history, future), atol=1e-8, rtol=1e-6)
    changed_unused = x.clone()
    changed_unused[..., 5] += 123  # no weekly calendar branch
    changed_unused[:, :-1, 4] += 456  # earlier clock phases never enter sensor MLP
    torch.testing.assert_close(model(x), model(changed_unused), atol=0, rtol=0)


def passing_rows():
    return [{"model": name, "execution_status": "completed", "metrics": {
        "rmse": 10., "mae": 8., "r2": .9, "mape": 2.}} for name in validator.MODELS]


def test_all_declared_standard_gates_pass():
    summary = validator.validation_summary(passing_rows())
    assert summary["all_passed"] and summary["all_runs_attempted"]
    assert all(summary["candidates"].values())


@pytest.mark.parametrize("name", validator.CANDIDATES)
@pytest.mark.parametrize("metric,value", [("rmse", 35.001), ("mae", 25.001),
                                          ("r2", .199), ("mape", 5.001)])
def test_each_candidate_including_ridge_requires_unchanged_absolute_gates(name, metric, value):
    rows = passing_rows()
    next(row for row in rows if row["model"] == name)["metrics"][metric] = value
    summary = validator.validation_summary(rows)
    assert not summary["all_passed"] and not summary["candidates"][name]


@pytest.mark.parametrize("name", validator.CANDIDATES)
@pytest.mark.parametrize("metric,value", [("rmse", 11.01), ("mae", 8.801),
                                          ("r2", .799), ("mape", 2.201)])
def test_all_candidates_including_ridge_require_base_regression_gates(name, metric, value):
    rows = passing_rows()
    next(row for row in rows if row["model"] == name)["metrics"][metric] = value
    summary = validator.validation_summary(rows)
    assert summary["acceptance"][name]["absolute_passed"]
    assert not summary["candidates"][name]


@pytest.mark.parametrize("status", ["pending", "running", "failed", "numerical_failure", "interrupted"])
def test_incomplete_reference_cannot_certify_candidate_regression(status):
    rows = passing_rows()
    rows[0]["execution_status"] = status
    assert not any(validator.validation_summary(rows)["candidates"].values())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_nonfinite_or_missing_standard_metrics_fail(value):
    rows = passing_rows()
    rows[-1]["metrics"]["rmse"] = value
    assert not validator.validation_summary(rows)["all_passed"]


def test_simple_references_have_finite_only_gate_with_errors_still_reported():
    rows = passing_rows()
    rows[-1]["metrics"].update(rmse=100., mae=70., r2=-3., mape=20.)
    summary = validator.validation_summary(rows)
    assert summary["all_passed"]
    assert not summary["acceptance"]["Seasonal"]["absolute_passed"]
    assert not summary["acceptance"]["Seasonal"]["diagnostic_regression_vs_base"]


def test_duplicate_and_undeclared_standard_rows_rejected():
    with pytest.raises(ValueError):
        validator.validation_summary(passing_rows() + passing_rows()[:1])
    with pytest.raises(ValueError):
        validator.validation_summary([{"model": "unknown"}])


def test_budget_fixed_and_output_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        validator.run_validation(tmp_path)
    with pytest.raises(ValueError, match="exactly 50"):
        validator.run_validation(tmp_path / "new", epochs=10)
    assert not (tmp_path / "new").exists()


def test_standard_failure_retained_with_every_other_model_attempted(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "source_manifest", lambda: {"revision": "test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_fit(name, model, data, device):
        assert device.type == "cpu"
        model.training_history = {"best_val_loss": 1., "validation_loss": [1.]}
        if name == "DynamicsLearned":
            raise FloatingPointError("injected failed fit")
        return model

    monkeypatch.setattr(validator, "_fit", fake_fit)
    monkeypatch.setattr(validator, "predict", lambda model, data, device: torch.zeros_like(data["y_test"]))
    monkeypatch.setattr(validator, "compute_benchmark_metrics", lambda *args: passing_rows()[0]["metrics"])
    monkeypatch.setattr(validator, "_reference_result", lambda name, data, frame, output: (
        torch.zeros_like(data["y_test"]), {"params": 0, "training_seconds": 0., "inference_seconds": 0.}))
    output = tmp_path / "new"
    summary = validator.run_validation(output)
    assert not summary["all_passed"] and summary["all_runs_attempted"]
    assert not summary["candidates"]["DynamicsLearned"]
    assert all(summary["candidates"][name] for name in validator.CANDIDATES if name != "DynamicsLearned")
    results = json.loads((output / "results.json").read_text())
    failed = next(row for row in results["runs"] if row["model"] == "DynamicsLearned")
    assert failed["execution_status"] == "numerical_failure"
    assert (output / "DynamicsLearned_42_failed_state.pt").exists()
    assert (output / "DynamicsLearned_42_initial_state.pt").exists()
    assert not (output / "DynamicsLearned_42_state.pt").exists()
    assert (output / "ResidualRidge_42_predictions.npz").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["validation_status"] == "failed"
    assert "src/dynamics_models_v1.py" in manifest["source_sha256"]


def test_cli_nonzero_status_follows_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "run_validation", lambda *args: {"all_passed": False, "candidates": {}})
    assert validator.main(["--output-dir", str(tmp_path / "new")]) == 1


def _reference_fixture():
    """Small independent history matrices for exact reference numerical checks."""
    rng = np.random.default_rng(5571)
    data = {"norm_stats": {"CO2": (600., 50.)}}
    for split, n in (("train", 30), ("val", 12), ("test", 8)):
        x = rng.normal(size=(n, 3, 6)).astype(np.float32)
        x[..., 4] = rng.integers(0, 24, size=(n, 3)) * (2 * np.pi / 24)
        y = x[:, -1, 2, None] + 0.4 * x[:, -2, 0, None] + np.linspace(0, .3, 12)[None, :]
        data[f"X_{split}"], data[f"y_{split}"] = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    times = pd.date_range("2026-01-05", periods=72, freq="1h", tz="UTC")
    frame = pd.DataFrame({"CO2": 600. + times.hour.to_numpy() + np.arange(72) / 20}, index=times)
    return data, frame


def test_real_ridge_reference_matches_independent_augmented_least_squares(tmp_path):
    data, frame = _reference_fixture()
    actual, record = validator._reference_result("ResidualRidge", data, frame, tmp_path)

    def features(split):
        # Independently assemble the information-equivalent CO2-first inputs.
        original = data[f"X_{split}"].numpy()
        sensor = np.stack((original[..., 2], original[..., 0], original[..., 1],
                           original[..., 3], np.ones(original.shape[:2]),
                           np.zeros(original.shape[:2])), axis=-1)
        return sensor.reshape(len(sensor), -1), sensor[:, -1, 0, None]

    x, anchor = features("train")
    vx, va = features("val")
    tx, ta = features("test")
    residual = data["y_train"].numpy().astype(np.float64) - anchor
    xm, ym = x.mean(0), residual.mean(0)
    candidates = []
    for penalty in validator.model_config("ResidualRidge")["penalties"]:
        augmented_x = np.vstack((x - xm, np.sqrt(len(x) * penalty) * np.eye(x.shape[1])))
        augmented_y = np.vstack((residual - ym, np.zeros((x.shape[1], 12))))
        weight = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        bias = ym - xm @ weight
        val_mse = np.mean((va + vx @ weight + bias - data["y_val"].numpy()) ** 2)
        candidates.append((val_mse, penalty, weight, bias))
    best = min(candidates, key=lambda candidate: candidate[0])
    np.testing.assert_allclose(actual.numpy(), ta + tx @ best[2] + best[3], atol=1e-10, rtol=1e-10)
    assert record["training"]["selected_penalty"] == best[1]
    assert record["params"] == x.shape[1] * 12 + 12
    assert (tmp_path / "ResidualRidge_coefficients.npz").exists()
    altered = copy.deepcopy(data)
    for split in ("train", "val", "test"):
        altered[f"X_{split}"][..., 4:] = -999  # clocks are not ridge information
    altered["y_test"].fill_(9999)  # reserved labels cannot change fitted predictions
    replay, _ = validator._reference_result("ResidualRidge", altered, frame, tmp_path)
    torch.testing.assert_close(replay, actual, atol=0, rtol=0)


def test_real_seasonal_uses_training_rows_only_and_persistence_uses_last_target(tmp_path):
    data, frame = _reference_fixture()
    prediction, _ = validator._reference_result("Seasonal", data, frame, tmp_path)
    altered = frame.copy()
    train_end = int(len(frame) * .8)
    altered.iloc[train_end:, 0] = -1e9
    replay, _ = validator._reference_result("Seasonal", data, altered, tmp_path)
    torch.testing.assert_close(prediction, replay, atol=0, rtol=0)
    persistence, record = validator._reference_result("Persistence", data, frame, tmp_path)
    torch.testing.assert_close(persistence, data["X_test"][:, -1, 2, None].expand(-1, 12), atol=0, rtol=0)
    assert record["params"] == 0
