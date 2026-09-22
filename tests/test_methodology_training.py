"""Small CPU checks of the preregistered adaptation and selection mechanics."""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.methodology_tasks import generate_task, initialization_bundle
from src.methodology_training import (
    FixedFeaturesV1, a_coordinate_diagnostics, b_geometry_diagnostics,
    closed_form_a, explicit_penalty, fit_ridge, make_a_model, make_b_model,
    parameter_count, parameter_manifest, regression_metrics, train_a, train_b,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/vm_methodology_v1.json").read_text())


@pytest.fixture(scope="module")
def a_task(config):
    return generate_task(config, "A", "pilot", "phase_only")


@pytest.fixture(scope="module")
def b_task(config):
    return generate_task(config, "B", "pilot", "smooth").train_prefix(32)


def small_arrays(task):
    """Use deterministic prefix arrays for unit tests, never study estimates."""
    arrays = task.as_arrays()
    return {name: value[:8].copy() if name.startswith(("train", "validation")) else value[::128].copy()
            for name, value in arrays.items()}


def test_a_known_start_conversion_counts_penalty_and_global_rng(a_task):
    torch.manual_seed(719)
    before = torch.get_rng_state().clone()
    phase = a_task.function.parameters["phase_before"]
    polar = make_a_model("polar", .02, phase, device="cpu")
    cartesian = make_a_model("cartesian", .02, phase, device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert parameter_count(polar) == parameter_count(cartesian) == 3
    x = torch.linspace(-math.pi, math.pi, 31).reshape(-1, 1)
    torch.testing.assert_close(polar(x), cartesian(x))
    torch.testing.assert_close(explicit_penalty(polar, "A", .001), explicit_penalty(cartesian, "A", .001))
    with torch.no_grad():
        polar.bias.fill_(100)
        polar.phase.add_(17)
    torch.testing.assert_close(explicit_penalty(polar, "A", .001), explicit_penalty(cartesian, "A", .001))


@pytest.mark.parametrize("model_id", ["learned_vm", "learned_geodesic", "sincos_mlp", "raw_angle_mlp"])
def test_b_counts_explicit_initialization_roles_and_rng(config, b_task, model_id):
    initialization = initialization_bundle(config, b_task)
    before = torch.get_rng_state().clone()
    model = make_b_model(model_id, initialization, device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert parameter_count(model) == 49
    rows = parameter_manifest(model, "B")
    assert sum(row["count"] for row in rows) == 49
    assert all(row["native_weight_decay"] == 0 for row in rows)
    assert [row["name"] for row in rows if row["explicit_penalty"] != "none"] == ["readout.weight"]
    expected = .0001 * model.readout.weight.square().sum() / 2
    torch.testing.assert_close(explicit_penalty(model, "B", .0001), expected)
    with torch.no_grad():
        model.readout.bias.add_(100)
    torch.testing.assert_close(explicit_penalty(model, "B", .0001), expected)
    result = b_geometry_diagnostics(model, torch.linspace(-math.pi, math.pi, 33).reshape(-1, 1).double())
    assert result["rotation_output_max_abs"] < 1e-10
    if model_id != "raw_angle_mlp":
        assert result["two_pi_output_max_abs"] < 1e-10
    else:
        assert result["two_pi_output_max_abs"] > 1e-4


def test_kernel_initialization_geometry_and_signed_readout_shared(config, b_task):
    init = initialization_bundle(config, b_task)
    vm, geo = [make_b_model(kind, init, device="cpu") for kind in ("learned_vm", "learned_geodesic")]
    torch.testing.assert_close(vm.bank.centers, geo.bank.centers)
    torch.testing.assert_close(vm.bank.concentrations, geo.bank.concentrations)
    torch.testing.assert_close(vm.readout.weight, geo.readout.weight)
    assert (vm.readout.weight < 0).any() and (vm.readout.weight > 0).any()
    assert not torch.equal(vm(torch.tensor([[0.]])), geo(torch.tensor([[0.]])))


def test_duplicate_optimizer_membership_rejected(config, b_task):
    model = make_b_model("sincos_mlp", initialization_bundle(config, b_task), device="cpu")
    model.alias = model.readout.weight
    with pytest.raises(ValueError, match="Duplicate"):
        parameter_manifest(model, "B")


@pytest.mark.parametrize("coordinate", ["polar", "cartesian"])
@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_a_exact_full_update_history_and_independent_curve_recompute(a_task, coordinate, optimizer):
    arrays = small_arrays(a_task)
    arrays["task_parameters"] = a_task.function.parameters
    model = make_a_model(coordinate, 1., a_task.function.parameters["phase_before"], device="cpu")
    bundle = train_a(model, arrays, optimizer_name=optimizer, learning_rate=.01, penalty=.001,
                     updates=4, record_every=2, device="cpu")
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    assert summary["status"] == "completed", summary.get("failure")
    assert summary["updates_completed"] == 4
    assert [row["step"] for row in summary["history"]] == [0, 2, 4]
    assert len(summary["update_history"]) == 4
    assert set(artifacts["recorded_states"]) == {"0", "2", "4"}
    curve = []
    for step in (0, 2, 4):
        prediction = artifacts["recorded_predictions"][str(step)]["grid"].numpy()
        curve.append(np.square(prediction - arrays["grid_clean"]).mean())
    expected = np.trapz(curve, [0, 2, 4]) / (4 * arrays["grid_clean"].var())
    assert summary["primary_error"] == pytest.approx(expected, rel=1e-14)
    assert summary["selected_step"] == 4
    assert summary["final_metrics"]["phase_error_defined"]
    assert summary["diagnostics"]["final"]["rotation_gradient_mapped_max_abs"] < 1e-10
    for row in summary["optimizer_parameters"]:
        assert row["native_weight_decay"] == 0
    json.dumps(summary, allow_nan=False)


def test_a_phase_undefined_threshold_and_historical_decay_diagnostic(a_task):
    arrays = small_arrays(a_task)
    arrays["task_parameters"] = a_task.function.parameters
    model = make_a_model("polar", .0001, .3, device="cpu")
    bundle = train_a(model, arrays, optimizer_name="Adam", learning_rate=.01, penalty=.001,
                     updates=0, device="cpu")
    summary = bundle["summary"]
    assert summary["status"] == "completed", summary.get("failure")
    assert summary["undefined_phase_count"] == 1
    assert summary["final_metrics"]["wrapped_phase_error"] is None
    diagnostic = summary["diagnostics"]["initial"]
    assert diagnostic["phase_shift_objective_abs"] < 1e-10
    assert diagnostic["phase_shift_gradient_max_abs"] < 1e-10
    assert abs(diagnostic["historical_decay_phase_shift_objective_difference"]) > .001
    assert diagnostic["historical_decay_phase_shift_gradient_difference"] == pytest.approx(.001 * 2 * math.pi)


def test_a_reference_solves_unpenalized_and_penalized_objectives():
    theta = np.linspace(-math.pi, math.pi, 40, endpoint=False)[:, None]
    y = 2 * np.cos(theta) - .7 * np.sin(theta) + .4
    arrays = {"train_x": theta, "train_y": y, "validation_x": theta, "validation_y": y,
              "grid_x": theta, "grid_clean": y, "grid_noisy": y}
    unpenalized = closed_form_a(arrays, 0)
    np.testing.assert_allclose(unpenalized["artifacts"]["coefficients"].numpy().flatten(), [2, -.7, .4], atol=1e-12)
    ridge = closed_form_a(arrays, .1)
    np.testing.assert_allclose(ridge["artifacts"]["coefficients"].numpy().flatten(), [2 / 1.2, -.7 / 1.2, .4], atol=1e-12)


def test_a_timeout_retains_initial_and_partial_evidence(a_task):
    model = make_a_model("polar", 1., 0., device="cpu")
    result = train_a(model, small_arrays(a_task), optimizer_name="Adam", learning_rate=.01,
                     penalty=0, updates=3, device="cpu", deadline=time.perf_counter() - 1)
    assert result["summary"]["status"] == "timeout"
    assert result["summary"]["updates_completed"] == 0
    assert len(result["summary"]["history"]) == 1
    assert "final_state" in result["artifacts"]


def test_a_nonfinite_gradients_retained_as_failure(a_task):
    model = make_a_model("polar", 1., 0., device="cpu")
    model.phase.register_hook(lambda g: g * float("nan"))
    result = train_a(model, small_arrays(a_task), optimizer_name="Adam", learning_rate=.01,
                     penalty=0, updates=3, device="cpu")
    assert result["summary"]["status"] == "numerical_failure"
    assert result["summary"]["failed_step"] == 1
    assert "gradient" in result["summary"]["failure"]
    assert "selected_state" not in result["artifacts"]


def test_b_initial_checkpoint_selected_and_final_state_preserved(config, b_task):
    model = make_b_model("sincos_mlp", initialization_bundle(config, b_task), device="cpu")
    arrays = small_arrays(b_task)
    with torch.no_grad():
        expected = model(torch.from_numpy(arrays["validation_x"]).float()).double().numpy()
    arrays["validation_y"] = expected
    initial = copy.deepcopy(model.state_dict())
    bundle = train_b(model, arrays, learning_rate=.1, batch_seed=92, max_updates=5,
                     validation_every=1, patience_evaluations=2, batch_size=4, device="cpu")
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    assert summary["status"] == "completed", summary.get("failure")
    assert summary["selected_step"] == 0
    assert summary["best_validation_mse"] == 0
    assert summary["updates_completed"] == 2
    assert summary["stop_reason"] == "patience"
    assert set(artifacts["checkpoint_states"]) == {"0", "1", "2"}
    assert any(not torch.equal(initial[name], artifacts["final_state"][name]) for name in initial)
    assert all(torch.equal(initial[name], model.state_dict()[name]) for name in initial)
    assert all(torch.equal(initial[name], artifacts["selected_state"][name]) for name in initial)
    for row in summary["history"]:
        value = artifacts["validation_predictions"][str(row["step"])].numpy()
        assert row["validation_mse"] == pytest.approx(np.square(value - expected).mean())
    json.dumps(summary, allow_nan=False)


def test_b_selection_and_stream_do_not_depend_on_test_outcomes(config, b_task):
    arrays = small_arrays(b_task)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["grid_clean"] = 20 * changed["grid_clean"] + 50
    changed["grid_noisy"] = -99 * changed["grid_noisy"]
    bundles = []
    for data in (arrays, changed):
        model = make_b_model("learned_vm", initialization_bundle(config, b_task), device="cpu")
        bundles.append(train_b(model, data, learning_rate=.01, batch_seed=29, max_updates=3,
                               validation_every=1, patience_evaluations=5, batch_size=4, device="cpu"))
    first, second = bundles
    assert first["summary"]["status"] == second["summary"]["status"] == "completed"
    assert first["summary"]["history"] == second["summary"]["history"]
    assert first["summary"]["update_history"] == second["summary"]["update_history"]
    assert first["summary"]["primary_error"] != second["summary"]["primary_error"]
    for name, value in first["artifacts"]["selected_state"].items():
        assert torch.equal(value, second["artifacts"]["selected_state"][name])
    assert torch.equal(first["artifacts"]["batch_indices"], second["artifacts"]["batch_indices"])


def test_b_explicit_batches_and_zero_budget(config, b_task):
    model = make_b_model("raw_angle_mlp", initialization_bundle(config, b_task), device="cpu")
    bundle = train_b(model, small_arrays(b_task), learning_rate=.01,
                     batch_indices=np.empty((0, 4), dtype=np.int64), max_updates=0, batch_size=4, device="cpu")
    assert bundle["summary"]["status"] == "completed", bundle["summary"].get("failure")
    assert bundle["summary"]["selected_step"] == 0
    assert bundle["summary"]["updates_completed"] == 0
    invalid = train_b(model, small_arrays(b_task), learning_rate=.01,
                      batch_indices=np.full((1, 4), 100), max_updates=1, batch_size=4, device="cpu")
    assert invalid["summary"]["status"] == "failed"
    assert "indices" in invalid["summary"]["failure"]


def test_b_nonfinite_failure_does_not_restore_favorable_checkpoint(config, b_task):
    model = make_b_model("learned_geodesic", initialization_bundle(config, b_task), device="cpu")
    model.readout.weight.register_hook(lambda g: g * float("nan"))
    bundle = train_b(model, small_arrays(b_task), learning_rate=.01, batch_seed=2, max_updates=3, device="cpu")
    assert bundle["summary"]["status"] == "numerical_failure"
    assert bundle["summary"]["updates_completed"] == 0
    assert "selected_state" in bundle["artifacts"]
    assert "metrics" not in bundle["summary"]


def test_b_small_improvements_select_checkpoint_but_do_not_reset_patience(config, b_task, monkeypatch):
    import src.methodology_training as training

    model = make_b_model("sincos_mlp", initialization_bundle(config, b_task), device="cpu")
    arrays = small_arrays(b_task)
    arrays["validation_y"][:] = 0
    validation_x = torch.from_numpy(arrays["validation_x"])
    original_predict, calls = training._predict, 0
    scores = [1., 1. - .4e-6, 1. - .8e-6]

    def controlled_validation(candidate, x):
        nonlocal calls
        if torch.equal(x, validation_x):
            score = scores[min(calls, 2)]
            calls += 1
            return torch.full_like(x, math.sqrt(score))
        return original_predict(candidate, x)

    monkeypatch.setattr(training, "_predict", controlled_validation)
    bundle = train_b(model, arrays, learning_rate=.01, batch_seed=1, max_updates=4,
                     validation_every=1, patience_evaluations=2, min_delta=1e-6, device="cpu")
    assert bundle["summary"]["status"] == "completed", bundle["summary"].get("failure")
    assert bundle["summary"]["selected_step"] == 2
    assert bundle["summary"]["updates_completed"] == 2
    assert [row["selected"] for row in bundle["summary"]["history"]] == [True, True, True]
    assert [row["significant_improvement"] for row in bundle["summary"]["history"]] == [True, False, False]


@pytest.mark.parametrize("stage", ["A", "B"])
def test_interruption_returns_partial_history_and_states(config, a_task, b_task, monkeypatch, stage):
    import src.methodology_training as training

    def interrupted_step(*args, **kwargs):
        raise KeyboardInterrupt("unit-test interruption")

    monkeypatch.setattr(training, "_gradient_step", interrupted_step)
    if stage == "A":
        model = make_a_model("polar", 1., 0., device="cpu")
        result = train_a(model, small_arrays(a_task), optimizer_name="Adam", learning_rate=.01,
                         penalty=0, updates=2, device="cpu")
    else:
        model = make_b_model("sincos_mlp", initialization_bundle(config, b_task), device="cpu")
        result = train_b(model, small_arrays(b_task), learning_rate=.01, batch_seed=2, max_updates=2, device="cpu")
    assert result["summary"]["status"] == "interrupted"
    assert result["summary"]["updates_completed"] == 0
    assert result["summary"]["history"][0]["step"] == 0
    assert "initial_state" in result["artifacts"] and "final_state" in result["artifacts"]


@pytest.mark.parametrize("model_id", ["fourier_ridge", "fixed_vm_ridge"])
def test_ridge_all_solves_float64_selection_and_geometry(b_task, model_id):
    arrays = small_arrays(b_task)
    bundle = fit_ridge(model_id, arrays)
    summary, artifacts = bundle["summary"], bundle["artifacts"]
    assert summary["status"] == "completed", summary.get("failure")
    assert summary["solves_completed"] == 4
    assert len(artifacts["candidate_states"]) == 4
    best = math.inf
    selected = None
    features = FixedFeaturesV1(model_id)(torch.from_numpy(arrays["validation_x"])).numpy()
    for row in summary["history"]:
        coefficients = artifacts["candidate_states"][str(row["lambda"])].numpy()
        assert coefficients.dtype == np.float64
        assert row["alpha"] == len(arrays["train_x"]) * row["lambda"]
        score = np.square(features @ coefficients[:-1] + coefficients[-1] - arrays["validation_y"]).mean()
        assert row["validation_mse"] == pytest.approx(score, abs=1e-14)
        if score < best - 1e-12:
            best, selected = score, row["lambda"]
    assert summary["selected_lambda"] == selected
    assert summary["boundary_audit"]["rotation_output_max_abs"] < 1e-10
    assert summary["boundary_audit"]["two_pi_output_max_abs"] < 1e-10
    json.dumps(summary, allow_nan=False)


def test_ridge_bias_unpenalized_and_smaller_lambda_tie_break(b_task):
    arrays = small_arrays(b_task)
    arrays["train_y"][:] = 2
    arrays["validation_y"][:] = 2
    bundle = fit_ridge("fixed_vm_ridge", arrays, lambdas=(1., .01))
    assert bundle["summary"]["status"] == "completed"
    assert bundle["summary"]["selected_lambda"] == .01
    for coefficients in bundle["artifacts"]["candidate_states"].values():
        np.testing.assert_allclose(coefficients.numpy()[:-1], 0, atol=1e-10)
        np.testing.assert_allclose(coefficients.numpy()[-1], 2, atol=1e-10)


def test_ridge_tie_is_relative_to_global_minimum(b_task, monkeypatch):
    import src.methodology_training as training

    arrays = small_arrays(b_task)
    arrays["validation_y"][:] = 0
    scores = {0.: 1., .0001: 1. - .6e-12, .01: 1. - 1.2e-12}

    def prescribed_coefficients(features, target, penalty):
        coefficients = np.zeros((49, 1), dtype=np.float64)
        coefficients[-1] = math.sqrt(scores[penalty])
        return coefficients

    monkeypatch.setattr(training, "_linear_solve", prescribed_coefficients)
    result = fit_ridge("fourier_ridge", arrays, lambdas=(0., .0001, .01))
    assert result["summary"]["status"] == "completed", result["summary"].get("failure")
    assert result["summary"]["selected_lambda"] == .0001


def test_ridge_timeout_and_nonfinite_arrays_are_not_successful(b_task):
    arrays = small_arrays(b_task)
    timeout = fit_ridge("fourier_ridge", arrays, deadline=time.perf_counter() - 1)
    assert timeout["summary"]["status"] == "timeout"
    assert timeout["summary"]["solves_completed"] == 0
    arrays["train_x"][0, 0] = np.nan
    failed = fit_ridge("fourier_ridge", arrays)
    assert failed["summary"]["status"] == "numerical_failure"
    assert failed["summary"]["solves_completed"] == 0


def test_metrics_accumulate_float64_and_boundary_is_declared():
    x = torch.tensor([[-math.pi], [0], [math.pi]], dtype=torch.float64)
    clean = torch.tensor([[1], [0], [-1]], dtype=torch.float64)
    pred = torch.tensor([[2], [0], [-2]], dtype=torch.float32)
    metrics = regression_metrics(pred, clean, clean, x)
    assert metrics["clean_mse"] == pytest.approx(2 / 3)
    assert metrics["normalized_clean_mse"] == 1
    assert metrics["boundary_mse"] == 1
    assert metrics["boundary_points"] == 2
