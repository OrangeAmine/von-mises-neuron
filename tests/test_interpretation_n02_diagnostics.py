"""Equation and intervention checks on disposable VM-N02 fixtures."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.interpretation_n02_cases import make_peak_unit, make_proxy_unit, fit_proxy
from src.interpretation_n02_diagnostics import (
    cartesian_model, circular_distance, contribution, derivative, equivalent_states,
    fwhm, match_components, peak_diagnostics, proxy_diagnostics, proxy_response, response,
)


@pytest.fixture
def config():
    return json.loads(Path("configs/vm_n02_interpretation_v1.json").read_text())


@pytest.fixture
def state():
    return {"centers": torch.tensor([0.4, 3.2], dtype=torch.float64),
            "kappa": torch.tensor([3.0, 6.0], dtype=torch.float64),
            "weights": torch.tensor([1.2, -0.6], dtype=torch.float64),
            "bias": torch.tensor(0.2, dtype=torch.float64)}


def test_derivative_matches_finite_difference_and_autograd(state):
    x = torch.linspace(-2, 8, 101, dtype=torch.float64)
    actual = derivative(state, x)
    finite = (response(state, x + 1e-6) - response(state, x - 1e-6)) / 2e-6
    torch.testing.assert_close(actual, finite, atol=1e-9, rtol=1e-7)
    from src.locality_models import chordal_features
    x.requires_grad_(True)
    y = chordal_features(x, state["centers"], state["kappa"]) @ state["weights"] + state["bias"]
    gradient, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(actual, gradient, atol=1e-12, rtol=1e-12)


def test_equivalent_functions_allow_incompatible_component_ablations(state):
    x = torch.linspace(0, 2 * math.pi, 256, dtype=torch.float64)
    for altered in equivalent_states(state).values():
        torch.testing.assert_close(response(state, x), response(altered, x), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(derivative(state, x), derivative(altered, x), atol=1e-12, rtol=1e-12)
    split = equivalent_states(state)["split"]
    torch.testing.assert_close(contribution(state, x)[:, 0], 2 * contribution(split, x)[:, 0])
    cancel = equivalent_states(state)["cancelling_pair"]
    assert torch.linalg.vector_norm(contribution(cancel, x)[:, -2]) > 1


def test_same_four_kernel_size_permits_arbitrary_cancelling_geometry(state):
    """Two real contributions plus a cancelling pair have nonunique geometry."""
    x = torch.linspace(0, 2 * math.pi, 256, dtype=torch.float64)
    first = equivalent_states(state)["cancelling_pair"]
    second = copy.deepcopy(first)
    second["centers"][-2:] = 2.1
    second["kappa"][-2:] = 19
    assert len(first["centers"]) == len(second["centers"]) == 4
    torch.testing.assert_close(response(first, x), response(second, x), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(derivative(first, x), derivative(second, x), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(contribution(first, x)[:, -2], contribution(second, x)[:, -2])


def test_match_permutation_and_phase_wrap(state):
    x = torch.linspace(0, 2 * math.pi, 512, dtype=torch.float64)
    permuted = equivalent_states(state)["permutation"]
    permuted["centers"] += 2 * math.pi
    matched = match_components(state, permuted, x, 1.0, .02)
    assert matched["cols"] == [1, 0]
    assert matched["center_distance"] < 1e-12
    assert matched["log_kappa_distance"] == 0
    assert matched["unmatched_active_count"] == 0
    assert abs(matched["unmatched_active_rms_mass"]) < 1e-12


def test_zero_or_flat_components_have_no_assigned_phase(state):
    x = torch.linspace(0, 2 * math.pi, 512, dtype=torch.float64)
    weak = copy.deepcopy(state)
    weak["weights"][:] = 0
    matched = match_components(weak, weak, x, 1.0, .02)
    assert matched["active_pair_count"] == 0
    assert matched["center_distance"] is None
    weak["weights"][:] = 1
    weak["kappa"][:] = 1e-9
    matched = match_components(weak, weak, x, 1.0, .02)
    assert matched["active_pair_count"] == 0
    assert matched["unmatched_active_count"] == 4


def test_fwhm_and_center_sign():
    assert fwhm([.01])[0] is None
    assert fwhm([9])[0] < fwhm([3])[0]
    assert float(circular_distance(0.1, 2 * math.pi + .1)) < 1e-14
    from src.circular_layers_v2 import PolarCircularLinearV2
    model = PolarCircularLinearV2(1, 1, dtype=torch.float64)
    with torch.no_grad():
        model.phase.fill_(.7)
    assert float(circular_distance(model.preferred_centers(), -.7)) < 1e-14
    assert float(model(torch.tensor([[-.7]], dtype=torch.float64))) > float(model(torch.tensor([[.7]], dtype=torch.float64)))


def test_peak_complete_identity_controls(config):
    unit = make_peak_unit(config, "fixture", 0, "isolated")
    states = []
    for seed in config["initializations"]:
        s = {k: v.clone() for k, v in unit["truth"].items()}
        s.update({"parameter_count": 4, "seconds": 0.0, "selected_step": 0, "init_seed": seed})
        states.append(s)
    metrics, details, arrays = peak_diagnostics(states, unit, config)
    assert metrics["recovered"] == 1
    assert metrics["curve_pair_nrmse"] == 0
    assert metrics["primary_deletion_delta_nmse"] > 1
    for key, value in metrics.items():
        if key.endswith("identity_error") or key.endswith("derivative_error"):
            assert value < 1e-10, key
    assert metrics["split_half_deletion_delta_nmse"] == pytest.approx(metrics["unsplit_first_deletion_delta_nmse"] / 4)
    assert arrays["curves"].shape == (3, config["samples"]["curve"])
    assert details["fits"][0]["validation_selected_component"] == 0


def test_proxy_target_is_not_recorded_clock(config):
    unit = make_proxy_unit(config, "fixture", 0)
    state = fit_proxy(unit, config)
    metrics, details, arrays = proxy_diagnostics(state, unit, config)
    assert metrics["generator_clock_direct_effect"] < 1e-12
    assert metrics["aligned_delete_delta_nmse"] > .1
    assert metrics["shifted_phase_plus_delta_nmse"] < -.1
    for key, value in metrics.items():
        if key.endswith("identity_error") or key.endswith("derivative_error"):
            assert value < 1e-10, key
    x = unit["evaluation"]["aligned"]["x"]
    model = cartesian_model(state["weights"], state["bias"])
    v = x.clone().requires_grad_(True)
    grad, = torch.autograd.grad(model(v).sum(), v)
    w = state["weights"]
    torch.testing.assert_close(grad[:, 1], -w[1] * x[:, 1].sin() + w[3] * x[:, 1].cos())
    assert arrays["response_surface"].shape == (32, 32)
    assert details["clock_phase_eligible"]


def test_proxy_rotation_is_not_input_rotation(config):
    unit = make_proxy_unit(config, "fixture", 1)
    state = fit_proxy(unit, config)
    x = unit["evaluation"]["aligned"]["x"]
    perturbed = x.clone()
    perturbed[:, 1] += .7
    assert not torch.allclose(proxy_response(state, x), proxy_response(state, perturbed))


@pytest.mark.parametrize("name", ["centers", "kappa", "weights"])
def test_diagnostics_do_not_mutate_fitted_state(state, name):
    old = state[name].clone()
    equivalent_states(state)
    response(state, torch.linspace(0, 3, 10, dtype=torch.float64))
    torch.testing.assert_close(state[name], old)
