"""Tests of units, delayed information, exact enclosure and action-path identity."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.decision_forecasts_v1 import CalendarSourceV1
from src.decision_plant_v1 import Plant
from src.source_interval_bridge_v1 import (
    FlowPlan, ReceivedState, SourceEnvelope, calibrate_source_envelope,
    classify_limit, project_source_envelope,
)
from scripts.demo_source_interval_bridge_v1 import example

STEP = 900_000_000_000


def fixture():
    """Construct a lagged state and a nine-interval source envelope."""
    point = np.full(9, 60000.)
    source = SourceEnvelope(100*STEP, 99*STEP, STEP, point, point-3000, point+3000, "fixture", 98*STEP, 20)
    state = ReceivedState(99*STEP, 100*STEP, 590., 610.)
    plan = FlowPlan("low", np.full(9, 45.), np.full(3, 45.))
    return source, state, plan, Plant()


@pytest.mark.parametrize("flow", [0., 1e-8, 45., 450.])
def test_bounds_enclose_independently_integrated_dynamics(flow):
    source, state, _, plant = fixture()
    plan = FlowPlan("specified", np.full(9, flow), np.full(3, flow))
    result = project_source_envelope(source, state, plan, plant)
    rng = np.random.default_rng(23)
    for _ in range(5):
        c = rng.uniform(state.lower_ppm, state.upper_ppm)
        true_source = rng.uniform(source.lower, source.upper)
        truth = []
        for s in true_source:
            solved = solve_ivp(lambda t, y: (s - flow*(y-plant.outdoor_ppm))/plant.volume_m3,
                               (0, plant.dt_hours), [c], rtol=1e-10, atol=1e-10)
            c = solved.y[0, -1]
            truth.append(c)
        assert np.all(np.array(truth)[1:] >= result.lower_ppm-1e-8)
        assert np.all(np.array(truth)[1:] <= result.upper_ppm+1e-8)
    assert np.all(result.endpoint_ns > source.issue_ns)
    assert len(result.endpoint_ns) == 8


def test_action_paths_have_distinct_envelopes_but_preserve_queue():
    source, state, low, plant = fixture()
    high = FlowPlan("high", np.r_[np.full(3, 45.), np.full(6, 450.)], low.locked_prefix_m3_h)
    a, b = [project_source_envelope(source, state, p, plant) for p in (low, high)]
    assert a.plan_id != b.plan_id
    np.testing.assert_array_equal(a.upper_ppm[:2], b.upper_ppm[:2])
    assert np.all(a.upper_ppm[2:] > b.upper_ppm[2:])
    broken = replace(high, flow_m3_h=np.full(9, 450.))
    with pytest.raises(ValueError, match="queued"):
        project_source_envelope(source, state, broken, plant)


@pytest.mark.parametrize("change", ["future_state", "wrong_start", "wrong_step", "short_prefix", "negative_flow"])
def test_temporal_and_actuator_contract_is_enforced(change):
    source, state, plan, plant = fixture()
    if change == "future_state":
        state = replace(state, arrival_ns=101*STEP)
    elif change == "wrong_start":
        state = replace(state, event_ns=98*STEP)
    elif change == "wrong_step":
        plant = replace(plant, dt_hours=.5)
    elif change == "short_prefix":
        plan = replace(plan, locked_prefix_m3_h=np.array([45.]))
    else:
        plan = replace(plan, flow_m3_h=np.full(9, -1.))
    with pytest.raises(ValueError):
        project_source_envelope(source, state, plan, plant)


@pytest.mark.parametrize("target,units", [("concentration", "ppm"), ("source_generation", "ppm"), ("CO2", "ppm*m3/h")])
def test_n05_concentration_intervals_are_rejected(target, units):
    source, _, _, _ = fixture()
    with pytest.raises(ValueError, match="incompatible"):
        replace(source, target=target, units=units)


def calibration_fixture():
    """Twenty historic paths, including the one-step sensor gap."""
    ci = np.arange(20, dtype=np.int64) * 12 * STEP
    ce = ci[:, None] + np.arange(9) * STEP
    cp = np.full((20, 9), 60000.)
    return dict(point=cp[0], issue_ns=1000*STEP, start_ns=999*STEP, step_ns=STEP,
                model_id="fixture", calibration_prediction=cp, calibration_truth=cp + np.arange(20)[:, None],
                calibration_issue_ns=ci, calibration_end_ns=ce, calibration_arrival_ns=ce+STEP)


def test_calibration_uses_whole_arrived_paths_and_ignores_future_truth():
    data = calibration_fixture()
    data["calibration_arrival_ns"][-1, -1] = 2000*STEP
    data["calibration_truth"][-1] = np.nan
    band = calibrate_source_envelope(**data)
    assert band.calibration_paths == 19
    # k=ceil(20*.9)=18; scores 0..18 -> radius 17.
    np.testing.assert_array_equal(band.upper-band.point, 17.)
    data["calibration_truth"][-1] = 1e20
    changed = calibrate_source_envelope(**data)
    np.testing.assert_array_equal(changed.upper, band.upper)


def test_sparse_calibration_remains_unbounded_through_projection():
    data = calibration_fixture()
    data["calibration_arrival_ns"][:] = np.iinfo(np.int64).max
    data["calibration_truth"][:] = np.nan
    band = calibrate_source_envelope(**data)
    assert band.calibration_paths == 0
    assert np.isinf(band.upper).all()
    state = ReceivedState(band.start_ns, band.issue_ns, 590, 610)
    plan = FlowPlan("base", np.full(9, 45.), np.full(3, 45.))
    result = project_source_envelope(band, state, plan, Plant())
    assert not result.bounded
    assert classify_limit(result, 900) == "unbounded: fallback required"


@pytest.mark.parametrize("bad", ["offset", "already_observed", "future_cutoff", "negative_source", "mutable_alias"])
def test_calibration_provenance_and_source_validation(bad):
    data = calibration_fixture()
    if bad == "offset":
        data["calibration_end_ns"] += STEP
    elif bad == "already_observed":
        data["calibration_arrival_ns"][0, 0] = data["calibration_issue_ns"][0]
    elif bad == "negative_source":
        data["calibration_truth"][0, 0] = -1
    elif bad == "future_cutoff":
        source, _, _, _ = fixture()
        with pytest.raises(ValueError):
            replace(source, calibration_cutoff_ns=source.issue_ns+1)
        return
    else:
        source, _, _, _ = fixture()
        original = source.point.copy()
        with pytest.raises(ValueError):
            source.point[0] = 0
        np.testing.assert_array_equal(source.point, original)
        return
    with pytest.raises(ValueError):
        calibrate_source_envelope(**data)


@pytest.mark.parametrize("mode", ["seasonal", "vm", "fourier"])
def test_unchanged_n06_calendar_source_outputs_fit_contract(mode):
    hours = np.arange(96*3)/4
    model = CalendarSourceV1(mode, {"ridge": .001, "features": 16, "kappa": 16})
    model.fit(hours, 50000+20000*np.cos(hours*2*np.pi/24))
    source, state, plan, plant = fixture()
    prediction = model.predict(np.arange(9)/4)
    band = replace(source, point=prediction, lower=np.maximum(0, prediction-3000), upper=prediction+3000,
                   model_id=f"N06-{mode}-fixture")
    assert project_source_envelope(band, state, plan, plant).bounded


def test_serializable_example_has_two_verified_action_paths():
    result = example()
    assert all(p["contained"] for p in result["plans"])
    assert result["plans"][0]["limit_signal"] != result["plans"][1]["limit_signal"]
