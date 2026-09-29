"""Independent physical, causal, pairing and forecast-target checks for N06."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import torch

from src.decision_plant_v1 import Plant, exact_update, interval_exposure, make_room, rollout_metrics
from src.decision_forecasts_v1 import CalendarSourceV1
from src.decision_policies_v1 import project_concentration, rollout, objective

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/vm_n06_v1.json').read_text())


@pytest.mark.parametrize('flow', [0, 1e-8, 45, 450])
@pytest.mark.parametrize('source', [0, 216000])
def test_exact_mass_balance_against_independent_ode(flow, source):
    p = Plant()
    numeric = solve_ivp(lambda t, c: [source / 150 + flow / 150 * (420 - c[0])],
                        [0, .25], [700], rtol=1e-11, atol=1e-11).y[0, -1]
    assert exact_update(700, source, flow, p) == pytest.approx(numeric, abs=2e-8)


def test_zero_flow_and_airflow_dependent_equilibrium():
    p = Plant()
    assert exact_update(600, 216000, 0, p) == 960
    for flow in (45, 450):
        eq = 420 + 216000 / flow
        assert exact_update(eq, 216000, flow, p) == pytest.approx(eq)
    full = exact_update(600, 216000, 450, p)
    half = exact_update(600, 216000, 450, p, .125)
    assert exact_update(half, 216000, 450, p, .125) == pytest.approx(full)


def test_exact_exposure_crossing_and_linear_limit():
    p = replace(Plant(), dt_hours=1)
    hours, severity = interval_exposure(1380, 0, 150, p)
    assert hours == pytest.approx(np.log(2))
    assert severity == pytest.approx(480 - 480 * np.log(2))
    assert interval_exposure(600, 90000, 0, p) == pytest.approx((.5, 75))
    assert interval_exposure(900, 0, 0, p) == (0, 0)


@pytest.mark.parametrize('delay', [0, 1, 4])
def test_causal_delay_applies_to_interval_not_left_state(delay):
    p = replace(Plant(), initial_ppm=1000, actuator_delay_steps=delay)
    result = rollout(np.zeros(12), np.zeros(13), p, 0)
    assert result['commands'][0] == 1
    assert np.all(result['actions'][:delay] == 0)
    assert result['actions'][delay] == 1
    assert result['states'][0] == 1000
    expected = exact_update(1000, 0, p.base_flow_m3h + (405 if delay == 0 else 0), p)
    assert result['states'][1] == pytest.approx(expected)


def test_unavailable_future_mutations_do_not_change_earlier_decisions():
    p = Plant()
    source, noise = np.full(30, 216000.), np.zeros(31)
    forecast = np.full(38, 180000.)
    original = rollout(source, noise, p, 100, forecast)
    noise[12:] = 10000
    source[12:] = 400000
    changed = rollout(source, noise, p, 100, forecast)
    np.testing.assert_array_equal(original['commands'][:13], changed['commands'][:13])
    np.testing.assert_array_equal(original['states'][:13], changed['states'][:13])
    assert original['observation_index'][12] == 11
    forecast[20:] *= 4
    future = rollout(source, noise, p, 100, forecast)
    np.testing.assert_array_equal(changed['commands'][:13], future['commands'][:13])


def test_observations_use_each_rollouts_own_controlled_history():
    source, noise, p = np.full(40, 216000.), np.zeros(41), Plant()
    a = rollout(source, noise, p, 0)
    b = rollout(source, noise, p, 200, np.full(48, 216000.))
    assert not np.array_equal(a['states'], b['states'])
    for run in (a, b):
        np.testing.assert_allclose(run['observations'][1:], run['states'][:-2])
        assert np.all((run['actions'] == 0) | (run['actions'] == 1))


def test_equal_policy_zero_horizon_and_zero_capacity_identity():
    source, noise = np.full(24, 200000.), np.zeros(25)
    a = rollout(source, noise, Plant(), 100)
    b = rollout(source, noise, Plant(), 100, np.full(24, 300000.), horizon=0)
    for key in ('states', 'commands', 'actions', 'observations'):
        np.testing.assert_array_equal(a[key], b[key])
    p = replace(Plant(), extra_flow_m3h=0)
    a = rollout(source, noise, p, 0)
    b = rollout(source, noise, p, 200, np.full(32, 200000.))
    np.testing.assert_array_equal(a['states'], b['states'])
    assert a['metrics']['additional_air_m3'] == b['metrics']['additional_air_m3'] == 0


def test_action_conditioned_and_no_action_forecast_targets_differ():
    p = Plant()
    source = np.full(8, 216000.)
    base = project_concentration(600, source, np.full(8, 45.), p)
    active = project_concentration(600, source, np.full(8, 450.), p)
    assert np.all(active < base)
    # The zero-command rollout recovers the counterfactual exactly.
    uncontrolled = rollout(source, np.zeros(9), p, -10000)
    np.testing.assert_allclose(base, uncontrolled['states'][1:])


def test_common_scenario_pairing_and_separate_training_stream():
    a = make_room(17, CONFIG)
    b = make_room(17, CONFIG, 3)
    np.testing.assert_array_equal(a['train_source'], b['train_source'])
    np.testing.assert_array_equal(a['noise'], b['noise'])
    np.testing.assert_array_equal(np.roll(a['source'], 12), b['source'])
    assert not np.array_equal(a['noise'], make_room(18, CONFIG)['noise'])


def test_metric_units_activation_switches_and_objective():
    p = Plant()
    actions = np.array([1, 1, 0, 1])
    source = np.zeros(4)
    states = np.r_[600, project_concentration(600, source, 45 + actions * 405, p)]
    values = rollout_metrics(states, source, actions, p)
    assert values['on_hours'] == .75
    assert values['additional_air_m3'] == 303.75
    assert values['total_air_m3'] == 348.75
    assert values['activations'] == 2 and values['switches'] == 3
    assert objective(values, 2) == .6075


@pytest.mark.parametrize('mode', ['seasonal', 'vm', 'fourier'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_only_source_forecasts_shape_finite_periodicity(mode, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    hours = np.arange(96 * 3) * .25
    labels = 100 + 20 * np.sin(hours * np.pi / 12)
    model = CalendarSourceV1(mode, CONFIG['forecast'], device).fit(hours, labels)
    times = np.array([[0., 1, 2], [23.75, 24, 48]])
    before = model.predict(times)
    assert before.shape == (2, 3) and np.isfinite(before).all()
    np.testing.assert_allclose(before, model.predict(times + 24), atol=1e-10)
    labels[:] = 100000
    np.testing.assert_array_equal(before, model.predict(times))
    assert model.record()['fitted_coefficients'] == (96 if mode == 'seasonal' else 17)
    if model.basis is not None:
        angles = torch.tensor([.1, .2], dtype=torch.float64, device=device, requires_grad=True)
        model.basis(angles).sum().backward()
        assert torch.isfinite(angles.grad).all() and torch.any(angles.grad != 0)


def test_cpu_cuda_calendar_parity():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    hours = np.arange(96) / 4
    labels = 100 + np.cos(hours)
    for mode in ('vm', 'fourier'):
        cpu = CalendarSourceV1(mode, CONFIG['forecast'], 'cpu').fit(hours, labels)
        gpu = CalendarSourceV1(mode, CONFIG['forecast'], 'cuda').fit(hours, labels)
        np.testing.assert_allclose(cpu.predict(hours), gpu.predict(hours), atol=1e-9)


@pytest.mark.parametrize('kwargs', [{'volume_m3': 0}, {'extra_flow_m3h': -1}, {'actuator_delay_steps': .5}])
def test_invalid_physical_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        Plant(**kwargs)


def test_development_selection_depends_on_objective_not_evaluation():
    from scripts.run_vm_n06_v1 import select
    config = {**CONFIG, 'arms': ['vm'], 'margins_ppm': [0, 100]}
    rows = [{'arm': 'vm', 'margin_ppm': 0, 'exceedance_hours': 1., 'additional_air_m3': 0},
            {'arm': 'vm', 'margin_ppm': 100, 'exceedance_hours': 0., 'additional_air_m3': 1000}]
    choices = select(rows, config)
    assert choices['exceedance_priority']['vm']['margin_ppm'] == 100
    assert choices['effort_priority']['vm']['margin_ppm'] == 0


def test_authored_timing_equal_errors_and_null_capacity():
    from scripts.analyze_vm_n06_v1 import teaching
    data = teaching(CONFIG)
    a, b = data['equal_error'].values()
    assert a['source_mae'] == b['source_mae']
    assert a['source_rmse'] == b['source_rmse']
    assert a['metrics']['exceedance_hours'] > b['metrics']['exceedance_hours']
    # No mistaken matched-effort claim for this authored example.
    assert a['metrics']['additional_air_m3'] != b['metrics']['additional_air_m3']
    null = data['constraints']['zero']['policies']
    np.testing.assert_array_equal(null['reactive']['states'], null['oracle']['states'])


def test_paired_analysis_resamples_rooms_and_preserves_failed_regimes():
    from scripts.analyze_vm_n06_v1 import paired_analysis
    config = {**CONFIG, 'evaluation_seeds': [1, 2], 'arms': ['vm', 'reactive'],
              'regimes': {'a': {}, 'b': {}}, 'practical_effect': {**CONFIG['practical_effect'], 'comparators': ['reactive']}}
    selection = {name: {arm: {'margin_ppm': 0} for arm in config['arms']} for name in config['objectives']}
    rows = [{'seed': seed, 'arm': arm, 'regime': regime, 'margin_ppm': 0,
             'exceedance_hours': (2 if arm == 'vm' and regime == 'b' else 1), 'additional_air_m3': 0}
            for seed in [1, 2] for arm in config['arms'] for regime in config['regimes']]
    result = paired_analysis(rows, selection, config)
    contrast = result['objectives']['exceedance_priority']['contrasts']['reactive']
    assert contrast['paired_95_interval'] == [-.5, -.5]
    assert contrast['paired_room_difference_vm_minus_control'] == [.5, .5]
    assert not result['primary_gate_passed']
