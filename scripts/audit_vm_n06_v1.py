"""Read-only saved-rollout checks using independent equations and quadrature."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def audit(study: Path, output: Path) -> None:
    """Check complete saved state/action ordering and sampled exact metrics."""
    began = time.perf_counter()
    record = json.loads((study / 'evaluation/summary.json').read_text())
    config = json.loads((study / 'source_lock.json').read_text())['config']
    assert record['status'] == 'complete' and len(record['rows']) == 360
    max_state_error, quadrature_cases, max_severity_error = 0., 0, 0.
    for row in record['rows']:
        path = study / 'evaluation' / row['artifact']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        a = np.load(path)
        case = np.load(study / 'evaluation' / f"{row['seed']}_{row['regime']}_scenario.npz")
        p = {**config['plant'], **config['regimes'][row['regime']]}
        states, source = a['states'], case['source']
        duty, commands = a['actions'], a['commands']
        dt, volume = p['dt_hours'], p['volume_m3']
        delay = p['actuator_delay_steps']
        expected_duty = commands if delay == 0 else np.r_[np.zeros(delay), commands[:-delay]]
        np.testing.assert_array_equal(duty, expected_duty)
        assert np.isin(duty, [0, 1]).all()
        flow = p['base_flow_m3h'] + duty * p['extra_flow_m3h']
        rate, eq = flow / volume, p['outdoor_ppm'] + source / flow
        expected_state = eq + (states[:-1] - eq) * np.exp(-rate * dt)
        max_state_error = max(max_state_error, float(np.max(np.abs(states[1:] - expected_state))))
        np.testing.assert_allclose(states[1:], expected_state, atol=2e-10, rtol=0)
        sensor_delay = p['sensor_delay_steps']
        indices = np.arange(len(source)) - sensor_delay
        expected_obs = np.array([p['initial_ppm'] if k < 0 else states[k] + case['noise'][k] for k in indices])
        np.testing.assert_array_equal(a['observations'], expected_obs)
        assert row['additional_air_m3'] == float(duty.sum() * dt * p['extra_flow_m3h'])
        assert row['activations'] == int(np.sum(duty > np.r_[0, duty[:-1]]))
        assert row['switches'] == int(np.sum(duty != np.r_[0, duty[:-1]]))
        if row['seed'] == config['evaluation_seeds'][0] and row['margin_ppm'] == 200:
            hours, severity = 0., 0.
            for j in range(len(source)):
                def excess(t: float) -> float:
                    return float(eq[j] + (states[j] - eq[j]) * np.exp(-rate[j] * t) - p['limit_ppm'])
                left, right = excess(0), excess(dt)
                if left * right < 0:
                    crossing = brentq(excess, 0, dt, xtol=1e-13)
                    low, high = (0, crossing) if left > 0 else (crossing, dt)
                elif max(left, right) > 0:
                    low, high = 0, dt
                else:
                    continue
                hours += high - low
                severity += quad(excess, low, high, epsabs=1e-9)[0]
            assert abs(hours - row['exceedance_hours']) < 1e-9
            max_severity_error = max(max_severity_error, abs(severity - row['severity_ppm_hours']))
            assert abs(severity - row['severity_ppm_hours']) < 1e-7
            quadrature_cases += 1
    receipt = {'status': 'passed', 'rollouts_checked': len(record['rows']),
               'intervals_checked': len(record['rows']) * len(source),
               'quadrature_rollouts': quadrature_cases, 'max_state_error_ppm': max_state_error,
               'max_quadrature_severity_error_ppm_hours': max_severity_error,
               'runtime_seconds': time.perf_counter() - began,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'Independent mass-balance formula, command delay, own-history observation, capacity, effort, starts/switches for all saved evaluation rollouts; root finding and numerical exposure quadrature for first room, all arms/regimes at margin 200. No refits or generator reruns.'}
    if output.exists():
        raise FileExistsError('Preserve the existing audit')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.study_dir, args.output)
