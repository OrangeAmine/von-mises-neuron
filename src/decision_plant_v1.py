"""N06 synthetic controlled mass balance, paired exogenous streams and metrics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Plant:
    """Known simulation constants; airflow in m3/h and all times in hours."""

    volume_m3: float = 150.0
    outdoor_ppm: float = 420.0
    base_flow_m3h: float = 45.0
    extra_flow_m3h: float = 405.0
    dt_hours: float = 0.25
    sensor_delay_steps: int = 1
    actuator_delay_steps: int = 2
    noise_ppm: float = 10.0
    initial_ppm: float = 600.0
    limit_ppm: float = 900.0

    def __post_init__(self) -> None:
        """Reject nonphysical constants and ambiguous delay counts."""
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Plant values must be finite')
        if min(self.volume_m3, self.dt_hours) <= 0:
            raise ValueError('Volume and interval must be positive')
        if min(self.base_flow_m3h, self.extra_flow_m3h, self.noise_ppm) < 0:
            raise ValueError('Flow and noise must be nonnegative')
        for value in (self.sensor_delay_steps, self.actuator_delay_steps):
            if type(value) is not int or value < 0:
                raise ValueError('Delays must be nonnegative integers')


def exact_update(c: float, source: float, flow: float, p: Plant,
                 duration: float | None = None) -> float:
    """Integrate constant source/flow exactly, including zero flow.

    Args:
        c: Left-boundary concentration, ppm.
        source: External generation, ppm*m3/h, independent of action.
        flow: Total balanced inflow/outflow, m3/h.
        p: Known room constants.
        duration: Optional interval length in hours.

    Returns:
        Right-boundary concentration in ppm.
    """
    dt = p.dt_hours if duration is None else duration
    if not all(math.isfinite(v) for v in (c, source, flow, dt)) or min(source, flow, dt) < 0:
        raise ValueError('Inputs must be finite with nonnegative source, flow and time')
    rate = flow / p.volume_m3
    if rate == 0:
        return c + source * dt / p.volume_m3
    decay = math.exp(-rate * dt)
    gain = -math.expm1(-rate * dt) / rate
    return p.outdoor_ppm + (c - p.outdoor_ppm) * decay + source / p.volume_m3 * gain


def interval_exposure(c: float, source: float, flow: float, p: Plant) -> tuple[float, float]:
    """Return exact above-limit hours and severity (ppm*h) for one interval."""
    dt, limit = p.dt_hours, p.limit_ppm
    end = exact_update(c, source, flow, p)
    if max(c, end) <= limit:
        return 0.0, 0.0
    rate, drive = flow / p.volume_m3, source / p.volume_m3
    lo, hi = 0.0, dt
    if min(c, end) < limit:
        if rate == 0:
            crossing = (limit - c) / drive
        else:
            eq = p.outdoor_ppm + drive / rate
            crossing = -math.log((limit - eq) / (c - eq)) / rate
        if c < limit:
            lo = crossing
        else:
            hi = crossing
    if rate == 0:
        integral = (c - limit) * (hi - lo) + drive * (hi * hi - lo * lo) / 2
    else:
        eq = p.outdoor_ppm + drive / rate
        integral = (eq - limit) * (hi - lo) + (c - eq) * (
            math.exp(-rate * lo) * -math.expm1(-rate * (hi - lo))) / rate
    return hi - lo, max(0.0, integral)


def make_room(seed: int, contract: dict[str, Any], shift_hours: float = 0.0) -> dict[str, Any]:
    """Generate independent train/scoring streams with paired regime interventions.

    Args:
        seed: Independent room identity; reused across regimes for pairing.
        contract: Frozen N06 JSON configuration.
        shift_hours: Scoring-only physical routine shift; clocks stay unchanged.

    Returns:
        Training/scoring source arrays, common scoring noise and room metadata.
    """
    spec = contract['source']
    dt = contract['plant']['dt_hours']
    per_day = round(24 / dt)
    room_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    people = int(room_rng.integers(spec['people_range'][0], spec['people_range'][1] + 1))
    arrival = round(room_rng.uniform(*spec['arrival_hour_range']) / dt)
    magnitude = people * spec['person_generation_m3h'] * 1e6

    def draw(days: int, stream: int, shift: float) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        result = np.zeros(days * per_day)
        for day in range(days):
            attending = rng.random() < spec['attendance_probability']
            jitter = int(rng.choice(spec['jitter_steps']))
            start = day * per_day + arrival + jitter + round(shift / dt)
            stop = start + round(spec['duration_hours'] / dt)
            if attending:
                result[start:stop] = magnitude
        return result

    train = draw(contract['training_days'], 1, 0.0)
    scoring = draw(contract['evaluation_days'], 2, shift_hours)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    noise = noise_rng.normal(0, contract['plant']['noise_ppm'], len(scoring) + 1)
    return {'train_source': train, 'source': scoring, 'noise': noise,
            'metadata': {'seed': seed, 'people': people, 'arrival_hour': arrival * dt,
                         'source_ppm_m3h': magnitude, 'shift_hours': shift_hours}}


def rollout_metrics(states: np.ndarray, source: np.ndarray, actions: np.ndarray,
                    p: Plant) -> dict[str, float]:
    """Measure each policy's achieved continuous curve and actual actuation."""
    if len(states) != len(source) + 1 or len(actions) != len(source):
        raise ValueError('State/action/source lengths differ')
    exposure = [interval_exposure(float(c), float(s), p.base_flow_m3h + float(u) * p.extra_flow_m3h, p)
                for c, s, u in zip(states[:-1], source, actions)]
    previous = np.r_[0, actions[:-1]]
    on_hours = float(actions.sum() * p.dt_hours)
    return {'exceedance_hours': float(sum(v[0] for v in exposure)),
            'severity_ppm_hours': float(sum(v[1] for v in exposure)),
            'additional_air_m3': on_hours * p.extra_flow_m3h,
            'total_air_m3': on_hours * p.extra_flow_m3h + len(actions) * p.dt_hours * p.base_flow_m3h,
            'on_hours': on_hours, 'activations': int(np.sum(actions > previous)),
            'switches': int(np.sum(actions != previous))}
