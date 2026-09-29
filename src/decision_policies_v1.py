"""Causal N06 policies with explicit no-new-command and action-conditioned targets."""
from __future__ import annotations

import numpy as np

from src.decision_plant_v1 import Plant, exact_update, rollout_metrics


def project_concentration(c: float, source: np.ndarray, flows: np.ndarray, p: Plant) -> np.ndarray:
    """Predict right-boundary concentrations under an explicit future flow path."""
    if source.ndim != 1 or source.shape != flows.shape:
        raise ValueError('Forecast source and action paths must be paired vectors')
    result = np.empty(len(source))
    for j, (s, q) in enumerate(zip(source, flows)):
        c = exact_update(c, float(s), float(q), p)
        result[j] = c
    return result


def decide(latest: float, peak: float, previous: int, trigger: float, hysteresis: float) -> int:
    """Apply one shared feedback-plus-forecast hysteresis rule."""
    signal = max(latest, peak)
    if signal >= trigger:
        return 1
    if signal < trigger - hysteresis:
        return 0
    return previous


def rollout(source: np.ndarray, noise: np.ndarray, p: Plant, margin: float,
            forecast_source: np.ndarray | None = None, horizon: int = 8,
            hysteresis: float = 50.0) -> dict:
    """Evolve a separate closed loop without exposing future truth to its policy.

    Args:
        source: Physical interval source; used only by the plant/scoring.
        noise: Paired additive measurement noise at state timestamps.
        p: Plant, sensor and actuator contract.
        margin: Trigger equals study limit minus this value in ppm.
        forecast_source: Issuable action-independent clock predictions, or an
            explicitly labeled oracle's realized source. None is reactive.
        horizon: Number of future concentration endpoints, zero is reactive.
        hysteresis: On/off separation in ppm.

    Returns:
        States, arrived observations, commands, applied actions, forecast peaks
        and metrics. No open-loop concentration accuracy is scored here.
    """
    source, noise = np.asarray(source), np.asarray(noise)
    n = len(source)
    if source.ndim != 1 or noise.shape != (n + 1,) or n < 1:
        raise ValueError('Expected N sources and N+1 noise samples')
    if not np.isfinite(source).all() or not np.isfinite(noise).all() or np.any(source < 0):
        raise ValueError('Source/noise must be finite and source nonnegative')
    if type(horizon) is not int or horizon < 0:
        raise ValueError('Horizon must be a nonnegative integer')
    if not np.isfinite([margin, hysteresis]).all() or hysteresis < 0:
        raise ValueError('Margin/hysteresis must be finite; hysteresis nonnegative')
    if forecast_source is not None:
        forecast_source = np.asarray(forecast_source)
        if forecast_source.ndim != 1 or len(forecast_source) < n + horizon:
            raise ValueError('Forecast path must cover scoring plus lookahead')
        if not np.isfinite(forecast_source).all() or np.any(forecast_source < 0):
            raise ValueError('Forecast source must be finite and nonnegative')
    states = np.empty(n + 1)
    states[0] = p.initial_ppm
    commands, actions = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    observations, peaks = np.empty(n), np.empty(n)
    observation_index = np.empty(n, dtype=int)
    previous = 0
    for k in range(n):
        arrived = k - p.sensor_delay_steps
        # C0 is known until the first sensor arrives. Future noise is never read.
        obs = p.initial_ppm if arrived < 0 else states[arrived] + noise[arrived]
        observations[k], observation_index[k] = obs, arrived
        peak = obs
        if forecast_source is not None and horizon > 0:
            start = max(0, arrived)
            estimate = obs
            # Past applied and irrevocably queued actions are known at issue k.
            for j in range(start, k + horizon):
                command_index = j - p.actuator_delay_steps
                pending = commands[command_index] if 0 <= command_index < k else 0
                q = p.base_flow_m3h + pending * p.extra_flow_m3h
                estimate = exact_update(estimate, float(forecast_source[j]), q, p)
                if j >= k:
                    peak = max(peak, estimate)
        peaks[k] = peak
        command = decide(obs, peak, previous, p.limit_ppm - margin, hysteresis)
        commands[k], previous = command, command
        delayed = k - p.actuator_delay_steps
        actions[k] = commands[delayed] if delayed >= 0 else 0
        states[k + 1] = exact_update(states[k], float(source[k]),
                                     p.base_flow_m3h + actions[k] * p.extra_flow_m3h, p)
    return {'states': states, 'commands': commands, 'actions': actions,
            'observations': observations, 'observation_index': observation_index,
            'forecast_peaks': peaks,
            'forecast_target': 'no_new_commands_given_pending_queue_and_current_controlled_history',
            'metrics': rollout_metrics(states, source, actions, p)}


def objective(metrics: dict, weight: float) -> float:
    """Score hours of exceedance plus declared added-air effort penalty."""
    return metrics['exceedance_hours'] + weight * metrics['additional_air_m3'] / 1000
