"""Causal persistence rows and an explicit delayed-error ledger for VM-N05."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.dynamics_data_v1 import (
    DynamicsConfig, Trajectory, _validate_trajectory, generate_trajectory,
    utc_daily_angles,
)

NEVER = np.iinfo(np.int64).max
DAY_NS = int(pd.Timedelta(days=1).value)


@dataclass
class ForecastRows:
    """One row per issue/horizon; unavailable truth is isolated behind a ledger.

    Times are integer UTC nanoseconds. ``available_ns=NEVER`` means no packet,
    not a scheduled delivery. ``truth`` is offline scoring evidence, not input.
    """

    issue_ns: np.ndarray
    target_ns: np.ndarray
    available_ns: np.ndarray
    horizon: np.ndarray
    point: np.ndarray
    truth: np.ndarray
    features: np.ndarray
    phase: np.ndarray
    origin_step: np.ndarray

    def subset(self, indices: np.ndarray) -> ForecastRows:
        """Select rows without altering their timing identities."""
        return ForecastRows(**{k: v[indices].copy() for k, v in vars(self).items()})

    def __len__(self) -> int:
        return len(self.issue_ns)


def scenario_trajectory(seed: int, config: dict, scenario: dict) -> Trajectory:
    """Apply the declared evaluation-only noise extension to the N04 generator."""
    settings = dict(config["data"])
    settings["split_days"] = tuple(settings["split_days"])
    settings.update(shifted=scenario["shifted"], outage=scenario["outage"])
    trajectory = generate_trajectory(seed, DynamicsConfig(**settings))
    start = config["timeline_days"]["evaluation_start"] * 1440 // settings["dt_minutes"]
    observed = trajectory.observed_ppm.copy()
    observed[start:] = trajectory.latent_ppm[start:] + scenario["evaluation_noise_multiplier"] * (
        observed[start:] - trajectory.latent_ppm[start:]
    )
    return replace(trajectory, observed_ppm=observed)


def origin_features(trajectory: Trajectory, origin: int, change_steps: int,
                    change_divisor: float) -> tuple[float, np.ndarray] | None:
    """Read a causal history, with both event and actual-arrival conditions.

    Past grid rows use only a predecessor event available by this issuance.
    No retrospective future measurement or unreceived outage truth is read.
    """
    cfg = trajectory.config
    if origin < cfg.lookback - 1 or change_steps >= cfg.lookback:
        return None
    rows = np.arange(origin - cfg.lookback + 1, origin + 1)
    # Earlier predecessors may fill the first requested row, within max age.
    candidates = np.arange(max(0, rows[0] - cfg.max_fill_age_steps), origin + 1)
    eligible = (
        trajectory.sensor_available[candidates]
        & ~trajectory.arrival_times[candidates].isna()
        & (trajectory.arrival_times.asi8[candidates] <= trajectory.event_times.asi8[origin])
        & (trajectory.event_times.asi8[candidates] <= trajectory.event_times.asi8[origin])
    )
    # Inspect finiteness only for values authorized by the two-time gate.
    available = candidates[eligible]
    eligible[eligible] = np.isfinite(trajectory.observed_ppm[available])
    source = np.maximum.accumulate(np.where(eligible, candidates, -1))[-cfg.lookback:]
    age = rows - source
    if np.any(source < 0) or np.any(age > cfg.max_fill_age_steps):
        return None
    values = trajectory.observed_ppm[source]
    delta = (values[-1] - values[-1 - change_steps]) / change_divisor
    features = np.array([delta, abs(delta), 1.0 - eligible[-cfg.lookback:].mean(),
                         age[-1] / cfg.max_fill_age_steps], dtype=np.float64)
    return float(values[-1]), features


def build_rows(trajectory: Trajectory, config: dict) -> tuple[ForecastRows, dict[str, int]]:
    """Issue all horizons on common causal origins; never filter by future truth."""
    _validate_trajectory(trajectory)
    horizons = np.asarray(config["horizon_steps"], dtype=np.int64)
    if horizons.ndim != 1 or len(set(horizons)) != len(horizons) or np.any(horizons <= 0):
        raise ValueError("horizons must be distinct positive steps")
    if horizons.max() > trajectory.config.horizon:
        raise ValueError("horizon exceeds declared full future span")
    records: dict[str, list[Any]] = {k: [] for k in ForecastRows.__annotations__}
    counts = {"candidate_origins": 0, "history_excluded": 0, "retained_origins": 0}
    phases = utc_daily_angles(trajectory.event_times).ravel()
    for origin in range(0, len(trajectory.event_times) - int(horizons.max()), trajectory.config.stride):
        counts["candidate_origins"] += 1
        context = origin_features(trajectory, origin, config["regimes"]["change_steps"],
                                  config["scale"]["change_divisor_ppm"])
        if context is None:
            counts["history_excluded"] += 1
            continue
        counts["retained_origins"] += 1
        point, features = context
        for horizon in horizons:
            target = origin + int(horizon)
            arrival = trajectory.arrival_times[target]
            actual = (arrival.value if trajectory.sensor_available[target] and not pd.isna(arrival)
                      else NEVER)
            values = dict(issue_ns=trajectory.event_times[origin].value,
                          target_ns=trajectory.event_times[target].value,
                          available_ns=actual, horizon=horizon, point=point,
                          truth=trajectory.observed_ppm[target], features=features.copy(),
                          phase=phases[target], origin_step=origin)
            for name, value in values.items():
                records[name].append(value)
    integer = {"issue_ns", "target_ns", "available_ns", "horizon", "origin_step"}
    arrays = {k: np.asarray(v, dtype=np.int64 if k in integer else np.float64)
              for k, v in records.items()}
    arrays["features"] = arrays["features"].reshape(-1, 4)
    return ForecastRows(**arrays), counts


class DelayedErrorLedger:
    """Order-independent, update-before-issuance access to matured labels.

    ``indices`` applies timing before inspecting label values. ``errors`` also
    checks timing so callers cannot accidentally fit on a split-crossing label.
    All forecasting/fitting/calibration paths must use this access boundary.
    """

    def __init__(self, rows: ForecastRows, step_ns: int):
        n = len(rows)
        if any(v.shape != ((n, 4) if k == "features" else (n,))
               for k, v in vars(rows).items()):
            raise ValueError("row arrays must be aligned")
        for name in ("issue_ns", "target_ns", "available_ns", "horizon"):
            if not np.issubdtype(getattr(rows, name).dtype, np.integer):
                raise ValueError("times and horizons must be integer coordinates")
        if step_ns <= 0 or np.any(rows.horizon <= 0):
            raise ValueError("step and horizons must be positive")
        if np.any(rows.target_ns - rows.issue_ns != rows.horizon * step_ns):
            raise ValueError("target must equal issue plus horizon")
        if np.any(rows.available_ns < rows.target_ns):
            raise ValueError("arrival cannot precede the target")
        if len(set(zip(rows.issue_ns.tolist(), rows.horizon.tolist()))) != n:
            raise ValueError("duplicate issue/horizon identity")
        if not np.isfinite(rows.point).all() or not np.isfinite(rows.phase).all():
            raise ValueError("point and known clock must be finite")
        self.rows = rows

    def timed_indices(self, cutoff_ns: int, issue_start: int, issue_stop: int,
                      horizon: int) -> np.ndarray:
        """Return mature rows sorted by target then issue, before label filtering."""
        r = self.rows
        take = np.flatnonzero((r.issue_ns >= issue_start) & (r.issue_ns < issue_stop)
                             & (r.issue_ns < cutoff_ns) & (r.horizon == horizon)
                             & (r.target_ns <= cutoff_ns) & (r.available_ns <= cutoff_ns)
                             & (r.available_ns != NEVER))
        return take[np.lexsort((r.issue_ns[take], r.target_ns[take]))]

    def indices(self, cutoff_ns: int, issue_start: int, issue_stop: int,
                horizon: int, limit: int | None = None) -> np.ndarray:
        """Return finite mature residual rows, optionally the latest positive count."""
        take = self.timed_indices(cutoff_ns, issue_start, issue_stop, horizon)
        take = take[np.isfinite(self.rows.truth[take])]
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("pool limit must be a positive integer")
            take = take[-limit:]
        return take

    def errors(self, indices: np.ndarray, cutoff_ns: int) -> np.ndarray:
        """Read absolute errors only after both clocks permit access."""
        r = self.rows
        if np.any((r.target_ns[indices] > cutoff_ns) | (r.available_ns[indices] > cutoff_ns)
                  | (r.issue_ns[indices] >= cutoff_ns) | (r.available_ns[indices] == NEVER)):
            raise ValueError("unmatured or unavailable residual requested")
        values = np.abs(r.truth[indices] - r.point[indices])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite matured residual")
        return values


def timeline_ns(config: dict) -> dict[str, int]:
    """Resolve the fixed UTC timeline with no split-relative phase reset."""
    epoch = pd.Timestamp(config["data"]["start_utc"]).value
    return {key: epoch + int(day) * DAY_NS for key, day in config["timeline_days"].items()}
