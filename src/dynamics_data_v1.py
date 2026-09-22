"""Versioned synthetic first-order forecasting data for VM-N04.

Clock coordinates are UTC, with a fixed Unix-epoch phase origin. Observations
have distinct event and arrival times. Sensor outages preserve the regular
grid; their hidden values remain available only as offline scoring targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DynamicsConfig:
    """One regime and the fixed acquisition/window contract.

    Arrival phase, irregular jitter, and event duration are rounded to the
    acquisition grid, so equilibrium is constant over each integration step.
    ``outage_test_days`` uses zero-based day indices within the test split.
    """

    tau_ratio: float = 1.0
    shifted: bool = False
    irregular: bool = False
    outage: bool = False
    dt_minutes: int = 15
    split_days: tuple[int, int, int] = (28, 14, 14)
    event_duration_hours: float = 2.0
    lookback: int = 24
    horizon: int = 16
    stride: int = 4
    max_fill_age_steps: int = 12
    delay_steps: int = 1
    noise_ppm: float = 10.0
    baseline_ppm: float = 420.0
    shift_hours: float = 3.0
    start_utc: str = "2026-01-05T00:00:00Z"
    outage_test_days: tuple[int, ...] = (2, 6, 10)

    def __post_init__(self) -> None:
        """Reject contracts that do not define a regular finite experiment."""
        integer_fields = (
            self.dt_minutes, self.lookback, self.horizon, self.stride,
            self.max_fill_age_steps, self.delay_steps,
        )
        if any(not isinstance(v, (int, np.integer)) for v in integer_fields):
            raise ValueError("grid and window sizes must be integers")
        if self.dt_minutes <= 0 or 1440 % self.dt_minutes:
            raise ValueError("dt_minutes must divide one UTC day")
        if min(self.lookback, self.horizon, self.stride, self.max_fill_age_steps) < 1:
            raise ValueError("window lengths, stride and fill age must be positive")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be nonnegative")
        if len(self.split_days) != 3 or any(
            not isinstance(v, (int, np.integer)) or v < 1 for v in self.split_days
        ):
            raise ValueError("split_days must contain three positive integer lengths")
        if any(not np.isfinite(v) for v in (
            self.tau_ratio, self.event_duration_hours, self.noise_ppm,
            self.baseline_ppm, self.shift_hours,
        )):
            raise ValueError("physical parameters must be finite")
        if self.tau_ratio <= 0 or not 0 < self.event_duration_hours <= 8:
            raise ValueError("response ratio must be positive and duration in (0, 8]")
        if self.noise_ppm < 0 or self.baseline_ppm <= 0:
            raise ValueError("noise must be nonnegative and baseline positive")
        if not 0 <= self.shift_hours <= 6:
            raise ValueError("shift_hours must be in [0, 6]")
        steps = self.event_duration_hours * 60 / self.dt_minutes
        if not np.isclose(steps, round(steps)) or steps < 1:
            raise ValueError("event duration must be a positive integer grid span")
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in self.outage_test_days):
            raise ValueError("outage days must be nonnegative integer indices")
        start = pd.Timestamp(self.start_utc)
        if pd.isna(start) or start.tzinfo is None:
            raise ValueError("start_utc must be timezone aware")
        if start.tz_convert("UTC") != start.tz_convert("UTC").normalize():
            raise ValueError("start_utc must be UTC midnight")


@dataclass
class Trajectory:
    """Full simulation truth and acquisition timestamps, before splitting."""

    event_times: pd.DatetimeIndex
    arrival_times: pd.DatetimeIndex
    latent_ppm: np.ndarray
    observed_ppm: np.ndarray
    equilibrium_ppm: np.ndarray
    sensor_available: np.ndarray
    attendance: np.ndarray
    boundary: np.ndarray
    outage: np.ndarray
    config: DynamicsConfig
    seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Normalizer:
    """Training-only observed-sensor mean and population standard deviation."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale values using the frozen training statistics."""
        return (np.asarray(values) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Return normalized predictions to ppm."""
        return np.asarray(values) * self.std + self.mean


@dataclass
class WindowBatch:
    """Common eligible origins and scoring truth for every comparison arm.

    ``sensor`` channels are normalized observation, direct-availability mask,
    and fill age divided by the allowed maximum. Angles are raw radians.
    Target timestamps are UTC Unix nanoseconds (shape ``N,H``); origins use
    an explicit UTC DatetimeIndex. Regime masks are reporting metadata only.
    """

    sensor: np.ndarray
    future_angles: np.ndarray
    y: np.ndarray
    y_ppm: np.ndarray
    clean_y_ppm: np.ndarray
    origin_indices: np.ndarray
    origin_times: pd.DatetimeIndex
    target_times: np.ndarray
    target_event: np.ndarray
    target_boundary: np.ndarray
    origin_outage: np.ndarray
    target_outage: np.ndarray
    exclusions: dict[str, int]


def first_order_update(
    previous: float | np.ndarray,
    equilibrium: float | np.ndarray,
    dt_hours: float,
    tau_hours: float,
) -> float | np.ndarray:
    """Integrate dc/dt=(equilibrium-c)/tau exactly for a constant interval.

    Args:
        previous: Concentration at the interval's left boundary, in ppm.
        equilibrium: Constant forcing equilibrium within that interval.
        dt_hours: Positive elapsed interval length in hours.
        tau_hours: Positive first-order response time in hours.

    Returns:
        Concentration at the right boundary of the interval.
    """
    if not np.isfinite(dt_hours) or not np.isfinite(tau_hours) or min(dt_hours, tau_hours) <= 0:
        raise ValueError("dt_hours and tau_hours must be positive and finite")
    return equilibrium + (previous - equilibrium) * np.exp(-dt_hours / tau_hours)


def utc_daily_angles(event_times: pd.DatetimeIndex) -> np.ndarray:
    """Return raw daily angles using UTC instants and the fixed Unix epoch."""
    if not isinstance(event_times, pd.DatetimeIndex) or event_times.tz is None:
        raise ValueError("event_times must be a timezone-aware DatetimeIndex")
    if event_times.hasnans:
        raise ValueError("event_times must not contain NaT")
    # Integer modulo avoids loss of subsecond precision from large epoch floats.
    day_ns = pd.Timedelta(days=1).value
    return (2 * np.pi * (event_times.asi8 % day_ns) / day_ns)[:, None]


def clock_consistency_diagnostic(event_times: pd.DatetimeIndex) -> dict[str, float]:
    """Check timezone representation and demonstrate a distinct +1h clock defect.

    A genuine routine shift changes forcing relative to unchanged clocks. This
    diagnostic instead shifts the encoded timestamps without moving forcing.
    The correct UTC encoding is invariant to a Budapest timezone representation,
    including DST, while the intentional timestamp defect is pi/12 radians.
    """
    angles = utc_daily_angles(event_times)
    corrected = utc_daily_angles(event_times.tz_convert("Europe/Budapest"))
    defective = utc_daily_angles(event_times + pd.Timedelta(hours=1))
    difference = lambda a, b: np.angle(np.exp(1j * (a - b)))
    return {
        "timezone_representation_max_error_rad": float(np.max(np.abs(difference(corrected, angles)), initial=0)),
        "intentional_plus1h_defect_mean_error_rad": float(np.mean(np.abs(difference(defective, angles)))) if len(angles) else 0.0,
    }


def generate_trajectory(seed: int, config: DynamicsConfig | None = None) -> Trajectory:
    """Generate independent room draws with regime-coupled randomness.

    Room parameters, daily attendance draws and sensor noise use independent
    streams spawned from ``seed``. Every regime consumes the same draws. Thus
    using a common room seed across regimes couples their environmental draws
    without making those regime variants independent replication units.
    """
    cfg = config or DynamicsConfig()
    room_stream, event_stream, noise_stream = np.random.SeedSequence(seed).spawn(3)
    room_rng = np.random.default_rng(room_stream)
    daily_rng = np.random.default_rng(event_stream)
    noise_rng = np.random.default_rng(noise_stream)
    steps_per_day = 1440 // cfg.dt_minutes
    days = sum(cfg.split_days)
    n_steps = days * steps_per_day
    dt_hours = cfg.dt_minutes / 60.0
    base_start_step = int(np.rint(room_rng.uniform(8.0, 10.0) / dt_hours))
    amplitude_ppm = float(room_rng.uniform(300.0, 600.0))
    present_draw = daily_rng.random(days) < 0.75
    jitter_steps = np.rint(daily_rng.uniform(-0.5, 0.5, days) / dt_hours).astype(int)
    amplitude_draw = daily_rng.uniform(0.8, 1.2, days)
    duration_steps = int(round(cfg.event_duration_hours / dt_hours))
    shift_steps = int(round(cfg.shift_hours / dt_hours))
    first_test_day = sum(cfg.split_days[:2])
    index = pd.date_range(
        pd.Timestamp(cfg.start_utc).tz_convert("UTC"), periods=n_steps,
        freq=pd.Timedelta(minutes=cfg.dt_minutes),
    )
    equilibrium = np.full(n_steps, cfg.baseline_ppm, dtype=np.float64)
    attendance = np.zeros(n_steps, dtype=bool)
    boundary = np.zeros(n_steps, dtype=bool)
    outage = np.zeros(n_steps, dtype=bool)
    boundary_radius = int(round(1.0 / dt_hours))
    daily_start_steps = []
    for day in range(days):
        nominal = day * steps_per_day + base_start_step
        if cfg.shifted and day >= first_test_day:
            nominal += shift_steps
        start = nominal + (jitter_steps[day] if cfg.irregular else 0)
        daily_start_steps.append(start)
        present = bool(present_draw[day]) if cfg.irregular else True
        if present:
            stop = start + duration_steps
            amplitude = amplitude_ppm * (amplitude_draw[day] if cfg.irregular else 1.0)
            equilibrium[start:stop] += amplitude
            attendance[start:stop] = True
            for transition in (start, stop):
                boundary[max(0, transition - boundary_radius):min(n_steps, transition + boundary_radius + 1)] = True
        # The outage is tied to the nominal routine clock, irrespective of the
        # realized presence/jitter; it is never attendance information.
        if cfg.outage and day - first_test_day in cfg.outage_test_days:
            outage[nominal:nominal + int(round(2.0 / dt_hours))] = True
    latent = np.empty(n_steps, dtype=np.float64)
    latent[0] = cfg.baseline_ppm
    alpha = np.exp(-dt_hours / (cfg.tau_ratio * cfg.event_duration_hours))
    for step in range(n_steps - 1):
        latent[step + 1] = equilibrium[step] + (latent[step] - equilibrium[step]) * alpha
    observed = latent + noise_rng.normal(0.0, cfg.noise_ppm, n_steps)
    arrivals = index + pd.Timedelta(minutes=cfg.delay_steps * cfg.dt_minutes)
    return Trajectory(
        event_times=index, arrival_times=arrivals, latent_ppm=latent,
        observed_ppm=observed, equilibrium_ppm=equilibrium,
        sensor_available=~outage, attendance=attendance, boundary=boundary,
        outage=outage, config=cfg, seed=int(seed), metadata={
            "base_start_hour_utc": base_start_step * dt_hours,
            "amplitude_ppm": amplitude_ppm,
            "tau_hours": cfg.tau_ratio * cfg.event_duration_hours,
            "daily_start_steps": daily_start_steps,
            "daily_presence_draw": present_draw.tolist(),
            "clock": "UTC, Unix-epoch daily phase; no DST change",
            "outage_arrival_semantics": "arrival times are scheduled; sensor_available=False means no packet arrives",
            "target": "noisy concentration in ppm; hidden outage targets retained for offline scoring",
        },
    )


def split_bounds(trajectory: Trajectory, split: SplitName) -> tuple[int, int]:
    """Return half-open integer bounds for one chronological split."""
    names = ("train", "validation", "test")
    if split not in names:
        raise ValueError(f"Unknown split {split!r}")
    steps = 1440 // trajectory.config.dt_minutes
    lengths = np.asarray(trajectory.config.split_days, dtype=np.int64) * steps
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    position = names.index(split)
    return int(boundaries[position]), int(boundaries[position + 1])


def _validate_trajectory(trajectory: Trajectory) -> None:
    """Reject removed rows or malformed timestamps instead of compressing gaps."""
    index = trajectory.event_times
    expected = sum(trajectory.config.split_days) * (1440 // trajectory.config.dt_minutes)
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans:
        raise ValueError("event grid must contain timezone-aware finite timestamps")
    if len(index) != expected or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("event grid must preserve every regular acquisition row")
    if not np.all(np.diff(index.asi8) == pd.Timedelta(minutes=trajectory.config.dt_minutes).value):
        raise ValueError("event grid contains a gap or irregular interval")
    if index[0] != pd.Timestamp(trajectory.config.start_utc):
        raise ValueError("event grid origin differs from the configured origin")
    arrivals = trajectory.arrival_times
    if not isinstance(arrivals, pd.DatetimeIndex) or arrivals.tz is None or len(arrivals) != expected:
        raise ValueError("arrival timestamps must match the aware event grid")
    if any(np.asarray(getattr(trajectory, name)).shape != (expected,) for name in (
        "latent_ppm", "observed_ppm", "equilibrium_ppm", "sensor_available",
        "attendance", "boundary", "outage",
    )):
        raise ValueError("trajectory arrays must match the acquisition grid")
    if np.any((arrivals.asi8 < index.asi8) & ~arrivals.isna()):
        raise ValueError("a measurement cannot arrive before its event")


def fit_normalizer(trajectory: Trajectory) -> Normalizer:
    """Fit only finite training sensor observations arrived by training end."""
    _validate_trajectory(trajectory)
    start, stop = split_bounds(trajectory, "train")
    available = (
        trajectory.sensor_available[start:stop]
        & np.isfinite(trajectory.observed_ppm[start:stop])
        & ~trajectory.arrival_times[start:stop].isna()
        & (trajectory.arrival_times[start:stop] <= trajectory.event_times[stop - 1])
    )
    values = trajectory.observed_ppm[start:stop][available]
    if not len(values):
        raise ValueError("no available training sensors for normalization")
    return Normalizer(float(values.mean()), max(float(values.std()), 1.0))


def sensor_history_at_origin(
    trajectory: Trajectory,
    origin_index: int,
    split: SplitName,
    normalizer: Normalizer,
) -> np.ndarray | None:
    """Construct split-local history using only packets available at origin.

    Past rows can use delayed packets received by the forecast origin, but a
    row's value never uses a measurement from a later event time. Forward fill
    is bounded by event-time age; no backward fill or cross-split fill occurs.
    Return ``None`` if any history value has no eligible sufficiently fresh
    predecessor, so all methods reject the entire origin together.
    """
    cfg = trajectory.config
    start, stop = split_bounds(trajectory, split)
    history_start = origin_index - cfg.lookback + 1
    if history_start < start or not start <= origin_index < stop:
        return None
    indices = np.arange(start, origin_index + 1)
    arrivals = trajectory.arrival_times[indices]
    eligible = (
        trajectory.sensor_available[indices]
        & np.isfinite(trajectory.observed_ppm[indices])
        & ~arrivals.isna()
        & (arrivals <= trajectory.event_times[origin_index])
        & (trajectory.event_times[indices] <= trajectory.event_times[origin_index])
    )
    # Accumulation selects the newest eligible event at or before each row.
    source_indices = np.maximum.accumulate(np.where(eligible, indices, -1))
    rows = np.arange(history_start, origin_index + 1)
    source = source_indices[history_start - start:]
    age = rows - source
    if np.any(source < start) or np.any(age > cfg.max_fill_age_steps):
        return None
    values = normalizer.transform(trajectory.observed_ppm[source])
    direct = eligible[history_start - start:].astype(np.float64)
    return np.column_stack((values, direct, age / cfg.max_fill_age_steps)).astype(np.float32)


def make_windows(
    trajectory: Trajectory,
    split: SplitName,
    normalizer: Normalizer,
) -> WindowBatch:
    """Construct common origin-aligned windows after chronological splitting.

    All candidates align to the split's fixed stride grid. Exclusions partition
    the candidate set into insufficient history, insufficient future horizon,
    stale/missing history, and nonfinite offline targets. Outage truth remains
    scored, so missing sensor packets alone never remove difficult targets.
    """
    _validate_trajectory(trajectory)
    if not np.isfinite(normalizer.mean) or not np.isfinite(normalizer.std) or normalizer.std <= 0:
        raise ValueError("normalizer must have finite mean and positive finite std")
    cfg = trajectory.config
    start, stop = split_bounds(trajectory, split)
    exclusions = dict(candidates=0, insufficient_history=0, insufficient_future=0,
                      label_unavailable_by_cutoff=0, stale_or_missing_history=0,
                      nonfinite_target=0, retained=0)
    origins: list[int] = []
    histories: list[np.ndarray] = []
    for origin in range(start, stop, cfg.stride):
        exclusions["candidates"] += 1
        if origin - cfg.lookback + 1 < start:
            exclusions["insufficient_history"] += 1
            continue
        if origin + cfg.horizon >= stop:
            exclusions["insufficient_future"] += 1
            continue
        targets = slice(origin + 1, origin + 1 + cfg.horizon)
        if split != "test" and not np.all(
            trajectory.sensor_available[targets]
            & ~trajectory.arrival_times[targets].isna()
            & (trajectory.arrival_times[targets] <= trajectory.event_times[stop - 1])
        ):
            exclusions["label_unavailable_by_cutoff"] += 1
            continue
        history = sensor_history_at_origin(trajectory, origin, split, normalizer)
        if history is None:
            exclusions["stale_or_missing_history"] += 1
            continue
        if not np.all(np.isfinite(trajectory.observed_ppm[origin + 1:origin + 1 + cfg.horizon])):
            exclusions["nonfinite_target"] += 1
            continue
        origins.append(origin)
        histories.append(history)
    origin_array = np.asarray(origins, dtype=np.int64)
    target_indices = origin_array[:, None] + np.arange(1, cfg.horizon + 1)[None, :]
    y_ppm = trajectory.observed_ppm[target_indices]
    all_angles = utc_daily_angles(trajectory.event_times)
    exclusions["retained"] = len(origins)
    return WindowBatch(
        sensor=np.stack(histories) if histories else np.empty((0, cfg.lookback, 3), dtype=np.float32),
        future_angles=all_angles[target_indices].astype(np.float32),
        y=normalizer.transform(y_ppm).astype(np.float32),
        y_ppm=y_ppm, clean_y_ppm=trajectory.latent_ppm[target_indices],
        origin_indices=origin_array, origin_times=trajectory.event_times[origin_array],
        target_times=trajectory.event_times.asi8[target_indices],
        target_event=trajectory.attendance[target_indices],
        target_boundary=trajectory.boundary[target_indices],
        origin_outage=np.asarray([
            np.any(trajectory.outage[i - cfg.lookback + 1:i + 1]) for i in origins
        ], dtype=bool),
        target_outage=trajectory.outage[target_indices], exclusions=exclusions,
    )


def aggregate_at_bin_end(
    observations: pd.Series,
    arrival_times: pd.DatetimeIndex | None = None,
    interval: str = "1h",
    *,
    sample_interval: str = "15min",
) -> pd.DataFrame:
    """Aggregate complete [left, right) bins with bin-end availability.

    Args:
        observations: Event-time indexed sensor values; missing values are NaN.
        arrival_times: Packet arrival timestamps, aligned with observations.
            Defaults to immediate arrival. A NaT arrival excludes that packet.
        interval: Fixed pandas resampling interval.
        sample_interval: Explicit expected acquisition cadence; never inferred
            from a potentially gapped input. It must divide ``interval``.

    Returns:
        Frame indexed by UTC bin-end event time, with ``value``, ``count``,
        ``complete``, ``expected_count``, and ``available_at``. Only a complete
        bin has a finite mean, available no earlier than its right edge and
        latest required packet arrival. Incomplete and empty bins remain on
        the grid with NaN value and NaT availability, retaining observed counts.
    """
    times = observations.index
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None or times.hasnans:
        raise ValueError("observations require finite aware event timestamps")
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("event timestamps must be unique and increasing")
    cadence_ns = pd.tseries.frequencies.to_offset(sample_interval).nanos
    bin_ns = pd.tseries.frequencies.to_offset(interval).nanos
    if cadence_ns <= 0 or bin_ns <= 0 or bin_ns % cadence_ns:
        raise ValueError("positive sample_interval must divide the fixed bin interval")
    if np.any(times.asi8 % cadence_ns):
        raise ValueError("events must lie on the expected acquisition grid")
    expected_count = bin_ns // cadence_ns
    arrivals = times if arrival_times is None else arrival_times
    if not isinstance(arrivals, pd.DatetimeIndex) or arrivals.tz is None or len(arrivals) != len(times):
        raise ValueError("arrival_times must align with aware event timestamps")
    if np.any((arrivals.asi8 < times.asi8) & ~arrivals.isna()):
        raise ValueError("arrival cannot precede event time")
    times, arrivals = times.tz_convert("UTC"), arrivals.tz_convert("UTC")
    frame = pd.DataFrame({"value": observations.to_numpy(), "arrival": arrivals}, index=times)
    frame.loc[arrivals.isna(), "value"] = np.nan
    finite = np.isfinite(frame["value"])
    frame.loc[~finite, "arrival"] = pd.NaT
    frame.loc[~finite, "value"] = np.nan
    grouped = frame.resample(interval, closed="left", label="right", origin="epoch")
    means = grouped["value"].mean()
    counts = grouped["value"].count()
    complete = counts == expected_count
    means = means.where(complete)
    latest = grouped["arrival"].max()
    available = [
        max(edge, arrival) if is_complete else pd.NaT
        for edge, arrival, is_complete in zip(means.index, latest, complete)
    ]
    return pd.DataFrame({"value": means, "count": counts,
                         "complete": complete, "expected_count": expected_count,
                         "available_at": pd.DatetimeIndex(available, tz=times.tz)})
