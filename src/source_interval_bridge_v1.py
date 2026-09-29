"""Explicit source-to-concentration interval contract for N05/N06 follow-up.

Bounds are conditional on a source path, an initial state interval, exact plant
parameters and a specified action path. They are not marginal CO2 coverage
claims and cannot accept the concentration intervals emitted by N05.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.decision_plant_v1 import Plant
from src.uncertainty_calibration_v1 import empirical_radius

SOURCE_TARGET = "source_generation"
SOURCE_UNITS = "ppm*m3/h"


def _source_identity(target: str, units: str) -> None:
    """Reject concentration forecasts and ambiguous units before arithmetic."""
    if target != SOURCE_TARGET or units != SOURCE_UNITS:
        raise ValueError("requires source_generation in ppm*m3/h; CO2 ppm intervals are incompatible")


def _integer_time(value: int, name: str) -> None:
    """Require nanosecond timestamps without silently rounding floats."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{name} must be integer UTC nanoseconds")


@dataclass(frozen=True)
class SourceEnvelope:
    """Bounds on constant source in consecutive [start, end) plant intervals."""

    issue_ns: int
    start_ns: int
    step_ns: int
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model_id: str
    calibration_cutoff_ns: int
    calibration_paths: int
    target: str = SOURCE_TARGET
    units: str = SOURCE_UNITS
    scope: str = "conditional simultaneous source path; empirical, no coverage guarantee"

    def __post_init__(self) -> None:
        """Validate units, ordering and nonnegative source support."""
        _source_identity(self.target, self.units)
        for name in ("issue_ns", "start_ns", "step_ns", "calibration_cutoff_ns"):
            _integer_time(getattr(self, name), name)
        if (self.step_ns <= 0 or self.start_ns > self.issue_ns
                or (self.issue_ns - self.start_ns) % self.step_ns
                or self.calibration_cutoff_ns > self.issue_ns):
            raise ValueError("invalid source timing or future calibration cutoff")
        if not self.model_id or not isinstance(self.calibration_paths, int) or self.calibration_paths < 0:
            raise ValueError("model identity and nonnegative calibration count required")
        arrays = [np.array(getattr(self, name), dtype=float, copy=True) for name in ("point", "lower", "upper")]
        p, lo, hi = arrays
        if p.ndim != 1 or not len(p) or any(a.shape != p.shape for a in arrays):
            raise ValueError("source arrays must share nonempty one-dimensional shape")
        if (not np.isfinite(p).all() or not np.isfinite(lo).all()
                or np.isnan(hi).any() or np.any(lo < 0) or np.any(lo > p) or np.any(p > hi)):
            raise ValueError("invalid nonnegative source envelope")
        if self.start_ns + len(p) * self.step_ns <= self.issue_ns:
            raise ValueError("source envelope must extend beyond issuance")
        for name, a in zip(("point", "lower", "upper"), arrays):
            a.setflags(write=False)
            object.__setattr__(self, name, a)


def calibrate_source_envelope(*, point: np.ndarray, issue_ns: int, start_ns: int,
                              step_ns: int, model_id: str, calibration_prediction: np.ndarray,
                              calibration_truth: np.ndarray, calibration_issue_ns: np.ndarray,
                              calibration_end_ns: np.ndarray, calibration_arrival_ns: np.ndarray,
                              alpha: float = 0.1, target: str = SOURCE_TARGET,
                              units: str = SOURCE_UNITS) -> SourceEnvelope:
    """Build an empirical whole-path source band using fully received paths.

    Calibration predictions must be archived forecasts from the named, frozen
    model and the same path length. The caller is responsible for provenance;
    array shape alone cannot establish an out-of-sample forecast. Source labels
    are assumed observable with the supplied delay. In this repository that is
    a synthetic assumption, not source inference from controlled CO2 readings.
    """
    _source_identity(target, units)
    p = np.asarray(point, dtype=float)
    cp, cy = np.asarray(calibration_prediction), np.asarray(calibration_truth)
    ci, ce, ca = map(np.asarray, (calibration_issue_ns, calibration_end_ns, calibration_arrival_ns))
    if p.ndim != 1 or cp.ndim != 2 or cp.shape[1:] != p.shape or cy.shape != cp.shape:
        raise ValueError("calibration path shapes must match the forecast horizon")
    if ci.shape != (len(cp),) or ce.shape != cp.shape or ca.shape != cp.shape:
        raise ValueError("calibration timestamp shape mismatch")
    if any(a.dtype.kind not in "iu" for a in (ci, ce, ca)):
        raise ValueError("calibration timestamps must be integer UTC nanoseconds")
    offsets = start_ns + np.arange(1, len(p) + 1, dtype=np.int64) * step_ns - issue_ns
    if (np.any(ce - ci[:, None] != offsets[None, :]) or np.any(ca < ce)
            or np.any(ca <= ci[:, None]) or not np.isfinite(cp).all()
            or np.any(cp < 0) or not 0 < alpha < 1):
        raise ValueError("invalid source calibration chronology, predictions or alpha")
    # Filter by receipts before inspecting truth: absent/future labels are opaque.
    eligible = (ci < issue_ns) & np.all(ca <= issue_ns, axis=1)
    observed = cy[eligible]
    if not np.isfinite(observed).all() or np.any(observed < 0):
        raise ValueError("received source calibration labels must be finite and nonnegative")
    scores = np.max(np.abs(cp[eligible] - observed), axis=1)
    radius = empirical_radius(scores, alpha)
    cutoff = int(ca[eligible].max()) if eligible.any() else int(issue_ns)
    return SourceEnvelope(issue_ns, start_ns, step_ns, p, np.maximum(0, p - radius),
                          p + radius, model_id, cutoff, int(eligible.sum()), target, units)


@dataclass(frozen=True)
class ReceivedState:
    """Bounds on the state at the measurement event, not at receipt time."""

    event_ns: int
    arrival_ns: int
    lower_ppm: float
    upper_ppm: float


@dataclass(frozen=True)
class FlowPlan:
    """Specified total airflow and independently supplied immutable prefix.

    The prefix contains already applied intervals followed by commands already
    queued through the actuator delay. It must come from the actuator ledger.
    """

    plan_id: str
    flow_m3_h: np.ndarray
    locked_prefix_m3_h: np.ndarray


@dataclass(frozen=True)
class ConcentrationEnvelope:
    """Conditional CO2 endpoints under one specified, identified action path."""

    issue_ns: int
    endpoint_ns: np.ndarray
    lower_ppm: np.ndarray
    upper_ppm: np.ndarray
    plan_id: str
    source_model_id: str
    bounded: bool
    scope: str = "conditional on source/state containment, exact plant, and specified flow path"


def project_source_envelope(source: SourceEnvelope, state: ReceivedState,
                            plan: FlowPlan, plant: Plant) -> ConcentrationEnvelope:
    """Propagate interval bounds using the monotone exact first-order solution.

    Positive coefficients preserve enclosure for known nonnegative airflow.
    The result includes only endpoints after issuance, while integrating the
    measurement-to-issuance gap using recorded and queued actions. Infinite
    source upper bounds remain infinite; there is no hidden clipping/fallback.
    """
    for name in ("event_ns", "arrival_ns"):
        _integer_time(getattr(state, name), name)
    if (state.event_ns != source.start_ns or state.event_ns > state.arrival_ns
            or state.arrival_ns > source.issue_ns):
        raise ValueError("state must start the source path and have arrived by issuance")
    if (not np.isfinite([state.lower_ppm, state.upper_ppm]).all()
            or state.lower_ppm > state.upper_ppm):
        raise ValueError("state bounds must be finite and ordered")
    if source.step_ns != round(plant.dt_hours * 3_600_000_000_000):
        raise ValueError("source cadence differs from the plant integration step")
    flows = np.asarray(plan.flow_m3_h, dtype=float)
    prefix = np.asarray(plan.locked_prefix_m3_h, dtype=float)
    lag = (source.issue_ns - source.start_ns) // source.step_ns
    locked = min(len(source.point), lag + plant.actuator_delay_steps)
    if (not plan.plan_id or flows.shape != source.point.shape or prefix.shape != (locked,)
            or not np.isfinite(flows).all() or np.any(flows < 0)
            or not np.array_equal(flows[:locked], prefix)):
        raise ValueError("invalid flow path or changed applied/queued actuator prefix")
    lower, upper = [], []
    lo, hi = state.lower_ppm, state.upper_ppm
    for j, q in enumerate(flows):
        rate = q / plant.volume_m3
        decay = np.exp(-rate * plant.dt_hours)
        # expm1 avoids cancellation; the zero-flow limit is dt / volume.
        gain = (plant.dt_hours if rate == 0 else -np.expm1(-rate * plant.dt_hours) / rate) / plant.volume_m3
        lo = plant.outdoor_ppm + (lo - plant.outdoor_ppm) * decay + source.lower[j] * gain
        hi = (float("inf") if np.isinf(hi) or np.isinf(source.upper[j]) else
              plant.outdoor_ppm + (hi - plant.outdoor_ppm) * decay + source.upper[j] * gain)
        lower.append(lo)
        upper.append(hi)
    endpoints = source.start_ns + np.arange(1, len(flows) + 1, dtype=np.int64) * source.step_ns
    future = endpoints > source.issue_ns
    lo_array, hi_array = np.array(lower)[future], np.array(upper)[future]
    return ConcentrationEnvelope(source.issue_ns, endpoints[future], lo_array, hi_array,
                                 plan.plan_id, source.model_id, bool(np.isfinite(hi_array).all()))


def classify_limit(envelope: ConcentrationEnvelope, limit_ppm: float) -> str:
    """Return a conditional decision signal without selecting or sending actions."""
    if not np.isfinite(limit_ppm):
        raise ValueError("limit must be finite")
    if not envelope.bounded:
        return "unbounded: fallback required"
    if np.all(envelope.upper_ppm <= limit_ppm):
        return "within limit conditional on envelope assumptions"
    if np.any(envelope.lower_ppm > limit_ppm):
        return "exceedance throughout the envelope at one or more endpoints"
    return "possible exceedance: envelope straddles the limit"
