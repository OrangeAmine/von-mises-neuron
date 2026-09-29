"""Causal discounted-error mixing of frozen forecasting experts for N07."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

NEVER = np.iinfo(np.int64).max


@dataclass(frozen=True)
class MixingResult:
    """Forecasts and auditable per-origin state before issuing each forecast."""

    prediction: np.ndarray
    weights: np.ndarray
    feedback_count: np.ndarray
    latest_feedback_ns: np.ndarray


def delayed_mix(experts: np.ndarray, truth: np.ndarray, issue_ns: np.ndarray,
                target_ns: np.ndarray, arrival_ns: np.ndarray,
                initial_loss: np.ndarray, initialization_cutoff_ns: int,
                scale: float, eta: float, half_life_updates: float = 24) -> MixingResult:
    """Mix frozen expert forecasts using only earlier forecasts' arrived errors.

    Args:
        experts: Finite predictions of shape (origins, horizons, experts).
        truth: Offline targets; values are accessed only after their arrival.
        issue_ns: Strictly increasing UTC issue timestamps, shape (origins,).
        target_ns: UTC target timestamps, shape (origins, horizons).
        arrival_ns: Observation receipt times; NEVER denotes an absent packet.
        initial_loss: Past validation MSE divided by scale squared, shape (H, E).
        initialization_cutoff_ns: Latest information used to initialize or fit.
        scale: Positive training-only target standard deviation.
        eta: Positive softmax inverse temperature.
        half_life_updates: Number of arrived labels to halve old loss weight.

    Returns:
        Convex predictions, weights, and feedback receipts before each issuance.

    No global finiteness check reads unavailable truth. Zero-delay labels can
    update at their event time, but a forecast must already have been issued.
    """
    p = np.asarray(experts, dtype=np.float64)
    y = np.asarray(truth)
    issue, target, arrival = map(np.asarray, (issue_ns, target_ns, arrival_ns))
    loss = np.array(initial_loss, dtype=np.float64, copy=True)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[2] < 2:
        raise ValueError("experts must have shape (N, H, E), E >= 2")
    n, h, e = p.shape
    if y.shape != (n, h) or target.shape != y.shape or arrival.shape != y.shape:
        raise ValueError("truth, target and arrival shapes must match (N, H)")
    if issue.shape != (n,) or loss.shape != (h, e):
        raise ValueError("issue or initial loss shape mismatch")
    if any(a.dtype.kind not in "iu" for a in (issue, target, arrival)):
        raise ValueError("timestamps must be integer UTC nanoseconds")
    if (np.any(np.diff(issue) <= 0) or np.any(target <= issue[:, None])
            or np.any(np.diff(target, axis=1) <= 0) or np.any(arrival < target)):
        raise ValueError("timestamps violate issuance/target/arrival chronology")
    if not isinstance(initialization_cutoff_ns, (int, np.integer)) or initialization_cutoff_ns > issue[0]:
        raise ValueError("initialization uses information after first issuance")
    if not np.isfinite(p).all() or not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError("expert forecasts and nonnegative initial losses must be finite")
    if not all(np.isfinite(v) and v > 0 for v in (scale, eta, half_life_updates)):
        raise ValueError("scale, eta and half life must be finite and positive")
    alpha = -np.expm1(-np.log(2) / half_life_updates)
    predictions = np.empty((n, h))
    weights = np.empty_like(p)
    counts = np.zeros(h, dtype=np.int64)
    latest = np.full(h, -1, dtype=np.int64)
    count_trace, latest_trace = np.empty((n, h), dtype=np.int64), np.empty((n, h), dtype=np.int64)
    queue: list[tuple[int, int, int, int]] = []
    for i, now in enumerate(issue):
        while queue and queue[0][0] <= now:
            received, _, old, horizon = heapq.heappop(queue)
            observed = float(y[old, horizon])
            if not np.isfinite(observed):
                raise ValueError("an arrived observation is nonfinite")
            errors = (p[old, horizon] - observed) / scale
            with np.errstate(over="raise", invalid="raise"):
                loss[horizon] = (1 - alpha) * loss[horizon] + alpha * errors ** 2
            counts[horizon] += 1
            latest[horizon] = received
        # Subtract the best loss before multiplying to avoid common offsets.
        with np.errstate(over="ignore"):
            logits = -eta * (loss - loss.min(axis=1, keepdims=True))
        w = np.exp(logits)
        w /= w.sum(axis=1, keepdims=True)
        weights[i] = w
        predictions[i] = (w * p[i]).sum(axis=1)
        count_trace[i], latest_trace[i] = counts, latest
        for j in range(h):
            if arrival[i, j] != NEVER:
                heapq.heappush(queue, (int(arrival[i, j]), int(target[i, j]), i, j))
    return MixingResult(predictions, weights, count_trace, latest_trace)


def paired_summary(records: list[dict], primary_step: int, seed: int,
                   replicates: int, minimum_gain: float, maximum_regression: float) -> dict:
    """Evaluate the fixed room-paired intersection gate; never resample windows."""
    rooms = sorted({r["seed"] for r in records})
    arms = ("sensor", "seasonal", "fixed", "adaptive")
    lookup = {(r["seed"], r["regime"], r["arm"]): r for r in records}
    if len(lookup) != len(records) or len(records) != len(rooms) * 2 * len(arms):
        raise ValueError("expected exactly one paired record per room/regime/arm")
    values = {arm: np.array([[lookup[(room, regime, arm)]["endpoints"][str(primary_step)]["rmse"]
                             / lookup[(room, regime, arm)]["normalization_std"]
                             for regime in ("stable", "shifted")] for room in rooms])
              for arm in arms}
    if not all(np.isfinite(a).all() and np.all(a > 0) for a in values.values()):
        raise ValueError("paired metrics must be finite and positive")
    draws = np.random.default_rng(seed).integers(0, len(rooms), (replicates, len(rooms)))
    contrasts = {}
    for arm in arms[:-1]:
        gains = 100 * (1 - values["adaptive"].mean(axis=1) / values[arm].mean(axis=1))
        ci = np.quantile(gains[draws].mean(axis=1), [0.025, 0.975])
        contrasts[arm] = {"mean_gain_percent": float(gains.mean()), "ci95": ci.tolist(),
                          "room_gains_percent": gains.tolist(),
                          "passed": bool(gains.mean() >= minimum_gain and ci[0] > 0)}
    regressions = {"stable_vs_seasonal": 100 * (values["adaptive"][:, 0].mean() / values["seasonal"][:, 0].mean() - 1),
                   "shifted_vs_sensor": 100 * (values["adaptive"][:, 1].mean() / values["sensor"][:, 1].mean() - 1)}
    return {"rooms": rooms, "contrasts": contrasts, "regression_percent": regressions,
            "overall_passed": bool(all(v["passed"] for v in contrasts.values())
                                   and all(v <= maximum_regression for v in regressions.values()))}
