"""Frozen positive circular scale factors and delayed interval calibration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import time

import numpy as np
import torch

from src.uncertainty_data_v1 import DelayedErrorLedger, ForecastRows, timeline_ns

CONDITIONAL = ("fourier", "hour_bin", "vm")


def phase_hours(phase: np.ndarray) -> np.ndarray:
    """Map circular coordinates to stable one-hour bins, including exact wrap."""
    turns = np.mod(np.asarray(phase, dtype=np.float64) / (2 * np.pi), 1.0)
    return (np.floor(np.round(turns * 24, 10)).astype(np.int64) % 24)


def design(features: np.ndarray, phase: np.ndarray, arm: str, settings: dict) -> np.ndarray:
    """Use identical causal covariates and 24 phase coordinates for each arm."""
    if features.shape != (len(phase), 4) or not np.isfinite(features).all() or not np.isfinite(phase).all():
        raise ValueError("finite aligned features and clock required")
    count = settings["basis_count"]
    theta = np.mod(phase, 2 * np.pi)
    if arm == "fourier":
        angle = theta[:, None] * np.arange(1, count // 2 + 1)
        clock = np.column_stack((np.cos(angle), np.sin(angle)))
    elif arm == "hour_bin":
        if count != 24:
            raise ValueError("hour-bin contract requires 24 features")
        clock = np.eye(24)[phase_hours(theta)]
    elif arm == "vm":
        centers = 2 * np.pi * np.arange(count) / count
        clock = np.exp(settings["kappa"] * (np.cos(theta[:, None] - centers) - 1))
    else:
        raise ValueError("unknown conditional arm")
    return np.column_stack((features, clock))


@dataclass
class ScaleModel:
    """One frozen horizon-specific scale regression, never a point forecaster."""

    arm: str
    settings: dict
    weight: np.ndarray
    bias: float
    global_scale: float
    phase_counts: np.ndarray
    fit_count: int
    fit_seconds: float
    device: str

    def predict(self, features: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return finite positive scales with explicit fallback/clipping flags."""
        if features.shape != (len(phase), 4) or not np.isfinite(phase).all():
            raise ValueError("aligned features and finite target clock required")
        finite = np.isfinite(features).all(axis=1)
        sparse = self.phase_counts[phase_hours(phase)] < self.settings["minimum_phase"]
        fallback = (~finite) | sparse | (self.fit_count < self.settings["minimum_fit"])
        log_scale = np.full(len(phase), np.log(self.global_scale))
        valid = ~fallback
        if valid.any():
            log_scale[valid] = design(features[valid], phase[valid], self.arm, self.settings) @ self.weight + self.bias
        numerical = ~np.isfinite(log_scale)
        fallback |= numerical
        log_scale[numerical] = np.log(self.global_scale)
        floor, ceiling = np.log(self.settings["floor_ppm"]), np.log(self.settings["ceiling_ppm"])
        flags = {"fallback": fallback, "floor": log_scale < floor, "ceiling": log_scale > ceiling}
        scales = np.exp(np.clip(log_scale, floor, ceiling))
        return scales, flags

    def record(self) -> dict:
        """Serialize factors, counts, numerical settings and measured fit cost."""
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


def fit_scale(ledger: DelayedErrorLedger, horizon: int, arm: str, config: dict,
              device: torch.device) -> ScaleModel:
    """Fit only authorized scale-origin labels available strictly before cutoff."""
    started = time.perf_counter()
    times, settings = timeline_ns(config), config["scale"]
    cutoff = times["scale_end"] - 1
    idx = ledger.indices(cutoff, times["scale_start"], times["scale_end"], horizon)
    r = ledger.rows
    idx = idx[np.isfinite(r.features[idx]).all(axis=1)]
    error = ledger.errors(idx, cutoff)
    floor, ceiling = settings["floor_ppm"], settings["ceiling_ppm"]
    global_scale = float(np.clip(np.median(error) if len(error) else floor, floor, ceiling))
    counts = np.bincount(phase_hours(r.phase[idx]), minlength=24)
    weight = np.zeros(4 + settings["basis_count"], dtype=np.float64)
    bias = np.log(global_scale)
    if len(idx) >= settings["minimum_fit"]:
        x = torch.as_tensor(design(r.features[idx], r.phase[idx], arm, settings), dtype=torch.float64, device=device)
        y = torch.as_tensor(np.log(np.maximum(error, floor)), dtype=torch.float64, device=device)
        xm, ym = x.mean(0), y.mean()
        xc, yc = x - xm, y - ym
        # Penalize coefficients on a mean-squared objective; intercept is free.
        gram = xc.T @ xc / len(x) + settings["ridge_penalty"] * torch.eye(x.shape[1], dtype=x.dtype, device=device)
        w = torch.linalg.solve(gram, xc.T @ yc / len(x))
        b = ym - xm @ w
        weight, bias = w.cpu().numpy(), float(b.cpu())
        if not np.isfinite(weight).all() or not np.isfinite(bias):
            raise FloatingPointError("nonfinite scale fit")
    return ScaleModel(arm, dict(settings), weight, float(bias), global_scale, counts,
                      len(idx), time.perf_counter() - started, str(device))


def empirical_radius(scores: np.ndarray, alpha: float) -> float:
    """Return the one-based ceil((n+1)*(1-alpha)) order statistic, without interpolation.

    Decimal arithmetic prevents a binary representation artifact at exact ranks.
    Tiny/empty pools whose rank exceeds n produce infinity, never a width cap.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be a finite nonnegative vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    k = int((Decimal(len(scores) + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else float("inf")


def calibrate(ledger: DelayedErrorLedger, models: dict[tuple[int, str], ScaleModel],
              config: dict, deadline: float = float("inf")) -> list[dict]:
    """Issue matched intervals after eligible arrivals, independent of row order."""
    r, times = ledger.rows, timeline_ns(config)
    alpha, limit = config["calibration"]["alpha"], config["calibration"]["rolling_count"]
    scales, flags = {}, {}
    for key, model in models.items():
        scales[key], flags[key] = model.predict(r.features, r.phase)
    records = []
    for h in config["horizon_steps"]:
        static = ledger.indices(times["evaluation_start"], times["calibration_start"], times["evaluation_start"], h)
        static_matched = static[-limit:]
        static_q = empirical_radius(ledger.errors(static, times["evaluation_start"]), alpha)
        matched_q = empirical_radius(ledger.errors(static_matched, times["evaluation_start"]), alpha)
        evaluate = np.flatnonzero((r.issue_ns >= times["evaluation_start"]) & (r.issue_ns < times["end"]) & (r.horizon == h))
        evaluate = evaluate[np.argsort(r.issue_ns[evaluate])]
        for row in evaluate:
            if time.perf_counter() >= deadline:
                raise TimeoutError("N05 prospective runtime cap reached")
            issue = int(r.issue_ns[row])
            pool = ledger.indices(issue, times["calibration_start"], issue, h, limit)
            error = ledger.errors(pool, issue)
            rolling_q = empirical_radius(error, alpha)
            for arm in config["arms"]:
                used, cutoff, radius = pool, issue, rolling_q
                scale, quantile = 1.0, rolling_q
                sparse = clipped_low = clipped_high = False
                small = False
                if arm in ("static", "static_matched"):
                    used = static if arm == "static" else static_matched
                    cutoff = times["evaluation_start"]
                    radius = quantile = static_q if arm == "static" else matched_q
                elif arm in CONDITIONAL:
                    small = len(pool) < config["calibration"]["minimum_pool"]
                    sparse = bool(flags[(h, arm)]["fallback"][row])
                    clipped_low = bool(flags[(h, arm)]["floor"][row])
                    clipped_high = bool(flags[(h, arm)]["ceiling"][row])
                    if not small:
                        scale = float(scales[(h, arm)][row])
                        quantile = empirical_radius(error / scales[(h, arm)][pool], alpha)
                        radius = scale * quantile
                records.append({
                    "row": int(row), "arm": arm, "horizon_steps": int(h),
                    "issue_ns": issue, "target_ns": int(r.target_ns[row]),
                    "observation_available_ns": int(r.available_ns[row]),
                    "point_ppm": float(r.point[row]), "lower_ppm": float(r.point[row] - radius),
                    "upper_ppm": float(r.point[row] + radius), "truth_ppm_offline": float(r.truth[row]),
                    "calibration_cutoff_ns": int(cutoff), "calibration_count": len(used),
                    "calibration_latest_target_ns": int(r.target_ns[used].max()) if len(used) else None,
                    "calibration_latest_arrival_ns": int(r.available_ns[used].max()) if len(used) else None,
                    "scale_ppm": scale, "quantile": quantile, "scale_fallback": sparse,
                    "scale_floor": clipped_low, "scale_ceiling": clipped_high,
                    "small_pool_fallback": small, "infinite_interval": not np.isfinite(radius),
                    "target_hour": int(phase_hours(r.phase[row:row + 1])[0]),
                    "recent_change_ppm": float(r.features[row, 0] * config["scale"]["change_divisor_ppm"]),
                    "stale_sensor": bool(r.features[row, 3] * config["data"]["max_fill_age_steps"] > config["data"]["delay_steps"]),
                })
    return records
