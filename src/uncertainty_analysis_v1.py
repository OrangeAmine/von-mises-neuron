"""Aligned interval metrics and paired room/block analysis for VM-N05."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.uncertainty_data_v1 import DAY_NS, timeline_ns


def metric_arrays(truth: np.ndarray, point: np.ndarray, lower: np.ndarray,
                  upper: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Calculate physical-unit scores, retaining infinite interval penalties."""
    truth, point, lower, upper = [np.asarray(a, dtype=np.float64) for a in (truth, point, lower, upper)]
    if truth.ndim != 1 or any(a.shape != truth.shape for a in (point, lower, upper)):
        raise ValueError("metric vectors must align exactly")
    if not np.isfinite(truth).all() or not np.isfinite(point).all():
        raise ValueError("offline truth and point must be finite; no silent exclusion")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("invalid interval bounds")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    width = upper - lower
    score = width + 2 / alpha * (np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0))
    return {"interval_score": score, "coverage": ((lower <= truth) & (truth <= upper)).astype(float),
            "width": width, "lower_miss": (truth < lower).astype(float),
            "upper_miss": (truth > upper).astype(float), "point_mae": abs(truth - point),
            "point_squared_error": (truth - point) ** 2}


def validate_alignment(frame: pd.DataFrame, config: dict) -> None:
    """Reject missing/duplicate arms, mismatched truth, point, times or horizons."""
    key = ["room_seed", "scenario", "issue_ns", "horizon_steps"]
    if frame.duplicated(key + ["arm"]).any():
        raise ValueError("duplicate interval identity")
    if set(frame.arm) != set(config["arms"]):
        raise ValueError("incorrect arm set")
    grouped = frame.groupby(key, sort=False)
    if not (grouped.size() == len(config["arms"])).all():
        raise ValueError("unmatched origins across arms")
    for column in ("point_ppm", "truth_ppm_offline", "target_ns", "observation_available_ns"):
        if not (grouped[column].nunique(dropna=False) == 1).all():
            raise ValueError("point/truth/timing alignment differs across arms")
    step = config["data"]["dt_minutes"] * 60 * 1_000_000_000
    if not (frame.target_ns - frame.issue_ns == frame.horizon_steps * step).all():
        raise ValueError("target horizon misalignment")
    if (frame.calibration_cutoff_ns > frame.issue_ns).any():
        raise ValueError("calibration cutoff exceeds issuance")
    for column in ("calibration_latest_target_ns", "calibration_latest_arrival_ns"):
        if (frame[column] > frame.calibration_cutoff_ns).any():
            raise ValueError("unmatured calibration evidence")


def cell_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Report all horizons, phases and origin-known regimes, including empty cuts."""
    records = []
    metrics = metric_arrays(frame.truth_ppm_offline.to_numpy(), frame.point_ppm.to_numpy(),
                            frame.lower_ppm.to_numpy(), frame.upper_ppm.to_numpy(), config["calibration"]["alpha"])
    enriched = frame.assign(**metrics)
    rise = config["regimes"]["recent_rise_ppm"]
    fall = config["regimes"]["recent_fall_ppm"]
    for (room, scenario, arm, h), group in enriched.groupby(["room_seed", "scenario", "arm", "horizon_steps"], sort=True):
        strata = {"all": np.ones(len(group), dtype=bool),
                  "recent_rise": (group.recent_change_ppm > rise).to_numpy(),
                  "recent_fall": (group.recent_change_ppm < fall).to_numpy(),
                  "quiet": ((group.recent_change_ppm >= fall) & (group.recent_change_ppm <= rise)).to_numpy(),
                  "stale_sensor": group.stale_sensor.to_numpy(dtype=bool)}
        for p in range(config["regimes"]["phase_bins"]):
            strata[f"phase_{p * 4:02d}_{(p + 1) * 4:02d}"] = (group.target_hour // 4 == p).to_numpy()
        for stratum, mask in strata.items():
            sample = group.loc[mask]
            row = dict(room_seed=int(room), scenario=scenario, arm=arm, horizon_steps=int(h), stratum=stratum, n=len(sample))
            for name in metrics:
                row[name] = float(sample[name].mean()) if len(sample) else None
            squared_error = row.pop("point_squared_error")
            row["point_rmse"] = np.sqrt(squared_error) if len(sample) else None
            row["scale_fallback_count"] = int(sample.scale_fallback.sum())
            row["scale_floor_count"] = int(sample.scale_floor.sum())
            row["scale_ceiling_count"] = int(sample.scale_ceiling.sum())
            row["small_pool_count"] = int(sample.small_pool_fallback.sum())
            row["infinite_count"] = int(sample.infinite_interval.sum())
            row["calibration_count_min"] = int(sample.calibration_count.min()) if len(sample) else None
            row["calibration_count_max"] = int(sample.calibration_count.max()) if len(sample) else None
            records.append(row)
    return pd.DataFrame(records)


def paired_summary(frame: pd.DataFrame, cells: pd.DataFrame, config: dict) -> dict:
    """Separate complete-pipeline room replication from fixed-fit block uncertainty."""
    settings = config["analysis"]
    h, comparator = settings["primary_horizon"], settings["primary_comparator"]
    primary = cells[(cells.stratum == "all") & (cells.horizon_steps == h)]
    room = primary.groupby(["room_seed", "arm"]).interval_score.mean().unstack("arm")
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = settings["bootstrap_draws"]
    take = rng.integers(0, len(room), (draws, len(room)))
    vm, control = room.vm.to_numpy(), room[comparator].to_numpy()
    finite = np.isfinite(vm).all() and np.isfinite(control).all() and np.all(control > 0)
    improvement = float(1 - vm.mean() / control.mean()) if finite else None
    effects = 1 - vm[take].mean(1) / control[take].mean(1) if finite else None
    ci = np.quantile(effects, [0.025, 0.975]).tolist() if finite else [None, None]
    by_scenario = primary.groupby(["scenario", "arm"])[["interval_score", "coverage"]].mean()
    guards = []
    for scenario in [s["name"] for s in config["scenarios"]]:
        candidate, reference = by_scenario.loc[(scenario, "vm")], by_scenario.loc[(scenario, comparator)]
        valid_score = (np.isfinite(candidate.interval_score) and np.isfinite(reference.interval_score)
                       and reference.interval_score > 0)
        regression = candidate.interval_score / reference.interval_score - 1 if valid_score else None
        guards.append({"scenario": scenario, "relative_score_regression": float(regression) if valid_score else None,
                       "vm_coverage": float(candidate.coverage),
                       "regression_passed": bool(valid_score and regression <= settings["maximum_scenario_regression"]),
                       "coverage_passed": bool(candidate.coverage >= settings["minimum_overall_coverage"])})
    # Conditional two-day block resampling holds issued intervals fixed, including
    # their fitted factors and online quantiles; it does not replay a new ledger.
    f = frame[frame.horizon_steps == h].copy()
    score = metric_arrays(f.truth_ppm_offline.to_numpy(), f.point_ppm.to_numpy(),
                           f.lower_ppm.to_numpy(), f.upper_ppm.to_numpy(), config["calibration"]["alpha"])["interval_score"]
    f["score"] = score
    f["block"] = ((f.issue_ns - timeline_ns(config)["evaluation_start"]) // (settings["conditional_block_days"] * DAY_NS)).astype(int)
    grouped = f.groupby(["room_seed", "scenario", "block", "arm"]).score.agg(["sum", "count"])
    rooms = sorted(f.room_seed.unique())
    scenarios = [s["name"] for s in config["scenarios"]]
    blocks = sorted(f.block.unique())
    conditional = np.zeros((draws, 2))
    for room_id in rooms:
        selected = rng.integers(0, len(blocks), (draws, len(blocks)))
        for a, arm in enumerate(("vm", comparator)):
            for scenario in scenarios:
                values = grouped.loc[(room_id, scenario)].xs(arm, level="arm").reindex(blocks)
                totals = values["sum"].to_numpy()[selected].sum(1)
                counts = values["count"].to_numpy()[selected].sum(1)
                conditional[:, a] += totals / counts / (len(rooms) * len(scenarios))
    conditional_effect = 1 - conditional[:, 0] / conditional[:, 1] if finite else None
    return {
        "primary_improvement": improvement, "paired_room_95_interval": ci,
        "room_effects": {str(i): float(1 - room.loc[i, "vm"] / room.loc[i, comparator]) for i in room.index} if finite else {},
        "room_effect_sample_sd": float(np.std(1 - vm / control, ddof=1)) if finite else None,
        "replication_unit": "independently generated and independently fitted room pipeline; all scenarios paired",
        "method_uncertainty_scope": "joint room/scale-fit/calibration variability; does not isolate factor variance from room heterogeneity",
        "conditional_fixed_interval_block_95_interval": np.quantile(conditional_effect, [0.025, 0.975]).tolist() if finite else [None, None],
        "conditional_scope": "two-day evaluation blocks, factors and issued quantiles fixed; omits refitting uncertainty",
        "scenario_guards": guards,
        "useful_effect_passed": bool(finite and improvement >= settings["minimum_improvement"]),
        "room_interval_passed": bool(finite and ci[0] > 0),
        "scientific_gate_passed": bool(finite and improvement >= settings["minimum_improvement"] and ci[0] > 0 and
                                       all(g["regression_passed"] and g["coverage_passed"] for g in guards)),
        "bootstrap_draws": draws, "bootstrap_seed": settings["bootstrap_seed"],
        "independent_rooms": len(room), "no_coverage_guarantee": True,
    }
