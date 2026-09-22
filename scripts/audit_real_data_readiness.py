"""Reproduce a cache-only readiness audit; no loader/download calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import _JENA_FEATURES


def audit(name: str, data_dir: Path) -> dict[str, Any]:
    """Audit a cached dataset, reproducing the historical loader preprocessing."""
    if name == "jena":
        path = data_dir / "jena/extracted/jena_climate_2009_2016.csv"
        raw = pd.read_csv(path)
        index = pd.to_datetime(raw.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
        raw = raw[_JENA_FEATURES]
        sentinels = (raw == -9999).sum().to_dict()
        raw = raw.replace(-9999, np.nan)
        expected_minutes = 10
    else:
        path = data_dir / "power/extracted/household_power_consumption.txt"
        raw = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
        index = pd.to_datetime(raw.pop("Date") + " " + raw.pop("Time"), format="%d/%m/%Y %H:%M:%S")
        raw = raw.apply(pd.to_numeric, errors="coerce")
        sentinels = {}
        expected_minutes = 1
    raw.index = pd.DatetimeIndex(index)
    numeric_columns = list(raw.columns)
    raw_steps = raw.index.to_series().diff().dt.total_seconds().div(60)
    grid = raw.resample("1h").mean()
    counts = raw.resample("1h").count()
    retained = grid.dropna(how="all")
    forward = retained.ffill() if name == "jena" else retained
    final = forward.bfill() if name == "jena" else retained
    ffilled = retained.isna() & forward.notna()
    bfilled = forward.isna() & final.notna()
    gap_hours = final.index.to_series().diff().dt.total_seconds().div(3600)
    bad_gaps = gap_hours[gap_hours.ne(1) & gap_hours.notna()]
    record = {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "raw_start": str(raw.index.min()), "raw_end": str(raw.index.max()),
        "raw_duplicate_timestamps": int(raw.index.duplicated().sum()),
        "raw_nonstandard_intervals": int((raw_steps.dropna() != expected_minutes).sum()),
        "raw_interval_minutes_min": float(raw_steps.min()), "raw_interval_minutes_max": float(raw_steps.max()),
        "raw_sentinel_counts": sentinels, "raw_missing_by_column": raw.isna().sum().to_dict(),
        "hourly_grid_rows": len(grid), "hourly_retained_rows": len(retained),
        "hourly_all_missing_dropped": int(grid.isna().all(axis=1).sum()),
        "hourly_missing_before_fill": retained.isna().sum().to_dict(),
        "hourly_ffill_cells": int(ffilled.sum().sum()), "hourly_ffill_rows": int(ffilled.any(axis=1).sum()),
        "hourly_ffill_by_column": ffilled.sum().to_dict(),
        "hourly_bfill_cells": int(bfilled.sum().sum()), "hourly_bfill_rows": int(bfilled.any(axis=1).sum()),
        "hourly_missing_after_fill": final.isna().sum().to_dict(),
        "hourly_gap_count": len(bad_gaps), "hourly_max_gap_hours": float(gap_hours.max()),
        "hourly_gap_records": [{"after_bin": str(final.index[final.index.get_loc(t) - 1]),
                                "next_bin": str(t), "elapsed_hours": float(v)} for t, v in bad_gaps.items()],
        "hourly_samples_per_target_min": int(counts[numeric_columns[0]][counts[numeric_columns[0]] > 0].min()),
        "hourly_samples_per_target_max": int(counts[numeric_columns[0]].max()),
        "splits": {},
    }
    if name == "jena":
        dup = raw[raw.index.duplicated(keep=False)]
        dedup_hourly = raw.loc[~raw.index.duplicated()].resample("1h").mean()
        temp_delta = (grid["T (degC)"] - dedup_hourly["T (degC)"]).abs()
        theta = np.deg2rad(raw["wd (deg)"])
        circular = np.mod(np.rad2deg(np.arctan2(np.sin(theta).resample("1h").mean(),
                                              np.cos(theta).resample("1h").mean())), 360)
        angle_diff = np.abs((circular - grid["wd (deg)"] + 180) % 360 - 180)
        record["jena_details"] = {
            "duplicates_with_differing_selected_values": int((dup.groupby(level=0).nunique(dropna=False)>1).any(axis=1).sum()),
            "hourly_temperature_bins_changed_by_deduplication": int((temp_delta > 1e-10).sum()),
            "max_temperature_deduplication_delta_celsius": float(temp_delta.max()),
            "ffilled_wind_values": {str(t): float(forward.loc[t, "wv (m/s)"])
                                    for t in retained.index[retained["wv (m/s)"].isna()]},
            "arithmetic_vs_circular_direction_bins_over_90_degrees": int((angle_diff > 90).sum()),
            "max_arithmetic_vs_circular_direction_difference": float(angle_diff.max()),
        }
    n = len(final)
    cutoffs = [0, int(n * .8), int(n * .9), n]
    for label, lo, hi in zip(["train", "validation", "test"], cutoffs[:-1], cutoffs[1:]):
        split = final.iloc[lo:hi]
        target = "T (degC)" if name == "jena" else "Global_active_power"
        width = 168 + 24
        valid = np.isfinite(split.to_numpy()).all(axis=1)
        bad = np.r_[0, (np.diff(split.index.asi8) != pd.Timedelta(hours=1).value).astype(int)]
        if len(split) >= width:
            bad_windows = np.lib.stride_tricks.sliding_window_view(bad, width)[:, 1:].any(axis=1)
            finite_windows = np.lib.stride_tricks.sliding_window_view(valid, width).all(axis=1)
            count = int((~bad_windows & finite_windows).sum())
        else:
            bad_windows = np.zeros(0)
            count = 0
        record["splits"][label] = {
            "rows": len(split), "start": str(split.index.min()), "end": str(split.index.max()),
            "legacy_candidate_windows": max(0, len(split)-width+1),
            "gap_crossing_windows": int(bad_windows.sum()), "contiguous_finite_windows": count,
            "ffill_cells": int(ffilled.iloc[lo:hi].sum().sum()),
            "target_mean": float(split[target].mean()), "target_std_population": float(split[target].std(ddof=0)),
        }
    return record


def main() -> None:
    """Write a fresh readiness audit without downloading or modifying raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["jena", "power"], default=["jena", "power"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {name: audit(name, args.data_dir) for name in args.datasets}
    output = args.output_dir / "real_data_readiness.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Readiness audit saved to {output}")


if __name__ == "__main__":
    main()
