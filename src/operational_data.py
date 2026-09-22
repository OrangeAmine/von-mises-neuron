"""Cache-only hourly data and causal windows for operational forecast evaluation.

Hourly observations are indexed by their availability time: a row labelled
01:00 averages raw readings in [00:00, 01:00). No missing value is imputed.
Naive source timestamps are interpreted literally, without assuming a timezone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.temporal import extract_angular_time_features


POWER_COLUMNS = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
OCCUPANCY_COLUMNS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def _fingerprint(path: Path) -> dict[str, Any]:
    """Describe the exact cached bytes used for a dataset."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove identical sensor repeats, rejecting conflicting timestamps."""
    duplicated = frame.index.duplicated(keep=False)
    if duplicated.any():
        distinct = frame.loc[duplicated].groupby(level=0).nunique(dropna=False)
        conflicts = distinct.index[(distinct > 1).any(axis=1)]
        if len(conflicts):
            raise ValueError(f"Conflicting sensor duplicates at {conflicts[0]}")
    retained = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    return retained, len(frame) - len(retained)


def load_cached_hourly(
    dataset: str,
    data_dir: Path = Path("data/raw"),
    *,
    min_coverage: float = 1 / 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load extracted occupancy or household power without network access.

    Args:
        dataset: ``power`` or ``occupancy``.
        data_dir: Existing cache root, containing dataset/extracted directories.
        min_coverage: Minimum finite readings per feature divided by 60. The
            default keeps every nonempty feature/hour; 0.8 requires 48 readings.
            Under-covered features become NaN while their hourly rows remain.

    Returns:
        Hourly sensor frame and JSON-compatible provenance/coverage metadata.

    Raises:
        FileNotFoundError: The expected extracted files are absent.
        ValueError: Unsupported dataset, invalid coverage or conflicting rows.
    """
    if dataset not in {"power", "occupancy"}:
        raise ValueError("dataset must be 'power' or 'occupancy'")
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be finite and in (0, 1]")
    extracted = Path(data_dir) / dataset / "extracted"
    frames, used_files = [], []
    if dataset == "power":
        paths = sorted(extracted.rglob("household_power_consumption.txt"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected exactly one cached power file in {extracted}")
        table = pd.read_csv(paths[0], sep=";", na_values=["?"], low_memory=False)
        index = pd.to_datetime(table["Date"] + " " + table["Time"],
                               format="%d/%m/%Y %H:%M:%S", errors="raise")
        frame = table[POWER_COLUMNS].apply(pd.to_numeric, errors="raise")
        frame.index = pd.DatetimeIndex(index)
        frames.append(frame)
        used_files.extend(paths)
    else:
        paths = sorted(extracted.rglob("*.txt")) + sorted(extracted.rglob("*.csv"))
        for path in paths:
            # UCI's three sensor files share the datatraining/datatest prefix.
            if not path.stem.lower().startswith(("datatraining", "datatest")):
                continue
            table = pd.read_csv(path, na_values=["?"])
            if not set(OCCUPANCY_COLUMNS + ["date"]).issubset(table.columns):
                raise ValueError(f"Missing expected occupancy columns in {path}")
            frame = table[OCCUPANCY_COLUMNS].apply(pd.to_numeric, errors="raise")
            frame.index = pd.DatetimeIndex(pd.to_datetime(table["date"], errors="raise"))
            frames.append(frame)
            used_files.append(path)
        if not frames:
            raise FileNotFoundError(f"No cached occupancy sensor files in {extracted}")
    raw = pd.concat(frames)
    if raw.empty or raw.index.hasnans or raw.index.tz is not None:
        raise ValueError("Raw data must contain nonempty, valid, timezone-naive timestamps")
    raw_rows = len(raw)
    raw, duplicates = _deduplicate(raw)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    grouped = raw.resample("1h", closed="left", label="right")
    counts = grouped.count()
    hourly = grouped.mean()
    threshold = int(np.ceil(60 * min_coverage - 1e-12))
    hourly = hourly.where(counts >= threshold)
    metadata = {
        "dataset": dataset, "raw_files": [_fingerprint(path) for path in used_files],
        "raw_rows": raw_rows, "deduplicated_rows": len(raw),
        "identical_duplicate_rows_removed": duplicates,
        "duplicate_policy": "identical selected sensor values deduplicated; conflicting values rejected",
        "clock_policy": "timezone-naive literal source wall clock; no timezone inferred",
        "bin_policy": "[t-1h,t) means indexed by right-edge availability t",
        "imputation": "none; full hourly grid retained",
        "raw_start": str(raw.index.min()), "raw_end": str(raw.index.max()),
        "hourly_start": str(hourly.index.min()), "hourly_end": str(hourly.index.max()),
        "hourly_rows": len(hourly), "sensor_columns": list(hourly.columns),
        "min_coverage": min_coverage, "minimum_readings_per_feature": threshold,
        "coverage_denominator_readings": 60,
        "all_features_missing_hours_before_coverage": int((counts == 0).all(axis=1).sum()),
        "all_features_missing_hours_after_coverage": int(hourly.isna().all(axis=1).sum()),
        "incomplete_feature_hours_after_coverage": int(hourly.isna().any(axis=1).sum()),
        "coverage": {
            column: {"raw_missing_readings": int(raw[column].isna().sum()),
                     "zero_reading_hours": int((counts[column] == 0).sum()),
                     "partial_hours": int(((counts[column] > 0) & (counts[column] < 60)).sum()),
                     "overfull_hours": int((counts[column] > 60).sum()),
                     "finite_reading_count_quantiles": {
                         str(q): float(counts[column].quantile(q)) for q in [0, .25, .5, .75, 1]}}
            for column in hourly.columns},
    }
    return hourly, metadata


def _target_stats(values: np.ndarray) -> dict[str, Any]:
    """Summarize finite observations in physical units."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": int(finite.size), "mean": float(finite.mean()),
            "std": float(finite.std()), "min": float(finite.min()), "max": float(finite.max())}


def prepare_forecast_windows(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback: int,
    horizon: int,
    target_col: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Build train-normalized windows with targets confined to each split.

    Args:
        frame: Numeric hourly sensors indexed by right-edge availability time.
        train_start: Inclusive first training availability timestamp.
        val_start: Exclusive train end and inclusive validation target start.
        test_start: Exclusive validation end and inclusive test target start.
        test_end: Exclusive final test target timestamp.
        lookback: Number of hourly observations available at each origin.
        horizon: Number of future hourly target values.
        target_col: Sensor forecast in both normalized and physical units.

    Returns:
        Split arrays plus serializable normalization, boundaries and diagnostics.
        Each ``X`` has normalized sensors first, then raw daily/weekly angles.
        ``X_sensor`` is a view of those sensor features. ``origins`` are final
        input availability timestamps; target timestamps are strictly later.
        Validation/test inputs may contain earlier splits. Within each split
        hourly origins overlap; all horizon targets must belong to that split.
        Future non-target sensors are never inspected to accept a window.

    Raises:
        ValueError: Invalid timestamps, boundaries, window sizes or training data.
    """
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or frame.index.tz is not None or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing or frame.empty):
        raise ValueError("frame needs a nonempty sorted unique timezone-naive DatetimeIndex")
    if not frame.columns.is_unique or target_col not in frame.columns:
        raise ValueError("Columns must be unique and include target_col")
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0
           for n in (lookback, horizon)):
        raise ValueError("lookback and horizon must be positive integers")
    boundaries = pd.DatetimeIndex([train_start, val_start, test_start, test_end])
    if (boundaries.hasnans or boundaries.tz is not None
            or not np.all(np.diff(boundaries.asi8) > 0)):
        raise ValueError("Boundaries must be strictly increasing timezone-naive timestamps")
    hour_ns = pd.Timedelta(hours=1).value
    if np.any(frame.index.asi8 % hour_ns) or np.any(boundaries.asi8 % hour_ns):
        raise ValueError("Data and boundaries must align to whole hours")
    index = pd.date_range(boundaries[0], boundaries[-1], freq="1h", inclusive="left")
    data = frame.reindex(index).replace([np.inf, -np.inf], np.nan)
    values = data.to_numpy(dtype=np.float64)
    train_values = values[index < boundaries[1]]
    stats: dict[str, list[float]] = {}
    fitted_counts: dict[str, int] = {}
    for j, column in enumerate(data.columns):
        finite = train_values[:, j][np.isfinite(train_values[:, j])]
        if not finite.size:
            raise ValueError(f"Training sensor {column} has no finite observations")
        mean, std = float(finite.mean()), float(finite.std(ddof=0))
        stats[column] = [mean, std if std > 1e-12 else 1.0]
        fitted_counts[column] = int(finite.size)
    means, scales = np.asarray(list(stats.values())).T
    normalized = (values - means) / scales
    features = np.concatenate([normalized, extract_angular_time_features(index, [24, 168])], axis=1)
    features = features.astype(np.float32)
    target_idx = list(data.columns).index(target_col)
    time_values = index.to_numpy(dtype="datetime64[ns]")
    splits, diagnostics = {}, {}
    for split, lower, upper in zip(("train", "val", "test"), boundaries[:-1], boundaries[1:]):
        candidates = 0
        rejected_context = 0
        rejected_input = 0
        rejected_target = 0
        positions = []
        for first in np.flatnonzero((index >= lower) & (index < upper)):
            if first + horizon > len(index) or index[first + horizon - 1] >= upper:
                continue
            candidates += 1
            if first < lookback:
                rejected_context += 1
                continue
            if not np.isfinite(features[first - lookback:first]).all():
                rejected_input += 1
                continue
            if not np.isfinite(normalized[first:first + horizon, target_idx]).all():
                rejected_target += 1
                continue
            positions.append(first)
        starts = np.asarray(positions, dtype=np.int64)
        input_rows = starts[:, None] + np.arange(-lookback, 0)
        target_rows = starts[:, None] + np.arange(horizon)
        x = features[input_rows]
        y_raw = values[target_rows, target_idx]
        splits[split] = {
            "X": x, "X_sensor": x[..., :len(data.columns)],
            "y": normalized[target_rows, target_idx].astype(np.float32),
            "y_raw": y_raw,
            "origins": time_values[starts - 1],
            "target_times": time_values[target_rows],
            "input_times": time_values[input_rows],
        }
        segment_mask = (index >= lower) & (index < upper)
        diagnostics[split] = {
            "rows": int(segment_mask.sum()), "candidate_windows": candidates,
            "insufficient_context_windows": rejected_context,
            "nonfinite_input_windows": rejected_input,
            "nonfinite_target_windows": rejected_target,
            "valid_windows": len(starts),
            "raw_target_rows": _target_stats(values[segment_mask, target_idx]),
            "pooled_window_targets": _target_stats(y_raw),
            "unique_target_hours": int(np.unique(target_rows).size),
        }
    metadata = {
        "sensor_columns": list(data.columns), "target_col": target_col, "target_idx": target_idx,
        "lookback": int(lookback), "horizon": int(horizon), "time_periods": [24, 168],
        "norm_stats": stats, "normalization_ddof": 0,
        "normalization_finite_training_counts": fitted_counts,
        "normalization_fit_policy": "each sensor's finite training rows only; constant scale=1",
        "boundaries": dict(zip(("train_start", "val_start", "test_start", "test_end"),
                               map(str, boundaries))),
        "boundary_policy": "half-open intervals on observation availability timestamps",
        "window_policy": "all input sensors and future targets finite; earlier split input context allowed",
        "missing_hour_rows_inserted": int((~index.isin(frame.index)).sum()),
        "splits": diagnostics,
    }
    return splits, metadata
