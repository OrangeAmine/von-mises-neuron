"""Preprocessing utilities: resampling, Savitzky-Golay smoothing, normalization.

Each function is designed to operate on a single DataFrame partition
(train, val, or test) so that pipelines can apply them independently
per split to avoid data leakage.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def resample_hourly(
    df: pd.DataFrame,
    interval: str = "1h",
    method: str = "mean",
) -> pd.DataFrame:
    """Resample sensor data to fixed-interval bins using aggregation.

    Each bin uses only data within its own time window (non-overlapping),
    so this is safe to apply before splitting. Supports arbitrary
    intervals (e.g. "1h", "5min", "30min"), not just hourly.

    Note: The function name ``resample_hourly`` is retained for backward
    compatibility despite supporting sub-hourly intervals.

    Args:
        df: DataFrame with DateTimeIndex and sensor columns.
        interval: Resampling interval (e.g., "1h", "5min", "30min").
        method: Aggregation method ("mean", "median", "max").

    Returns:
        Resampled DataFrame.
    """
    if method == "mean":
        resampled = df.resample(interval).mean()
    elif method == "median":
        resampled = df.resample(interval).median()
    elif method == "max":
        resampled = df.resample(interval).max()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    # Drop rows that are entirely NaN (empty bins with no raw data)
    resampled = resampled.dropna(how="all")

    print(f"  Resampled {len(df)} rows -> {len(resampled)} rows "
          f"(interval={interval}, method={method})")
    return resampled


def apply_savgol(
    df: pd.DataFrame,
    window_length: int = 7,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Apply Savitzky-Golay filter independently to each sensor column.

    Handles NaN values by:
      1. Temporarily interpolating NaNs (linear) to allow filtering
      2. Applying the Savitzky-Golay filter
      3. Re-masking original NaN positions so we don't fabricate data

    This function should be called on individual splits (train/val/test)
    to avoid cross-split leakage.

    Args:
        df: DataFrame with sensor columns.
        window_length: Length of the filter window (must be odd, in samples).
            At 1h resolution, window_length=7 means a 7-hour window.
        polyorder: Order of the polynomial used to fit samples.

    Returns:
        Smoothed DataFrame with original NaN positions preserved.
    """
    # Ensure window_length is odd (required by savgol_filter)
    if window_length % 2 == 0:
        window_length += 1
        print(f"  Adjusted window_length to {window_length} (must be odd)")

    df = df.copy()

    for col in df.columns:
        nan_mask = df[col].isna()
        n_valid = (~nan_mask).sum()

        # Need at least window_length valid points to apply the filter
        if n_valid < window_length:
            print(f"  {col}: skipping Savitzky-Golay "
                  f"(only {n_valid} valid points < window={window_length})")
            continue

        # Temporarily fill NaNs for filtering
        filled = df[col].interpolate(method="linear", limit_direction="both")

        # Handle any remaining NaNs at edges with forward/backward fill
        filled = filled.ffill().bfill()

        # Apply the filter
        smoothed = savgol_filter(filled.values, window_length, polyorder)
        df[col] = smoothed

        # Re-mask original NaN positions
        df.loc[nan_mask, col] = np.nan

    print(f"  Applied Savitzky-Golay filter "
          f"(window={window_length}, polyorder={polyorder})")
    return df


def normalize(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Z-score normalize sensor columns.

    If stats is None (training set), compute mean and std from the data.
    If stats is provided (val/test set), apply those stats without
    recomputing — this prevents data leakage.

    Args:
        df: DataFrame with sensor columns.
        stats: Optional dict of {column: (mean, std)}. If None, stats
            are computed from df (use only for training set).

    Returns:
        Tuple of (normalized_df, stats_dict).

    Example:
        # Fit on train, transform all splits
        train_norm, stats = normalize(train_df)
        val_norm, _ = normalize(val_df, stats=stats)
        test_norm, _ = normalize(test_df, stats=stats)
    """
    df = df.copy()
    computed_stats: Dict[str, Tuple[float, float]] = {}

    for col in df.columns:
        if stats is None:
            # Compute from this data (training set)
            mean = df[col].mean()
            std = df[col].std()
            # Avoid division by zero for constant columns
            if std == 0 or np.isnan(std):
                std = 1.0
        else:
            mean, std = stats[col]

        df[col] = (df[col] - mean) / std
        computed_stats[col] = (float(mean), float(std))

    if stats is None:
        print("  Normalization stats computed from data (training set):")
    else:
        print("  Normalization stats applied from training set:")

    for col, (m, s) in computed_stats.items():
        print(f"    {col}: mean={m:.4f}, std={s:.4f}")

    return df, computed_stats
