"""Chronological train/validation/test splitting for time series data.

Splits are strictly temporal — no shuffling. This is essential for
time series to prevent future information from leaking into training.
"""

from typing import List, Tuple

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    ratios: List[float] = [0.8, 0.1, 0.1],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train/val/test by chronological order.

    Args:
        df: DataFrame with a sorted DateTimeIndex.
        ratios: List of three floats summing to 1.0 for
            [train_ratio, val_ratio, test_ratio].

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to ~1.0 or df is empty.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame")

    n = len(df)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Report split boundaries
    print(f"\n  Chronological Split (ratios={ratios}):")
    print(f"  {'Set':<8} {'Rows':>7}  {'Start':<25} {'End':<25}")
    print(f"  {'-' * 70}")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:<8} {len(split):>7}  "
              f"{str(split.index.min()):<25} {str(split.index.max()):<25}")
    print()

    return train_df, val_df, test_df
