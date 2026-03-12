"""Sliding window sequence creation and PyTorch tensor export.

Converts tabular sensor DataFrames into (X, y) sequence pairs suitable
for training temporal deep learning models (LSTM, Transformer, TCN, etc.).

X: (N, lookback, n_features)  -- all sensors as input
y: (N, horizon)               -- CO2 forecast target
t_indices: (N,)               -- integer index into original DataFrame
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def create_sequences(
    df: pd.DataFrame,
    lookback: int = 72,
    horizon: int = 24,
    target_col: str = "CO2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding window sequences from a sensor DataFrame.

    Windows containing any NaN are skipped entirely -- we never feed
    partial or fabricated data to the network.

    Args:
        df: Normalized DataFrame with sensor columns and DateTimeIndex.
        lookback: Number of past timesteps for the input window.
        horizon: Number of future timesteps to predict.
        target_col: Column name for the prediction target.

    Returns:
        Tuple of (X, y, t_indices) where:
            X: np.ndarray of shape (N, lookback, n_features)
            y: np.ndarray of shape (N, horizon)
            t_indices: np.ndarray of shape (N,) -- integer index of the
                last lookback step for each sequence (into the original df).
        N is the number of valid (NaN-free) windows.

    Raises:
        ValueError: If target_col is not in the DataFrame columns.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    values = df.values  # (T, n_features)
    target_idx = list(df.columns).index(target_col)
    n_samples = len(df) - lookback - horizon + 1

    if n_samples <= 0:
        print(f"  WARNING: Not enough data for sequences "
              f"(len={len(df)}, need >= {lookback + horizon})")
        return (
            np.empty((0, lookback, df.shape[1])),
            np.empty((0, horizon)),
            np.empty((0,), dtype=np.int64),
        )

    X_list = []
    y_list = []
    t_list = []
    skipped = 0

    for i in range(n_samples):
        x_window = values[i : i + lookback]                      # (lookback, n_features)
        y_window = values[i + lookback : i + lookback + horizon, target_idx]  # (horizon,)

        # Skip windows with any NaN
        if np.isnan(x_window).any() or np.isnan(y_window).any():
            skipped += 1
            continue

        X_list.append(x_window)
        y_list.append(y_window)
        # Timestamp index = position of the last lookback step
        t_list.append(i + lookback - 1)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, lookback, df.shape[1]))
    y = np.stack(y_list, axis=0) if y_list else np.empty((0, horizon))
    t_indices = np.array(t_list, dtype=np.int64)

    print(f"  Sequences: {len(X_list)} valid, {skipped} skipped (contained NaN)")
    print(f"    X shape: {X.shape}  (N, lookback={lookback}, features={df.shape[1]})")
    print(f"    y shape: {y.shape}  (N, horizon={horizon})")

    return X, y, t_indices


def save_tensors(
    X: np.ndarray,
    y: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[dict] = None,
    t_indices: Optional[np.ndarray] = None,
) -> None:
    """Save (X, y) arrays as a PyTorch .pt file with optional metadata.

    The file contains a dict with keys "X", "y" (float32 tensors),
    optionally "metadata" (dict), and optionally "timestamps_idx"
    (int64 tensor of DataFrame indices for each sequence).

    Args:
        X: Input array of shape (N, lookback, n_features).
        y: Target array of shape (N, horizon).
        path: Output file path (should end in .pt).
        metadata: Optional dict of configuration parameters to embed
            (e.g., lookback, horizon, target, normalization stats).
        t_indices: Optional array of integer indices into the original
            DataFrame, shape (N,). Each entry is the position of the
            last lookback step for that sequence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32),
    }
    if metadata is not None:
        data["metadata"] = metadata
    if t_indices is not None:
        data["timestamps_idx"] = torch.tensor(t_indices, dtype=torch.int64)

    torch.save(data, path)
    print(f"  Saved tensors to {path}")
    print(f"    X: {data['X'].shape}, y: {data['y'].shape}")
