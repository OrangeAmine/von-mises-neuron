"""Benchmark the Von Mises neuron forecaster on a synthetic time series.

Generates a deterministic sinusoidal signal (daily + weekly periodicities
with Gaussian noise) to validate that the VonMisesForecaster can learn a
structured periodic pattern and produce reasonable forecasts.

The synthetic signal mimics indoor CO2 dynamics at 1-hour resolution:
    y(t) = 600 + 50*sin(2*pi*t/24) + 15*sin(2*pi*t/168) + eps
where:
    24  = samples per day   (1h resolution)
    168 = samples per week  (7d x 24h)
    eps ~ N(0, 5)

Expected outcomes for a 12-hour forecast:
    - RMSE  < 35 ppm   (signal range ~550-700 ppm)
    - MAE   < 25 ppm
    - R2    > 0.20
    - MAPE  < 5.0%

Usage:
    python scripts/benchmark_standard_dataset.py
    python scripts/benchmark_standard_dataset.py --epochs 100
    python scripts/benchmark_standard_dataset.py --horizon 24 --lookback 48
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.von_mises_neuron import VonMisesForecaster
from src.baselines import (
    LSTMForecaster,
    TransformerForecaster,
    HandCraftedLinearForecaster,
)
from src.preprocessing import normalize
from src.sequences import create_sequences
from src.split import chronological_split
from src.temporal import (
    extract_angular_time_features,
    TemporalVonMisesForecaster,
)


# ---------------------------------------------------------------------------
# Acceptance thresholds for each metric (12h horizon on synthetic data)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "rmse": 35.0,
    "mae": 25.0,
    "r2": 0.20,
    "mape": 5.0,
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_co2(
    n_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic CO2-like time series with known structure.

    Signal components:
        - Base level:   600 ppm
        - Daily cycle:  50 * sin(2*pi*t / 24)
        - Weekly cycle: 15 * sin(2*pi*t / 168)
        - Noise:        N(0, 5)

    Additional synthetic features mimic a real sensor set:
        - Temperature: daily cycle (phase-shifted) + noise
        - Humidity: anti-correlated with temperature
        - Noise (dB): loosely correlated with CO2 activity

    Args:
        n_days: Number of days to generate.
        seed: Random seed for the noise component.

    Returns:
        DataFrame with hourly DatetimeIndex and columns
        [Temperature, Humidity, CO2, Noise].
    """
    rng = np.random.default_rng(seed)

    samples_per_day = 24
    samples_per_week = 7 * samples_per_day
    n_samples = n_days * samples_per_day

    t = np.arange(n_samples, dtype=np.float64)

    # Target: CO2
    daily_cycle = 50.0 * np.sin(2.0 * np.pi * t / samples_per_day)
    weekly_cycle = 15.0 * np.sin(2.0 * np.pi * t / samples_per_week)
    noise = rng.normal(0, 5, size=n_samples)
    co2 = 600.0 + daily_cycle + weekly_cycle + noise

    # Synthetic features
    temperature = (
        22.0
        + 3.0 * np.sin(2.0 * np.pi * t / samples_per_day - np.pi / 4)
        + rng.normal(0, 0.5, n_samples)
    )
    humidity = (
        50.0
        - 5.0 * np.sin(2.0 * np.pi * t / samples_per_day - np.pi / 4)
        + rng.normal(0, 1.0, n_samples)
    )
    noise_db = (
        45.0
        + 3.0 * np.sin(2.0 * np.pi * t / samples_per_day + 0.5)
        + rng.normal(0, 1.0, n_samples)
    )

    start_date = pd.Timestamp("2021-01-01", tz="Europe/Paris")
    datetimes = pd.date_range(start=start_date, periods=n_samples, freq="1h")

    df = pd.DataFrame(
        {
            "Temperature": temperature,
            "Humidity": humidity,
            "CO2": co2,
            "Noise": noise_db,
        },
        index=datetimes,
    )

    return df


def augment_with_time_features(
    df: pd.DataFrame,
    periods: List[int] = None,
) -> pd.DataFrame:
    """Add sin/cos time features for each specified period.

    Computes a continuous time index from the DataFrame's DatetimeIndex
    (hours since the first timestamp), then generates sin(2*pi*t/P) and
    cos(2*pi*t/P) for each period P.  These are the "hand-crafted" features
    that a domain expert would engineer — the VM neuron should learn them
    automatically.

    Args:
        df: DataFrame with DatetimeIndex.
        periods: List of periods in hours (default [24, 168] for daily/weekly).

    Returns:
        DataFrame with additional sin_P and cos_P columns for each period P.
    """
    if periods is None:
        periods = [24, 168]
    df = df.copy()
    # Hours since start — robust to irregular sampling
    t_hours = (df.index - df.index[0]).total_seconds() / 3600.0
    for p in periods:
        df[f"sin_{p}"] = np.sin(2.0 * np.pi * t_hours / p)
        df[f"cos_{p}"] = np.cos(2.0 * np.pi * t_hours / p)
    return df


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def prepare_benchmark_data(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    device: torch.device,
    target_col: str = "CO2",
    split_ratios: List[float] = None,
) -> Dict[str, Any]:
    """Split, normalize, and window data for benchmarking.

    Args:
        df: DataFrame with sensor columns and DatetimeIndex.
        lookback: Number of past timesteps per input window.
        horizon: Number of future timesteps to predict.
        device: PyTorch device (cuda or cpu).
        target_col: Column name for the prediction target.
        split_ratios: Train/val/test ratios (default [0.8, 0.1, 0.1]).

    Returns:
        Dict with X/y tensors, norm_stats, horizon, lookback, target_col.
    """
    if split_ratios is None:
        split_ratios = [0.8, 0.1, 0.1]

    train_df, val_df, test_df = chronological_split(
        df, ratios=split_ratios
    )

    train_norm, stats = normalize(train_df, stats=None)
    val_norm, _ = normalize(val_df, stats=stats)
    test_norm, _ = normalize(test_df, stats=stats)

    X_train_np, y_train_np, _ = create_sequences(
        train_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_val_np, y_val_np, _ = create_sequences(
        val_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_test_np, y_test_np, _ = create_sequences(
        test_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )

    data = {
        "X_train": torch.tensor(X_train_np, dtype=torch.float32).to(device),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32).to(device),
        "X_val": torch.tensor(X_val_np, dtype=torch.float32).to(device),
        "y_val": torch.tensor(y_val_np, dtype=torch.float32).to(device),
        "X_test": torch.tensor(X_test_np, dtype=torch.float32).to(device),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32).to(device),
        "norm_stats": stats,
        "horizon": horizon,
        "lookback": lookback,
        "target_col": target_col,
    }

    return data


def prepare_temporal_data(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    device: torch.device,
    time_periods: List[int] = None,
    target_col: str = "CO2",
    split_ratios: List[float] = None,
) -> Dict[str, Any]:
    """Split, normalize, add angular time features, and window data.

    Angular time features are added AFTER normalization to preserve their
    [0, 2π) range. Sensor features are z-scored normally.

    Args:
        df: DataFrame with sensor columns and DatetimeIndex.
        lookback: Number of past timesteps per input window.
        horizon: Number of future timesteps to predict.
        device: PyTorch device.
        time_periods: Periods in hours for angular features.
        target_col: Column name for the prediction target.
        split_ratios: Train/val/test ratios.

    Returns:
        Dict with X/y tensors, norm_stats, config.
    """
    if time_periods is None:
        time_periods = [24, 168]
    if split_ratios is None:
        split_ratios = [0.8, 0.1, 0.1]

    # Capture sensor column count BEFORE augmentation
    sensor_size = len(df.columns)

    train_df, val_df, test_df = chronological_split(df, ratios=split_ratios)

    # Normalize sensor features only
    train_norm, stats = normalize(train_df, stats=None)
    val_norm, _ = normalize(val_df, stats=stats)
    test_norm, _ = normalize(test_df, stats=stats)

    # Add angular time features AFTER normalization (bypass z-scoring)
    augmented = []
    for split_norm in [train_norm, val_norm, test_norm]:
        aug = split_norm.copy()
        angles = extract_angular_time_features(aug.index, periods=time_periods)
        for i, p in enumerate(time_periods):
            aug[f"angle_{p}"] = angles[:, i]
        augmented.append(aug)
    train_aug, val_aug, test_aug = augmented

    X_train_np, y_train_np, _ = create_sequences(
        train_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_val_np, y_val_np, _ = create_sequences(
        val_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_test_np, y_test_np, _ = create_sequences(
        test_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )

    data = {
        "X_train": torch.tensor(X_train_np, dtype=torch.float32).to(device),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32).to(device),
        "X_val": torch.tensor(X_val_np, dtype=torch.float32).to(device),
        "y_val": torch.tensor(y_val_np, dtype=torch.float32).to(device),
        "X_test": torch.tensor(X_test_np, dtype=torch.float32).to(device),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32).to(device),
        "norm_stats": stats,
        "horizon": horizon,
        "lookback": lookback,
        "target_col": target_col,
        "sensor_size": sensor_size,
        "time_periods": time_periods,
    }

    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_benchmark_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict[str, Tuple[float, float]],
    target_col: str = "CO2",
) -> Dict[str, float]:
    """Compute all benchmark metrics in the original (denormalized) scale.

    Args:
        y_pred: Predicted values, shape (N, horizon), z-scored.
        y_true: Ground truth values, shape (N, horizon), z-scored.
        norm_stats: Normalization statistics dict {col: (mean, std)}.
        target_col: Column name of the target variable (for denormalization).

    Returns:
        Dict with keys: rmse, mae, r2, mape.
    """
    mean, std = norm_stats[target_col]
    y_pred_orig = y_pred * std + mean
    y_true_orig = y_true * std + mean

    # RMSE
    mse = ((y_pred_orig - y_true_orig) ** 2).mean().item()
    rmse = mse ** 0.5

    # MAE
    mae = torch.abs(y_pred_orig - y_true_orig).mean().item()

    # R2 = 1 - SS_res / SS_tot
    ss_res = ((y_pred_orig - y_true_orig) ** 2).sum().item()
    ss_tot = ((y_true_orig - y_true_orig.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAPE
    abs_true_safe = torch.clamp(torch.abs(y_true_orig), min=1.0)
    mape = (
        torch.abs(y_pred_orig - y_true_orig) / abs_true_safe
    ).mean().item() * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    max_epochs: int = 50,
    batch_size: int = 64,
    patience: int = 15,
    gradient_clip: float = 1.0,
) -> nn.Module:
    """Train a neural model with early stopping and LR scheduling."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_von_mises(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the Von Mises neuron forecaster."""
    horizon = data["horizon"]
    input_size = data["X_train"].shape[2]

    model = VonMisesForecaster(
        input_size=input_size,
        hidden_size=64,
        num_vm_layers=2,
        dropout=0.1,
        output_size=horizon,
        init_kappa=1.0,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])


def benchmark_lstm(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the LSTM-only baseline (parameter-matched)."""
    horizon = data["horizon"]
    input_size = data["X_train"].shape[2]

    # hidden_size=68 approximately matches VonMisesForecaster(hidden=64) param count
    # VM total: 43,148 params → LSTM total: 43,668 params (+1.2%)
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=68,
        num_layers=2,
        dropout=0.1,
        output_size=horizon,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])


def benchmark_transformer(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the Transformer baseline."""
    horizon = data["horizon"]
    input_size = data["X_train"].shape[2]

    # d_model=48 with dim_feedforward=128 gives ~44,956 params (~4% over VM)
    model = TransformerForecaster(
        input_size=input_size,
        d_model=48,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_size=horizon,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])


def benchmark_handcrafted(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the linear + hand-crafted features baseline."""
    horizon = data["horizon"]
    lookback = data["lookback"]
    input_size = data["X_train"].shape[2]  # includes sin/cos columns

    model = HandCraftedLinearForecaster(
        input_size=input_size,
        lookback=lookback,
        output_size=horizon,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])


def benchmark_temporal_von_mises(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the Temporal Von Mises forecaster."""
    horizon = data["horizon"]
    sensor_size = data["sensor_size"]
    time_periods = data["time_periods"]

    model = TemporalVonMisesForecaster(
        sensor_size=sensor_size,
        time_periods=time_periods,
        hidden_size=64,
        num_vm_layers=2,
        freq_bands=[0.5, 2.0, 8.0],
        dropout=0.1,
        output_size=horizon,
        use_temporal_conv=True,
        conv_kernel_size=3,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])


# ---------------------------------------------------------------------------
# Threshold checking and reporting
# ---------------------------------------------------------------------------
def check_thresholds(
    metrics: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """Check whether metrics meet acceptance thresholds."""
    failures = []

    if metrics["rmse"] > THRESHOLDS["rmse"]:
        failures.append(
            f"  RMSE {metrics['rmse']:.2f} > {THRESHOLDS['rmse']} (threshold)"
        )
    if metrics["mae"] > THRESHOLDS["mae"]:
        failures.append(
            f"  MAE  {metrics['mae']:.2f} > {THRESHOLDS['mae']} (threshold)"
        )
    if metrics["r2"] < THRESHOLDS["r2"]:
        failures.append(
            f"  R2   {metrics['r2']:.4f} < {THRESHOLDS['r2']} (threshold)"
        )
    if metrics["mape"] > THRESHOLDS["mape"]:
        failures.append(
            f"  MAPE {metrics['mape']:.2f}% > {THRESHOLDS['mape']}% (threshold)"
        )

    return len(failures) == 0, failures


def check_regression(
    new_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
    max_regression_pct: float = 10.0,
) -> Tuple[bool, List[str]]:
    """Check that new model does not regress vs. base beyond a threshold.

    For RMSE, MAE, MAPE: new must not exceed base by more than max_regression_pct.
    For R2: new must not be below base by more than max_regression_pct (absolute).
    """
    failures = []
    for metric in ["rmse", "mae", "mape"]:
        base_val = base_metrics[metric]
        new_val = new_metrics[metric]
        if base_val > 0:
            pct_change = (new_val - base_val) / base_val * 100
            if pct_change > max_regression_pct:
                failures.append(
                    f"  {metric.upper()}: {new_val:.2f} is {pct_change:.1f}% worse "
                    f"than base {base_val:.2f} (max allowed: {max_regression_pct}%)"
                )
    # R2: higher is better, check absolute difference
    r2_diff = base_metrics["r2"] - new_metrics["r2"]
    if r2_diff > max_regression_pct / 100:
        failures.append(
            f"  R2: {new_metrics['r2']:.4f} is {r2_diff:.4f} below "
            f"base {base_metrics['r2']:.4f}"
        )
    return len(failures) == 0, failures


def print_report(
    metrics: Dict[str, float],
    elapsed: float,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> bool:
    """Print a formatted report and return pass/fail."""
    print("\n")
    print("=" * 68)
    print("  BENCHMARK RESULTS -- Von Mises Forecaster")
    print("=" * 68)
    print(f"\n  Signal: y(t) = 600 + 50*sin(daily) + 15*sin(weekly) + N(0,5)")
    print(f"  Resolution: 1 hour")
    print(f"  Lookback: {lookback}, Horizon: {horizon}")

    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"  Device: {device_str}")

    print(f"\n  Acceptance thresholds:")
    print(f"    RMSE  <= {THRESHOLDS['rmse']} ppm")
    print(f"    MAE   <= {THRESHOLDS['mae']} ppm")
    print(f"    R2    >= {THRESHOLDS['r2']}")
    print(f"    MAPE  <= {THRESHOLDS['mape']}%")

    passed, failures = check_thresholds(metrics)
    status_str = "PASS" if passed else "FAIL"

    print(f"\n  {'Metric':<8s} {'Value':>10s} {'Threshold':>12s}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 12}")
    print(f"  {'RMSE':<8s} {metrics['rmse']:>9.2f}  {'<= ' + str(THRESHOLDS['rmse']):>12s}")
    print(f"  {'MAE':<8s} {metrics['mae']:>9.2f}  {'<= ' + str(THRESHOLDS['mae']):>12s}")
    print(f"  {'R2':<8s} {metrics['r2']:>9.4f}  {'>= ' + str(THRESHOLDS['r2']):>12s}")
    print(f"  {'MAPE':<8s} {metrics['mape']:>8.2f}%  {'<= ' + str(THRESHOLDS['mape']) + '%':>12s}")
    print(f"\n  Time: {elapsed:.1f}s")

    if failures:
        for f in failures:
            print(f"    {f}")

    print(f"\n  {'=' * 68}")
    print(f"  RESULT: {status_str}")
    print(f"  {'=' * 68}\n")

    return passed


def print_comparison_report(
    all_results: Dict[str, Dict[str, Any]],
    lookback: int,
    horizon: int,
    device: torch.device,
    vm_passed: bool,
) -> None:
    """Print a side-by-side comparison table of all models."""
    print("\n")
    print("=" * 78)
    print("  BENCHMARK COMPARISON -- Von Mises vs. Baselines")
    print("=" * 78)
    print(f"\n  Signal: y(t) = 600 + 50*sin(daily) + 15*sin(weekly) + N(0,5)")
    print(f"  Resolution: 1 hour | Lookback: {lookback} | Horizon: {horizon}")

    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"  Device: {device_str}")

    # Table header
    print(f"\n  {'Model':<26s} {'Params':>8s} {'RMSE':>8s} {'MAE':>8s}"
          f" {'R2':>8s} {'MAPE':>8s} {'Time':>8s}")
    print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 8}"
          f" {'-' * 8} {'-' * 8} {'-' * 8}")

    for name, res in all_results.items():
        m = res["metrics"]
        params = res["params"]
        t = res["time_seconds"]
        print(
            f"  {name:<26s} {params:>8,d} {m['rmse']:>8.2f} {m['mae']:>8.2f}"
            f" {m['r2']:>8.4f} {m['mape']:>7.2f}% {t:>7.1f}s"
        )

    status_str = "PASS" if vm_passed else "FAIL"
    print(f"\n  VM acceptance thresholds: RMSE<={THRESHOLDS['rmse']}, "
          f"MAE<={THRESHOLDS['mae']}, R2>={THRESHOLDS['r2']}, "
          f"MAPE<={THRESHOLDS['mape']}%")
    print(f"  VM threshold check: {status_str}")
    print(f"  {'=' * 78}\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def _count_params(model: nn.Module) -> int:
    """Return total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


def _run_and_report(
    name: str,
    benchmark_fn,
    data: Dict[str, Any],
    epochs: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a single benchmark, print a summary line, and return results."""
    t0 = time.time()
    metrics = benchmark_fn(data, epochs, device)
    elapsed = time.time() - t0
    print(
        f"  {name}: RMSE={metrics['rmse']:.2f}  "
        f"MAE={metrics['mae']:.2f}  "
        f"R2={metrics['r2']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  "
        f"({elapsed:.1f}s)"
    )
    return {"metrics": metrics, "time_seconds": round(elapsed, 2)}


def main() -> None:
    """Entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark the Von Mises neuron forecaster on synthetic data"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)"
    )
    parser.add_argument(
        "--n-days", type=int, default=90,
        help="Days of synthetic data to generate (default: 90)"
    )
    parser.add_argument(
        "--horizon", type=int, default=12,
        help="Forecast horizon in steps (default: 12)"
    )
    parser.add_argument(
        "--lookback", type=int, default=72,
        help="Lookback window in steps (default: 72)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/benchmark",
        help="Directory for benchmark results JSON (default: results/benchmark)"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline models (only run VM forecaster)"
    )
    args = parser.parse_args()

    run_baselines = not args.skip_baselines
    n_phases = 8 if run_baselines else 5

    seed_everything(42)

    # --- Phase 1: Generate synthetic dataset ---
    print(f"\n[1/{n_phases}] Generating synthetic dataset...")
    df = generate_synthetic_co2(n_days=args.n_days)
    print(f"  Generated {len(df)} samples ({args.n_days} days @ 1h resolution)")
    print(f"  CO2 range: [{df['CO2'].min():.0f}, {df['CO2'].max():.0f}] ppm")

    # --- Phase 2: Prepare data pipelines ---
    print(f"\n[2/{n_phases}] Preparing data pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"  Device: {device_str}")

    data = prepare_benchmark_data(df, args.lookback, args.horizon, device)
    print(f"  Train: {data['X_train'].shape[0]} sequences")
    print(f"  Val:   {data['X_val'].shape[0]} sequences")
    print(f"  Test:  {data['X_test'].shape[0]} sequences")

    # Augmented pipeline for hand-crafted baseline (adds sin/cos columns)
    if run_baselines:
        df_augmented = augment_with_time_features(df, periods=[24, 168])
        data_augmented = prepare_benchmark_data(
            df_augmented, args.lookback, args.horizon, device
        )
        print(f"  Augmented features: {data_augmented['X_train'].shape[2]} "
              f"(+{data_augmented['X_train'].shape[2] - data['X_train'].shape[2]} crafted)")

    # Temporal pipeline for TemporalVonMisesForecaster (angular time features)
    data_temporal = prepare_temporal_data(
        df, args.lookback, args.horizon, device,
        time_periods=[24, 168],
    )
    print(f"  Temporal features: {data_temporal['X_train'].shape[2]} "
          f"(+{data_temporal['X_train'].shape[2] - data['X_train'].shape[2]} angular)")

    # --- Phase 3: Train & evaluate VonMisesForecaster ---
    print(f"\n[3/{n_phases}] Training VonMisesForecaster...")
    seed_everything(42)
    vm_result = _run_and_report(
        "VonMises", benchmark_von_mises, data, args.epochs, device
    )

    # Compute VM param count for reference
    vm_model_tmp = VonMisesForecaster(
        input_size=data["X_train"].shape[2], hidden_size=64,
        num_vm_layers=2, dropout=0.1, output_size=data["horizon"],
    )
    vm_result["params"] = _count_params(vm_model_tmp)
    del vm_model_tmp

    # Check VM thresholds (the only model with acceptance criteria)
    vm_passed, vm_failures = check_thresholds(vm_result["metrics"])

    # Collect all results for comparison
    all_results: Dict[str, Dict[str, Any]] = {"VonMises": vm_result}

    # --- Phase 4: Train & evaluate TemporalVonMisesForecaster ---
    print(f"\n[4/{n_phases}] Training TemporalVonMisesForecaster...")
    seed_everything(42)
    tvm_result = _run_and_report(
        "TemporalVM", benchmark_temporal_von_mises,
        data_temporal, args.epochs, device
    )
    tvm_model_tmp = TemporalVonMisesForecaster(
        sensor_size=data_temporal["sensor_size"],
        time_periods=data_temporal["time_periods"],
        hidden_size=64, num_vm_layers=2,
        freq_bands=[0.5, 2.0, 8.0], dropout=0.1,
        output_size=data["horizon"],
    )
    tvm_result["params"] = _count_params(tvm_model_tmp)
    del tvm_model_tmp
    all_results["TemporalVM"] = tvm_result

    # Check SC3: TemporalVM must not regress >10% vs. base VonMises
    tvm_ok, tvm_failures = check_regression(
        tvm_result["metrics"], vm_result["metrics"]
    )
    if not tvm_ok:
        print("\n  WARNING: TemporalVM regressed vs. base VonMises:")
        for f in tvm_failures:
            print(f"    {f}")

    if run_baselines:
        # --- Phase 5: Train & evaluate LSTM baseline ---
        print(f"\n[5/{n_phases}] Training LSTMForecaster (baseline)...")
        seed_everything(42)
        lstm_result = _run_and_report(
            "LSTM", benchmark_lstm, data, args.epochs, device
        )
        lstm_model_tmp = LSTMForecaster(
            input_size=data["X_train"].shape[2], hidden_size=68,
            num_layers=2, dropout=0.1, output_size=data["horizon"],
        )
        lstm_result["params"] = _count_params(lstm_model_tmp)
        del lstm_model_tmp
        all_results["LSTM"] = lstm_result

        # --- Phase 6: Train & evaluate Transformer baseline ---
        print(f"\n[6/{n_phases}] Training TransformerForecaster (baseline)...")
        seed_everything(42)
        transformer_result = _run_and_report(
            "Transformer", benchmark_transformer, data, args.epochs, device
        )
        tf_model_tmp = TransformerForecaster(
            input_size=data["X_train"].shape[2], d_model=48, nhead=4,
            num_layers=2, dim_feedforward=128, dropout=0.1,
            output_size=data["horizon"],
        )
        transformer_result["params"] = _count_params(tf_model_tmp)
        del tf_model_tmp
        all_results["Transformer"] = transformer_result

        # --- Phase 7: Train & evaluate HandCrafted baseline ---
        print(f"\n[7/{n_phases}] Training HandCraftedLinear (baseline)...")
        seed_everything(42)
        hc_result = _run_and_report(
            "HandCrafted", benchmark_handcrafted,
            data_augmented, args.epochs, device
        )
        hc_model_tmp = HandCraftedLinearForecaster(
            input_size=data_augmented["X_train"].shape[2],
            lookback=args.lookback, output_size=data["horizon"],
        )
        hc_result["params"] = _count_params(hc_model_tmp)
        del hc_model_tmp
        all_results["HandCrafted"] = hc_result

    # --- Final phase: Report ---
    print(f"\n[{n_phases}/{n_phases}] Generating report...")

    if run_baselines:
        print_comparison_report(
            all_results, args.lookback, args.horizon, device, vm_passed
        )
    else:
        print_report(
            vm_result["metrics"], vm_result["time_seconds"],
            args.lookback, args.horizon, device,
        )

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.json"

    json_results = {
        "config": {
            "n_days": args.n_days,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "seed": 42,
            "epochs": args.epochs,
            "device": str(device),
        },
        "thresholds": THRESHOLDS,
        "results": {},
        "all_passed": vm_passed,
    }

    for name, res in all_results.items():
        is_vm = name == "VonMises"
        json_results["results"][name] = {
            **res["metrics"],
            "params": res["params"],
            "time_seconds": res["time_seconds"],
            "passed": vm_passed if is_vm else None,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to: {output_path}")

    sys.exit(0 if vm_passed else 1)


if __name__ == "__main__":
    main()
