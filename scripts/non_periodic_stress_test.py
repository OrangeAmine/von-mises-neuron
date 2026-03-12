"""Stress test: VonMisesForecaster on data with NO periodic structure.

The VM neuron is designed for periodic signals. This script tests its
behavior on signals that have *no* periodicity, answering two questions:

1. **Does it hurt?** On a random walk / AR(1) process, the VM encoder
   should be neutral at worst -- it can learn broad kappa (flat response)
   that effectively passes inputs through. If it hurts, the cos()
   nonlinearity is distorting useful information.

2. **Does it distort trends?** On a trend-dominated signal (exponential
   growth + noise), the VM encoder must learn to "pass through" the
   monotonic pattern. Since cos() wraps every 2*pi, a large trend in
   z-scored space could be destructively folded.

We compare VonMisesForecaster vs LSTMForecaster (same architecture, no
VM layers) on both signals, then inspect the learned kappa values.

Expected outcomes:
- LSTM should match or beat VM on both non-periodic signals.
- VM's learned kappa should converge to LOW values (broad response),
  since sharp selectivity is counterproductive without periodicity.
- VM should NOT catastrophically fail -- if it does, cos() wrapping
  is a liability on non-periodic data.

Usage:
    python scripts/non_periodic_stress_test.py
    python scripts/non_periodic_stress_test.py --epochs 50
"""

import argparse
import json
import math
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
from src.baselines import LSTMForecaster
from src.preprocessing import normalize
from src.sequences import create_sequences
from src.split import chronological_split


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
# Synthetic non-periodic data generators
# ---------------------------------------------------------------------------
def generate_random_walk(n_samples: int = 2160, seed: int = 42) -> pd.DataFrame:
    """Generate a pure random walk with correlated auxiliary features.

    The target is a random walk (cumulative sum of Gaussian increments),
    which has NO periodic structure -- it's the canonical non-stationary,
    non-periodic process.

    Features:
        - Target: random walk, y(t) = y(t-1) + N(0, 1)
        - Momentum: exponential moving average of Target (lagging indicator)
        - Volatility: rolling std of increments (heteroscedasticity proxy)
        - Noise: pure i.i.d. Gaussian (uninformative control)

    Args:
        n_samples: Number of time steps (default 2160 = 90 days hourly).
        seed: Random seed.

    Returns:
        DataFrame with hourly DatetimeIndex and 4 columns.
    """
    rng = np.random.default_rng(seed)

    increments = rng.normal(0, 1, n_samples)
    target = np.cumsum(increments)

    # Exponential moving average (span=24h) as a correlated lagging feature
    alpha = 2.0 / (24 + 1)
    momentum = np.zeros(n_samples)
    momentum[0] = target[0]
    for i in range(1, n_samples):
        momentum[i] = alpha * target[i] + (1 - alpha) * momentum[i - 1]

    # Rolling volatility (24h window) -- captures changing variance
    volatility = np.zeros(n_samples)
    for i in range(n_samples):
        start = max(0, i - 23)
        volatility[i] = increments[start:i + 1].std() if i > 0 else 1.0

    noise = rng.normal(0, 1, n_samples)

    idx = pd.date_range("2021-01-01", periods=n_samples, freq="1h")
    return pd.DataFrame({
        "Target": target,
        "Momentum": momentum,
        "Volatility": volatility,
        "Noise": noise,
    }, index=idx)


def generate_trend_signal(n_samples: int = 2160, seed: int = 42) -> pd.DataFrame:
    """Generate a trend-dominated signal with NO periodic structure.

    The target follows exponential growth + mean-reverting noise,
    mimicking stock-price-like behavior. There is no daily/weekly
    cycle -- just a persistent upward drift.

    Signal: y(t) = 100 * exp(0.001 * t) + AR(1) noise
    The AR(1) noise has rho=0.95 (highly persistent but NOT periodic).

    Features:
        - Target: exponential growth + AR(1) noise
        - Trend: the pure exponential component (leaky oracle)
        - Deviation: Target - Trend (the AR(1) residual)
        - Noise: pure i.i.d. Gaussian (uninformative control)

    Args:
        n_samples: Number of time steps (default 2160 = 90 days hourly).
        seed: Random seed.

    Returns:
        DataFrame with hourly DatetimeIndex and 4 columns.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples, dtype=float)

    # Exponential growth: 100 * exp(0.001 * t)
    # At t=2160, this reaches ~100 * exp(2.16) = ~868
    trend = 100.0 * np.exp(0.001 * t)

    # AR(1) noise: x(t) = 0.95 * x(t-1) + N(0, 2)
    # High autocorrelation but no periodicity
    ar_noise = np.zeros(n_samples)
    for i in range(1, n_samples):
        ar_noise[i] = 0.95 * ar_noise[i - 1] + rng.normal(0, 2)

    target = trend + ar_noise

    deviation = ar_noise.copy()
    noise = rng.normal(0, 1, n_samples)

    idx = pd.date_range("2021-01-01", periods=n_samples, freq="1h")
    return pd.DataFrame({
        "Target": target,
        "Trend": trend,
        "Deviation": deviation,
        "Noise": noise,
    }, index=idx)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def prepare_data(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    device: torch.device,
    target_col: str = "Target",
) -> Dict[str, Any]:
    """Split, normalize, and window data for benchmarking.

    Standard pipeline: chronological split -> z-score -> sliding windows.
    """
    train_df, val_df, test_df = chronological_split(
        df, ratios=[0.8, 0.1, 0.1]
    )

    train_norm, stats = normalize(train_df, stats=None)
    val_norm, _ = normalize(val_df, stats=stats)
    test_norm, _ = normalize(test_df, stats=stats)

    X_train, y_train, _ = create_sequences(
        train_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_val, y_val, _ = create_sequences(
        val_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_test, y_test, _ = create_sequences(
        test_norm, lookback=lookback, horizon=horizon, target_col=target_col
    )

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32).to(device),
        "y_train": torch.tensor(y_train, dtype=torch.float32).to(device),
        "X_val": torch.tensor(X_val, dtype=torch.float32).to(device),
        "y_val": torch.tensor(y_val, dtype=torch.float32).to(device),
        "X_test": torch.tensor(X_test, dtype=torch.float32).to(device),
        "y_test": torch.tensor(y_test, dtype=torch.float32).to(device),
        "norm_stats": stats,
        "target_col": target_col,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 50,
    batch_size: int = 64,
    patience: int = 15,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
) -> nn.Module:
    """Train with early stopping (same loop as benchmark script)."""
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

            if not torch.isfinite(loss):
                # Diverged -- return current state
                model.eval()
                return model

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if not math.isfinite(val_loss):
            break

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
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict[str, Tuple[float, float]],
    target_col: str = "Target",
) -> Dict[str, float]:
    """Compute RMSE, MAE, R2 in original scale."""
    mean, std = norm_stats[target_col]
    yp = y_pred * std + mean
    yt = y_true * std + mean

    mse = ((yp - yt) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = torch.abs(yp - yt).mean().item()

    ss_res = ((yp - yt) ** 2).sum().item()
    ss_tot = ((yt - yt.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_experiment(
    signal_name: str,
    df: pd.DataFrame,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train VM and LSTM on a signal, compare metrics and inspect kappa."""

    lookback = 72
    horizon = 12

    data = prepare_data(df, lookback, horizon, device)
    input_size = data["X_train"].shape[2]

    results = {}

    # --- VonMises ---
    seed_everything(seed)
    vm_model = VonMisesForecaster(
        input_size=input_size,
        hidden_size=64,
        num_vm_layers=2,
        dropout=0.1,
        output_size=horizon,
        init_kappa=1.0,
    ).to(device)

    t0 = time.time()
    vm_model = _train_model(
        vm_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )
    vm_time = time.time() - t0

    with torch.no_grad():
        vm_pred = vm_model(data["X_test"])
    vm_metrics = compute_metrics(vm_pred, data["y_test"], data["norm_stats"])

    # Inspect learned kappa -- key diagnostic
    vm_kappas = vm_model.get_layer_concentrations()
    kappa_layer0 = vm_kappas[0].cpu().numpy()  # shape (64, 4)

    results["VonMises"] = {
        "metrics": vm_metrics,
        "params": _count_params(vm_model),
        "time": round(vm_time, 2),
        "kappa_stats": {
            "mean": float(kappa_layer0.mean()),
            "std": float(kappa_layer0.std()),
            "max": float(kappa_layer0.max()),
            "min": float(kappa_layer0.min()),
        },
    }

    # --- LSTM baseline ---
    seed_everything(seed)
    lstm_model = LSTMForecaster(
        input_size=input_size,
        hidden_size=68,  # parameter-matched to VM
        num_layers=2,
        dropout=0.1,
        output_size=horizon,
    ).to(device)

    t0 = time.time()
    lstm_model = _train_model(
        lstm_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )
    lstm_time = time.time() - t0

    with torch.no_grad():
        lstm_pred = lstm_model(data["X_test"])
    lstm_metrics = compute_metrics(lstm_pred, data["y_test"], data["norm_stats"])

    results["LSTM"] = {
        "metrics": lstm_metrics,
        "params": _count_params(lstm_model),
        "time": round(lstm_time, 2),
    }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_signal_report(
    signal_name: str,
    signal_desc: str,
    results: Dict[str, Any],
) -> None:
    """Print comparison table for one signal type."""
    print(f"\n{'=' * 72}")
    print(f"  SIGNAL: {signal_name}")
    print(f"  {signal_desc}")
    print(f"{'=' * 72}")

    print(f"\n  {'Model':<12s} {'Params':>8s} {'RMSE':>10s} {'MAE':>10s}"
          f" {'R2':>8s} {'Time':>7s}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10}"
          f" {'-' * 8} {'-' * 7}")

    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<12s} {res['params']:>8,d} {m['rmse']:>10.4f} "
              f"{m['mae']:>10.4f} {m['r2']:>8.4f} {res['time']:>6.1f}s")

    # VM vs LSTM comparison
    vm = results["VonMises"]["metrics"]
    lstm = results["LSTM"]["metrics"]
    rmse_delta = ((vm["rmse"] - lstm["rmse"]) / lstm["rmse"]) * 100
    r2_delta = vm["r2"] - lstm["r2"]

    print(f"\n  VM vs LSTM: RMSE {rmse_delta:+.1f}%,  R2 {r2_delta:+.4f}")

    if abs(rmse_delta) < 5.0:
        verdict = "NEUTRAL -- VM neither helps nor hurts"
    elif rmse_delta > 0:
        verdict = "VM HURTS -- cos() wrapping distorts non-periodic patterns"
    else:
        verdict = "VM HELPS -- unexpected on non-periodic data"

    print(f"  Verdict: {verdict}")

    # Kappa analysis
    ks = results["VonMises"]["kappa_stats"]
    print(f"\n  LEARNED KAPPA (first VM layer):")
    print(f"    Mean: {ks['mean']:.4f}  (init: 1.0)")
    print(f"    Std:  {ks['std']:.4f}")
    print(f"    Min:  {ks['min']:.4f}  Max: {ks['max']:.4f}")

    if ks["mean"] < 0.8:
        kappa_verdict = "kappa DECREASED -- neurons broadened (good: learned to be non-selective)"
    elif ks["mean"] > 1.2:
        kappa_verdict = "kappa INCREASED -- neurons sharpened (bad: overfitting to noise patterns)"
    else:
        kappa_verdict = "kappa STABLE -- neurons stayed near initialization"

    print(f"    {kappa_verdict}")
    print()


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the non-periodic stress test."""
    parser = argparse.ArgumentParser(
        description="Stress test VonMisesForecaster on non-periodic data"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/stress_test",
        help="Directory for results JSON (default: results/stress_test)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"

    print(f"\n  NON-PERIODIC STRESS TEST")
    print(f"  Device: {device_str}")
    print(f"  Epochs: {args.epochs}")

    all_results = {}

    # --- Signal 1: Random Walk ---
    print(f"\n  [1/2] Generating random walk signal...")
    rw_df = generate_random_walk(n_samples=2160, seed=args.seed)
    print(f"    Target range: [{rw_df['Target'].min():.1f}, "
          f"{rw_df['Target'].max():.1f}]")

    print(f"  [1/2] Training models on random walk...")
    rw_results = run_experiment(
        "random_walk", rw_df, args.epochs, device, args.seed
    )
    all_results["random_walk"] = rw_results

    print_signal_report(
        "Random Walk (AR(0) -- no periodicity)",
        "y(t) = y(t-1) + N(0,1) -- pure Brownian motion",
        rw_results,
    )

    # --- Signal 2: Trend ---
    print(f"\n  [2/2] Generating trend signal...")
    trend_df = generate_trend_signal(n_samples=2160, seed=args.seed)
    print(f"    Target range: [{trend_df['Target'].min():.1f}, "
          f"{trend_df['Target'].max():.1f}]")

    print(f"  [2/2] Training models on trend signal...")
    trend_results = run_experiment(
        "trend", trend_df, args.epochs, device, args.seed
    )
    all_results["trend"] = trend_results

    print_signal_report(
        "Exponential Trend + AR(1) Noise (no periodicity)",
        "y(t) = 100*exp(0.001*t) + AR(1, rho=0.95) noise",
        trend_results,
    )

    # --- Summary ---
    print(f"{'=' * 72}")
    print(f"  STRESS TEST SUMMARY")
    print(f"{'=' * 72}")

    for signal_name, results in all_results.items():
        vm_rmse = results["VonMises"]["metrics"]["rmse"]
        lstm_rmse = results["LSTM"]["metrics"]["rmse"]
        delta = ((vm_rmse - lstm_rmse) / lstm_rmse) * 100
        kappa_mean = results["VonMises"]["kappa_stats"]["mean"]

        print(f"\n  {signal_name}:")
        print(f"    VM RMSE={vm_rmse:.4f}  LSTM RMSE={lstm_rmse:.4f}  "
              f"delta={delta:+.1f}%")
        print(f"    Learned mean kappa: {kappa_mean:.4f} (init: 1.0)")

    # Final verdict
    print(f"\n  OVERALL ASSESSMENT:")
    rw_delta = ((all_results["random_walk"]["VonMises"]["metrics"]["rmse"]
                 - all_results["random_walk"]["LSTM"]["metrics"]["rmse"])
                / all_results["random_walk"]["LSTM"]["metrics"]["rmse"]) * 100
    trend_delta = ((all_results["trend"]["VonMises"]["metrics"]["rmse"]
                    - all_results["trend"]["LSTM"]["metrics"]["rmse"])
                   / all_results["trend"]["LSTM"]["metrics"]["rmse"]) * 100

    max_hurt = max(rw_delta, trend_delta)
    if max_hurt < 5.0:
        print(f"    VM is SAFE on non-periodic data (max degradation: "
              f"{max_hurt:+.1f}%)")
    elif max_hurt < 15.0:
        print(f"    VM shows MILD degradation on non-periodic data "
              f"({max_hurt:+.1f}%)")
    else:
        print(f"    VM shows SIGNIFICANT degradation on non-periodic data "
              f"({max_hurt:+.1f}%)")
        print(f"    cos() wrapping is a liability when inputs have no periodicity")

    print(f"\n{'=' * 72}\n")

    # --- Save JSON ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stress_test_results.json"

    json_output = {
        "config": {
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
            "lookback": 72,
            "horizon": 12,
        },
        "signals": {},
    }
    for signal_name, results in all_results.items():
        json_output["signals"][signal_name] = {
            model_name: {
                "metrics": res["metrics"],
                "params": res["params"],
                "time_seconds": res["time"],
                **({"kappa_stats": res["kappa_stats"]}
                   if "kappa_stats" in res else {}),
            }
            for model_name, res in results.items()
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
