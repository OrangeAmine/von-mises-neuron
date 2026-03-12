"""Benchmark the Von Mises neuron forecaster on real-world datasets.

Downloads, preprocesses, and evaluates VonMisesForecaster + baselines on
three public datasets where periodicity is natural but noisy:

1. **UCI Occupancy Detection** -- indoor CO2 with daily occupancy cycles.
2. **UCI Household Electric Power** -- daily + weekly power load patterns.
3. **Jena Climate** -- temperature with diurnal + seasonal cycles.

Each dataset exercises the VM neuron's ability to learn periodic patterns
from real (non-synthetic) signals, testing whether its learnable kappa
and mu parameters outperform fixed sin/cos features and standard LSTM/
Transformer architectures.

Usage:
    python scripts/benchmark_real_datasets.py
    python scripts/benchmark_real_datasets.py --datasets occupancy jena
    python scripts/benchmark_real_datasets.py --epochs 100 --skip-baselines
    python scripts/benchmark_real_datasets.py --data-dir data/raw --output-dir results/real
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
    compute_param_matched_hidden_size,
)
from src.preprocessing import normalize
from src.sequences import create_sequences
from src.split import chronological_split
from src.data_loading import load_dataset, list_datasets, DATASET_CONFIGS


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
# Time feature augmentation (for HandCrafted baseline)
# ---------------------------------------------------------------------------
def augment_with_time_features(
    df: pd.DataFrame,
    periods: List[int],
) -> pd.DataFrame:
    """Add sin/cos time features for each specified period.

    Computes a continuous time index (hours since first timestamp),
    then generates sin(2*pi*t/P) and cos(2*pi*t/P) for each period P.

    Args:
        df: DataFrame with DatetimeIndex.
        periods: List of periods in hours.

    Returns:
        DataFrame with additional sin_P and cos_P columns.
    """
    df = df.copy()
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
    target_col: str,
    split_ratios: List[float],
) -> Dict[str, Any]:
    """Split, normalize, and window data for benchmarking.

    Args:
        df: DataFrame with sensor columns and DatetimeIndex.
        lookback: Number of past timesteps per input window.
        horizon: Number of future timesteps to predict.
        device: PyTorch device (cuda or cpu).
        target_col: Column name for the prediction target.
        split_ratios: Train/val/test ratios.

    Returns:
        Dict with X/y tensors, norm_stats, horizon, lookback, target_col.
    """
    train_df, val_df, test_df = chronological_split(df, ratios=split_ratios)

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_benchmark_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict[str, Tuple[float, float]],
    target_col: str,
) -> Dict[str, float]:
    """Compute benchmark metrics in the original (denormalized) scale.

    Args:
        y_pred: Predicted values (z-scored), shape (N, horizon).
        y_true: Ground truth values (z-scored), shape (N, horizon).
        norm_stats: Normalization statistics {col: (mean, std)}.
        target_col: Column name for denormalization.

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

    # MAPE (clamp denominator to avoid division by near-zero targets)
    abs_true_safe = torch.clamp(torch.abs(y_true_orig), min=1.0)
    mape = (
        torch.abs(y_pred_orig - y_true_orig) / abs_true_safe
    ).mean().item() * 100.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Batched inference (avoids CUDA OOM on large datasets)
# ---------------------------------------------------------------------------
def _batched_predict(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run inference in mini-batches to avoid CUDA OOM.

    Large datasets (e.g. Jena with 70k sequences of lookback=168) create
    huge intermediate tensors in VonMisesLinear's broadcasting step.
    Processing in chunks of 256 keeps peak GPU memory manageable.
    """
    if X.shape[0] <= batch_size:
        return model(X)

    chunks = []
    for i in range(0, X.shape[0], batch_size):
        chunks.append(model(X[i : i + batch_size]))
    return torch.cat(chunks, dim=0)


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

        # Validate in mini-batches to avoid CUDA OOM on large datasets
        model.eval()
        with torch.no_grad():
            val_pred = _batched_predict(model, X_val)
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
# Model helpers
# ---------------------------------------------------------------------------
def _count_params(model: nn.Module) -> int:
    """Return total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


def _build_vm(input_size: int, horizon: int, device: torch.device) -> VonMisesForecaster:
    """Build a VonMisesForecaster with standard hyperparameters."""
    return VonMisesForecaster(
        input_size=input_size,
        hidden_size=64,
        num_vm_layers=2,
        dropout=0.1,
        output_size=horizon,
        init_kappa=1.0,
    ).to(device)


def _build_lstm(input_size: int, horizon: int, vm_params: int, device: torch.device) -> LSTMForecaster:
    """Build a parameter-matched LSTMForecaster."""
    h = compute_param_matched_hidden_size(
        vm_params, input_size=input_size, output_size=horizon
    )
    return LSTMForecaster(
        input_size=input_size,
        hidden_size=h,
        num_layers=2,
        dropout=0.1,
        output_size=horizon,
    ).to(device)


def _build_transformer(input_size: int, horizon: int, device: torch.device) -> TransformerForecaster:
    """Build a TransformerForecaster with standard hyperparameters."""
    return TransformerForecaster(
        input_size=input_size,
        d_model=48,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_size=horizon,
    ).to(device)


def _build_handcrafted(input_size: int, lookback: int, horizon: int, device: torch.device) -> HandCraftedLinearForecaster:
    """Build a HandCraftedLinearForecaster."""
    return HandCraftedLinearForecaster(
        input_size=input_size,
        lookback=lookback,
        output_size=horizon,
    ).to(device)


# ---------------------------------------------------------------------------
# Single-model benchmark runner
# ---------------------------------------------------------------------------
def _run_single(
    name: str,
    model: nn.Module,
    data: Dict[str, Any],
    epochs: int,
) -> Dict[str, Any]:
    """Train, evaluate, and return results for a single model."""
    t0 = time.time()

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = _batched_predict(model, data["X_test"])

    metrics = compute_benchmark_metrics(
        y_pred, data["y_test"], data["norm_stats"], data["target_col"]
    )
    elapsed = time.time() - t0
    params = _count_params(model)

    print(
        f"    {name:<22s}  "
        f"RMSE={metrics['rmse']:>8.3f}  "
        f"MAE={metrics['mae']:>8.3f}  "
        f"R2={metrics['r2']:>7.4f}  "
        f"MAPE={metrics['mape']:>6.2f}%  "
        f"Params={params:>7,d}  "
        f"({elapsed:.1f}s)"
    )

    return {
        "metrics": metrics,
        "params": params,
        "time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------
def benchmark_dataset(
    dataset_name: str,
    epochs: int,
    device: torch.device,
    data_dir: Path,
    skip_baselines: bool = False,
) -> Dict[str, Any]:
    """Run the full benchmark suite on a single dataset.

    Args:
        dataset_name: One of "occupancy", "power", "jena".
        epochs: Maximum training epochs.
        device: PyTorch device.
        data_dir: Root data directory for downloads.
        skip_baselines: If True, only run VonMises model.

    Returns:
        Dict with dataset config and per-model results.
    """
    # 1. Load dataset
    df, config = load_dataset(dataset_name, data_dir=data_dir)

    target_col = config["target_col"]
    lookback = config["lookback"]
    horizon = config["horizon"]
    unit = config["unit"]
    split_ratios = config["split_ratios"]
    periods = config["handcrafted_periods"]

    print(f"\n  Target: {target_col} ({unit})")
    print(f"  Lookback: {lookback}h, Horizon: {horizon}h")
    print(f"  Data: {len(df)} hourly rows, {df.shape[1]} features")

    # 2. Prepare standard data pipeline
    print(f"\n  Preparing data pipeline...")
    data = prepare_benchmark_data(
        df, lookback, horizon, device,
        target_col=target_col,
        split_ratios=split_ratios,
    )
    print(f"  Train: {data['X_train'].shape[0]} sequences")
    print(f"  Val:   {data['X_val'].shape[0]} sequences")
    print(f"  Test:  {data['X_test'].shape[0]} sequences")

    # Augmented pipeline for HandCrafted baseline
    data_augmented = None
    if not skip_baselines:
        df_aug = augment_with_time_features(df, periods)
        data_augmented = prepare_benchmark_data(
            df_aug, lookback, horizon, device,
            target_col=target_col,
            split_ratios=split_ratios,
        )
        n_crafted = data_augmented["X_train"].shape[2] - data["X_train"].shape[2]
        print(f"  Augmented features: {data_augmented['X_train'].shape[2]} "
              f"(+{n_crafted} crafted)")

    # 3. Train & evaluate models
    input_size = data["X_train"].shape[2]
    results: Dict[str, Dict[str, Any]] = {}

    print(f"\n  Training models (max {epochs} epochs):")

    # VonMises
    seed_everything(42)
    vm_model = _build_vm(input_size, horizon, device)
    results["VonMises"] = _run_single("VonMises", vm_model, data, epochs)

    if not skip_baselines:
        # LSTM (parameter-matched)
        seed_everything(42)
        vm_params = results["VonMises"]["params"]
        lstm_model = _build_lstm(input_size, horizon, vm_params, device)
        results["LSTM"] = _run_single("LSTM", lstm_model, data, epochs)

        # Transformer
        seed_everything(42)
        tf_model = _build_transformer(input_size, horizon, device)
        results["Transformer"] = _run_single("Transformer", tf_model, data, epochs)

        # HandCrafted (uses augmented data)
        seed_everything(42)
        hc_input_size = data_augmented["X_train"].shape[2]
        hc_model = _build_handcrafted(hc_input_size, lookback, horizon, device)
        results["HandCrafted"] = _run_single(
            "HandCrafted", hc_model, data_augmented, epochs
        )

    return {
        "dataset": dataset_name,
        "config": config,
        "data_shape": {
            "total_rows": len(df),
            "n_features": df.shape[1],
            "train_sequences": data["X_train"].shape[0],
            "val_sequences": data["X_val"].shape[0],
            "test_sequences": data["X_test"].shape[0],
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_dataset_report(
    ds_result: Dict[str, Any],
) -> None:
    """Print a formatted comparison table for one dataset."""
    name = ds_result["dataset"]
    config = ds_result["config"]
    results = ds_result["results"]

    print(f"\n{'=' * 78}")
    print(f"  {name.upper()} -- {config['description']}")
    print(f"  Target: {config['target_col']} ({config['unit']})")
    print(f"  Lookback: {config['lookback']}h | Horizon: {config['horizon']}h")
    print(f"{'=' * 78}")

    # Table header
    print(f"\n  {'Model':<22s} {'Params':>8s} "
          f"{'RMSE':>10s} {'MAE':>10s} {'R2':>8s} {'MAPE':>8s} {'Time':>8s}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")

    unit = config["unit"]
    for model_name, res in results.items():
        m = res["metrics"]
        print(
            f"  {model_name:<22s} {res['params']:>8,d} "
            f"{m['rmse']:>9.3f}  {m['mae']:>9.3f}  "
            f"{m['r2']:>7.4f}  {m['mape']:>6.2f}%  {res['time_seconds']:>6.1f}s"
        )

    # Quick comparison: VM vs best baseline
    if len(results) > 1:
        vm_rmse = results["VonMises"]["metrics"]["rmse"]
        baselines = {k: v for k, v in results.items() if k != "VonMises"}
        best_bl_name = min(baselines, key=lambda k: baselines[k]["metrics"]["rmse"])
        best_bl_rmse = baselines[best_bl_name]["metrics"]["rmse"]
        delta_pct = (vm_rmse - best_bl_rmse) / best_bl_rmse * 100
        direction = "worse" if delta_pct > 0 else "better"
        print(f"\n  VM vs best baseline ({best_bl_name}): "
              f"{abs(delta_pct):.1f}% {direction} RMSE")

    print(f"{'=' * 78}")


def print_cross_dataset_summary(
    all_results: List[Dict[str, Any]],
) -> None:
    """Print a summary table across all datasets."""
    print(f"\n\n{'#' * 78}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'#' * 78}")

    # Collect model names from first result that has all models
    model_names = list(all_results[0]["results"].keys())

    # Header
    print(f"\n  {'Dataset':<14s}", end="")
    for mn in model_names:
        print(f"  {mn:>14s}", end="")
    print(f"  {'VM vs Best':>12s}")

    print(f"  {'-' * 14}", end="")
    for _ in model_names:
        print(f"  {'-' * 14}", end="")
    print(f"  {'-' * 12}")

    # One row per dataset, showing RMSE values
    for ds in all_results:
        name = ds["dataset"]
        unit = ds["config"]["unit"]
        print(f"  {name:<14s}", end="")

        rmse_values = {}
        for mn in model_names:
            if mn in ds["results"]:
                rmse = ds["results"][mn]["metrics"]["rmse"]
                rmse_values[mn] = rmse
                print(f"  {rmse:>13.3f}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")

        # VM vs best baseline
        if "VonMises" in rmse_values and len(rmse_values) > 1:
            vm_rmse = rmse_values["VonMises"]
            bl_rmses = {k: v for k, v in rmse_values.items() if k != "VonMises"}
            best_bl = min(bl_rmses.values())
            delta = (vm_rmse - best_bl) / best_bl * 100
            sign = "+" if delta > 0 else ""
            print(f"  {sign}{delta:>10.1f}%", end="")
        print()

    print(f"\n  (RMSE in original units; negative % = VM is better)")
    print(f"{'#' * 78}\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the real-world dataset benchmark."""
    available = list_datasets()

    parser = argparse.ArgumentParser(
        description="Benchmark Von Mises forecaster on real-world datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=available, default=available,
        help=f"Datasets to benchmark (default: all). Choices: {available}"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs per model (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/real_benchmark",
        help="Directory for result JSON (default: results/real_benchmark)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Root directory for dataset downloads (default: data/raw)"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline models (only run VM forecaster)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"

    print(f"\n{'#' * 78}")
    print(f"  REAL-WORLD DATASET BENCHMARK")
    print(f"  Device: {device_str}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Baselines: {'skipped' if args.skip_baselines else 'enabled'}")
    print(f"{'#' * 78}")

    data_dir = Path(args.data_dir)
    all_results: List[Dict[str, Any]] = []

    for ds_name in args.datasets:
        try:
            seed_everything(args.seed)
            ds_result = benchmark_dataset(
                ds_name,
                epochs=args.epochs,
                device=device,
                data_dir=data_dir,
                skip_baselines=args.skip_baselines,
            )
            print_dataset_report(ds_result)
            all_results.append(ds_result)

        except Exception as e:
            print(f"\n  WARNING: Failed to benchmark '{ds_name}': {e}")
            print(f"  Skipping to next dataset...\n")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n  ERROR: No datasets were successfully benchmarked.")
        sys.exit(1)

    # Cross-dataset summary
    if len(all_results) > 1:
        print_cross_dataset_summary(all_results)

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_benchmark_results.json"

    # Build JSON-serializable output
    json_output = {
        "config": {
            "datasets": args.datasets,
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
        },
        "datasets": {},
    }

    for ds in all_results:
        ds_name = ds["dataset"]
        json_output["datasets"][ds_name] = {
            "config": ds["config"],
            "data_shape": ds["data_shape"],
            "results": {},
        }
        for model_name, res in ds["results"].items():
            json_output["datasets"][ds_name]["results"][model_name] = {
                **res["metrics"],
                "params": res["params"],
                "time_seconds": res["time_seconds"],
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
