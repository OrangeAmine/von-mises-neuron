"""Generate presentation figures comparing VonMises vs baseline forecasters.

Trains 4 models (VonMises, LSTM, Transformer, HandCrafted) on 4 datasets
(synthetic CO2, UCI Occupancy, UCI Power, Jena Climate) and produces 6
categories of publication-quality figures for a lab presentation.

Figure categories:
    1. Architecture diagram — VM neuron vs standard neuron response
    2. Prediction overlays — all models vs ground truth on test windows
    3. Metric bar charts — RMSE, MAE, R², MAPE grouped by model
    4. VM response curves — top learned neurons' receptive fields
    5. Training curves — loss over epochs for all models
    6. Phase/concentration heatmaps — learned μ and κ matrices

Usage:
    python scripts/generate_presentation_figures.py
    python scripts/generate_presentation_figures.py --epochs 80
    python scripts/generate_presentation_figures.py --skip-real
"""

import argparse
import csv
import importlib.util
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.von_mises_neuron import VonMisesForecaster, VonMisesLinear
from src.baselines import (
    LSTMForecaster,
    TransformerForecaster,
    HandCraftedLinearForecaster,
    compute_param_matched_hidden_size,
)
from src.preprocessing import normalize
from src.sequences import create_sequences
from src.split import chronological_split
from src.data_loading import load_dataset, DATASET_CONFIGS


# ---------------------------------------------------------------------------
# Reuse synthetic data generation from the benchmark script (importlib to
# avoid triggering its main())
# ---------------------------------------------------------------------------
def _load_benchmark_module():
    """Dynamically import benchmark_standard_dataset to reuse its helpers."""
    spec = importlib.util.spec_from_file_location(
        "benchmark", str(PROJECT_ROOT / "scripts" / "benchmark_standard_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bench = _load_benchmark_module()
generate_synthetic_co2 = _bench.generate_synthetic_co2
augment_with_time_features = _bench.augment_with_time_features


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------
DATASET_LABELS = {
    "synthetic": "Synthetic CO2",
    "occupancy": "UCI Occupancy (CO2)",
    "power": "Household Power",
    "jena": "Jena Climate (Temperature)",
}
DATASET_SUFFIXES = ["a", "b", "c", "d"]
DATASET_UNITS = {
    "synthetic": "ppm",
    "occupancy": "ppm",
    "power": "kW",
    "jena": "degC",
}

MODEL_COLORS = {
    "VonMises": "tab:blue",
    "LSTM": "tab:orange",
    "Transformer": "tab:green",
    "HandCrafted": "tab:red",
}

# Feature names per dataset (for heatmap axis labels)
FEATURE_NAMES = {
    "synthetic": ["Temperature", "Humidity", "CO2", "Noise"],
    "occupancy": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
    "power": [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ],
    "jena": ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "wd (deg)"],
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
# Data pipeline
# ---------------------------------------------------------------------------
def prepare_data(
    df,
    lookback: int,
    horizon: int,
    device: torch.device,
    target_col: str,
    split_ratios: List[float],
) -> Dict[str, Any]:
    """Split, normalize, window data. Returns dict with tensors + stats."""
    train_df, val_df, test_df = chronological_split(df, ratios=split_ratios)
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
        "horizon": horizon,
        "lookback": lookback,
        "target_col": target_col,
    }


def load_all_datasets(
    device: torch.device,
    real_datasets: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load and prepare synthetic + selected real-world datasets."""
    if real_datasets is None:
        real_datasets = ["occupancy", "jena"]
    datasets = {}

    # --- Synthetic ---
    print("\n[Dataset] Generating synthetic CO2...")
    df_syn = generate_synthetic_co2(n_days=90)
    data_syn = prepare_data(df_syn, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
    df_syn_aug = augment_with_time_features(df_syn, [24, 168])
    data_syn_aug = prepare_data(df_syn_aug, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
    datasets["synthetic"] = {
        "data": data_syn, "data_aug": data_syn_aug, "df": df_syn,
    }

    # --- Real-world datasets ---
    for ds_name in real_datasets:
        print(f"\n[Dataset] Loading {ds_name}...")
        df, config = load_dataset(ds_name)
        data = prepare_data(
            df, config["lookback"], config["horizon"], device,
            config["target_col"], config["split_ratios"],
        )
        df_aug = augment_with_time_features(df, config["handcrafted_periods"])
        data_aug = prepare_data(
            df_aug, config["lookback"], config["horizon"], device,
            config["target_col"], config["split_ratios"],
        )
        datasets[ds_name] = {"data": data, "data_aug": data_aug, "df": df}

    return datasets


# ---------------------------------------------------------------------------
# Training loop with loss history recording
# ---------------------------------------------------------------------------
def _batched_predict(
    model: nn.Module, X: torch.Tensor, batch_size: int = 256
) -> torch.Tensor:
    """Run inference in mini-batches to avoid CUDA OOM."""
    if X.shape[0] <= batch_size:
        return model(X)
    chunks = []
    for i in range(0, X.shape[0], batch_size):
        chunks.append(model(X[i : i + batch_size]))
    return torch.cat(chunks, dim=0)


def train_model_with_history(
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
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train with early stopping; return (model, loss history dict)."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = _batched_predict(model, X_val)
            val_loss = criterion(val_pred, y_val).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
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
    return model, history


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict,
    target_col: str,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R², MAPE in denormalized scale."""
    mean, std = norm_stats[target_col]
    yp = y_pred * std + mean
    yt = y_true * std + mean

    rmse = ((yp - yt) ** 2).mean().item() ** 0.5
    mae = torch.abs(yp - yt).mean().item()
    ss_res = ((yp - yt) ** 2).sum().item()
    ss_tot = ((yt - yt.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    abs_safe = torch.clamp(torch.abs(yt), min=1.0)
    mape = (torch.abs(yp - yt) / abs_safe).mean().item() * 100.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Train all 4 models on one dataset
# ---------------------------------------------------------------------------
def train_all_models(
    ds_name: str,
    datasets: Dict[str, Dict],
    epochs: int,
    device: torch.device,
    model_names: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train selected models on one dataset. Returns per-model results dict."""
    if model_names is None:
        model_names = list(MODEL_COLORS.keys())
    ds = datasets[ds_name]
    data = ds["data"]
    data_aug = ds["data_aug"]
    input_size = data["X_train"].shape[2]
    horizon = data["horizon"]
    lookback = data["lookback"]

    results = {}

    # --- VonMises ---
    if "VonMises" in model_names:
        print("  Training VonMises...")
        seed_everything(42)
        vm_model = VonMisesForecaster(
            input_size=input_size, hidden_size=64, num_vm_layers=2,
            dropout=0.1, output_size=horizon, init_kappa=1.0,
        ).to(device)
        vm_model, vm_hist = train_model_with_history(
            vm_model, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], max_epochs=epochs,
        )
        with torch.no_grad():
            vm_pred = _batched_predict(vm_model, data["X_test"])
        vm_metrics = compute_metrics(
            vm_pred, data["y_test"], data["norm_stats"], data["target_col"]
        )
        results["VonMises"] = {
            "model": vm_model, "history": vm_hist, "metrics": vm_metrics,
            "predictions": vm_pred, "params": _count_params(vm_model),
        }
        print(f"    RMSE={vm_metrics['rmse']:.3f}  R2={vm_metrics['r2']:.4f}")

    # Param count for LSTM matching (use VM if available, else default)
    vm_params = results["VonMises"]["params"] if "VonMises" in results else 43148

    # --- LSTM (parameter-matched) ---
    if "LSTM" in model_names:
        print("  Training LSTM...")
        seed_everything(42)
        h_lstm = compute_param_matched_hidden_size(vm_params, input_size, horizon)
        lstm_model = LSTMForecaster(
            input_size=input_size, hidden_size=h_lstm, num_layers=2,
            dropout=0.1, output_size=horizon,
        ).to(device)
        lstm_model, lstm_hist = train_model_with_history(
            lstm_model, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], max_epochs=epochs,
        )
        with torch.no_grad():
            lstm_pred = _batched_predict(lstm_model, data["X_test"])
        lstm_metrics = compute_metrics(
            lstm_pred, data["y_test"], data["norm_stats"], data["target_col"]
        )
        results["LSTM"] = {
            "model": lstm_model, "history": lstm_hist, "metrics": lstm_metrics,
            "predictions": lstm_pred, "params": _count_params(lstm_model),
        }
        print(f"    RMSE={lstm_metrics['rmse']:.3f}  R2={lstm_metrics['r2']:.4f}")

    # --- Transformer ---
    if "Transformer" in model_names:
        print("  Training Transformer...")
        seed_everything(42)
        tf_model = TransformerForecaster(
            input_size=input_size, d_model=48, nhead=4, num_layers=2,
            dim_feedforward=128, dropout=0.1, output_size=horizon,
        ).to(device)
        tf_model, tf_hist = train_model_with_history(
            tf_model, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], max_epochs=epochs,
        )
        with torch.no_grad():
            tf_pred = _batched_predict(tf_model, data["X_test"])
        tf_metrics = compute_metrics(
            tf_pred, data["y_test"], data["norm_stats"], data["target_col"]
        )
        results["Transformer"] = {
            "model": tf_model, "history": tf_hist, "metrics": tf_metrics,
            "predictions": tf_pred, "params": _count_params(tf_model),
        }
        print(f"    RMSE={tf_metrics['rmse']:.3f}  R2={tf_metrics['r2']:.4f}")

    # --- HandCrafted (uses augmented data with sin/cos features) ---
    if "HandCrafted" in model_names:
        print("  Training HandCrafted...")
        seed_everything(42)
        hc_input_size = data_aug["X_train"].shape[2]
        hc_model = HandCraftedLinearForecaster(
            input_size=hc_input_size, lookback=lookback, output_size=horizon,
        ).to(device)
        hc_model, hc_hist = train_model_with_history(
            hc_model, data_aug["X_train"], data_aug["y_train"],
            data_aug["X_val"], data_aug["y_val"], max_epochs=epochs,
        )
        with torch.no_grad():
            hc_pred = _batched_predict(hc_model, data_aug["X_test"])
        hc_metrics = compute_metrics(
            hc_pred, data_aug["y_test"], data_aug["norm_stats"],
            data_aug["target_col"],
        )
        results["HandCrafted"] = {
            "model": hc_model, "history": hc_hist, "metrics": hc_metrics,
            "predictions": hc_pred, "params": _count_params(hc_model),
        }
        print(f"    RMSE={hc_metrics['rmse']:.3f}  R2={hc_metrics['r2']:.4f}")

    return results


# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure 1: Architecture diagram
# ---------------------------------------------------------------------------
def plot_architecture_diagram(output_dir: Path) -> None:
    """VM neuron response vs standard neuron response over theta.

    Shows exp(kappa * cos(theta + mu)) for several (kappa, mu) pairs alongside
    a standard neuron's ReLU(w*theta + b) to illustrate periodicity preservation.
    """
    theta = np.linspace(0, 4 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: standard neuron — ReLU(w*theta + b) breaks periodicity
    ax = axes[0]
    for w, b, label in [
        (1.0, -2.0, "w=1, b=−2"),
        (0.5, 0.0, "w=0.5, b=0"),
        (2.0, -5.0, "w=2, b=−5"),
    ]:
        ax.plot(theta, np.maximum(0, w * theta + b), label=label, linewidth=1.5)
    ax.set_title("Standard Neuron: ReLU(wθ + b)", fontsize=13)
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.set_xlim(0, 4 * np.pi)
    # Mark 2pi boundaries
    for x in [2 * np.pi, 4 * np.pi]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.4)

    # Right: VM neuron — exp(kappa * cos(theta + mu)) is 2pi-periodic
    ax = axes[1]
    for kappa, mu, label in [
        (2.0, 0.0, "κ=2, μ=0"),
        (3.0, np.pi / 2, "κ=3, μ=π/2"),
        (1.0, np.pi, "κ=1, μ=π"),
        (5.0, np.pi / 4, "κ=5, μ=π/4"),
    ]:
        ax.plot(
            theta, np.exp(kappa * np.cos(theta + mu)),
            label=label, linewidth=1.5,
        )
    for x in [2 * np.pi, 4 * np.pi]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Von Mises Neuron: exp(κ·cos(θ + μ))", fontsize=13)
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.set_xlim(0, 4 * np.pi)

    fig.suptitle(
        "Periodicity Preservation: Standard vs Von Mises Neuron",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "01_architecture_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_architecture_diagram.png")


# ---------------------------------------------------------------------------
# Figure 2: Prediction overlays
# ---------------------------------------------------------------------------
def plot_predictions(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    data: Dict[str, Any],
    output_dir: Path,
    n_steps: int = 100,
) -> None:
    """Ground truth vs all models' predictions on a representative test window.

    Picks a window from the middle of the test set and plots the first
    forecast step (h=0) for visual clarity.
    """
    norm_stats = data["norm_stats"]
    target_col = data["target_col"]
    mean, std = norm_stats[target_col]

    # Denormalize ground truth — first step of each forecast window
    y_true_orig = (data["y_test"][:, 0] * std + mean).cpu().numpy()

    # Pick a window from the middle of the test set
    total = len(y_true_orig)
    start = max(0, total // 2 - n_steps // 2)
    end = min(total, start + n_steps)
    idx = slice(start, end)
    t = np.arange(end - start)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        t, y_true_orig[idx], color="black", linewidth=2,
        label="Ground Truth", zorder=5,
    )

    for model_name in results:
        color = MODEL_COLORS.get(model_name, "tab:gray")
        pred = results[model_name]["predictions"]
        pred_orig = (pred[:, 0] * std + mean).cpu().numpy()
        ax.plot(
            t, pred_orig[idx], color=color, linewidth=1.2,
            alpha=0.85, label=model_name,
        )

    label = DATASET_LABELS.get(ds_name, ds_name)
    unit = DATASET_UNITS.get(ds_name, "")
    ax.set_title(f"Prediction Comparison — {label} (1-step ahead)", fontsize=13)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel(f"{target_col} ({unit})")
    ax.legend(loc="best")
    fig.tight_layout()

    fname = f"02{suffix}_predictions_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 3: Metric bar charts
# ---------------------------------------------------------------------------
def plot_metrics(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart of RMSE, MAE, R², MAPE for all models."""
    model_names = list(results.keys())
    metrics_keys = ["rmse", "mae", "r2", "mape"]
    metric_labels = ["RMSE", "MAE", "R²", "MAPE (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, key, mlabel in zip(axes, metrics_keys, metric_labels):
        values = [results[m]["metrics"][key] for m in model_names]
        colors = [MODEL_COLORS.get(m, "tab:gray") for m in model_names]
        bars = ax.bar(
            model_names, values, color=colors, edgecolor="white", linewidth=0.5
        )

        # Annotate numeric values on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if key == "r2" else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt, ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(mlabel, fontsize=12)
        ax.tick_params(axis="x", rotation=30)

    label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(f"Model Comparison — {label}", fontsize=14, y=1.02)
    fig.tight_layout()

    fname = f"03{suffix}_metrics_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 4: VM response curves
# ---------------------------------------------------------------------------
def plot_vm_response_curves(
    ds_name: str,
    suffix: str,
    vm_model: VonMisesForecaster,
    output_dir: Path,
    top_k: int = 8,
) -> None:
    """Response curves for the top-k VM neurons (by concentration).

    Plots exp(kappa * cos(theta + mu)) using learned parameters from the
    first VM layer. 'Top' neurons have the highest mean concentration —
    they respond most sharply to specific inputs.
    """
    vm_layers = vm_model.get_vm_layers()
    if not vm_layers:
        return
    layer = vm_layers[0]
    kappas = layer.get_concentrations().cpu().numpy()  # (out, in)
    mus = layer.get_phases().cpu().numpy()

    # Rank neurons by mean concentration across inputs
    mean_kappa = kappas.mean(axis=1)
    top_indices = np.argsort(mean_kappa)[-top_k:][::-1]

    theta = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis

    for rank, neuron_idx in enumerate(top_indices):
        # Use the input dimension with highest kappa for this neuron
        best_input = np.argmax(kappas[neuron_idx])
        k = kappas[neuron_idx, best_input]
        m = mus[neuron_idx, best_input]
        response = np.exp(k * np.cos(theta + m))
        color = cmap(rank / max(top_k - 1, 1))
        ax.plot(
            theta, response, color=color, linewidth=1.5,
            label=f"n{neuron_idx} (κ={k:.2f}, μ={m:.2f})",
        )

    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("exp(κ·cos(θ + μ))")
    label = DATASET_LABELS.get(ds_name, ds_name)
    ax.set_title(f"Top-{top_k} VM Neuron Response Curves — {label}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    fig.tight_layout()

    fname = f"04{suffix}_vm_response_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 5: Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Train and validation loss curves for all 4 models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training loss
    ax = axes[0]
    for model_name in results:
        color = MODEL_COLORS.get(model_name, "tab:gray")
        losses = results[model_name]["history"]["train_loss"]
        ax.plot(losses, color=color, label=model_name, linewidth=1.2)
    ax.set_title("Training Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()

    # Right: validation loss
    ax = axes[1]
    for model_name in results:
        color = MODEL_COLORS.get(model_name, "tab:gray")
        losses = results[model_name]["history"]["val_loss"]
        ax.plot(losses, color=color, label=model_name, linewidth=1.2)
    ax.set_title("Validation Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()

    label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(f"Training Dynamics — {label}", fontsize=14, y=1.02)
    fig.tight_layout()

    fname = f"05{suffix}_training_curves_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Figure 6: Phase & concentration heatmaps
# ---------------------------------------------------------------------------
def plot_phase_kappa_heatmaps(
    ds_name: str,
    suffix: str,
    vm_model: VonMisesForecaster,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Heatmaps of learned mu (phase) and kappa (concentration).

    Each cell [j, i] represents neuron j's response to input feature i.
    Phase μ uses a circular colourmap (twilight) since it wraps at 2π.
    Concentration κ uses a sequential colourmap (magma) since higher = sharper.
    """
    vm_layers = vm_model.get_vm_layers()
    if not vm_layers:
        return
    layer = vm_layers[0]
    kappas = layer.get_concentrations().cpu().numpy()
    mus = layer.get_phases().cpu().numpy()

    n_neurons, n_features = kappas.shape
    # Match feature names to actual input dimension
    if len(feature_names) >= n_features:
        feat_labels = feature_names[:n_features]
    else:
        feat_labels = feature_names + [
            f"f{i}" for i in range(len(feature_names), n_features)
        ]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(5, n_neurons * 0.25 + 2))
    )

    # Left: phase shifts mu
    sns.heatmap(
        mus, ax=axes[0], cmap="twilight", vmin=0, vmax=2 * np.pi,
        xticklabels=feat_labels,
        yticklabels=[str(i) for i in range(n_neurons)],
        cbar_kws={"label": "μ (radians)"},
    )
    axes[0].set_title("Phase Shifts (μ)", fontsize=12)
    axes[0].set_xlabel("Input Feature")
    axes[0].set_ylabel("Neuron Index")

    # Right: concentrations kappa
    sns.heatmap(
        kappas, ax=axes[1], cmap="magma",
        xticklabels=feat_labels,
        yticklabels=[str(i) for i in range(n_neurons)],
        cbar_kws={"label": "κ"},
    )
    axes[1].set_title("Concentrations (κ)", fontsize=12)
    axes[1].set_xlabel("Input Feature")
    axes[1].set_ylabel("Neuron Index")

    label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(
        f"Learned VM Parameters (Layer 1) — {label}", fontsize=14, y=1.02
    )
    fig.tight_layout()

    fname = f"06{suffix}_phase_kappa_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Metrics CSV export
# ---------------------------------------------------------------------------
def save_metrics_csv(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
) -> None:
    """Save all metrics to a CSV for easy reference in slides."""
    rows = []
    for ds_name, model_results in all_results.items():
        for model_name, res in model_results.items():
            m = res["metrics"]
            rows.append({
                "dataset": ds_name,
                "model": model_name,
                "params": res["params"],
                "rmse": f"{m['rmse']:.4f}",
                "mae": f"{m['mae']:.4f}",
                "r2": f"{m['r2']:.4f}",
                "mape": f"{m['mape']:.2f}",
            })

    csv_path = output_dir / "metrics_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "model", "params", "rmse", "mae", "r2", "mape"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("  Saved: metrics_summary.csv")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation figures for VonMises vs baselines"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs per model (default: 50)",
    )
    parser.add_argument(
        "--skip-real", action="store_true",
        help="Only run synthetic dataset (faster for testing)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["occupancy", "jena"],
        help="Real-world datasets to include (default: occupancy jena)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/presentation_figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to train (default: all). E.g. --models VonMises LSTM",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"\nDevice: {device_str}")
    print(f"Output: {output_dir}")

    t_start = time.time()

    # --- Figure 1: Architecture diagram (no training needed) ---
    print("\n[1/3] Generating architecture diagram...")
    plot_architecture_diagram(output_dir)

    # --- Load datasets ---
    print("\n[2/3] Loading datasets...")
    if args.skip_real:
        seed_everything(42)
        df_syn = generate_synthetic_co2(n_days=90)
        data_syn = prepare_data(df_syn, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
        df_syn_aug = augment_with_time_features(df_syn, [24, 168])
        data_syn_aug = prepare_data(
            df_syn_aug, 72, 12, device, "CO2", [0.8, 0.1, 0.1]
        )
        datasets = {
            "synthetic": {
                "data": data_syn, "data_aug": data_syn_aug, "df": df_syn,
            },
        }
    else:
        datasets = load_all_datasets(device, real_datasets=args.datasets)

    dataset_names = list(datasets.keys())

    # --- Train & generate per-dataset figures ---
    print(f"\n[3/3] Training models and generating figures...")
    all_results: Dict[str, Dict[str, Dict]] = {}

    for i, ds_name in enumerate(dataset_names):
        suffix = DATASET_SUFFIXES[i]
        label = DATASET_LABELS.get(ds_name, ds_name)
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {label} ({i + 1}/{len(dataset_names)})")
        print(f"{'=' * 60}")

        # Train all 4 models
        seed_everything(42)
        results = train_all_models(ds_name, datasets, args.epochs, device, args.models)
        all_results[ds_name] = results

        data = datasets[ds_name]["data"]

        # Generate all 5 per-dataset figure types
        print("\n  Generating figures...")
        plot_predictions(ds_name, suffix, results, data, output_dir)
        plot_metrics(ds_name, suffix, results, output_dir)
        plot_vm_response_curves(
            ds_name, suffix, results["VonMises"]["model"], output_dir
        )
        plot_training_curves(ds_name, suffix, results, output_dir)

        feat_names = FEATURE_NAMES.get(
            ds_name, [f"f{i}" for i in range(data["X_train"].shape[2])]
        )
        plot_phase_kappa_heatmaps(
            ds_name, suffix, results["VonMises"]["model"], feat_names, output_dir
        )

    # Metrics CSV
    save_metrics_csv(all_results, output_dir)

    elapsed = time.time() - t_start
    n_figures = 1 + 5 * len(dataset_names)
    print(f"\nDone! {n_figures} figures + metrics CSV saved to {output_dir}")
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
