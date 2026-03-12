# Presentation Figure Generator — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/generate_presentation_figures.py` — a single script that trains 4 models on 4 datasets and generates 21 presentation-ready figures + a metrics CSV.

**Architecture:** Single self-contained script following existing patterns from `benchmark_standard_dataset.py` and `interpretability_analysis.py`. Reuses all existing model classes, data loaders, and pipeline utilities. Training loop extended to record per-epoch losses and return trained models. Six plotting functions, each generating one figure category.

**Tech Stack:** Python, PyTorch, matplotlib, seaborn, numpy, pandas

---

## Chunk 1: Script skeleton + data pipeline + training

### Task 1: Script skeleton with imports and infrastructure

**Files:**
- Create: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Create the script with module docstring, imports, and constants**

```python
"""Generate presentation figures comparing VonMises vs baseline forecasters.

Trains 4 models (VonMises, LSTM, Transformer, HandCrafted) on 4 datasets
(synthetic CO2, UCI Occupancy, UCI Power, Jena Climate) and produces 6
categories of publication-quality figures for a lab presentation.

Usage:
    python scripts/generate_presentation_figures.py
    python scripts/generate_presentation_figures.py --epochs 80
    python scripts/generate_presentation_figures.py --skip-real  # synthetic only
"""

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
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

# Reuse synthetic data generation from the benchmark script
from scripts.benchmark_standard_dataset import (
    generate_synthetic_co2,
    augment_with_time_features,
)

# Figure output directory
OUTPUT_DIR = Path("results/presentation_figures")

# Dataset display names and suffixes for filenames
DATASET_LABELS = {
    "synthetic": "Synthetic CO2",
    "occupancy": "UCI Occupancy (CO2)",
    "power": "Household Power",
    "jena": "Jena Climate (Temperature)",
}
DATASET_SUFFIXES = ["a", "b", "c", "d"]

# Model display config: name -> color
MODEL_COLORS = {
    "VonMises": "tab:blue",
    "LSTM": "tab:orange",
    "Transformer": "tab:green",
    "HandCrafted": "tab:red",
}
```

- [ ] **Step 2: Add seeding utility**

```python
def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Task 2: Data pipeline — prepare all 4 datasets

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add data preparation function (reuses existing pipeline)**

```python
def prepare_data(
    df: pd.DataFrame,
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

    X_train, y_train, _ = create_sequences(train_norm, lookback, horizon, target_col)
    X_val, y_val, _ = create_sequences(val_norm, lookback, horizon, target_col)
    X_test, y_test, _ = create_sequences(test_norm, lookback, horizon, target_col)

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
```

- [ ] **Step 2: Add function to load all 4 datasets into a unified dict**

```python
def load_all_datasets(device: torch.device) -> Dict[str, Dict[str, Any]]:
    """Load and prepare all 4 datasets. Returns {name: {data, data_aug, config}}."""
    datasets = {}

    # --- Synthetic ---
    print("\n[Dataset] Generating synthetic CO2...")
    df_syn = generate_synthetic_co2(n_days=90)
    syn_config = {"target_col": "CO2", "lookback": 72, "horizon": 12,
                  "split_ratios": [0.8, 0.1, 0.1], "unit": "ppm",
                  "handcrafted_periods": [24, 168]}
    data_syn = prepare_data(df_syn, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
    df_syn_aug = augment_with_time_features(df_syn, [24, 168])
    data_syn_aug = prepare_data(df_syn_aug, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
    datasets["synthetic"] = {"data": data_syn, "data_aug": data_syn_aug,
                             "config": syn_config, "df": df_syn}

    # --- Real-world datasets ---
    for ds_name in ["occupancy", "power", "jena"]:
        print(f"\n[Dataset] Loading {ds_name}...")
        df, config = load_dataset(ds_name)
        data = prepare_data(df, config["lookback"], config["horizon"], device,
                           config["target_col"], config["split_ratios"])
        df_aug = augment_with_time_features(df, config["handcrafted_periods"])
        data_aug = prepare_data(df_aug, config["lookback"], config["horizon"],
                               device, config["target_col"], config["split_ratios"])
        datasets[ds_name] = {"data": data, "data_aug": data_aug,
                             "config": config, "df": df}

    return datasets
```

### Task 3: Training loop with loss recording

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add training function that records per-epoch train+val losses**

```python
def _batched_predict(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
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
    """Train with early stopping; return (model, {"train_loss": [...], "val_loss": [...]})."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

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

        history["train_loss"].append(np.mean(epoch_losses))
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
```

### Task 4: Model building + full training pipeline

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add model builders and metrics computation**

```python
def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict,
    target_col: str,
) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, MAPE in denormalized scale."""
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


def train_all_models(
    ds_name: str,
    datasets: Dict[str, Dict],
    epochs: int,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Train all 4 models on one dataset. Returns per-model results dict."""
    ds = datasets[ds_name]
    data = ds["data"]
    data_aug = ds["data_aug"]
    input_size = data["X_train"].shape[2]
    horizon = data["horizon"]
    lookback = data["lookback"]

    results = {}

    # --- VonMises ---
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
    vm_metrics = compute_metrics(vm_pred, data["y_test"], data["norm_stats"], data["target_col"])
    results["VonMises"] = {
        "model": vm_model, "history": vm_hist, "metrics": vm_metrics,
        "predictions": vm_pred, "params": _count_params(vm_model),
    }

    vm_params = results["VonMises"]["params"]

    # --- LSTM ---
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
    lstm_metrics = compute_metrics(lstm_pred, data["y_test"], data["norm_stats"], data["target_col"])
    results["LSTM"] = {
        "model": lstm_model, "history": lstm_hist, "metrics": lstm_metrics,
        "predictions": lstm_pred, "params": _count_params(lstm_model),
    }

    # --- Transformer ---
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
    tf_metrics = compute_metrics(tf_pred, data["y_test"], data["norm_stats"], data["target_col"])
    results["Transformer"] = {
        "model": tf_model, "history": tf_hist, "metrics": tf_metrics,
        "predictions": tf_pred, "params": _count_params(tf_model),
    }

    # --- HandCrafted (uses augmented data) ---
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
    hc_metrics = compute_metrics(hc_pred, data_aug["y_test"], data_aug["norm_stats"], data_aug["target_col"])
    results["HandCrafted"] = {
        "model": hc_model, "history": hc_hist, "metrics": hc_metrics,
        "predictions": hc_pred, "params": _count_params(hc_model),
    }

    return results
```

- [ ] **Step 2: Verify skeleton runs**

Run: `python scripts/generate_presentation_figures.py --help`
Expected: Prints usage, no import errors.

---

## Chunk 2: Plotting functions (figures 1-3)

### Task 5: Figure 1 — Architecture diagram

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add architecture diagram plotting function**

```python
def plot_architecture_diagram(output_dir: Path) -> None:
    """Figure 1: VM neuron response vs standard neuron response over theta.

    Shows exp(kappa * cos(theta + mu)) for several (kappa, mu) pairs alongside
    a standard neuron's ReLU(w*theta + b) to illustrate periodicity preservation.
    """
    theta = np.linspace(0, 4 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Standard neuron — ReLU(w*theta + b) breaks periodicity
    ax = axes[0]
    for w, b, label in [(1.0, -2.0, "w=1, b=-2"), (0.5, 0.0, "w=0.5, b=0"),
                         (2.0, -5.0, "w=2, b=-5")]:
        ax.plot(theta, np.maximum(0, w * theta + b), label=label)
    ax.set_title("Standard Neuron: ReLU(wθ + b)", fontsize=13)
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.set_xlim(0, 4 * np.pi)

    # Right: VM neuron — exp(kappa * cos(theta + mu)) is 2pi-periodic
    ax = axes[1]
    for kappa, mu, label in [(2.0, 0.0, "κ=2, μ=0"), (3.0, np.pi/2, "κ=3, μ=π/2"),
                              (1.0, np.pi, "κ=1, μ=π"), (5.0, np.pi/4, "κ=5, μ=π/4")]:
        ax.plot(theta, np.exp(kappa * np.cos(theta + mu)), label=label)
    # Mark 2pi boundaries
    for x in [2 * np.pi, 4 * np.pi]:
        ax.axvline(x, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Von Mises Neuron: exp(κ·cos(θ + μ))", fontsize=13)
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Output")
    ax.legend()
    ax.set_xlim(0, 4 * np.pi)

    fig.suptitle("Periodicity Preservation: Standard vs Von Mises Neuron", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "01_architecture_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_architecture_diagram.png")
```

### Task 6: Figure 2 — Prediction overlays

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add prediction overlay plotting function**

```python
def plot_predictions(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    data: Dict[str, Any],
    output_dir: Path,
    n_steps: int = 100,
) -> None:
    """Figure 2: Ground truth vs all models' predictions on a test window.

    Picks a representative window from the middle of the test set and plots
    the first forecast step (h=0) for visual clarity, showing how each model
    tracks the true signal.
    """
    norm_stats = data["norm_stats"]
    target_col = data["target_col"]
    mean, std = norm_stats[target_col]
    y_true = data["y_test"]

    # Denormalize ground truth — first step of each forecast window
    y_true_orig = (y_true[:, 0] * std + mean).cpu().numpy()

    # Pick a window from the middle of the test set for a representative view
    total = len(y_true_orig)
    start = max(0, total // 2 - n_steps // 2)
    end = min(total, start + n_steps)
    idx = slice(start, end)
    t = np.arange(end - start)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, y_true_orig[idx], color="black", linewidth=2, label="Ground Truth", zorder=5)

    for model_name, color in MODEL_COLORS.items():
        pred = results[model_name]["predictions"]
        pred_orig = (pred[:, 0] * std + mean).cpu().numpy()
        ax.plot(t, pred_orig[idx], color=color, linewidth=1.2, alpha=0.85, label=model_name)

    label = DATASET_LABELS.get(ds_name, ds_name)
    config = data
    ax.set_title(f"Prediction Comparison — {label} (h=1 step ahead)", fontsize=13)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel(f"{target_col} ({_get_unit(ds_name)})")
    ax.legend(loc="best")
    fig.tight_layout()

    fname = f"02{suffix}_predictions_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def _get_unit(ds_name: str) -> str:
    """Return the unit string for a dataset."""
    units = {"synthetic": "ppm", "occupancy": "ppm", "power": "kW", "jena": "°C"}
    return units.get(ds_name, "")
```

### Task 7: Figure 3 — Metric bar charts

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add metric bar chart plotting function**

```python
def plot_metrics(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Figure 3: Grouped bar chart of RMSE, MAE, R², MAPE for all models."""
    model_names = list(results.keys())
    metrics_to_plot = ["rmse", "mae", "r2", "mape"]
    metric_labels = ["RMSE", "MAE", "R²", "MAPE (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric_key, metric_label in zip(axes, metrics_to_plot, metric_labels):
        values = [results[m]["metrics"][metric_key] for m in model_names]
        colors = [MODEL_COLORS[m] for m in model_names]
        bars = ax.bar(model_names, values, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate values on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if metric_key == "r2" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9)

        ax.set_title(metric_label, fontsize=12)
        ax.tick_params(axis="x", rotation=30)

    label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(f"Model Comparison — {label}", fontsize=14, y=1.02)
    fig.tight_layout()

    fname = f"03{suffix}_metrics_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
```

- [ ] **Step 2: Verify figures 1-3 generate correctly**

Add a temporary `if __name__ == "__main__"` block, run the script on synthetic data only, and visually check the first 3 figure types. Then remove the temporary block.

---

## Chunk 3: Plotting functions (figures 4-6) + main

### Task 8: Figure 4 — VM response curves

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add VM response curve plotting function**

```python
def plot_vm_response_curves(
    ds_name: str,
    suffix: str,
    vm_model: VonMisesForecaster,
    output_dir: Path,
    top_k: int = 8,
) -> None:
    """Figure 4: Response curves for the top-k VM neurons (by concentration).

    Plots exp(kappa * cos(theta + mu)) for each neuron, using the learned
    kappa and mu from the first VM layer. 'Top' neurons are those with the
    highest mean concentration — they respond most sharply to specific inputs.
    """
    vm_layers = vm_model.get_vm_layers()
    if not vm_layers:
        return
    layer = vm_layers[0]  # first VM layer
    kappas = layer.get_concentrations().cpu().numpy()  # (out, in)
    mus = layer.get_phases().cpu().numpy()              # (out, in)

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
        ax.plot(theta, response, color=color, linewidth=1.5,
                label=f"n{neuron_idx} (κ={k:.2f}, μ={m:.2f})")

    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("exp(κ·cos(θ + μ))")
    label = DATASET_LABELS.get(ds_name, ds_name)
    ax.set_title(f"Top-{top_k} VM Neuron Response Curves — {label}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    fig.tight_layout()

    fname = f"04{suffix}_vm_response_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
```

### Task 9: Figure 5 — Training curves

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add training curve plotting function**

```python
def plot_training_curves(
    ds_name: str,
    suffix: str,
    results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Figure 5: Train and validation loss curves for all 4 models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training loss
    ax = axes[0]
    for model_name, color in MODEL_COLORS.items():
        losses = results[model_name]["history"]["train_loss"]
        ax.plot(losses, color=color, label=model_name, linewidth=1.2)
    ax.set_title("Training Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()

    # Right: validation loss
    ax = axes[1]
    for model_name, color in MODEL_COLORS.items():
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
```

### Task 10: Figure 6 — Phase and concentration heatmaps

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add phase/concentration heatmap plotting function**

```python
def plot_phase_kappa_heatmaps(
    ds_name: str,
    suffix: str,
    vm_model: VonMisesForecaster,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Figure 6: Heatmaps of learned mu (phase) and kappa (concentration).

    Shows what periodic patterns each VM neuron has learned to detect.
    Each cell [j, i] represents neuron j's response to input feature i.
    """
    vm_layers = vm_model.get_vm_layers()
    if not vm_layers:
        return
    layer = vm_layers[0]
    kappas = layer.get_concentrations().cpu().numpy()
    mus = layer.get_phases().cpu().numpy()

    n_neurons, n_features = kappas.shape
    # Truncate feature names if layer has different input size
    feat_labels = feature_names[:n_features] if len(feature_names) >= n_features \
        else feature_names + [f"f{i}" for i in range(len(feature_names), n_features)]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_neurons * 0.25 + 2)))

    # Left: phase shifts mu
    sns.heatmap(mus, ax=axes[0], cmap="twilight", vmin=0, vmax=2*np.pi,
                xticklabels=feat_labels, yticklabels=range(n_neurons),
                cbar_kws={"label": "μ (radians)"})
    axes[0].set_title("Phase Shifts (μ)", fontsize=12)
    axes[0].set_xlabel("Input Feature")
    axes[0].set_ylabel("Neuron Index")

    # Right: concentrations kappa
    sns.heatmap(kappas, ax=axes[1], cmap="magma",
                xticklabels=feat_labels, yticklabels=range(n_neurons),
                cbar_kws={"label": "κ"})
    axes[1].set_title("Concentrations (κ)", fontsize=12)
    axes[1].set_xlabel("Input Feature")
    axes[1].set_ylabel("Neuron Index")

    label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(f"Learned VM Parameters (Layer 1) — {label}", fontsize=14, y=1.02)
    fig.tight_layout()

    fname = f"06{suffix}_phase_kappa_{ds_name}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
```

### Task 11: Metrics CSV export

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add CSV export function**

```python
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
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "params",
                                                "rmse", "mae", "r2", "mape"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: metrics_summary.csv")
```

### Task 12: Main entry point + CLI

**Files:**
- Modify: `scripts/generate_presentation_figures.py`

- [ ] **Step 1: Add feature name lookup and main function**

```python
# Feature names per dataset (for heatmap axis labels)
FEATURE_NAMES = {
    "synthetic": ["Temperature", "Humidity", "CO2", "Noise"],
    "occupancy": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
    "power": ["Global_active_power", "Global_reactive_power", "Voltage",
              "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
    "jena": ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "wd (deg)"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation figures for VonMises vs baselines"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--skip-real", action="store_true",
                        help="Only run synthetic dataset (faster for testing)")
    parser.add_argument("--output-dir", type=str,
                        default="results/presentation_figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Figure 1: Architecture diagram (no training needed) ---
    print("\n[1/3] Generating architecture diagram...")
    plot_architecture_diagram(output_dir)

    # --- Load datasets ---
    print("\n[2/3] Loading datasets...")
    if args.skip_real:
        # Only synthetic
        seed_everything(42)
        df_syn = generate_synthetic_co2(n_days=90)
        syn_config = {"target_col": "CO2", "lookback": 72, "horizon": 12,
                      "split_ratios": [0.8, 0.1, 0.1], "unit": "ppm",
                      "handcrafted_periods": [24, 168]}
        data_syn = prepare_data(df_syn, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
        df_syn_aug = augment_with_time_features(df_syn, [24, 168])
        data_syn_aug = prepare_data(df_syn_aug, 72, 12, device, "CO2", [0.8, 0.1, 0.1])
        datasets = {"synthetic": {"data": data_syn, "data_aug": data_syn_aug,
                                   "config": syn_config, "df": df_syn}}
    else:
        datasets = load_all_datasets(device)

    dataset_names = list(datasets.keys())

    # --- Train & generate per-dataset figures ---
    print(f"\n[3/3] Training models and generating figures...")
    all_results = {}

    for i, ds_name in enumerate(dataset_names):
        suffix = DATASET_SUFFIXES[i]
        print(f"\n{'='*60}")
        print(f"  Dataset: {DATASET_LABELS.get(ds_name, ds_name)} ({i+1}/{len(dataset_names)})")
        print(f"{'='*60}")

        # Train all 4 models
        seed_everything(42)
        results = train_all_models(ds_name, datasets, args.epochs, device)
        all_results[ds_name] = results

        data = datasets[ds_name]["data"]

        # Figure 2: Prediction overlays
        plot_predictions(ds_name, suffix, results, data, output_dir)

        # Figure 3: Metric bar charts
        plot_metrics(ds_name, suffix, results, output_dir)

        # Figure 4: VM response curves
        plot_vm_response_curves(ds_name, suffix, results["VonMises"]["model"], output_dir)

        # Figure 5: Training curves
        plot_training_curves(ds_name, suffix, results, output_dir)

        # Figure 6: Phase/concentration heatmaps
        feat_names = FEATURE_NAMES.get(ds_name, [f"f{i}" for i in range(data["X_train"].shape[2])])
        plot_phase_kappa_heatmaps(ds_name, suffix, results["VonMises"]["model"], feat_names, output_dir)

    # Metrics CSV
    save_metrics_csv(all_results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
    print(f"Total figures: {1 + 5 * len(dataset_names)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full script on synthetic only to verify end-to-end**

Run: `python scripts/generate_presentation_figures.py --skip-real --epochs 10`
Expected: 6 figures in `results/presentation_figures/` + metrics CSV, exit 0.

- [ ] **Step 3: Run full script on all 4 datasets**

Run: `python scripts/generate_presentation_figures.py --epochs 50`
Expected: 21 figures + CSV in `results/presentation_figures/`, exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_presentation_figures.py
git commit -m "feat: add presentation figure generator for VM vs baseline comparison"
```
