"""Interpretability analysis of learned Von Mises neuron parameters.

Trains a VonMisesForecaster on synthetic CO2 data while logging the
concentration (kappa) and phase (mu) parameters at every epoch, then
generates publication-quality analysis plots.

Three analyses:

1. **Phase Alignment** -- do learned mu values cluster near true signal peaks?
2. **Concentration Evolution** -- how does kappa change during training?
3. **VM Response Curves** -- receptive field visualization for top neurons.

Important caveat on phase interpretation:

    The inputs to the VM layer are z-scored time series features (mean=0,
    std=1), NOT angles.  The VM layer computes cos(x_i + mu_ji), which
    peaks when x_i = -mu.  So mu encodes the preferred *z-score value*
    that triggers peak neuron response, not a true angular direction.
    Z-scored inputs range roughly [-3, 3], which is a subset of
    [0, 2*pi) ~ [0, 6.28).  Some mu values may correspond to z-scores
    never actually seen during training.

Usage:
    python scripts/interpretability_analysis.py
    python scripts/interpretability_analysis.py --epochs 50
    python scripts/interpretability_analysis.py --output-dir results/interpretability
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend -- must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.von_mises_neuron import VonMisesForecaster

# Column order from generate_synthetic_co2() DataFrame
FEATURE_NAMES = ["Temperature", "Humidity", "CO2", "Noise"]


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
def _load_benchmark_module():
    """Dynamically import the benchmark script to reuse its data functions.

    Uses importlib to avoid triggering the benchmark's main() and to keep
    this script self-contained (same pattern as ablation_study.py).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark", str(PROJECT_ROOT / "scripts" / "benchmark_standard_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
# Training with per-epoch parameter logging
# ---------------------------------------------------------------------------
def train_with_logging(
    model: VonMisesForecaster,
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
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Train a VonMisesForecaster and log kappa/mu at every epoch.

    Mirrors _train_model() from benchmark_standard_dataset.py (same
    optimizer, scheduler, early stopping, gradient clipping) plus
    per-epoch snapshots of the first VM layer's parameters.

    Only the first VM layer (layer 0) is logged because it directly
    processes the 4 input features.  Layer 1's inputs are 64-dim
    hidden features, making them harder to interpret.

    Args:
        model: VonMisesForecaster to train.
        X_train, y_train: Training data tensors.
        X_val, y_val: Validation data tensors.
        lr: Learning rate (default 0.001).
        weight_decay: L2 regularization (default 0.0001).
        max_epochs: Max training epochs (default 50).
        batch_size: Mini-batch size (default 64).
        patience: Early stopping patience (default 15).
        gradient_clip: Max gradient norm (default 1.0).

    Returns:
        Tuple of (trained_model, epoch_logs) where epoch_logs is a list
        of dicts with keys: epoch, train_loss, val_loss, kappa, mu.
        kappa and mu are numpy arrays of shape (out_features, in_features).
    """
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
    epoch_logs: List[Dict[str, Any]] = []

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        # --- Snapshot VM parameters ---
        # Capture kappa and mu from the first VM layer at every epoch.
        # Memory: ~4 KB per epoch (two 64x4 float64 arrays) -- negligible.
        vm_layer_0 = model.get_vm_layers()[0]
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "kappa": vm_layer_0.get_concentrations().cpu().numpy().copy(),
            "mu": vm_layer_0.get_phases().cpu().numpy().copy(),
        })

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopped at epoch {epoch + 1} "
                  f"(best val loss: {best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, epoch_logs


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_concentration_heatmap(
    kappa: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Plot final kappa matrix as a heatmap (neurons x features).

    Each cell shows how sharply neuron j responds to feature i.
    High kappa = sharp, selective response; low = broad, diffuse.

    Args:
        kappa: Shape (n_neurons, n_features) with kappa > 0.
        feature_names: Column labels for the features.
        output_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        kappa,
        xticklabels=feature_names,
        yticklabels=False,  # 64 neuron labels would be unreadable
        cmap="YlOrRd",
        cbar_kws={"label": "kappa (concentration)"},
        ax=ax,
    )
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Concentration (kappa) -- First VM Layer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_concentration_evolution(
    epoch_logs: List[Dict[str, Any]],
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Plot mean kappa per feature across training epochs.

    Each line shows how the average concentration for one input feature
    evolves during training.  Rising lines indicate neurons sharpening
    their selectivity; flat lines suggest kappa is stable.

    Args:
        epoch_logs: List of epoch dicts with "kappa" arrays.
        feature_names: List of feature names for the legend.
        output_path: Where to save the PNG.
    """
    epochs = [log["epoch"] for log in epoch_logs]
    n_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(n_features):
        mean_kappa = [log["kappa"][:, i].mean() for log in epoch_logs]
        ax.plot(epochs, mean_kappa, label=feature_names[i],
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean kappa (across all neurons)")
    ax.set_title("Concentration Evolution During Training")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_phase_distribution(
    mu: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Plot histograms of learned mu values, one subplot per feature.

    Non-uniform distributions suggest neurons have discovered preferred
    operating points.  Uniform distributions indicate the feature is not
    strongly periodic or the neurons haven't specialized.

    Args:
        mu: Shape (n_neurons, n_features) in [0, 2*pi).
        feature_names: Feature names for subplot titles.
        output_path: Where to save the PNG.
    """
    n_features = len(feature_names)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_features):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.hist(
            mu[:, i], bins=20, range=(0, 2 * np.pi),
            edgecolor="black", alpha=0.7, color="#4C72B0",
        )
        ax.set_xlabel("mu (radians)")
        ax.set_ylabel("Count")
        ax.set_title(feature_names[i])
        ax.set_xlim(0, 2 * np.pi)
        # Mark pi and 2*pi for reference
        ax.axvline(np.pi, color="gray", linestyle="--", alpha=0.5, label="pi")

    # Hide unused subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Phase (mu) Distribution -- First VM Layer\n"
        "(cos peaks at x = -mu; inputs are z-scored, not angles)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_vm_response_curves(
    kappa: np.ndarray,
    mu: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 5,
) -> None:
    """Plot exp(kappa * cos(theta + mu)) for the top-kappa neurons.

    This visualizes each neuron's "receptive field" -- how it responds
    to different input values.  High kappa produces a sharp, narrow peak;
    low kappa produces a broad, nearly flat response.

    Args:
        kappa: Shape (n_neurons, n_features) -- final concentrations.
        mu: Shape (n_neurons, n_features) -- final phase shifts.
        feature_names: Feature names for column titles.
        output_path: Where to save the PNG.
        top_n: Number of top neurons to visualize (default 5).
    """
    n_features = len(feature_names)
    # Rank neurons by their maximum kappa across any feature
    neuron_max_kappa = kappa.max(axis=1)
    top_indices = np.argsort(neuron_max_kappa)[-top_n:][::-1]

    theta = np.linspace(-np.pi, np.pi, 500)

    fig, axes = plt.subplots(
        top_n, n_features,
        figsize=(4 * n_features, 3 * top_n),
        sharex=True,
    )
    if top_n == 1:
        axes = axes.reshape(1, -1)

    for row, neuron_idx in enumerate(top_indices):
        for col in range(n_features):
            ax = axes[row, col]
            k = kappa[neuron_idx, col]
            m = mu[neuron_idx, col]

            # The von Mises response: exp(kappa * cos(theta + mu))
            response = np.exp(k * np.cos(theta + m))
            ax.plot(theta, response, color="#1f77b4", linewidth=1.5)
            ax.fill_between(theta, response, alpha=0.15, color="#1f77b4")

            # Annotate with kappa value
            ax.text(
                0.95, 0.95, f"k={k:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(feature_names[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Neuron {neuron_idx}", fontsize=10)
            if row == top_n - 1:
                ax.set_xlabel("theta")

    fig.suptitle(
        f"VM Response Curves -- Top {top_n} Neurons by Max Concentration",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_attention(
    kappa: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> None:
    """Bar chart of mean kappa per feature (which features get most attention).

    Higher mean kappa indicates the VM layer dedicates more neuron
    selectivity to that feature -- a proxy for "importance" in the
    periodic feature encoding.

    Args:
        kappa: Shape (n_neurons, n_features) -- final concentrations.
        feature_names: Feature names for x-axis labels.
        output_path: Where to save the PNG.
    """
    mean_kappa = kappa.mean(axis=0)
    std_kappa = kappa.std(axis=0)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        feature_names, mean_kappa,
        yerr=std_kappa, capsize=5,
        color=colors[:len(feature_names)],
        edgecolor="black", alpha=0.85,
    )
    ax.set_ylabel("Mean kappa (across all neurons)")
    ax.set_xlabel("Input Feature")
    ax.set_title("Feature Attention -- Mean Concentration Per Input Feature")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars, mean_kappa):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------
def save_json_summary(
    epoch_logs: List[Dict[str, Any]],
    feature_names: List[str],
    output_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Save a JSON file with key numerical findings from the analysis.

    Converts numpy arrays to summary statistics (mean, std, max, min)
    since raw arrays are not JSON-serializable.

    Args:
        epoch_logs: Per-epoch parameter logs.
        feature_names: Feature names for labeling.
        output_dir: Directory to write the JSON file.
        config: Run configuration (epochs, seed, etc.).
    """
    final_kappa = epoch_logs[-1]["kappa"]
    final_mu = epoch_logs[-1]["mu"]
    initial_kappa = epoch_logs[0]["kappa"]

    # Per-feature kappa statistics
    kappa_stats = {}
    for i, name in enumerate(feature_names):
        col = final_kappa[:, i]
        kappa_stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "max": float(col.max()),
            "min": float(col.min()),
        }

    # Per-feature mu statistics (including circular mean)
    mu_stats = {}
    for i, name in enumerate(feature_names):
        col = final_mu[:, i]
        # Circular mean: arctan2(mean(sin), mean(cos)) wrapped to [0, 2*pi)
        circ_mean = float(
            np.arctan2(np.sin(col).mean(), np.cos(col).mean()) % (2 * np.pi)
        )
        mu_stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "circular_mean": circ_mean,
        }

    # Feature attention ranking (by mean kappa)
    mean_kappa_per_feature = final_kappa.mean(axis=0)
    ranking_indices = np.argsort(mean_kappa_per_feature)[::-1]
    feature_ranking = [feature_names[i] for i in ranking_indices]

    # Top 5 neuron-feature pairs by kappa
    flat_indices = np.argsort(final_kappa.ravel())[::-1][:5]
    top_pairs = []
    for flat_idx in flat_indices:
        neuron_idx, feat_idx = divmod(int(flat_idx), len(feature_names))
        top_pairs.append({
            "neuron": neuron_idx,
            "feature": feature_names[feat_idx],
            "kappa": float(final_kappa[neuron_idx, feat_idx]),
            "mu": float(final_mu[neuron_idx, feat_idx]),
        })

    # Concentration evolution summary
    initial_mean = float(initial_kappa.mean())
    final_mean = float(final_kappa.mean())
    if final_mean > initial_mean * 1.05:
        trend = "increasing (neurons sharpened)"
    elif final_mean < initial_mean * 0.95:
        trend = "decreasing (neurons broadened)"
    else:
        trend = "stable (roughly unchanged)"

    summary = {
        "config": config,
        "final_kappa_stats": kappa_stats,
        "final_mu_stats": mu_stats,
        "feature_attention_ranking": feature_ranking,
        "top_5_neuron_feature_pairs": top_pairs,
        "concentration_evolution": {
            "initial_mean_kappa": initial_mean,
            "final_mean_kappa": final_mean,
            "trend": trend,
        },
        "training": {
            "epochs_completed": len(epoch_logs),
            "final_train_loss": epoch_logs[-1]["train_loss"],
            "final_val_loss": epoch_logs[-1]["val_loss"],
            "best_val_loss": min(log["val_loss"] for log in epoch_logs),
        },
    }

    output_path = output_dir / "interpretability_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary saved to: {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_console_summary(
    epoch_logs: List[Dict[str, Any]],
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Print ASCII-only console summary of key findings."""
    final_kappa = epoch_logs[-1]["kappa"]
    final_mu = epoch_logs[-1]["mu"]
    initial_kappa = epoch_logs[0]["kappa"]

    n_neurons, n_features = final_kappa.shape

    print(f"\n{'=' * 68}")
    print(f"  INTERPRETABILITY ANALYSIS -- Von Mises Forecaster")
    print(f"{'=' * 68}")

    print(f"\n  Trained for {len(epoch_logs)} epochs")
    print(f"  First VM layer shape: {n_neurons} neurons x {n_features} features")

    # Feature attention ranking
    mean_kappa = final_kappa.mean(axis=0)
    ranking = np.argsort(mean_kappa)[::-1]

    print(f"\n  FEATURE ATTENTION (mean kappa per feature):")
    print(f"    {'Feature':<15s}  {'Mean kappa':>10s}  {'Rank':>4s}")
    print(f"    {'-' * 15}  {'-' * 10}  {'-' * 4}")
    for rank, idx in enumerate(ranking):
        print(f"    {feature_names[idx]:<15s}  {mean_kappa[idx]:>10.4f}  "
              f"{rank + 1:>4d}")

    # Concentration evolution
    initial_mean = initial_kappa.mean()
    final_mean = final_kappa.mean()
    if final_mean > initial_mean * 1.05:
        trend = "INCREASED (neurons sharpened)"
    elif final_mean < initial_mean * 0.95:
        trend = "DECREASED (neurons broadened)"
    else:
        trend = "STABLE (roughly unchanged)"

    print(f"\n  CONCENTRATION EVOLUTION:")
    print(f"    Initial mean kappa: {initial_mean:.4f}")
    print(f"    Final mean kappa:   {final_mean:.4f}")
    print(f"    Trend: {trend}")

    # Top 5 neuron-feature pairs
    flat_indices = np.argsort(final_kappa.ravel())[::-1][:5]
    print(f"\n  TOP 5 NEURON-FEATURE PAIRS (highest kappa):")
    print(f"    {'Neuron':>6s}  {'Feature':<15s}  {'kappa':>7s}  {'mu (rad)':>8s}")
    print(f"    {'-' * 6}  {'-' * 15}  {'-' * 7}  {'-' * 8}")
    for flat_idx in flat_indices:
        n_idx, f_idx = divmod(int(flat_idx), n_features)
        print(f"    {n_idx:>6d}  {feature_names[f_idx]:<15s}  "
              f"{final_kappa[n_idx, f_idx]:>7.3f}  "
              f"{final_mu[n_idx, f_idx]:>8.4f}")

    print(f"\n  NOTE ON PHASE INTERPRETATION:")
    print(f"    Inputs are z-scored (mean=0, std=1), NOT angles.")
    print(f"    The VM layer computes cos(x_i + mu_ji).")
    print(f"    cos peaks when x_i + mu = 0, i.e., x_i = -mu.")
    print(f"    So mu encodes the z-score value at which a neuron responds")
    print(f"    most strongly (after wrapping to [0, 2*pi)).")

    print(f"\n  Plots saved to: {output_dir}")
    print(f"{'=' * 68}\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the interpretability analysis."""
    parser = argparse.ArgumentParser(
        description="Interpretability analysis of VonMisesForecaster parameters"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/interpretability",
        help="Directory for output plots and JSON (default: results/interpretability)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-days", type=int, default=90,
        help="Days of synthetic data (default: 90)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"

    print(f"\n  Device: {device_str}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {output_dir}")

    seed_everything(args.seed)

    # --- Load data ---
    print(f"\n  Loading synthetic CO2 data ({args.n_days} days)...")
    benchmark_mod = _load_benchmark_module()
    df = benchmark_mod.generate_synthetic_co2(n_days=args.n_days, seed=args.seed)
    data = benchmark_mod.prepare_benchmark_data(
        df, lookback=72, horizon=12, device=device
    )

    input_size = data["X_train"].shape[2]
    print(f"  Input features: {input_size} ({', '.join(FEATURE_NAMES)})")
    print(f"  Train: {data['X_train'].shape[0]:,d} sequences")

    # --- Build model ---
    model = VonMisesForecaster(
        input_size=input_size,
        hidden_size=64,
        num_vm_layers=2,
        dropout=0.1,
        output_size=data["horizon"],
        init_kappa=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: VonMisesForecaster ({n_params:,d} params)")

    # --- Train with logging ---
    print(f"\n  Training with per-epoch parameter logging...")
    t0 = time.time()
    model, epoch_logs = train_with_logging(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=args.epochs,
    )
    elapsed = time.time() - t0
    print(f"  Training complete: {len(epoch_logs)} epochs in {elapsed:.1f}s")
    print(f"  Best val loss: {min(log['val_loss'] for log in epoch_logs):.6f}")

    # --- Extract final parameters ---
    final_kappa = epoch_logs[-1]["kappa"]  # shape (64, 4)
    final_mu = epoch_logs[-1]["mu"]        # shape (64, 4)

    # --- Generate plots ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating plots...")

    plot_concentration_heatmap(
        final_kappa, FEATURE_NAMES,
        output_dir / "concentration_heatmap.png",
    )
    print(f"    [1/5] concentration_heatmap.png")

    plot_concentration_evolution(
        epoch_logs, FEATURE_NAMES,
        output_dir / "concentration_evolution.png",
    )
    print(f"    [2/5] concentration_evolution.png")

    plot_phase_distribution(
        final_mu, FEATURE_NAMES,
        output_dir / "phase_distribution.png",
    )
    print(f"    [3/5] phase_distribution.png")

    plot_vm_response_curves(
        final_kappa, final_mu, FEATURE_NAMES,
        output_dir / "vm_response_curves.png",
    )
    print(f"    [4/5] vm_response_curves.png")

    plot_feature_attention(
        final_kappa, FEATURE_NAMES,
        output_dir / "feature_attention.png",
    )
    print(f"    [5/5] feature_attention.png")

    # --- Save JSON summary ---
    config = {
        "epochs_max": args.epochs,
        "epochs_trained": len(epoch_logs),
        "hidden_size": 64,
        "num_vm_layers": 2,
        "input_features": input_size,
        "feature_names": FEATURE_NAMES,
        "n_days": args.n_days,
        "seed": args.seed,
        "device": str(device),
        "training_time_seconds": round(elapsed, 2),
    }
    save_json_summary(epoch_logs, FEATURE_NAMES, output_dir, config)

    # --- Print console summary ---
    print_console_summary(epoch_logs, FEATURE_NAMES, output_dir)


if __name__ == "__main__":
    main()
