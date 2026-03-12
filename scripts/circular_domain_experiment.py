"""Circular domain experiment — VM neuron on natively angular data.

All prior experiments fed z-scored real-valued data into cos(x + mu), where
the VM neuron's 2π-periodicity was arguably unnecessary.  This experiment
tests the VM neuron on data where angles are the native representation:

  1. Wind direction   — hourly forecasting, direction in radians
  2. Protein angles   — φ→ψ regression on the Ramachandran plot
  3. Robotics joints  — trajectory forecasting across 0/2π boundary

Key design choices:
  - Sin/cos output representation (model outputs 2×horizon: sin + cos parts)
  - Circular metrics (CMAE, CRMSE, CR²) instead of standard RMSE/MAE
  - Mixed normalization: angular cols raw [0, 2π), real cols z-scored
  - Circular loss for angle reconstruction

Usage:
    python scripts/circular_domain_experiment.py --epochs 10               # smoke test
    python scripts/circular_domain_experiment.py --epochs 100              # full run
    python scripts/circular_domain_experiment.py --domains wind robotics   # subset
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path so `src.*` imports work when run as a script
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.angular_data import (
    generate_ramachandran_data,
    generate_robot_trajectory,
    generate_wind_data,
)
from src.circular import (
    compute_circular_metrics,
    mixed_normalize,
    sincos_mse_loss,
    sincos_to_angle,
)
from src.sequences import create_sequences
from src.split import chronological_split
from src.von_mises_neuron import VonMisesForecaster, VonMisesLinear
from src.vm_variants import VonMisesAttentionForecaster
from src.baselines import LSTMForecaster, TransformerForecaster


# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


@dataclass
class DomainConfig:
    """Configuration for a circular domain experiment."""
    name: str
    task_type: str             # "sequence" or "regression"
    lookback: int = 24
    horizon: int = 6
    angular_cols: List[str] = field(default_factory=list)
    target_col: str = ""
    batch_size: int = 64
    lr: float = 1e-3


DOMAIN_CONFIGS = {
    "wind": DomainConfig(
        name="wind",
        task_type="sequence",
        lookback=24,
        horizon=6,
        angular_cols=["wind_direction"],
        target_col="wind_direction",
        batch_size=64,
        lr=1e-3,
    ),
    "protein": DomainConfig(
        name="protein",
        task_type="regression",
        lookback=0,   # not a sequence task
        horizon=1,    # predict 1 angle (psi)
        angular_cols=[],
        target_col="psi",
        batch_size=128,
        lr=1e-3,
    ),
    "robotics": DomainConfig(
        name="robotics",
        task_type="sequence",
        lookback=20,
        horizon=10,
        angular_cols=["joint1_angle", "joint2_angle"],
        target_col="joint1_angle",
        batch_size=64,
        lr=1e-3,
    ),
}


# ---------------------------------------------------------------------------
# Experiment-specific models (protein regression)
# ---------------------------------------------------------------------------


class VonMisesRegressor(nn.Module):
    """VM-based angular regressor for phi → psi prediction.

    Architecture: VonMisesLinear(1, H) → LN → VonMisesLinear(H, H) → LN → Linear(H, 2)

    Outputs (sin(psi), cos(psi)) to avoid the 0/2π discontinuity.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        # VM layers in log-space for inter-layer stacking
        self.vm1 = VonMisesLinear(1, hidden_size, log_output=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.vm2 = VonMisesLinear(hidden_size, hidden_size, log_output=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        # Output head: 2 values (sin, cos)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1) → (B, 2) as (sin_pred, cos_pred)."""
        h = self.ln1(self.vm1(x))
        h = self.ln2(self.vm2(h))
        return self.head(h)


class MLPRegressor(nn.Module):
    """Standard MLP regressor baseline for phi → psi prediction.

    Architecture: Linear(1, H) → ReLU → Linear(H, H) → ReLU → Linear(H, 2)

    Parameter-matched to VonMisesRegressor for fair comparison.
    """

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1) → (B, 2) as (sin_pred, cos_pred)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_models(
    config: DomainConfig,
    input_size: int,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """Build all models for a given domain.

    For sequence tasks: VonMisesForecaster, VonMisesAttentionForecaster,
                        LSTMForecaster, TransformerForecaster
    For regression:     VonMisesRegressor, MLPRegressor

    All sequence models output 2*horizon (sin/cos pairs).
    """
    models = {}

    if config.task_type == "sequence":
        # Output size = 2 * horizon (sin + cos for each predicted step)
        out_size = 2 * config.horizon
        h = 64

        models["VonMises+LSTM"] = VonMisesForecaster(
            input_size=input_size,
            hidden_size=h,
            num_vm_layers=2,
            output_size=out_size,
        )

        models["VM-Attention"] = VonMisesAttentionForecaster(
            input_size=input_size,
            d_model=h,
            nhead=4,
            num_layers=2,
            output_size=out_size,
        )

        models["LSTM"] = LSTMForecaster(
            input_size=input_size,
            hidden_size=h,
            num_layers=2,
            output_size=out_size,
        )

        models["Transformer"] = TransformerForecaster(
            input_size=input_size,
            d_model=h,
            nhead=4,
            num_layers=2,
            output_size=out_size,
        )

    else:  # regression
        h = 32
        models["VonMises"] = VonMisesRegressor(hidden_size=h)
        models["MLP"] = MLPRegressor(hidden_size=h)

    # Move all to device
    for name, model in models.items():
        models[name] = model.to(device)

    return models


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_wind_data(
    config: DomainConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Prepare wind direction data with mixed normalization.

    Returns (train_loader, val_loader, test_loader, n_features).
    Target y is stored as sin/cos pairs: (N, 2*horizon).
    """
    print("\n=== Wind Direction Data ===")
    df = generate_wind_data()

    # Mixed normalize: direction stays in [0, 2pi), speed+temp z-scored
    train_df, val_df, test_df = chronological_split(df)
    train_df, stats = mixed_normalize(train_df, angular_cols=config.angular_cols)
    val_df, _ = mixed_normalize(val_df, angular_cols=config.angular_cols, stats=stats)
    test_df, _ = mixed_normalize(test_df, angular_cols=config.angular_cols, stats=stats)

    loaders = []
    for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        X, y, _ = create_sequences(
            split_df, lookback=config.lookback,
            horizon=config.horizon, target_col=config.target_col,
        )
        # y is raw angles in [0, 2π) — convert to sin/cos
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_sincos = torch.cat([torch.sin(y_tensor), torch.cos(y_tensor)], dim=-1)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_sincos)
        shuffle = (name == "train")
        loaders.append(DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle))
        print(f"  {name}: X={X_tensor.shape}, y_sincos={y_sincos.shape}")

    return loaders[0], loaders[1], loaders[2], X.shape[-1]


def prepare_protein_data(
    config: DomainConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Prepare protein dihedral angle data.

    Non-sequential: phi → predict psi (angular regression).
    Returns (train_loader, val_loader, test_loader, n_features=1).
    """
    print("\n=== Protein Dihedral Angles ===")
    phi, psi = generate_ramachandran_data()

    # Random split (i.i.d. samples, not chronological)
    n = len(phi)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": perm[:n_train],
        "val":   perm[n_train:n_train + n_val],
        "test":  perm[n_train + n_val:],
    }

    loaders = []
    for name in ["train", "val", "test"]:
        idx = splits[name]
        # X = phi as (N, 1), y = psi target angle
        X_t = torch.tensor(phi[idx], dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        y_angle = torch.tensor(psi[idx], dtype=torch.float32)             # (N,)
        # Convert target to sin/cos: (N, 2)
        y_sincos = torch.stack([torch.sin(y_angle), torch.cos(y_angle)], dim=-1)

        dataset = TensorDataset(X_t, y_sincos)
        shuffle = (name == "train")
        loaders.append(DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle))
        print(f"  {name}: X={X_t.shape}, y_sincos={y_sincos.shape}")

    return loaders[0], loaders[1], loaders[2], 1


def prepare_robotics_data(
    config: DomainConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Prepare robotics joint trajectory data.

    Both columns are angular — no z-score at all.
    Returns (train_loader, val_loader, test_loader, n_features).
    """
    print("\n=== Robotics Joint Trajectory ===")
    df = generate_robot_trajectory()

    # Create a synthetic DatetimeIndex for chronological_split compatibility
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="min")

    # All columns are angular — no z-score
    train_df, val_df, test_df = chronological_split(df)
    # mixed_normalize with all angular cols → just wraps to [0, 2pi)
    train_df, stats = mixed_normalize(
        train_df, angular_cols=config.angular_cols,
    )
    val_df, _ = mixed_normalize(
        val_df, angular_cols=config.angular_cols, stats=stats,
    )
    test_df, _ = mixed_normalize(
        test_df, angular_cols=config.angular_cols, stats=stats,
    )

    loaders = []
    for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        X, y, _ = create_sequences(
            split_df, lookback=config.lookback,
            horizon=config.horizon, target_col=config.target_col,
        )
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_sincos = torch.cat([torch.sin(y_tensor), torch.cos(y_tensor)], dim=-1)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_sincos)
        shuffle = (name == "train")
        loaders.append(DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle))
        print(f"  {name}: X={X_tensor.shape}, y_sincos={y_sincos.shape}")

    return loaders[0], loaders[1], loaders[2], X.shape[-1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DomainConfig,
    epochs: int,
    device: torch.device,
    task_type: str = "sequence",
) -> Dict:
    """Train a model with sin/cos MSE loss and circular validation metrics.

    Returns a dict with training history and best validation metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    best_val_cmae = float("inf")
    best_epoch = 0
    patience_counter = 0
    max_patience = 25  # early stopping patience

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if task_type == "sequence":
                out = model(X_batch)  # (B, 2*horizon)
            else:
                out = model(X_batch)  # (B, 2)

            # Sin/cos MSE loss: compare model output with sin/cos targets
            loss = nn.functional.mse_loss(out, y_batch)

            # NaN guard
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch + 1}, stopping")
                return {"stopped": "nan_loss", "epoch": epoch}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # --- Validate ---
        val_metrics = evaluate_model(model, val_loader, config, device, task_type)
        val_cmae = val_metrics["cmae"]
        scheduler.step(val_cmae)

        # Early stopping
        if val_cmae < best_val_cmae:
            best_val_cmae = val_cmae
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs}: "
                  f"loss={avg_train_loss:.6f}  "
                  f"val_CMAE={val_cmae:.4f} rad ({math.degrees(val_cmae):.1f}deg)  "
                  f"val_CR2={val_metrics['cr2']:.4f}")

        if patience_counter >= max_patience:
            print(f"    Early stopping at epoch {epoch + 1} "
                  f"(best was epoch {best_epoch})")
            break

    return {
        "best_val_cmae": best_val_cmae,
        "best_epoch": best_epoch,
        "final_epoch": epoch + 1,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: DomainConfig,
    device: torch.device,
    task_type: str = "sequence",
) -> Dict[str, float]:
    """Evaluate a model with circular metrics.

    Reconstructs angles from sin/cos outputs via atan2, then computes
    CMAE, CRMSE, CR² against ground truth angles.
    """
    model.eval()
    all_pred_angles = []
    all_true_angles = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            out = model(X_batch)

            if task_type == "sequence":
                horizon = config.horizon
                # out: (B, 2*horizon) → split into sin and cos halves
                sin_pred = out[:, :horizon]
                cos_pred = out[:, horizon:]
                # Target: same structure
                sin_true = y_batch[:, :horizon]
                cos_true = y_batch[:, horizon:]
            else:
                # out: (B, 2) → sin, cos
                sin_pred = out[:, 0]
                cos_pred = out[:, 1]
                sin_true = y_batch[:, 0]
                cos_true = y_batch[:, 1]

            # Reconstruct angles via atan2
            pred_angle = sincos_to_angle(sin_pred, cos_pred)
            true_angle = sincos_to_angle(sin_true, cos_true)

            all_pred_angles.append(pred_angle.cpu())
            all_true_angles.append(true_angle.cpu())

    pred = torch.cat(all_pred_angles).flatten()
    true = torch.cat(all_true_angles).flatten()

    return compute_circular_metrics(pred, true)


# ---------------------------------------------------------------------------
# Kappa diagnostics
# ---------------------------------------------------------------------------


def extract_kappa_diagnostics(model: nn.Module, name: str) -> Optional[Dict]:
    """Extract concentration (κ) statistics from VM-based models."""
    kappas = []

    if hasattr(model, "get_vm_layers"):
        # VonMisesForecaster
        for vm in model.get_vm_layers():
            kappas.append(vm.get_concentrations().cpu())
    elif hasattr(model, "get_attention_concentrations"):
        # VonMisesAttentionForecaster
        kappas = [k.cpu() for k in model.get_attention_concentrations()]
    elif isinstance(model, VonMisesRegressor):
        kappas.append(model.vm1.get_concentrations().cpu())
        kappas.append(model.vm2.get_concentrations().cpu())
    else:
        return None

    if not kappas:
        return None

    all_k = torch.cat([k.flatten() for k in kappas])
    return {
        "mean": float(all_k.mean()),
        "std": float(all_k.std()),
        "min": float(all_k.min()),
        "max": float(all_k.max()),
        "n_layers": len(kappas),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_domain(
    domain_name: str,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    """Run all models on a single domain and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = DOMAIN_CONFIGS[domain_name]
    print(f"\n{'=' * 70}")
    print(f"  DOMAIN: {domain_name.upper()}")
    print(f"  Task: {config.task_type}, lookback={config.lookback}, "
          f"horizon={config.horizon}")
    print(f"{'=' * 70}")

    # --- Data preparation ---
    if domain_name == "wind":
        train_loader, val_loader, test_loader, n_features = \
            prepare_wind_data(config, device)
    elif domain_name == "protein":
        train_loader, val_loader, test_loader, n_features = \
            prepare_protein_data(config, device)
    elif domain_name == "robotics":
        train_loader, val_loader, test_loader, n_features = \
            prepare_robotics_data(config, device)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    # --- Build and train models ---
    models = build_models(config, n_features, device)
    domain_results = {}

    for model_name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {model_name} ({n_params:,} params) ---")

        # Reset seed for each model so they see the same data order
        torch.manual_seed(seed)

        t0 = time.time()
        train_info = train_model(
            model, train_loader, val_loader, config,
            epochs=epochs, device=device, task_type=config.task_type,
        )
        train_time = time.time() - t0

        # Test evaluation
        test_metrics = evaluate_model(
            model, test_loader, config, device, task_type=config.task_type,
        )

        # κ diagnostics
        kappa_info = extract_kappa_diagnostics(model, model_name)

        domain_results[model_name] = {
            "metrics": test_metrics,
            "params": n_params,
            "time_seconds": round(train_time, 2),
            "train_info": train_info,
        }
        if kappa_info is not None:
            domain_results[model_name]["kappa_stats"] = kappa_info

        print(f"  Test CMAE = {test_metrics['cmae']:.4f} rad "
              f"({test_metrics['cmae_deg']:.1f} deg)")
        print(f"  Test CRMSE = {test_metrics['crmse']:.4f} rad "
              f"({test_metrics['crmse_deg']:.1f} deg)")
        print(f"  Test CR2 = {test_metrics['cr2']:.4f}")
        if kappa_info:
            print(f"  kappa: mean={kappa_info['mean']:.3f}, "
                  f"std={kappa_info['std']:.3f}, "
                  f"range=[{kappa_info['min']:.3f}, {kappa_info['max']:.3f}]")

    return domain_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_domain_table(domain_name: str, results: Dict) -> None:
    """Print a formatted comparison table for one domain."""
    print(f"\n{'=' * 75}")
    print(f"  {domain_name.upper()} -- Test Results")
    print(f"{'=' * 75}")

    header = f"  {'Model':<25} {'CMAE(deg)':>10} {'CRMSE(deg)':>11} {'CR2':>8} {'Params':>8} {'Time(s)':>8}"
    print(header)
    print(f"  {'-' * 73}")

    # Sort by CMAE (best first)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["cmae"],
    )

    best_cmae = sorted_models[0][1]["metrics"]["cmae"]

    for model_name, info in sorted_models:
        m = info["metrics"]
        cmae_deg = m["cmae_deg"]
        crmse_deg = m["crmse_deg"]
        cr2 = m["cr2"]
        params = info["params"]
        t = info["time_seconds"]

        # Delta vs best
        delta = ((m["cmae"] / best_cmae) - 1) * 100 if best_cmae > 0 else 0
        delta_str = f" (+{delta:.1f}%)" if delta > 0.5 else " (best)"

        print(f"  {model_name:<25} {cmae_deg:>9.1f}  {crmse_deg:>10.1f}  "
              f"{cr2:>8.4f} {params:>8,} {t:>8.1f}{delta_str}")


def print_cross_domain_summary(all_results: Dict) -> None:
    """Print a cross-domain comparison summary."""
    print(f"\n{'=' * 75}")
    print(f"  CROSS-DOMAIN SUMMARY -- VM vs Best Baseline")
    print(f"{'=' * 75}")

    print(f"  {'Domain':<12} {'Best VM Model':<25} {'VM CMAE':>9} "
          f"{'Best Baseline':<18} {'Base CMAE':>10} {'Delta':>8}")
    print(f"  {'-' * 85}")

    for domain_name, results in all_results.items():
        # Separate VM and baseline models
        vm_models = {k: v for k, v in results.items()
                     if "vonmises" in k.lower() or "vm" in k.lower()}
        baseline_models = {k: v for k, v in results.items()
                          if k not in vm_models}

        if not vm_models or not baseline_models:
            continue

        best_vm_name = min(vm_models, key=lambda k: vm_models[k]["metrics"]["cmae"])
        best_base_name = min(baseline_models, key=lambda k: baseline_models[k]["metrics"]["cmae"])

        vm_cmae = vm_models[best_vm_name]["metrics"]["cmae_deg"]
        base_cmae = baseline_models[best_base_name]["metrics"]["cmae_deg"]
        delta = ((vm_cmae / base_cmae) - 1) * 100 if base_cmae > 0 else 0
        delta_str = f"{delta:+.1f}%"

        print(f"  {domain_name:<12} {best_vm_name:<25} {vm_cmae:>8.1f}  "
              f"{best_base_name:<18} {base_cmae:>9.1f}  {delta_str:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Circular domain experiment — VM on angular data",
    )
    parser.add_argument(
        "--domains", nargs="+",
        default=["wind", "protein", "robotics"],
        choices=["wind", "protein", "robotics"],
        help="Domains to run (default: all three)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Domains: {args.domains}")
    print(f"Epochs: {args.epochs}")

    all_results = {}
    total_t0 = time.time()

    for domain_name in args.domains:
        domain_results = run_domain(
            domain_name, epochs=args.epochs, device=device, seed=args.seed,
        )
        all_results[domain_name] = domain_results
        print_domain_table(domain_name, domain_results)

    # Cross-domain summary
    if len(all_results) > 1:
        print_cross_domain_summary(all_results)

    total_time = time.time() - total_t0
    print(f"\n  Total experiment time: {total_time:.1f}s")

    # --- Save results ---
    output_dir = Path("results/circular_domains")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert metrics to JSON-serializable format
    json_results = {
        "config": {
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
            "domains": args.domains,
        },
        "domains": {},
    }
    for domain_name, results in all_results.items():
        json_results["domains"][domain_name] = {}
        for model_name, info in results.items():
            json_results["domains"][domain_name][model_name] = {
                "metrics": info["metrics"],
                "params": info["params"],
                "time_seconds": info["time_seconds"],
            }
            if "kappa_stats" in info:
                json_results["domains"][domain_name][model_name]["kappa_stats"] = \
                    info["kappa_stats"]

    json_path = output_dir / "circular_domain_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
