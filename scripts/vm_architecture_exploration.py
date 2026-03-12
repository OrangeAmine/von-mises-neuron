"""VM Architecture Exploration -- LSTM-free variant comparison.

Compares four architectures on synthetic periodic CO2 data:
  1. VonMisesForecaster (reference): VM encoder + LSTM
  2. LSTMForecaster (reference): Linear encoder + LSTM
  3. DeepVMPoolForecaster: Deep residual VM stack + temporal conv + mean pool
  4. VonMisesAttentionForecaster: VM kernel attention (periodic Transformer)

Goal: Explore whether the VM neuron can model temporal dynamics *without*
an LSTM backbone.  This is research exploration, not an attempt to beat LSTM.

Usage:
    python scripts/vm_architecture_exploration.py --epochs 5    # smoke test
    python scripts/vm_architecture_exploration.py --epochs 50   # full run
    python scripts/vm_architecture_exploration.py --groups baselines vm_attention
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project root for importlib and results
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.von_mises_neuron import VonMisesForecaster
from src.baselines import LSTMForecaster
from src.vm_variants import DeepVMPoolForecaster, VonMisesAttentionForecaster


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureConfig:
    """Configuration for a single architecture experiment."""

    name: str
    model_type: str  # vm_forecaster | lstm | deep_vm_pool | vm_attention

    # Common
    hidden_size: int = 64
    dropout: float = 0.1

    # DeepVMPool-specific
    num_blocks: int = 4
    conv_kernel_size: int = 3
    use_residual: bool = True
    use_conv: bool = True

    # VonMisesAttention-specific
    nhead: int = 4
    num_attention_layers: int = 2
    dim_feedforward: int = 128

    # VM-common
    init_kappa: float = 1.0
    num_vm_layers: int = 2  # for VonMisesForecaster baseline


# Experiment registry: groups of configs to compare
EXPERIMENTS: Dict[str, List[ArchitectureConfig]] = {
    "baselines": [
        ArchitectureConfig(
            name="VonMisesForecaster (ref)",
            model_type="vm_forecaster",
        ),
        ArchitectureConfig(
            name="LSTMForecaster (ref)",
            model_type="lstm",
        ),
    ],
    "deep_vm_depth": [
        ArchitectureConfig(
            name="DeepVM-4",
            model_type="deep_vm_pool",
            num_blocks=4,
        ),
        ArchitectureConfig(
            name="DeepVM-8",
            model_type="deep_vm_pool",
            num_blocks=8,
        ),
        ArchitectureConfig(
            name="DeepVM-16",
            model_type="deep_vm_pool",
            num_blocks=16,
        ),
    ],
    "deep_vm_ablation": [
        ArchitectureConfig(
            name="DeepVM-4 (no residual)",
            model_type="deep_vm_pool",
            num_blocks=4,
            use_residual=False,
        ),
        ArchitectureConfig(
            name="DeepVM-4 (no conv)",
            model_type="deep_vm_pool",
            num_blocks=4,
            use_conv=False,
        ),
        ArchitectureConfig(
            name="DeepVM-4 (no res, no conv)",
            model_type="deep_vm_pool",
            num_blocks=4,
            use_residual=False,
            use_conv=False,
        ),
    ],
    "vm_attention": [
        ArchitectureConfig(
            name="VM-Attn-2L",
            model_type="vm_attention",
            num_attention_layers=2,
        ),
        ArchitectureConfig(
            name="VM-Attn-4L",
            model_type="vm_attention",
            num_attention_layers=4,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Data loading (reuse from benchmark_standard_dataset.py)
# ---------------------------------------------------------------------------

def _load_benchmark_module():
    """Dynamically load the benchmark module for data generation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark", str(PROJECT_ROOT / "scripts" / "benchmark_standard_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_synthetic_data(
    device: torch.device,
    lookback: int = 72,
    horizon: int = 12,
) -> Dict[str, Any]:
    """Load synthetic CO2 data using the benchmark pipeline."""
    mod = _load_benchmark_module()
    df = mod.generate_synthetic_co2(n_days=90, seed=42)
    data = mod.prepare_benchmark_data(df, lookback, horizon, device)
    return data


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(
    config: ArchitectureConfig,
    input_size: int,
    output_size: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate a model from an ArchitectureConfig."""
    if config.model_type == "vm_forecaster":
        return VonMisesForecaster(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_vm_layers=config.num_vm_layers,
            dropout=config.dropout,
            output_size=output_size,
            init_kappa=config.init_kappa,
        ).to(device)

    elif config.model_type == "lstm":
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_vm_layers,
            dropout=config.dropout,
            output_size=output_size,
        ).to(device)

    elif config.model_type == "deep_vm_pool":
        return DeepVMPoolForecaster(
            input_size=input_size,
            d_model=config.hidden_size,
            num_blocks=config.num_blocks,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            output_size=output_size,
            init_kappa=config.init_kappa,
            use_residual=config.use_residual,
            use_conv=config.use_conv,
        ).to(device)

    elif config.model_type == "vm_attention":
        return VonMisesAttentionForecaster(
            input_size=input_size,
            d_model=config.hidden_size,
            nhead=config.nhead,
            num_layers=config.num_attention_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            output_size=output_size,
            init_kappa=config.init_kappa,
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


# ---------------------------------------------------------------------------
# Training (same optimizer/scheduler as benchmark, with NaN detection)
# ---------------------------------------------------------------------------

def _train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 64,
    patience: int = 15,
    gradient_clip: float = 1.0,
) -> Tuple[nn.Module, bool]:
    """Train model with early stopping and NaN detection.

    Returns:
        Tuple of (trained model, diverged flag).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
    )

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    diverged = False

    for epoch in range(max_epochs):
        # -- Train --
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            # NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        if diverged:
            break

        # -- Validate --
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if math.isnan(val_loss) or math.isinf(val_loss):
            diverged = True
            break

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best checkpoint
    if best_state is not None and not diverged:
        model.load_state_dict(best_state)
        model.to(X_train.device)

    return model, diverged


# ---------------------------------------------------------------------------
# Metrics (denormalized)
# ---------------------------------------------------------------------------

def _batched_predict(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run inference in batches to avoid CUDA OOM."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i + batch_size]))
    return torch.cat(preds, dim=0)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_stats: Dict[str, Any],
    target_col: str = "CO2",
) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, MAPE on denormalized predictions."""
    # norm_stats stores {col: (mean, std)} tuples
    mean, std = norm_stats[target_col]

    pred_np = pred.cpu().numpy() * std + mean
    target_np = target.cpu().numpy() * std + mean

    residuals = pred_np - target_np
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((target_np - target_np.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    denom = np.clip(np.abs(target_np), 1.0, None)
    mape = float(np.mean(np.abs(residuals) / denom) * 100)

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Kappa diagnostics
# ---------------------------------------------------------------------------

def extract_kappa_diagnostics(
    model: nn.Module,
    config: ArchitectureConfig,
) -> List[Dict[str, Any]]:
    """Extract kappa statistics from VM-based models.

    Returns a list of dicts, one per layer/head, with kappa stats.
    """
    diagnostics = []

    if config.model_type in ("vm_forecaster", "deep_vm_pool"):
        kappas = model.get_layer_concentrations()
        for i, k in enumerate(kappas):
            k_np = k.cpu().numpy()
            diagnostics.append({
                "layer": i,
                "type": "vm_layer",
                "mean": float(k_np.mean()),
                "std": float(k_np.std()),
                "min": float(k_np.min()),
                "max": float(k_np.max()),
            })

    elif config.model_type == "vm_attention":
        kappas = model.get_attention_concentrations()
        for i, k in enumerate(kappas):
            k_np = k.cpu().numpy()
            diagnostics.append({
                "layer": i,
                "type": "attn_head",
                "mean": float(k_np.mean()),
                "std": float(k_np.std()),
                "min": float(k_np.min()),
                "max": float(k_np.max()),
            })

    return diagnostics


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: ArchitectureConfig,
    data: Dict[str, Any],
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate one architecture configuration.

    Returns dict with name, metrics, params, time, kappa diagnostics.
    """
    seed_everything(seed)

    input_size = data["X_train"].shape[-1]
    output_size = data["y_train"].shape[-1]

    model = build_model(config, input_size, output_size, device)
    params = _count_params(model)

    t0 = time.time()
    model, diverged = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )
    train_time = time.time() - t0

    result = {
        "name": config.name,
        "model_type": config.model_type,
        "params": params,
        "time_seconds": round(train_time, 1),
        "diverged": diverged,
        "config": asdict(config),
    }

    if diverged:
        result["metrics"] = {"rmse": float("inf"), "mae": float("inf"),
                             "r2": float("-inf"), "mape": float("inf")}
        result["kappa_diagnostics"] = []
    else:
        pred = _batched_predict(model, data["X_test"])
        metrics = compute_metrics(
            pred, data["y_test"], data["norm_stats"],
            target_col=data.get("target_col", "CO2"),
        )
        result["metrics"] = metrics
        result["kappa_diagnostics"] = extract_kappa_diagnostics(model, config)

    return result


# ---------------------------------------------------------------------------
# Reporting (ASCII-only for Windows cp932)
# ---------------------------------------------------------------------------

def print_results_table(
    results: List[Dict[str, Any]],
    title: str,
) -> None:
    """Print a formatted metrics comparison table."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()

    header = (
        f"  {'Model':<30s} {'Params':>8s} {'RMSE':>8s} {'MAE':>8s}"
        f" {'R2':>8s} {'MAPE':>8s} {'Time':>7s}"
    )
    print(header)
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

    for r in results:
        m = r["metrics"]
        if r["diverged"]:
            print(f"  {r['name']:<30s} {r['params']:>8,d}   ** DIVERGED **")
        else:
            print(
                f"  {r['name']:<30s} {r['params']:>8,d}"
                f" {m['rmse']:>8.2f} {m['mae']:>8.2f}"
                f" {m['r2']:>8.4f} {m['mape']:>7.2f}%"
                f" {r['time_seconds']:>6.1f}s"
            )

    print()


def print_kappa_table(results: List[Dict[str, Any]]) -> None:
    """Print kappa diagnostics for VM-based models."""
    # Filter to models with kappa diagnostics
    vm_results = [r for r in results if r.get("kappa_diagnostics")]
    if not vm_results:
        return

    print("  KAPPA DIAGNOSTICS (learned concentration):")
    print(
        f"  {'Model':<30s} {'Layer':>6s} {'Mean':>8s} {'Std':>8s}"
        f" {'Min':>8s} {'Max':>8s}"
    )
    print(f"  {'-' * 30} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for r in vm_results:
        for i, kd in enumerate(r["kappa_diagnostics"]):
            label = r["name"] if i == 0 else ""
            layer_label = f"L{kd['layer']}"
            print(
                f"  {label:<30s} {layer_label:>6s}"
                f" {kd['mean']:>8.4f} {kd['std']:>8.4f}"
                f" {kd['min']:>8.4f} {kd['max']:>8.4f}"
            )

    print()


def print_comparison_summary(results: List[Dict[str, Any]]) -> None:
    """Print relative performance vs the best baseline."""
    # Find reference models
    ref_vm = next((r for r in results if r["model_type"] == "vm_forecaster"), None)
    ref_lstm = next((r for r in results if r["model_type"] == "lstm"), None)

    if not ref_vm or ref_vm["diverged"]:
        return

    ref_rmse = ref_vm["metrics"]["rmse"]

    print("  RELATIVE PERFORMANCE (vs VonMisesForecaster reference):")
    print(f"  {'Model':<30s} {'RMSE':>8s} {'Delta':>10s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 10}")

    for r in results:
        if r["diverged"]:
            print(f"  {r['name']:<30s}      --   DIVERGED")
            continue
        rmse = r["metrics"]["rmse"]
        delta = ((rmse - ref_rmse) / ref_rmse) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {r['name']:<30s} {rmse:>8.2f} {sign}{delta:>8.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VM Architecture Exploration",
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
        help="Experiment groups to run (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/architecture_exploration",
        help="Directory for results output",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    print()
    print("  VM ARCHITECTURE EXPLORATION")
    print(f"  Device: {device} ({device_name})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Groups: {', '.join(args.groups)}")

    # Load data
    print()
    print("  Loading synthetic CO2 data...")
    data = _load_synthetic_data(device)
    input_size = data["X_train"].shape[-1]
    output_size = data["y_train"].shape[-1]
    print(f"  Input features: {input_size}, Horizon: {output_size}")
    print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    # Collect all configs to run (deduplicate by name)
    all_configs = []
    seen_names = set()
    for group in args.groups:
        for config in EXPERIMENTS[group]:
            if config.name not in seen_names:
                all_configs.append((group, config))
                seen_names.add(config.name)

    print(f"\n  Running {len(all_configs)} experiments...")
    print()

    # Run experiments
    all_results = []
    group_results: Dict[str, List[Dict]] = {g: [] for g in args.groups}

    for i, (group, config) in enumerate(all_configs, 1):
        print(f"  [{i}/{len(all_configs)}] {config.name}...", end="", flush=True)
        result = run_single_experiment(
            config, data, args.epochs, device, args.seed,
        )
        all_results.append(result)

        # Add to each group this config belongs to
        for g in args.groups:
            if any(c.name == config.name for c in EXPERIMENTS[g]):
                group_results[g].append(result)

        if result["diverged"]:
            print(f" DIVERGED ({result['time_seconds']}s)")
        else:
            m = result["metrics"]
            print(
                f" RMSE={m['rmse']:.2f}, R2={m['r2']:.4f}"
                f" ({result['params']:,} params, {result['time_seconds']}s)"
            )

    # Print per-group tables
    for group in args.groups:
        results = group_results[group]
        if results:
            title = f"{group.upper().replace('_', ' ')} -- Synthetic CO2 (Periodic)"
            print_results_table(results, title)
            print_kappa_table(results)

    # Print overall comparison
    print_results_table(all_results, "ALL ARCHITECTURES -- Synthetic CO2 (Periodic)")
    print_kappa_table(all_results)
    print_comparison_summary(all_results)

    # Save JSON results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "exploration_results.json"

    json_output = {
        "config": {
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
            "device_name": device_name,
            "input_size": input_size,
            "output_size": output_size,
        },
        "experiments": {
            group: group_results[group] for group in args.groups
        },
        "all_results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(json_output, f, indent=2, default=str)

    print(f"  Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
