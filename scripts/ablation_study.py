"""Ablation study for the VonMisesForecaster architecture.

Isolates five architectural dimensions to understand what matters in the
VM architecture and what is incidental:

1. **VM layer depth** (1-4): Does stacking VM layers help, or does one suffice?
2. **κ initialization** (0.1-10.0): Does init matter, or does training recover?
3. **Log vs exp stacking** (True/False): Is log-space essential for stability?
4. **LayerNorm** (True/False): Does removing it cause high-κ neuron domination?
5. **Temporal model** (LSTM/mean pool): How much does the LSTM contribute?

Each ablation varies one dimension while holding all others at the default
(2 layers, κ=1.0, log=True, LN=True, LSTM=True).  This gives 16 unique
configurations total.

Usage:
    python scripts/ablation_study.py                          # all, synthetic, 50 epochs
    python scripts/ablation_study.py --ablations vm_depth     # single group
    python scripts/ablation_study.py --dataset jena           # real-world
    python scripts/ablation_study.py --epochs 5               # quick smoke test
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.von_mises_neuron import VonMisesForecaster


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
# Ablation configuration
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    """One experiment's architectural settings.

    All fields have defaults matching the standard VonMisesForecaster config.
    Ablation sweeps modify exactly one field at a time.
    """
    name: str = "default"
    num_vm_layers: int = 2
    init_kappa: float = 1.0
    log_output: bool = True
    use_layernorm: bool = True
    use_lstm: bool = True
    hidden_size: int = 64
    dropout: float = 0.1

    def is_default(self) -> bool:
        """Check if this config matches the default settings."""
        return (
            self.num_vm_layers == 2
            and self.init_kappa == 1.0
            and self.log_output is True
            and self.use_layernorm is True
            and self.use_lstm is True
        )


# The default config appears in every ablation group as the reference point.
DEFAULT_CONFIG = AblationConfig(name="default (2L, k=1.0, log, LN, LSTM)")


# ---------------------------------------------------------------------------
# Ablation registry: 5 groups, 16 unique configs
# ---------------------------------------------------------------------------
ABLATION_REGISTRY: Dict[str, List[AblationConfig]] = {
    "vm_depth": [
        AblationConfig(name="1 VM layer", num_vm_layers=1),
        DEFAULT_CONFIG,  # 2 layers (default)
        AblationConfig(name="3 VM layers", num_vm_layers=3),
        AblationConfig(name="4 VM layers", num_vm_layers=4),
    ],
    "kappa_init": [
        AblationConfig(name="k=0.1 (very broad)", init_kappa=0.1),
        AblationConfig(name="k=0.5 (broad)", init_kappa=0.5),
        DEFAULT_CONFIG,  # k=1.0 (default)
        AblationConfig(name="k=2.0 (moderate)", init_kappa=2.0),
        AblationConfig(name="k=5.0 (sharp)", init_kappa=5.0),
        AblationConfig(name="k=10.0 (very sharp)", init_kappa=10.0),
    ],
    "log_vs_exp": [
        DEFAULT_CONFIG,  # log-space (default)
        AblationConfig(name="exp-space stacking", log_output=False),
    ],
    "layernorm": [
        DEFAULT_CONFIG,  # with LayerNorm (default)
        AblationConfig(name="no LayerNorm", use_layernorm=False),
    ],
    "temporal_model": [
        DEFAULT_CONFIG,  # with LSTM (default)
        AblationConfig(name="mean pool (no LSTM)", use_lstm=False),
    ],
}


# ---------------------------------------------------------------------------
# Data loading — reuse existing infrastructure
# ---------------------------------------------------------------------------
def _load_synthetic_data(
    device: torch.device,
    lookback: int = 72,
    horizon: int = 12,
) -> Dict[str, Any]:
    """Generate synthetic CO2 data and prepare train/val/test tensors.

    Imports generate_synthetic_co2 and prepare_benchmark_data from the
    existing benchmark script to ensure identical data pipeline.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark", str(PROJECT_ROOT / "scripts" / "benchmark_standard_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = mod.generate_synthetic_co2(n_days=90, seed=42)
    data = mod.prepare_benchmark_data(df, lookback, horizon, device)
    return data


def _load_real_data(
    dataset_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a real-world dataset and prepare train/val/test tensors.

    Uses the data_loading registry and prepare_benchmark_data pipeline.
    """
    import importlib.util
    from src.data_loading import load_dataset

    spec = importlib.util.spec_from_file_location(
        "benchmark", str(PROJECT_ROOT / "scripts" / "benchmark_standard_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df, config = load_dataset(dataset_name)
    data = mod.prepare_benchmark_data(
        df,
        lookback=config["lookback"],
        horizon=config["horizon"],
        device=device,
        target_col=config["target_col"],
        split_ratios=config["split_ratios"],
    )
    return data


# ---------------------------------------------------------------------------
# Metrics (reuse from benchmark script)
# ---------------------------------------------------------------------------
def _compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    norm_stats: Dict[str, Tuple[float, float]],
    target_col: str = "CO2",
) -> Dict[str, float]:
    """Compute RMSE, MAE, R², MAPE in the original (denormalized) scale."""
    mean, std = norm_stats[target_col]
    y_pred_orig = y_pred * std + mean
    y_true_orig = y_true * std + mean

    mse = ((y_pred_orig - y_true_orig) ** 2).mean().item()
    rmse = mse ** 0.5
    mae = torch.abs(y_pred_orig - y_true_orig).mean().item()

    ss_res = ((y_pred_orig - y_true_orig) ** 2).sum().item()
    ss_tot = ((y_true_orig - y_true_orig.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    abs_true_safe = torch.clamp(torch.abs(y_true_orig), min=1.0)
    mape = (
        torch.abs(y_pred_orig - y_true_orig) / abs_true_safe
    ).mean().item() * 100.0

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Training with NaN/Inf safety
# ---------------------------------------------------------------------------
def _count_params(model: nn.Module) -> int:
    """Return total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


def _batched_predict(
    model: nn.Module, X: torch.Tensor, batch_size: int = 256
) -> torch.Tensor:
    """Run inference in mini-batches to avoid CUDA OOM on large datasets."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            chunks.append(model(X[i:i + batch_size]))
    return torch.cat(chunks, dim=0)


def _train_model_with_nan_check(
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
) -> Tuple[nn.Module, bool]:
    """Train a model with early stopping, LR scheduling, and NaN detection.

    Returns:
        Tuple of (model, diverged) where diverged=True if training produced
        NaN/Inf losses (expected for exp-space stacking ablation).
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
    diverged = False

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            # NaN/Inf detection — exp-space stacking can cause this
            if not torch.isfinite(loss):
                diverged = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if diverged:
            break

        # Validation with batched inference for CUDA safety
        model.eval()
        with torch.no_grad():
            val_pred = _batched_predict(model, X_val, batch_size=256)
            val_loss = criterion(val_pred, y_val).item()

        if not math.isfinite(val_loss):
            diverged = True
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

    if best_state is not None and not diverged:
        model.load_state_dict(best_state)
    model.eval()

    return model, diverged


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_single_experiment(
    config: AblationConfig,
    data: Dict[str, Any],
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate a single ablation config.

    Returns:
        Dict with metrics, params, time_seconds, diverged flag, and config.
    """
    seed_everything(seed)

    input_size = data["X_train"].shape[2]
    horizon = data.get("horizon", data["y_train"].shape[1])
    target_col = data.get("target_col", "CO2")

    model = VonMisesForecaster(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_vm_layers=config.num_vm_layers,
        dropout=config.dropout,
        output_size=horizon,
        init_kappa=config.init_kappa,
        log_output=config.log_output,
        use_layernorm=config.use_layernorm,
        use_lstm=config.use_lstm,
    ).to(device)

    params = _count_params(model)

    t0 = time.time()
    model, diverged = _train_model_with_nan_check(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )
    elapsed = time.time() - t0

    if diverged:
        return {
            "name": config.name,
            "diverged": True,
            "metrics": {"rmse": None, "mae": None, "r2": None, "mape": None},
            "params": params,
            "time_seconds": round(elapsed, 2),
            "config": asdict(config),
        }

    # Evaluate on test set with batched inference
    y_pred = _batched_predict(model, data["X_test"], batch_size=256)

    metrics = _compute_metrics(
        y_pred, data["y_test"], data["norm_stats"], target_col
    )

    return {
        "name": config.name,
        "diverged": False,
        "metrics": metrics,
        "params": params,
        "time_seconds": round(elapsed, 2),
        "config": asdict(config),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_ablation_table(
    group_name: str,
    results: List[Dict[str, Any]],
) -> None:
    """Print a formatted comparison table for one ablation group."""
    print(f"\n{'=' * 80}")
    print(f"  ABLATION: {group_name}")
    print(f"{'=' * 80}")

    # Header — use ASCII 'R2' to avoid cp932 encoding issues on Windows
    print(f"\n  {'Config':<35s} {'Params':>7s} {'RMSE':>8s} {'MAE':>8s}"
          f" {'R2':>8s} {'MAPE':>8s} {'Time':>7s}")
    print(f"  {'-' * 35} {'-' * 7} {'-' * 8} {'-' * 8}"
          f" {'-' * 8} {'-' * 8} {'-' * 7}")

    for res in results:
        name = res["name"]
        # Mark default config with asterisk
        if AblationConfig(**res["config"]).is_default():
            name = f"* {name}"
        else:
            name = f"  {name}"

        if res["diverged"]:
            print(
                f"  {name:<35s} {res['params']:>7,d} "
                f"{'DIVERGED':>8s} {'---':>8s} {'---':>8s} "
                f"{'---':>8s} {res['time_seconds']:>6.1f}s"
            )
        else:
            m = res["metrics"]
            print(
                f"  {name:<35s} {res['params']:>7,d} "
                f"{m['rmse']:>8.2f} {m['mae']:>8.2f} "
                f"{m['r2']:>8.4f} {m['mape']:>7.2f}% "
                f"{res['time_seconds']:>6.1f}s"
            )

    print()


def print_summary(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print a cross-group summary highlighting key findings."""
    print(f"\n{'=' * 80}")
    print(f"  ABLATION STUDY SUMMARY")
    print(f"{'=' * 80}")

    for group_name, results in all_results.items():
        # Find the default config's RMSE as baseline
        default_rmse = None
        for res in results:
            if AblationConfig(**res["config"]).is_default() and not res["diverged"]:
                default_rmse = res["metrics"]["rmse"]
                break

        if default_rmse is None:
            print(f"\n  {group_name}: default config diverged, cannot compare")
            continue

        # Find best and worst non-diverged configs
        valid = [r for r in results if not r["diverged"]]
        if not valid:
            print(f"\n  {group_name}: all configs diverged")
            continue

        best = min(valid, key=lambda r: r["metrics"]["rmse"])
        worst = max(valid, key=lambda r: r["metrics"]["rmse"])
        n_diverged = sum(1 for r in results if r["diverged"])

        best_delta = ((best["metrics"]["rmse"] - default_rmse) / default_rmse) * 100
        worst_delta = ((worst["metrics"]["rmse"] - default_rmse) / default_rmse) * 100

        print(f"\n  {group_name}:")
        print(f"    Best:  {best['name']:<30s} RMSE={best['metrics']['rmse']:.2f} "
              f"({best_delta:+.1f}% vs default)")
        print(f"    Worst: {worst['name']:<30s} RMSE={worst['metrics']['rmse']:.2f} "
              f"({worst_delta:+.1f}% vs default)")
        if n_diverged > 0:
            print(f"    Diverged: {n_diverged}/{len(results)} configs")

    print(f"\n{'=' * 80}\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the ablation study."""
    parser = argparse.ArgumentParser(
        description="Run ablation study on the VonMisesForecaster architecture"
    )
    parser.add_argument(
        "--ablations", type=str, nargs="+",
        choices=list(ABLATION_REGISTRY.keys()),
        default=list(ABLATION_REGISTRY.keys()),
        help="Which ablation groups to run (default: all)"
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "occupancy", "power", "jena"],
        help="Dataset to use (default: synthetic)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs per config (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation",
        help="Directory for results JSON (default: results/ablation)"
    )
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" ({torch.cuda.get_device_name(0)})"
    print(f"\n  Device: {device_str}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Ablation groups: {args.ablations}")

    # --- Load data ---
    print(f"\n  Loading data...")
    if args.dataset == "synthetic":
        data = _load_synthetic_data(device)
    else:
        data = _load_real_data(args.dataset, device)

    print(f"  Train: {data['X_train'].shape[0]:,d} sequences, "
          f"{data['X_train'].shape[2]} features")
    print(f"  Val:   {data['X_val'].shape[0]:,d} sequences")
    print(f"  Test:  {data['X_test'].shape[0]:,d} sequences")

    # --- Run ablations ---
    # Track configs already run to avoid re-running the default config
    # when it appears in multiple groups
    cache: Dict[str, Dict[str, Any]] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    total_configs = sum(len(ABLATION_REGISTRY[g]) for g in args.ablations)
    # Deduplicate: the default config appears in every group
    unique_configs = set()
    for group in args.ablations:
        for cfg in ABLATION_REGISTRY[group]:
            # Use config fields (excluding name) as the cache key
            key = (cfg.num_vm_layers, cfg.init_kappa, cfg.log_output,
                   cfg.use_layernorm, cfg.use_lstm)
            unique_configs.add(key)
    n_unique = len(unique_configs)

    print(f"\n  Total configs: {total_configs} "
          f"({n_unique} unique after deduplication)")

    config_idx = 0
    for group_name in args.ablations:
        configs = ABLATION_REGISTRY[group_name]
        group_results: List[Dict[str, Any]] = []

        print(f"\n{'-' * 60}")
        print(f"  Ablation group: {group_name} ({len(configs)} configs)")
        print(f"{'-' * 60}")

        for cfg in configs:
            # Cache key from config fields (name is just for display)
            cache_key = (cfg.num_vm_layers, cfg.init_kappa, cfg.log_output,
                         cfg.use_layernorm, cfg.use_lstm)

            if cache_key in cache:
                # Reuse result from a previous group (same default config)
                result = cache[cache_key].copy()
                result["name"] = cfg.name  # might have different display name
                tag = "cached"
            else:
                config_idx += 1
                print(f"\n  [{config_idx}/{n_unique}] {cfg.name}...")
                result = run_single_experiment(
                    cfg, data, args.epochs, device, args.seed
                )
                cache[cache_key] = result
                tag = "done"

            status = "DIVERGED" if result["diverged"] else (
                f"RMSE={result['metrics']['rmse']:.2f}"
            )
            print(f"    -> {status} ({result['time_seconds']:.1f}s) [{tag}]")

            group_results.append(result)

        all_results[group_name] = group_results
        print_ablation_table(group_name, group_results)

    # --- Summary ---
    print_summary(all_results)

    # --- Save JSON ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_results.json"

    json_output = {
        "config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "seed": args.seed,
            "device": str(device),
        },
        "ablations": {},
    }
    for group_name, results in all_results.items():
        json_output["ablations"][group_name] = results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
