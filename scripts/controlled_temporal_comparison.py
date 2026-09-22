"""Run paired, budget-controlled synthetic VM comparisons with source manifests.

Data seed and training seeds are separate. Test metrics are descriptive; failed
ablations are retained and never used to select a new hyperparameter setting.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_standard_dataset import (
    THRESHOLDS, _train_model, augment_with_time_features, check_regression,
    check_thresholds, compute_benchmark_metrics, generate_synthetic_co2,
    prepare_benchmark_data, prepare_temporal_data, seed_everything,
)
from src.baselines import HandCraftedLinearForecaster
from src.sincos import SinCosForecaster
from src.split import chronological_split
from src.temporal import TemporalVonMisesForecaster
from src.von_mises_neuron import VonMisesForecaster

VARIANTS = [
    "VonMises", "SinCos", "VonMisesTime", "SinCosTime", "TemporalVM",
    "NoTime", "UnitScales", "LearnedScales", "NoConv", "Pooling", "HandCrafted",
]
CORE_VARIANTS = {"VonMises", "SinCos", "VonMisesTime", "SinCosTime", "TemporalVM"}
COMPARISON_PAIRS = [
    ("SinCos", "VonMises"), ("SinCosTime", "VonMisesTime"),
    ("VonMisesTime", "VonMises"), ("SinCosTime", "SinCos"),
    ("TemporalVM", "VonMises"), ("TemporalVM", "SinCos"),
    ("TemporalVM", "VonMisesTime"), ("TemporalVM", "SinCosTime"),
    *[(name, "TemporalVM") for name in
      ["NoTime", "UnitScales", "LearnedScales", "NoConv", "Pooling", "HandCrafted"]],
    ("HandCrafted", "VonMises"), ("HandCrafted", "SinCos"),
]


def apply_calendar_clock(frame: pd.DataFrame, clock: str = "local") -> pd.DataFrame:
    """Select calendar coordinates while preserving observations and instants.

    Args:
        frame: Synthetic data with its original timezone-aware Paris index.
        clock: ``local`` preserves historical DST behavior; ``fixed`` converts
            the index to UTC+01:00, matching winter training calendar phases.

    Returns:
        The original frame for local mode, or a copy with a fixed-offset index.
        Sensor values and elapsed-time hand-crafted features are unchanged.

    Raises:
        ValueError: If the clock option is unknown or the index is timezone-naive.
    """
    if clock not in {"local", "fixed"}:
        raise ValueError("calendar clock must be local or fixed")
    if frame.index.tz is None:
        raise ValueError("calendar clock requires a timezone-aware index")
    if clock == "local":
        return frame
    converted = frame.copy()
    converted.index = converted.index.tz_convert(timezone(timedelta(hours=1)))
    return converted


def source_manifest() -> dict[str, Any]:
    """Record the executable source tree and environment, including dirty files."""
    paths = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")])
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "numpy", "pandas", "scipy", "scikit-learn", "pytest"]},
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def json_safe(value: Any) -> Any:
    """Replace nonfinite numeric values with null while retaining failure text."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically save strict JSON so completed runs survive interruptions."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_model(name: str, data: dict[str, Any]) -> torch.nn.Module:
    """Initialize paired controls from a shared reference before training.

    Temporal removals retain all compatible reference tensors. UnitScales
    absorbs fixed amplitudes into projection columns, matching its initial
    phase function. It tests amplitude parameterization and optimization, not
    frequency content; the reference bank's initial zero bias is essential.
    """
    common = dict(hidden_size=64, num_vm_layers=2, dropout=0.1,
                  output_size=data["horizon"])
    if name in {"VonMises", "SinCos", "VonMisesTime", "SinCosTime"}:
        base = VonMisesForecaster(input_size=data["X_train"].shape[-1], **common)
        return SinCosForecaster.from_von_mises(base) if name.startswith("SinCos") else base
    if name == "HandCrafted":
        return HandCraftedLinearForecaster(input_size=data["X_train"].shape[-1],
                                          lookback=data["lookback"], output_size=data["horizon"])
    temporal_common = dict(sensor_size=data["sensor_size"], time_periods=data["time_periods"],
                           **common)
    reference = TemporalVonMisesForecaster(freq_bands=[0.5, 2.0, 8.0], **temporal_common)
    if name == "TemporalVM":
        return reference
    variant = TemporalVonMisesForecaster(
        freq_bands=[1.0, 1.0, 1.0] if name == "UnitScales" else [0.5, 2.0, 8.0],
        use_time_conditioning=name != "NoTime", learnable_kappa=name == "LearnedScales",
        use_temporal_conv=name != "NoConv", use_lstm=name != "Pooling", **temporal_common,
    )
    target_state = variant.state_dict()
    shared_state = {
        key: tensor for key, tensor in reference.state_dict().items()
        if key in target_state and tensor.shape == target_state[key].shape
        and not (name == "UnitScales" and key == "encoder.freq_bank.kappa")
    }
    variant.load_state_dict(shared_state, strict=False)
    if name == "UnitScales":
        with torch.no_grad():
            for projection in variant.encoder.delta_mu_projections:
                projection.weight.mul_(reference.encoder.freq_bank.get_kappa().unsqueeze(0))
    return variant


def reference_predictions(data: dict[str, Any], target_index: int,
                          season: int = 24) -> dict[str, torch.Tensor]:
    """Predict using only observations available in each lookback window."""
    x = data["X_test"][..., target_index]
    horizon = data["horizon"]
    if x.shape[1] < season:
        raise ValueError("Seasonal persistence requires a full season of lookback")
    indices = x.shape[1] - season + torch.arange(horizon, device=x.device) % season
    return {"Persistence": x[:, -1:].expand(-1, horizon),
            "SeasonalPersistence": x[:, indices]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain failures and disclose the number of finite seeds in each statistic."""
    summary = {}
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [row for row in rows if row["model"] == name]
        summary[name] = {"n_seeds": len(group), "params": group[0]["params"],
                         "numerical_failures": sum(r.get("execution_status") == "numerical_failure" for r in group),
                         "execution_failures": sum(r.get("execution_status") == "failed" for r in group),
                         "absolute_failures": sum(not r["absolute_passed"] for r in group),
                         "absolute_passed_all": all(r["absolute_passed"] for r in group)}
        for metric in THRESHOLDS:
            values = [row["metrics"][metric] for row in group
                      if row["metrics"][metric] is not None and math.isfinite(row["metrics"][metric])]
            summary[name][metric] = describe_values(values)
    return summary


def describe_values(values: list[float]) -> dict[str, Any]:
    """Describe finite seed replicates, leaving unavailable estimates null."""
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def paired_differences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report candidate-minus-reference differences only for matched finite seeds."""
    lookup = {(row["model"], row["seed"]): row for row in rows}
    pairs = {}
    for candidate, reference in COMPARISON_PAIRS:
        seeds = sorted({seed for model, seed in lookup if model == candidate}
                       & {seed for model, seed in lookup if model == reference})
        if not seeds:
            continue
        metrics = {}
        for metric in THRESHOLDS:
            differences, excluded = {}, []
            for seed in seeds:
                a = lookup[candidate, seed]["metrics"][metric]
                b = lookup[reference, seed]["metrics"][metric]
                if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
                    excluded.append(seed)
                else:
                    differences[str(seed)] = a - b
            metrics[metric] = {**describe_values(list(differences.values())),
                               "by_seed": differences, "excluded_seeds": excluded}
        pairs[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate, "reference": reference, "metrics": metrics,
        }
    return pairs


def validation_summary(rows: list[dict[str, Any]], models: list[str],
                       seeds: list[int]) -> dict[str, Any]:
    """Separate required core gates from descriptive ablation outcomes."""
    lookup = {(row["model"], row["seed"]): row for row in rows}
    required = [name for name in models if name in CORE_VARIANTS]
    core_checks = {
        name: all((name, seed) in lookup and lookup[name, seed]["absolute_passed"] for seed in seeds)
        for name in required
    }
    regression = {}
    if "TemporalVM" in models and "VonMises" in models:
        for seed in seeds:
            if ("TemporalVM", seed) not in lookup or ("VonMises", seed) not in lookup:
                regression[str(seed)] = {"passed": False, "failures": ["Run not completed"]}
                continue
            def metrics(name: str) -> dict[str, float]:
                return {key: value if value is not None else math.nan
                        for key, value in lookup[name, seed]["metrics"].items()}
            passed, failures = check_regression(metrics("TemporalVM"), metrics("VonMises"))
            regression[str(seed)] = {"passed": passed, "failures": failures}
    complete = len(rows) == len(models) * len(seeds)
    return {
        "all_runs_attempted": complete, "required_core_models": required,
        "core_absolute_checks": core_checks,
        "core_passed": all(core_checks.values()) and all(r["passed"] for r in regression.values()),
        "all_absolute_passed": complete and all(row["absolute_passed"] for row in rows),
        "temporal_regression": regression,
        "exploratory_absolute_checks": {
            name: all((name, seed) in lookup and lookup[name, seed]["absolute_passed"] for seed in seeds)
            for name in models if name not in CORE_VARIANTS
        },
    }


def _run_comparison(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Execute the protocol, persisting every attempted model before proceeding."""
    with zipfile.ZipFile(args.output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ["src/*.py", "scripts/*.py", "tests/*.py", "requirements.txt"]:
            for path in sorted(ROOT.glob(pattern)):
                archive.write(path, path.relative_to(ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = generate_synthetic_co2(n_days=90, seed=args.data_seed)
    df = apply_calendar_clock(df, args.calendar_clock)
    data = prepare_benchmark_data(df, 72, 12, device)
    temporal = prepare_temporal_data(df, 72, 12, device, time_periods=[24, 168])
    augmented = prepare_benchmark_data(augment_with_time_features(df, [24, 168]), 72, 12, device)
    split_info = {}
    for name, split in zip(["train", "val", "test"], chronological_split(df, [0.8, 0.1, 0.1])):
        split_info[name] = {"rows": len(split), "first": str(split.index[0]),
                            "last": str(split.index[-1]), "windows": len(data[f"X_{name}"])}
    manifest.update({
        "environment": source_manifest(), "data_seed": args.data_seed,
        "training_seeds": args.seeds, "models": args.models, "epochs": args.epochs,
        "lookback": 72, "horizon": 12, "target": "CO2", "unit": "ppm",
        "split": split_info, "normalization": data["norm_stats"],
        "sensor_columns": list(df.columns), "time_periods": [24, 168],
        "calendar_clock": args.calendar_clock,
        "calendar_timezone": str(df.index.tz),
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest(),
        "training_rng": "reset to seed after construction for every model",
        "thresholds": THRESHOLDS,
    })
    write_json(args.output_dir / "manifest.json", manifest)
    references = {}
    for name, prediction in reference_predictions(data, list(df.columns).index("CO2")).items():
        references[name] = compute_benchmark_metrics(prediction, data["y_test"], data["norm_stats"])
    rows: list[dict[str, Any]] = []
    def persist_results() -> dict[str, Any]:
        validation = validation_summary(rows, args.models, args.seeds)
        write_json(args.output_dir / "results.json", {
            "runs": rows, "summary": summarize(rows), "references": references,
            "paired_differences": paired_differences(rows), "validation": validation,
            "temporal_regression": validation["temporal_regression"],
        })
        return validation
    persist_results()
    for seed in args.seeds:
        for name in args.models:
            model_data = augmented if name == "HandCrafted" else (
                data if name in {"VonMises", "SinCos"} else temporal)
            start = time.perf_counter()
            model = None
            row = {"model": name, "seed": seed, "params": None,
                   "metrics": {metric: None for metric in THRESHOLDS},
                   "absolute_passed": False, "absolute_failures": [],
                   "execution_status": "running"}
            try:
                seed_everything(seed)
                model = build_model(name, model_data).to(device)
                row.update(params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           architecture=repr(model))
                seed_everything(seed)
                start = time.perf_counter()
                model = _train_model(model, model_data["X_train"], model_data["y_train"],
                                     model_data["X_val"], model_data["y_val"], max_epochs=args.epochs)
                history = model.training_history
                row["training"] = json_safe(history)
                if not all(math.isfinite(value) for value in history["validation_loss"]):
                    raise FloatingPointError("Nonfinite validation loss during training")
                if not math.isfinite(history["best_val_loss"]):
                    raise FloatingPointError("No finite validation checkpoint")
                with torch.no_grad():
                    prediction = model(model_data["X_test"])
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite predictions")
                metrics = compute_benchmark_metrics(prediction, model_data["y_test"], model_data["norm_stats"])
                row["metrics"] = json_safe(metrics)
                passed, failures = check_thresholds(metrics)
                row.update(absolute_passed=passed, absolute_failures=failures)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite evaluation metrics")
                row["execution_status"] = "completed"
                np.savez_compressed(args.output_dir / f"{name}_{seed}_predictions.npz",
                                    prediction=prediction.cpu().numpy(), target=model_data["y_test"].cpu().numpy())
                torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                           args.output_dir / f"{name}_{seed}_state.pt")
            except FloatingPointError as exc:
                row["execution_status"] = "numerical_failure"
                row["absolute_failures"].append(str(exc))
            except BaseException as exc:
                row["execution_status"] = "failed"
                row["absolute_passed"] = False
                row["absolute_failures"].append(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                row["time_seconds"] = time.perf_counter() - start
                if model is not None and hasattr(model, "training_history"):
                    row["training"] = json_safe(model.training_history)
                rows.append(row)
                persist_results()
            rmse = row["metrics"]["rmse"]
            print(f"{name} seed={seed} params={row['params']} RMSE={rmse} "
                  f"absolute_passed={row['absolute_passed']} status={row['execution_status']}", flush=True)
            del model
    return persist_results()


def main() -> None:
    """Persist execution status separately from core validation and exit gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--calendar-clock", choices=["local", "fixed"], default="local",
                        help="Calendar index: historical Paris/DST (local, default) or fixed UTC+01:00")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("Epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        parser.error("Seeds and model names must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "execution_status": "running",
                "models": args.models, "training_seeds": args.seeds,
                "calendar_clock": args.calendar_clock}
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        validation = _run_comparison(args, manifest)
    except BaseException as exc:
        manifest.update(status="failed", execution_status="failed",
                        validation_status="incomplete",
                        error=f"{type(exc).__name__}: {exc}",
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    manifest.update(status="completed", execution_status="completed", validation=validation,
                    validation_status=("not_applicable" if not validation["required_core_models"]
                                       else "passed" if validation["core_passed"] else "failed"),
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / "manifest.json", manifest)
    sys.exit(0 if validation["core_passed"] else 1)


if __name__ == "__main__":
    main()
