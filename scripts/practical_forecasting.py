"""Evaluate practical power forecasts with frozen development and final blocks."""

import argparse
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_standard_dataset import seed_everything
from scripts.controlled_temporal_comparison import source_manifest, write_json
from src.baselines import LSTMForecaster
from src.forecasting import PersistenceResidualForecaster
from src.forecast_training import batched_predict, train_forecaster
from src.operational_data import load_cached_hourly, prepare_forecast_windows
from src.practical_evaluation import (
    fit_ridge_residual, forecast_metrics, select_candidate, simple_predictions,
)
from src.sincos import SinCosForecaster
from src.temporal import TemporalVonMisesForecaster

NEURAL = ["ResidualLSTM", "ResidualSinCos", "ResidualTemporal"]
SIMPLE = ["Persistence", "Seasonal24", "Seasonal168"]
DEVELOPMENT_STARTS = ["2008-02-01", "2008-08-01", "2009-02-01"]
FINAL_START = "2010-10-29"
TARGET = "Global_active_power"
IMPLEMENTATION_PATHS = [
    "src/operational_data.py", "src/practical_evaluation.py", "src/forecasting.py",
    "src/forecast_training.py", "src/baselines.py", "src/sincos.py", "src/temporal.py",
    "src/von_mises_neuron.py", "scripts/practical_forecasting.py",
    "scripts/benchmark_standard_dataset.py", "scripts/controlled_temporal_comparison.py",
]


def fold_boundaries(test_start: str) -> dict[str, pd.Timestamp]:
    """Construct the prespecified half-open 180/28/28-day intervals."""
    start = pd.Timestamp(test_start)
    val = start - pd.Timedelta(days=28)
    return dict(train_start=val - pd.Timedelta(days=180), val_start=val,
                test_start=start, test_end=start + pd.Timedelta(days=28))


def make_neural(name: str, metadata: dict[str, Any]) -> torch.nn.Module:
    """Construct a persistence-initialized correction model with shared inputs."""
    sensors = len(metadata["sensor_columns"])
    if name == "ResidualLSTM":
        body = LSTMForecaster(sensors + 2, hidden_size=68, num_layers=2,
                              dropout=0.1, output_size=24)
    elif name == "ResidualSinCos":
        body = SinCosForecaster(sensors + 2, hidden_size=64, num_vm_layers=2,
                                dropout=0.1, output_size=24)
    elif name == "ResidualTemporal":
        body = TemporalVonMisesForecaster(sensor_size=sensors, time_periods=[24, 168],
                                          hidden_size=64, num_vm_layers=2, dropout=0.1,
                                          output_size=24)
    else:
        raise ValueError(f"Unknown neural family {name}")
    return PersistenceResidualForecaster(body, sensors, metadata["target_idx"])


def save_forecasts(path: Path, predictions: np.ndarray, split: dict[str, Any]) -> None:
    """Save common physical-unit targets and exact forecast-origin timestamps."""
    np.savez_compressed(path, prediction=predictions, target=split["y_raw"],
                        origins=split["origins"], target_times=split["target_times"])


def evaluate_fold(frame: pd.DataFrame, test_start: str, output: Path,
                  families: list[str], seeds: list[int], epochs: int,
                  *, prepare_only: bool = False) -> dict[str, Any]:
    """Fit using training/validation only, then score one untouched fold block."""
    output.mkdir(parents=True, exist_ok=False)
    splits, meta = prepare_forecast_windows(frame, **fold_boundaries(test_start),
                                           lookback=168, horizon=24, target_col=TARGET)
    if any(len(split["X"]) == 0 for split in splits.values()):
        raise ValueError("Every split needs valid windows")
    write_json(output / "data_manifest.json", meta)
    report: dict[str, Any] = {"status": "prepared" if prepare_only else "running",
                              "test_start": test_start, "runs": [], "families": {}}
    write_json(output / "results.json", report)
    if prepare_only:
        return report
    target_index = meta["target_idx"]
    mean, scale = meta["norm_stats"][TARGET]
    train, val, test = (splits[key] for key in ["train", "val", "test"])
    # Scoring happens only after candidates/settings for this invocation were frozen.
    for name, normalized in simple_predictions(test["X"], target_index, 24).items():
        physical = normalized.astype(np.float64) * scale + mean
        report["families"][name] = forecast_metrics(physical, test["y_raw"])
        save_forecasts(output / f"{name}.npz", physical, test)
    write_json(output / "results.json", report)
    if "RidgeResidual" in families:
        start = time.perf_counter()
        ridge_row = {"family": "RidgeResidual", "seed": None, "status": "running"}
        report["runs"].append(ridge_row)
        try:
            ridge, fit_meta = fit_ridge_residual(train["X"], train["y"], val["X"], val["y"],
                                             target_index=target_index)
            physical = ridge.predict(test["X"]).astype(np.float64) * scale + mean
            report["families"]["RidgeResidual"] = forecast_metrics(physical, test["y_raw"])
            ridge_row.update(status="completed", training=fit_meta)
            save_forecasts(output / "RidgeResidual.npz", physical, test)
            if ridge.estimator is not None:
                np.savez_compressed(output / "RidgeResidual_coefficients.npz",
                                coefficients=ridge.estimator.coef_, intercept=ridge.estimator.intercept_)
        except FloatingPointError as error:
            ridge_row.update(status="numerical_failure", error=str(error))
        except BaseException as error:
            ridge_row.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["status"] = "failed"
            raise
        finally:
            ridge_row["time_seconds"] = time.perf_counter() - start
            write_json(output / "results.json", report)
        print(f"{test_start} RidgeResidual status={ridge_row['status']} "
              f"RMSE={report['families'].get('RidgeResidual', {}).get('rmse')}", flush=True)
    x_train, y_train = torch.from_numpy(train["X"]), torch.from_numpy(train["y"])
    x_val, y_val = torch.from_numpy(val["X"]), torch.from_numpy(val["y"])
    x_test = torch.from_numpy(test["X"])
    for name in [family for family in families if family in NEURAL]:
        predictions = []
        for seed in seeds:
            row: dict[str, Any] = {"family": name, "seed": seed, "status": "running"}
            report["runs"].append(row)
            model = None
            start = time.perf_counter()
            try:
                seed_everything(seed)
                model = make_neural(name, meta)
                row.update(params=sum(p.numel() for p in model.parameters()), architecture=repr(model))
                seed_everything(seed)
                model = train_forecaster(model, x_train, y_train, x_val, y_val,
                                           max_epochs=epochs, source_config={"family": name, "seed": seed})
                normalized = batched_predict(model, x_test).numpy()
                physical = normalized.astype(np.float64) * scale + mean
                row.update(status="completed", metrics=forecast_metrics(physical, test["y_raw"]))
                save_forecasts(output / f"{name}_{seed}.npz", physical, test)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           output / f"{name}_{seed}_state.pt")
                predictions.append(physical)
            except FloatingPointError as error:
                row.update(status="numerical_failure", error=str(error))
            except BaseException as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                report["status"] = "failed"
                raise
            finally:
                if model is not None and hasattr(model, "training_history"):
                    row["training"] = model.training_history
                if model is not None and row["status"] != "completed":
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                               output / f"{name}_{seed}_failed_state.pt")
                row["time_seconds"] = time.perf_counter() - start
                write_json(output / "results.json", report)
                del model
            print(f"{test_start} {name} seed={seed} status={row['status']} "
                  f"RMSE={row.get('metrics', {}).get('rmse')} epoch={row.get('training', {}).get('best_epoch')}", flush=True)
        # Never form a favorable survivor ensemble after a failed seed.
        if len(predictions) == len(seeds):
            ensemble = np.mean(predictions, axis=0)
            report["families"][name] = forecast_metrics(ensemble, test["y_raw"])
            save_forecasts(output / f"{name}_ensemble.npz", ensemble, test)
        write_json(output / "results.json", report)
    report["status"] = "completed"
    report["all_fits_finite"] = all(row["status"] == "completed" for row in report["runs"])
    write_json(output / "results.json", report)
    return report


def main() -> None:
    """Run development or the once-selected final candidate with full provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--phase", choices=["development", "final"], default="development")
    parser.add_argument("--synthetic-validation", type=Path)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs != 50 or args.seeds != [42, 43, 44]:
        parser.error("Frozen v1 protocol requires epochs=50 and seeds=42 43 44")
    if args.phase == "development" and not args.prepare_only and args.synthetic_validation is None:
        parser.error("Development fitting requires --synthetic-validation")
    if args.phase == "final" and args.selection_dir is None:
        parser.error("Final evaluation requires a completed --selection-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"status": "running", "phase": args.phase,
                                "created_utc": datetime.now(timezone.utc).isoformat(),
                                "epochs": args.epochs, "seeds": args.seeds}
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        frame, source = load_cached_hourly("power", args.data_dir, min_coverage=0.75)
        manifest.update(environment=source_manifest(), data=source, protocol="practical_forecasting_v1",
                        target=TARGET, unit="kW", coverage=0.75, lookback=168, horizon=24,
                        protocol_sha256=hashlib.sha256((ROOT / "docs/practical_forecasting_protocol.md").read_bytes()).hexdigest())
        manifest["implementation_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in IMPLEMENTATION_PATHS}
        with zipfile.ZipFile(args.output_dir / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for pattern in ["src/*.py", "scripts/*.py", "tests/*.py", "requirements.txt",
                            "docs/practical_forecasting_protocol.md"]:
                for path in sorted(ROOT.glob(pattern)):
                    archive.write(path, path.relative_to(ROOT))
        families = ["RidgeResidual", *NEURAL]
        starts = DEVELOPMENT_STARTS
        if args.phase == "development" and not args.prepare_only:
            validation = json.loads(args.synthetic_validation.read_text(encoding="utf-8"))
            candidates = validation["candidates"]
            families = ["RidgeResidual", *[name for name in NEURAL if candidates[name] is True]]
            manifest["synthetic_validation"] = {"path": str(args.synthetic_validation.resolve()),
                                                "sha256": hashlib.sha256(args.synthetic_validation.read_bytes()).hexdigest(),
                                                "candidates": candidates}
        elif args.phase == "final":
            previous = json.loads((args.selection_dir / "manifest.json").read_text(encoding="utf-8"))
            selection = json.loads((args.selection_dir / "selection.json").read_text(encoding="utf-8"))
            if previous["status"] != "completed" or previous["phase"] != "development":
                raise ValueError("Selection must come from completed development evaluation")
            if previous["epochs"] != args.epochs or previous["seeds"] != args.seeds:
                raise ValueError("Final refit must preserve development training settings")
            for key in ["protocol_sha256", "implementation_sha256", "data"]:
                if previous[key] != manifest[key]:
                    raise ValueError(f"Final refit must preserve development {key}")
            if previous["environment"]["packages"] != manifest["environment"]["packages"]:
                raise ValueError("Final refit must preserve package versions")
            archived_folds = {start: json.loads((args.selection_dir / start / "results.json").read_text())
                              for start in DEVELOPMENT_STARTS}
            common = set.intersection(*(set(report["families"]) for report in archived_folds.values()))
            expected = select_candidate({start: {name: metrics for name, metrics in report["families"].items()
                                                  if name in common} for start, report in archived_folds.items()},
                                        family_order=[*SIMPLE, "RidgeResidual", *NEURAL])
            if selection["selected_family"] != expected["selected_family"]:
                raise ValueError("Saved selection does not match frozen development selection rule")
            families = [selection["selected_family"]]
            starts = [FINAL_START]
            manifest["selection"] = selection
            manifest["selection_sha256"] = hashlib.sha256((args.selection_dir / "selection.json").read_bytes()).hexdigest()
        manifest.update(families=families, test_starts=starts)
        write_json(args.output_dir / "manifest.json", manifest)
        reports = {}
        with threadpool_limits(limits=4):
            for start in starts:
                reports[start] = evaluate_fold(frame, start, args.output_dir / start, families,
                                               args.seeds, args.epochs, prepare_only=args.prepare_only)
        if not args.prepare_only:
            if args.phase == "development":
                # A failed family is removed from eligibility, with all failed rows retained.
                common = set.intersection(*(set(report["families"]) for report in reports.values()))
                fold_metrics = {key: {name: report["families"][name] for name in report["families"] if name in common}
                                for key, report in reports.items()}
                selection = select_candidate(fold_metrics, family_order=[*SIMPLE, "RidgeResidual", *NEURAL])
                selection["incomplete_families"] = sorted(set(families) - common)
                write_json(args.output_dir / "selection.json", selection)
            else:
                metrics = reports[FINAL_START]["families"]
                chosen = families[0]
                best_simple = min(SIMPLE, key=lambda name: metrics[name]["rmse"])
                available = chosen in metrics
                improved = available and chosen not in SIMPLE and metrics[chosen]["rmse"] <= 0.95 * metrics[best_simple]["rmse"]
                write_json(args.output_dir / "final_assessment.json", {
                    "selected_family": chosen, "best_final_simple": best_simple,
                    "selected_metrics": metrics.get(chosen), "best_simple_metrics": metrics[best_simple],
                    "learned_5pct_improvement_confirmed": improved,
                    "note": "No family reselection or retry based on this final block",
                })
                if not available:
                    raise FloatingPointError("Selected final family failed; no survivor substitution permitted")
        manifest.update(status="prepared" if args.prepare_only else "completed",
                        all_fits_finite=all(report.get("all_fits_finite", True) for report in reports.values()),
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
