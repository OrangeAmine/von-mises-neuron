"""Independently audit completed practical-forecast artifacts without fitting.

This script reads only the explicitly requested run/fold. It never selects a
family, accesses a sibling final run, or changes archived experiment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import torch


def independent_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Recompute physical-unit metrics without using the production evaluator."""
    assert prediction.shape == target.shape and prediction.ndim == 2
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    squared = np.square(difference)
    absolute = np.abs(difference)
    variation = np.sum(np.square(target - np.mean(target)))
    denominator = np.sum(np.abs(target))
    return {
        "rmse": float(np.sqrt(np.mean(squared))), "mae": float(np.mean(absolute)),
        "r2": None if variation == 0 else float(1 - np.sum(squared) / variation),
        "wape": None if denominator == 0 else float(100 * np.sum(absolute) / denominator),
        "mape": float(np.mean(100 * absolute / np.clip(np.abs(target), .1, None))),
        "mape_denominator_floor_kw": .1,
        "per_horizon_rmse": np.sqrt(np.mean(squared, axis=0)).tolist(),
        "sse": float(np.sum(squared)), "n_targets": int(target.size),
    }


def _compare(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require tight agreement of every archived forecast metric."""
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None, key
        else:
            np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-12,
                                       err_msg=f"Metric mismatch: {key}")


def audit_history(history: dict[str, Any]) -> dict[str, Any]:
    """Replay checkpoint choice from epoch-zero validation history."""
    assert history["status"] == "completed"
    losses = np.asarray(history["validation_loss"], dtype=np.float64)
    epochs = history["epochs_completed"]
    assert len(losses) == epochs + 1 and np.isfinite(losses).all()
    assert history["validation_epochs"] == list(range(epochs + 1))
    assert len(history["training_loss"]) == len(history["learning_rate"]) == epochs
    assert np.isfinite(history["training_loss"]).all()
    assert np.isfinite(history["learning_rate"]).all()
    best = 0
    for epoch in range(1, epochs + 1):
        if losses[epoch] < losses[best] - history["min_delta"]:
            best = epoch
    assert best == history["best_epoch"]
    assert losses[best] == history["best_val_loss"] and losses[best] <= losses[0]
    return {"epochs_completed": epochs, "best_epoch": best,
            "initial_validation_mse": float(losses[0]),
            "selected_validation_mse": float(losses[best])}


def audit_fold(directory: Path) -> dict[str, Any]:
    """Audit one completed fold's metrics, alignments, ensembles and histories."""
    directory = Path(directory)
    report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed", "Fold is not complete"
    common = None
    forecasts: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}

    def check_archive(label: str, path: Path, metrics: dict[str, Any]) -> None:
        nonlocal common
        with np.load(path, allow_pickle=False) as archive:
            prediction, target = archive["prediction"], archive["target"]
            origins, times = archive["origins"], archive["target_times"]
        assert origins.shape == (len(target),) and times.shape == target.shape
        expected_times = origins[:, None] + np.arange(1, target.shape[1] + 1).astype("timedelta64[h]")
        np.testing.assert_array_equal(times, expected_times)
        assert np.all(times >= np.datetime64(metadata["boundaries"]["test_start"]))
        assert np.all(times < np.datetime64(metadata["boundaries"]["test_end"]))
        if common is None:
            common = target, origins, times
        else:
            for actual, expected in zip((target, origins, times), common):
                np.testing.assert_array_equal(actual, expected)
        recomputed = independent_metrics(prediction, target)
        _compare(recomputed, metrics)
        forecasts[label] = prediction
        summary[label] = {**recomputed, "predicted_mean_kw": float(prediction.mean()),
                          "observed_mean_kw": float(target.mean()),
                          "mean_error_kw": float((prediction - target).mean()),
                          "predicted_min_kw": float(prediction.min()),
                          "predicted_max_kw": float(prediction.max()),
                          "negative_prediction_fraction": float(np.mean(prediction < 0))}

    for family, metrics in report["families"].items():
        suffix = "_ensemble" if family.startswith("Residual") else ""
        check_archive(family, directory / f"{family}{suffix}.npz", metrics)
    histories = {}
    seeds_by_family: dict[str, list[dict[str, Any]]] = {}
    ridge_persistence = None
    for row in report["runs"]:
        if row["status"] != "completed":
            continue
        if row["family"] == "RidgeResidual":
            history = row["training"]
            scores = history["validation_scores"]
            ridge_persistence = scores[0]["validation_mse"]
            best = scores[0]
            for score in scores[1:]:
                if score["validation_mse"] < best["validation_mse"] - 1e-12:
                    best = score
            assert best["alpha"] == history["selected_alpha"]
            assert best["validation_mse"] == history["best_val_loss"]
            continue
        label = f"{row['family']}_{row['seed']}"
        check_archive(label, directory / f"{label}.npz", row["metrics"])
        histories[label] = audit_history(row["training"])
        checkpoint = torch.load(directory / f"{label}_state.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint.values())
        assert checkpoint, "Empty checkpoint"
        seeds_by_family.setdefault(row["family"], []).append(row)
    for history in histories.values():
        if ridge_persistence is not None:
            np.testing.assert_allclose(history["initial_validation_mse"], ridge_persistence,
                                       rtol=1e-12, atol=1e-12)
    variability = {}
    for family, rows in seeds_by_family.items():
        if family not in forecasts:
            continue
        labels = [f"{family}_{row['seed']}" for row in rows]
        ensemble = np.mean([forecasts[label] for label in labels], axis=0)
        np.testing.assert_array_equal(ensemble, forecasts[family])
        errors = [row["metrics"]["rmse"] for row in rows]
        variability[family] = {"seeds": [row["seed"] for row in rows],
                               "individual_rmse_mean_kw": float(np.mean(errors)),
                               "individual_rmse_sample_sd_kw": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
                               "ensemble_rmse_kw": summary[family]["rmse"]}
    assert common is not None
    assert len(common[0]) == metadata["splits"]["test"]["valid_windows"]
    np.testing.assert_allclose(common[0].mean(), metadata["splits"]["test"]["pooled_window_targets"]["mean"],
                               rtol=1e-12, atol=1e-12)
    return {"passed": True, "fold": report["test_start"], "archives_checked": len(forecasts),
            "origins": len(common[1]), "unique_future_hours": int(np.unique(common[2]).size),
            "metrics_and_levels": summary, "checkpoint_selection": histories,
            "seed_variability": variability, "normalization": metadata["norm_stats"],
            "limitations": "Checkpoint contents checked finite; checkpoint forward replay not performed. Independent raw normalization is a separate optional audit section."}


def audit_provenance(directory: Path) -> dict[str, Any]:
    """Verify archived implementation hashes against the saved source ZIP."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        for path, expected in manifest["implementation_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected, path
        protocol = "docs/practical_forecasting_protocol.md"
        assert hashlib.sha256(archive.read(protocol)).hexdigest() == manifest["protocol_sha256"]
    for source in manifest["data"]["raw_files"]:
        digest = hashlib.sha256()
        path = Path(source["path"])
        assert path.stat().st_size == source["bytes"]
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == source["sha256"]
    return {"passed": True, "implementation_files": len(manifest["implementation_sha256"]),
            "protocol_sha256": manifest["protocol_sha256"], "raw_data": manifest["data"]["raw_files"]}


def audit_normalization(directory: Path, folds: list[str]) -> dict[str, Any]:
    """Rebuild only requested training-period hourly statistics from raw CSV."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["data"]["raw_files"]
    assert len(sources) == 1 and manifest["data"]["dataset"] == "power"
    first_metadata = json.loads((directory / folds[0] / "data_manifest.json").read_text(encoding="utf-8"))
    columns = first_metadata["sensor_columns"]
    raw = pd.read_csv(sources[0]["path"], sep=";", na_values="?",
                      usecols=["Date", "Time", *columns])
    timestamps = pd.to_datetime(raw.pop("Date") + " " + raw.pop("Time"), format="%d/%m/%Y %H:%M:%S")
    raw.index = pd.DatetimeIndex(timestamps)
    assert raw.index.is_unique, "Independent normalization audit expects this cache's unique power timestamps"
    raw = raw[columns].replace([np.inf, -np.inf], np.nan)
    result = {}
    for fold in folds:
        metadata = json.loads((directory / fold / "data_manifest.json").read_text(encoding="utf-8"))
        lower = pd.Timestamp(metadata["boundaries"]["train_start"])
        upper = pd.Timestamp(metadata["boundaries"]["val_start"])
        # Construct availability labels directly instead of using production's
        # resampling/window helpers; no requested test observations participate.
        segment = raw.loc[(raw.index >= lower - pd.Timedelta(hours=1)) &
                          (raw.index < upper - pd.Timedelta(hours=1))]
        labels = segment.index.floor("h") + pd.Timedelta(hours=1)
        groups = segment.groupby(labels)
        means = groups.mean().where(groups.count() >= 45)
        checked = {}
        for column in columns:
            finite = means[column].dropna().to_numpy()
            mean, scale = float(finite.mean()), float(finite.std(ddof=0))
            scale = scale if scale > 1e-12 else 1.0
            np.testing.assert_allclose([mean, scale], metadata["norm_stats"][column], rtol=1e-12, atol=1e-12)
            assert len(finite) == metadata["normalization_finite_training_counts"][column]
            checked[column] = {"mean": mean, "scale": scale, "finite_training_hours": len(finite)}
        result[fold] = {"passed": True, "recomputed_training_statistics": checked}
    return result


def main() -> None:
    """Write a separate audit artifact without modifying experiment results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-raw-normalization", action="store_true")
    args = parser.parse_args()
    result = {"provenance": audit_provenance(args.run_dir),
              "folds": [audit_fold(args.run_dir / fold) for fold in args.fold]}
    if args.verify_raw_normalization:
        result["independent_raw_normalization"] = audit_normalization(args.run_dir, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "folds": args.fold, "output": str(args.output)}))


if __name__ == "__main__":
    main()
