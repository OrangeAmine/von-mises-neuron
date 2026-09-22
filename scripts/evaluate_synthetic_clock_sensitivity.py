"""Evaluate saved synthetic checkpoints with a fixed UTC+01:00 calendar.

This exploratory sensitivity uses no training or checkpoint selection. Original
predictions must reproduce before corrected-clock scores are calculated.
"""

import argparse
import hashlib
import json
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_standard_dataset import (
    check_regression, check_thresholds, compute_benchmark_metrics,
    generate_synthetic_co2, prepare_temporal_data, seed_everything,
)
from scripts.controlled_temporal_comparison import build_model, source_manifest, write_json
from src.temporal import extract_angular_time_features

FIXED_CLOCK = timezone(timedelta(hours=1))
UNCHANGED_MODELS = {"VonMises", "SinCos", "HandCrafted"}


def fingerprint(path: Path) -> str:
    """Hash the actual artifact bytes used for evaluation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash a tensor's contiguous CPU value bytes."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def clock_window_groups(
    index: pd.DatetimeIndex, lookback: int = 72, horizon: int = 12,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Classify windows by target-bin clock mismatch, without inspecting scores."""
    wall = extract_angular_time_features(index, [24, 168])
    fixed = extract_angular_time_features(index.tz_convert(FIXED_CLOCK), [24, 168])
    changed = np.abs(np.angle(np.exp(1j * (wall - fixed)))).max(axis=1) > 1e-10
    target = np.lib.stride_tricks.sliding_window_view(changed, horizon)[lookback:]
    inputs = np.lib.stride_tricks.sliding_window_view(changed, lookback)[:-horizon]
    groups = {
        "all": np.ones(len(target), dtype=bool),
        "pre_transition_targets": ~target.any(axis=1),
        "straddling_targets": target.any(axis=1) & ~target.all(axis=1),
        "post_transition_targets": target.all(axis=1),
    }
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    counts.update(changed_test_rows=int(changed.sum()), changed_target_entries=int(target.sum()),
                  target_entries=int(target.size), inputs_with_changed_clock=int(inputs.any(axis=1).sum()),
                  inputs_entirely_changed_clock=int(inputs.all(axis=1).sum()))
    return groups, counts


def prepare_clock_pair(
    frame: pd.DataFrame, device: torch.device, lookback: int = 72, horizon: int = 12,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Prepare original and corrected clocks while requiring unchanged fitting data.

    Args:
        frame: Original synthetic frame indexed in Europe/Paris.
        device: Selected inference device.
        lookback: Historical checkpoint input window.
        horizon: Historical checkpoint output window.

    Returns:
        Original and fixed-clock temporal datasets plus value fingerprints.

    Raises:
        ValueError: If clock correction changes training/validation inputs or
            sensor/target tensors; such data require a different protocol.
    """
    fixed_frame = frame.copy()
    fixed_frame.index = fixed_frame.index.tz_convert(FIXED_CLOCK)
    original = prepare_temporal_data(frame, lookback, horizon, device, time_periods=[24, 168])
    corrected = prepare_temporal_data(fixed_frame, lookback, horizon, device, time_periods=[24, 168])
    sensor_size = original["sensor_size"]
    hashes = {}
    for split in ("train", "val", "test"):
        old_x, new_x = original[f"X_{split}"], corrected[f"X_{split}"]
        if not torch.equal(original[f"y_{split}"], corrected[f"y_{split}"]):
            raise ValueError("Clock correction changed target tensors")
        if not torch.equal(old_x[..., :sensor_size], new_x[..., :sensor_size]):
            raise ValueError("Clock correction changed sensor tensors")
        if split != "test" and not torch.equal(old_x, new_x):
            raise ValueError("Clock correction changed fitting inputs; checkpoint-only protocol is invalid")
        hashes[split] = {"original_input": tensor_fingerprint(old_x),
                         "corrected_input": tensor_fingerprint(new_x),
                         "unchanged_target": tensor_fingerprint(original[f"y_{split}"])}
    return original, corrected, hashes


def require_reproduction(
    prediction: torch.Tensor, target: torch.Tensor, artifact: Path,
) -> float:
    """Require saved target equality and tight original-prediction agreement."""
    with np.load(artifact, allow_pickle=False) as saved:
        previous = saved["prediction"]
        saved_target = saved["target"]
    actual = prediction.detach().cpu().numpy()
    if not np.array_equal(target.detach().cpu().numpy(), saved_target):
        raise ValueError(f"Saved targets do not match: {artifact.name}")
    if previous.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(previous).all():
        raise ValueError(f"Invalid original predictions: {artifact.name}")
    if not np.allclose(actual, previous, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Original predictions failed reproduction: {artifact.name}")
    return float(np.max(np.abs(actual - previous)))


def grouped_metrics(
    prediction: torch.Tensor, data: dict[str, Any], groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Score prespecified window groups using the original target normalization."""
    results = {}
    for name, selected in groups.items():
        mask = torch.as_tensor(selected, device=prediction.device)
        results[name] = compute_benchmark_metrics(prediction[mask], data["y_test"][mask],
                                                 data["norm_stats"]) if selected.any() else None
    return results


def evaluate(source_dir: Path, output_dir: Path) -> None:
    """Reproduce original checkpoints, then save paired corrected-clock evidence."""
    source_manifest_path = source_dir / "manifest.json"
    source_results_path = source_dir / "results.json"
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("execution_status", source.get("status")) != "completed":
        raise ValueError("Source comparison must finish before frozen-checkpoint evaluation")
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = generate_synthetic_co2(n_days=90, seed=source["data_seed"])
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_hash != source["data_sha256"]:
        raise ValueError("Regenerated dataset fingerprint does not match source run")
    original, corrected, input_hashes = prepare_clock_pair(frame, device, source["lookback"], source["horizon"])
    test_index = frame.index[int(len(frame) * .9):]
    groups, group_counts = clock_window_groups(test_index, source["lookback"], source["horizon"])
    manifest = {
        "status": "running", "environment": source_manifest(),
        "label": "Exploratory frozen-checkpoint clock sensitivity; no training or checkpoint selection",
        "source_directory": str(source_dir.resolve()),
        "source_manifest_sha256": fingerprint(source_manifest_path),
        "source_results_sha256": fingerprint(source_results_path), "data_sha256": data_hash,
        "input_sha256": input_hashes, "device": str(device),
        "original_clock": "Europe/Paris", "corrected_clock": "Fixed UTC+01:00, aligned to winter training clock",
        "prediction_reproduction_tolerance": {"absolute": 1e-5, "relative": 1e-5, "units": "normalized target"},
        "groups": group_counts,
        "unchanged_references": {name: value for name, value in source_results.get("summary", {}).items()
                                 if name in UNCHANGED_MODELS},
    }
    write_json(output_dir / "manifest.json", manifest)
    rows, excluded = [], []
    lookup = {(row["model"], row["seed"]): row for row in source_results["runs"]}
    try:
        for source_row in source_results["runs"]:
            name, seed = source_row["model"], source_row["seed"]
            if name in UNCHANGED_MODELS:
                continue
            if source_row["execution_status"] != "completed":
                excluded.append({"model": name, "seed": seed, "source_status": source_row["execution_status"]})
                continue
            base = lookup.get(("VonMises", seed))
            if base is None or base["execution_status"] != "completed":
                raise ValueError(f"Missing completed unchanged base VM for seed {seed}")
            base_artifact = source_dir / f"VonMises_{seed}_predictions.npz"
            with np.load(base_artifact, allow_pickle=False) as saved_base:
                if not np.array_equal(saved_base["target"], original["y_test"].cpu().numpy()):
                    raise ValueError(f"Base VM targets do not match for seed {seed}")
                base_prediction = torch.as_tensor(saved_base["prediction"], device=device)
            base_metrics = compute_benchmark_metrics(base_prediction, original["y_test"], original["norm_stats"])
            if not all(np.isclose(base_metrics[key], value, atol=1e-5, rtol=1e-5)
                       for key, value in base["metrics"].items()):
                raise ValueError(f"Saved base VM metrics do not reproduce for seed {seed}")
            checkpoint = source_dir / f"{name}_{seed}_state.pt"
            artifact = source_dir / f"{name}_{seed}_predictions.npz"
            seed_everything(seed)
            model = build_model(name, original).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            model.eval()
            with torch.no_grad():
                old_prediction = model(original["X_test"])
            max_reproduction_error = require_reproduction(old_prediction, original["y_test"], artifact)
            # No corrected-clock score is available until original inference passes.
            with torch.no_grad():
                new_prediction = model(corrected["X_test"])
            if not torch.isfinite(new_prediction).all():
                raise FloatingPointError(f"Nonfinite corrected prediction for {name}, seed {seed}")
            if name == "NoTime" and not torch.equal(old_prediction, new_prediction):
                raise ValueError("NoTime predictions changed despite disabled time conditioning")
            old_scores = grouped_metrics(old_prediction, original, groups)
            new_scores = grouped_metrics(new_prediction, original, groups)
            if not all(np.isfinite(value) for scores in (old_scores, new_scores)
                       for group in scores.values() if group is not None for value in group.values()):
                raise FloatingPointError(f"Nonfinite clock sensitivity metrics for {name}, seed {seed}")
            differences = {group: {metric: new_scores[group][metric] - old_scores[group][metric]
                                   for metric in old_scores[group]} if old_scores[group] is not None else None
                           for group in groups}
            absolute, absolute_failures = check_thresholds(new_scores["all"])
            regression, regression_failures = check_regression(new_scores["all"], base_metrics)
            original_absolute, _ = check_thresholds(old_scores["all"])
            original_regression, _ = check_regression(old_scores["all"], base_metrics)
            row = {
                "model": name, "seed": seed, "checkpoint_sha256": fingerprint(checkpoint),
                "original_predictions_sha256": fingerprint(artifact),
                "base_predictions_sha256": fingerprint(base_artifact),
                "base_checkpoint_sha256": fingerprint(source_dir / f"VonMises_{seed}_state.pt"),
                "max_original_reproduction_error_normalized": max_reproduction_error,
                "max_clock_prediction_change_normalized": float((new_prediction-old_prediction).abs().max().item()),
                "original_metrics": old_scores, "corrected_metrics": new_scores,
                "corrected_minus_original": differences, "unchanged_base_metrics": base_metrics,
                "original_absolute_passed": original_absolute,
                "original_regression_vs_base_passed": original_regression,
                "corrected_absolute_passed": absolute, "corrected_absolute_failures": absolute_failures,
                "corrected_regression_vs_base_passed": regression,
                "corrected_regression_vs_base_failures": regression_failures,
            }
            rows.append(row)
            np.savez_compressed(output_dir / f"{name}_{seed}_predictions.npz",
                                original=old_prediction.cpu().numpy(), corrected=new_prediction.cpu().numpy(),
                                target=original["y_test"].cpu().numpy())
            write_json(output_dir / "results.json", {"runs": rows, "excluded": excluded})
            print(f"{name} seed={seed}: original RMSE={old_scores['all']['rmse']:.5f}, "
                  f"corrected RMSE={new_scores['all']['rmse']:.5f}", flush=True)
            del model
        summary = {}
        for name in dict.fromkeys(row["model"] for row in rows):
            selected = [row for row in rows if row["model"] == name]
            summary[name] = {"seeds": [row["seed"] for row in selected], "paired_changes": {}}
            for metric in ("rmse", "mae", "r2", "mape"):
                values = [row["corrected_minus_original"]["all"][metric] for row in selected]
                summary[name]["paired_changes"][metric] = {
                    "mean": float(np.mean(values)),
                    "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
        write_json(output_dir / "results.json", {"runs": rows, "excluded": excluded, "summary": summary})
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output_dir / "manifest.json", manifest)
        raise
    manifest["status"] = "completed"
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    """Run inference only into a fresh directory after the source fit finishes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    evaluate(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
