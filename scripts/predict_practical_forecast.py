"""Reproduce a selected final power forecaster on an observed hourly CSV.

CSV timestamps are timezone-naive right-edge availability times for hourly means.
Values must already satisfy the training data's 45-readings-per-sensor/hour
coverage policy: this CSV interface cannot independently verify minute coverage.
This is bounded offline inference, not a production-readiness certification.
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.controlled_temporal_comparison import write_json
from scripts.practical_forecasting import (
    FINAL_START, IMPLEMENTATION_PATHS, NEURAL, SIMPLE, TARGET, make_neural,
)
from src.forecast_training import batched_predict
from src.practical_evaluation import _ridge_features, simple_predictions
from src.temporal import extract_angular_time_features


def sha256(path: Path) -> str:
    """Hash exact bytes of a source or inference artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_run(run_dir: Path) -> dict[str, Any]:
    """Read a completed frozen final run and verify reconstruction source hashes.

    Args:
        run_dir: Directory containing the completed final manifest and fold.

    Returns:
        Selected family, metadata, selected training rows, and artifact hashes.

    Raises:
        ValueError: The run is incomplete, inconsistent, or its source changed.
    """
    artifacts: dict[str, str] = {}
    def read(relative: str) -> dict[str, Any]:
        path = run_dir / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        artifacts[relative] = sha256(path)
        return value
    manifest = read("manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("phase") != "final"
            or manifest.get("all_fits_finite") is not True):
        raise ValueError("Inference requires a completed final run with finite fits")
    if (manifest.get("protocol") != "practical_forecasting_v1"
            or manifest.get("lookback") != 168 or manifest.get("horizon") != 24
            or manifest.get("target") != TARGET or manifest.get("unit") != "kW"
            or manifest.get("coverage") != 0.75
            or manifest.get("test_starts") != [FINAL_START]):
        raise ValueError("Final manifest does not match the frozen power protocol")
    selected = manifest.get("selection", {}).get("selected_family")
    if selected not in [*SIMPLE, "RidgeResidual", *NEURAL] or manifest.get("families") != [selected]:
        raise ValueError("Final manifest must identify exactly its selected family")
    hashes = manifest.get("implementation_sha256", {})
    for relative in IMPLEMENTATION_PATHS:
        if hashes.get(relative) != sha256(ROOT / relative):
            raise ValueError(f"Reconstruction implementation changed: {relative}")
    assessment = read("final_assessment.json")
    if assessment.get("selected_family") != selected or assessment.get("selected_metrics") is None:
        raise ValueError("Final assessment does not contain the selected forecast")
    meta = read(f"{FINAL_START}/data_manifest.json")
    if (meta.get("lookback") != 168 or meta.get("horizon") != 24
            or meta.get("time_periods") != [24, 168] or meta.get("target_col") != TARGET):
        raise ValueError("Stored data dimensions/target do not match inference")
    columns = meta.get("sensor_columns", [])
    if not columns or len(set(columns)) != len(columns) or TARGET not in columns:
        raise ValueError("Stored sensor columns are invalid")
    if meta.get("target_idx") != columns.index(TARGET):
        raise ValueError("Stored target index does not match sensor order")
    report = read(f"{FINAL_START}/results.json")
    if (report.get("status") != "completed" or report.get("all_fits_finite") is not True
            or selected not in report.get("families", {})):
        raise ValueError("Selected final fold did not complete successfully")
    rows = [row for row in report.get("runs", []) if row.get("family") == selected]
    if selected in NEURAL:
        if (manifest.get("seeds") != [42, 43, 44] or len(rows) != 3
                or sorted(row.get("seed", -1) for row in rows) != [42, 43, 44]
                or any(row.get("status") != "completed" for row in rows)):
            raise ValueError("Selected neural family requires all three completed seeds")
    elif selected == "RidgeResidual":
        if len(rows) != 1 or rows[0].get("status") != "completed":
            raise ValueError("Selected ridge fit did not complete")
    return {"manifest": manifest, "metadata": meta, "family": selected, "rows": rows,
            "assessment": assessment, "artifact_sha256": artifacts}


def prepare_history(csv_path: Path, metadata: dict[str, Any],
                    timestamp_column: str = "timestamp") -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Validate observed hourly inputs and apply only stored training statistics.

    Args:
        csv_path: At least 168 consecutive hourly sensor rows, in physical units.
        metadata: Final fold sensor order, target index, and train normalization.
        timestamp_column: CSV field containing literal availability timestamps.

    Returns:
        Float32 input ``(1, 168, sensors+2)`` and the selected history timestamps.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique")
    frame = pd.read_csv(csv_path)
    columns = metadata["sensor_columns"]
    if timestamp_column not in frame or any(column not in frame for column in columns):
        raise ValueError("CSV requires its timestamp column and every stored sensor column")
    index = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_column], errors="raise"))
    hour = pd.Timedelta(hours=1).value
    if (len(index) < 168 or index.hasnans or index.tz is not None
            or not index.is_unique or not index.is_monotonic_increasing
            or np.any(index.asi8 % hour) or np.any(np.diff(index.asi8) != hour)):
        raise ValueError("History requires at least 168 ordered unique consecutive naive whole hours")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Every supplied sensor value must be finite")
    stats = np.asarray([metadata["norm_stats"][column] for column in columns], dtype=np.float64)
    if stats.shape != (len(columns), 2) or not np.isfinite(stats).all() or np.any(stats[:, 1] <= 0):
        raise ValueError("Stored normalization must contain finite means and positive scales")
    normalized = (values[-168:] - stats[:, 0]) / stats[:, 1]
    history_index = index[-168:]
    angles = extract_angular_time_features(history_index, [24, 168])
    features = np.concatenate((normalized, angles), axis=1).astype(np.float32)[None]
    if not np.isfinite(features).all():
        raise ValueError("Normalized inputs exceed the supported finite float32 range")
    return features, history_index


def predict_selected(run_dir: Path, bundle: dict[str, Any], x: np.ndarray,
                     device: str = "auto") -> np.ndarray:
    """Reconstruct selected artifacts and return 24 original-unit power values.

    Args:
        run_dir: Completed final run directory.
        bundle: Validated final-run metadata from ``load_final_run``.
        x: Prepared float32 history, with normalized sensors then raw angles.
        device: ``auto``, ``cpu`` or ``cuda`` for neural inference.

    Returns:
        A finite length-24 forecast in kW, without clipping or model reselection.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    family, meta = bundle["family"], bundle["metadata"]
    bundle["inference_device"] = "cpu"
    index = meta["target_idx"]
    meanscale = meta["norm_stats"][TARGET]
    def artifact(relative: str) -> Path:
        path = run_dir / relative
        bundle["artifact_sha256"][relative] = sha256(path)
        return path
    if family in SIMPLE:
        normalized = simple_predictions(x, index, 24)[family]
    elif family == "RidgeResidual":
        training = bundle["rows"][0]["training"]
        normalized = np.repeat(x[:, -1, index, None].astype(np.float64), 24, axis=1)
        fallback = training.get("persistence_selected")
        if fallback is True and training.get("selected_candidate") == "Persistence":
            pass
        elif fallback is False and training.get("selected_candidate") == "RidgeResidual":
            path = artifact(f"{FINAL_START}/RidgeResidual_coefficients.npz")
            with np.load(path, allow_pickle=False) as saved:
                coefficients, intercept = saved["coefficients"], saved["intercept"]
            features = _ridge_features(x, index, 2)
            if (coefficients.shape != (24, features.shape[1]) or intercept.shape != (24,)
                    or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()
                    or training.get("n_features") != features.shape[1]):
                raise ValueError("Ridge coefficient dimensions or values are invalid")
            normalized += features @ coefficients.T + intercept
        else:
            raise ValueError("Ridge persistence-fallback metadata is inconsistent")
    else:
        selected_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        bundle["inference_device"] = str(selected_device)
        predictions = []
        for seed in (42, 43, 44):
            model = make_neural(family, meta).to(selected_device)
            path = artifact(f"{FINAL_START}/{family}_{seed}_state.pt")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Selected neural checkpoint contains nonfinite values")
            model.load_state_dict(state, strict=True)
            predictions.append(batched_predict(model, torch.from_numpy(x)).numpy().astype(np.float64))
        normalized = np.mean(predictions, axis=0)
    physical = normalized.astype(np.float64) * meanscale[1] + meanscale[0]
    if physical.shape != (1, 24) or not np.isfinite(physical).all():
        raise ValueError("Selected model did not produce a finite 24-hour forecast")
    return physical[0]


def run_prediction(run_dir: Path, input_csv: Path, output_dir: Path,
                   timestamp_column: str = "timestamp", device: str = "auto") -> Path:
    """Write a new forecast CSV and full inference provenance without overwriting.

    Args:
        run_dir: Completed final training run.
        input_csv: Observed hourly sensor history.
        output_dir: New directory for forecast and provenance artifacts.
        timestamp_column: Availability-time column in the input CSV.
        device: Neural inference device selection.

    Returns:
        Path to the new forecast CSV.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    bundle = load_final_run(run_dir)
    features, timestamps = prepare_history(input_csv, bundle["metadata"], timestamp_column)
    prediction = predict_selected(run_dir, bundle, features, device)
    future = pd.date_range(timestamps[-1] + pd.Timedelta(hours=1), periods=24, freq="1h")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "forecast.csv"
    pd.DataFrame({"timestamp": future, "Global_active_power_kW": prediction}).to_csv(path, index=False)
    write_json(output_dir / "provenance.json", {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_family": bundle["family"], "origin": str(timestamps[-1]),
        "history_start": str(timestamps[0]), "lookback": 168, "horizon": 24,
        "input_csv": str(input_csv.resolve()), "input_sha256": sha256(input_csv),
        "final_run": str(run_dir.resolve()), "artifact_sha256": bundle["artifact_sha256"],
        "inference_source_sha256": sha256(Path(__file__)), "forecast_sha256": sha256(path),
        "sensor_columns": bundle["metadata"]["sensor_columns"],
        "normalization": "stored final-training statistics; no fitting",
        "clock": "literal naive wall clock; timestamps are hourly-bin right-edge availability",
        "coverage": "Input hourly means must satisfy >=45 minute readings per sensor/hour; minute coverage is not independently checked by this CSV interface",
        "device_requested": device, "device_used": bundle["inference_device"],
        "torch": torch.__version__, "numpy": np.__version__,
        "clipping": "none", "reselection": "none",
        "learned_5pct_improvement_confirmed": bundle["assessment"].get("learned_5pct_improvement_confirmed"),
        "scope": "offline inference reproduction; not production-readiness certification",
    })
    return path


def main() -> None:
    """Parse the bounded offline-inference command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(run_prediction(args.final_run, args.input_csv, args.output_dir,
                         args.timestamp_column, args.device))


if __name__ == "__main__":
    main()
