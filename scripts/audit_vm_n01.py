"""Independent CPU observer of saved VM-N01 arrays and all fit checkpoints.

This observer imports no study model, trainer, task generator or analysis code.
It never generates a function, fits parameters or modifies source artifacts.
The observer's own hash is recorded separately from the scientific source lock.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> Any:
    """Read a saved JSON artifact without changing it."""
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    """Hash exact artifact bytes incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    """Independently match the protocol's canonical JSON identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Checks:
    """Track checked assertions and worst numeric discrepancy in the receipt."""

    def __init__(self) -> None:
        self.count = 0
        self.maximum_absolute_difference = 0.

    def require(self, condition: bool, label: str) -> None:
        """Record one check and fail with a concrete artifact location."""
        self.count += 1
        if not condition:
            raise ValueError(label)

    def close(self, actual: Any, expected: Any, label: str) -> None:
        """Compare finite numeric values allowing CPU/CUDA rounding only."""
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label+": shape/nonfinite")
        difference = float(np.max(np.abs(a-b))) if a.size else 0.
        self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
        self.require(bool(np.allclose(a, b, atol=2e-10, rtol=2e-8)), f"{label}: discrepancy {difference}")


def numpy_prediction(state: Mapping[str, Any], spec: Mapping[str, Any], theta: np.ndarray,
                     checks: Checks, label: str) -> np.ndarray:
    """Reconstruct the explicit feature equation from CPU checkpoint tensors."""
    checks.require(state["specification"] == spec, label+": checkpoint model specification")
    checks.require(state["basis_count"] == spec["features"] and state["fitted_parameter_count"] == spec["parameters"], label+": capacity")
    leaves = {}
    for name, value in state["state_dict"].items():
        checks.require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                       and value.dtype == torch.float64 and bool(torch.isfinite(value).all()), label+": tensor "+name)
        leaves[name] = value.numpy()
    x = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    if spec.get("unrestricted", False):
        centers = leaves["basis.centers"].reshape(1, -1)
        width = float(leaves["basis.width"])
        checks.close(centers.ravel(), np.linspace(-math.pi, math.pi, spec["line_rbf_count"]), label+": line centers")
        checks.close(width, spec["line_rbf_width"], label+": line width")
        features = np.column_stack((x/math.pi, np.exp(-.5*((x-centers)/width)**2)))
    else:
        frequencies = leaves["basis.frequencies"]
        checks.close(frequencies, np.arange(1, spec["harmonics"]+1), label+": integer harmonics")
        pieces = []
        if spec["harmonics"]:
            pieces.extend((np.cos(x*frequencies), np.sin(x*frequencies)))
        if spec["kernels"]:
            centers = leaves["basis.bank.centers"].reshape(1, -1)
            widths = (np.logaddexp(0., leaves["basis.bank.raw_kappa"]) if spec["kind"] != "ridge"
                      else leaves["basis.bank.fixed_kappa"])
            checks.require(centers.shape == (1, spec["kernels"]) and widths.shape == (spec["kernels"],)
                           and np.all(widths > 0), label+": kernel geometry")
            # Independent direct cosine equation, versus stable half-angle
            # evaluation in the fitting implementation.
            pieces.append(np.exp(widths*(np.cos(x-centers)-1)))
        features = np.concatenate(pieces, axis=1)
    weight, bias = leaves["readout_weight"], leaves["readout_bias"]
    checks.require(features.shape == (len(x), spec["features"]) and weight.shape == (spec["features"], 1)
                   and bias.shape == (1,), label+": feature/readout shape")
    prediction = features @ weight + bias
    checks.require(bool(np.isfinite(prediction).all()), label+": finite prediction")
    return prediction


def expected_matrix(config: Mapping[str, Any], phase: str, rates: Mapping[str, float]) -> list[dict[str, Any]]:
    """Independently enumerate all prospective fit identities."""
    matrix = []
    task = config["tasks"]
    budgets = task["pilot_label_budgets"] if phase == "pilot" else task["label_budgets"]
    for family in task["families"]:
        for replicate in range(task["replicates"][phase]):
            for labels in budgets:
                for initialization in range(task["initializations"][phase]):
                    for spec in config["models"]:
                        choices = [None] if spec["kind"] == "ridge" else (config["training"]["pilot_learning_rates"] if phase == "pilot" else [rates[spec["id"]]])
                        for rate in choices:
                            matrix.append({"fit_id": f"fit_{len(matrix):05d}", "phase": phase,
                                           "family": family, "replicate": replicate, "labels": labels,
                                           "initialization": initialization, "model_id": spec["id"], "learning_rate": rate})
    return matrix


def aggregate(rows: list[dict[str, Any]], config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Average replayed metrics within function before equal family averaging."""
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["family"], row["replicate"], row["allowance"], row["learning_rate"])].append(row["normalized_mse"])
    functions = [{"model_id": key[0], "family": key[1], "replicate": key[2], "allowance": key[3],
                  "learning_rate": key[4], "normalized_mse": float(np.mean(values)), "fit_count": len(values)}
                 for key, values in grouped.items()]
    models = {}
    for spec in config["models"]:
        rates = config["training"]["pilot_learning_rates"] if phase == "pilot" and spec["kind"] != "ridge" else [None]
        for rate in rates:
            matching = [row for row in functions if row["model_id"] == spec["id"]
                        and row["allowance"] == config["training"]["primary_time_seconds"]
                        and (phase != "pilot" or row["learning_rate"] == rate)]
            family_errors = {family: float(np.mean([row["normalized_mse"] for row in matching if row["family"] == family]))
                             for family in config["tasks"]["families"]}
            key = spec["id"] + (f"@{rate:g}" if rate is not None else "")
            models[key] = {"family_errors": family_errors,
                           "balanced_error": float(np.mean([family_errors[family] for family in config["tasks"]["periodic_families"]]))}
    return {"functions": functions, "models": models}


def contrast(functions: list[dict[str, Any]], config: Mapping[str, Any], phase: str,
             hybrid: str, comparator: str) -> dict[str, Any]:
    """Independently resample complete paired functions, stratified by family."""
    families = config["tasks"]["periodic_families"]
    n = config["tasks"]["replicates"][phase]
    points = config["analysis"]["bootstrap_resamples"]
    lookup = {(row["model_id"], row["family"], row["replicate"]): row["normalized_mse"]
              for row in functions if row["allowance"] == config["training"]["primary_time_seconds"]}
    rng = np.random.default_rng(np.random.SeedSequence([config["seeds"]["analysis"], 101]))
    boots = np.zeros((2, points))
    means = np.zeros((2, len(families)))
    for f, family in enumerate(families):
        sampled = rng.integers(0, n, size=(points, n))
        for m, model in enumerate((hybrid, comparator)):
            values = np.array([lookup[(model, family, replicate)] for replicate in range(n)])
            means[m, f] = np.mean(values)
            boots[m] += values[sampled].mean(axis=1)/len(families)
    improvement = float(1-means[0].mean()/means[1].mean())
    tail = (1-config["analysis"]["confidence"])/2
    interval = np.quantile(1-boots[0]/boots[1], [tail, 1-tail]).tolist()
    smooth_index = families.index("smooth")
    smooth_regression = float(means[0, smooth_index]/means[1, smooth_index]-1)
    improving_local = sum(means[0, families.index(family)] < means[1, families.index(family)] for family in ("narrow", "multimodal", "mixed"))
    gate = config["analysis"]["gate"]
    passed = (improvement >= gate["each_comparison_min_relative_improvement"]
              and interval[0] > gate["each_comparison_lower_interval_above"]
              and smooth_regression <= gate["each_comparison_max_smooth_regression"]
              and improving_local >= gate["each_comparison_min_improving_local_families"])
    return {"relative_improvement": improvement, "interval": interval, "smooth_regression": smooth_regression,
            "improving_local_families": int(improving_local), "passed": bool(passed)}


def audit_phase(directory: Path, config: Mapping[str, Any], checks: Checks,
                progress: dict[str, Any]) -> dict[str, Any]:
    """Audit complete stored data and checkpoints, returning replayed summaries."""
    summary, plan, receipt = [read(directory/name) for name in ("summary.json", "plan.json", "receipt.json")]
    phase = summary["phase"]
    checks.require(phase in {"pilot", "development", "confirmation"} and plan["phase"] == phase, "phase identity")
    checks.require(summary["complete"] and receipt["complete"] and receipt["within_ceiling"], "phase not complete")
    checks.require(receipt["summary_sha256"] == file_hash(directory/"summary.json"), "summary hash")
    checks.require(0 <= receipt["elapsed_seconds"] <= config["phases"]["ceiling_seconds"][phase], "phase ceiling")
    checks.require(summary["config_sha256"] == plan["config_sha256"] == object_hash(config), "configuration identity")
    matrix = expected_matrix(config, phase, summary["selected_rates"])
    checks.require(plan["matrix"] == matrix and len(matrix) == config["phases"]["planned_fits"][phase]
                   and summary["attempted_count"] == len(matrix), "exact prospective matrix")
    expected_names = {f"{row['fit_id']}.{suffix}" for row in matrix for suffix in ("json", "pt")}
    checks.require({p.name for p in directory.glob("fit_*")} == expected_names, "missing/extra fit artifacts")
    for relative, expected in summary["sources"].items():
        path = (ROOT/relative).resolve()
        checks.require(path.is_relative_to(ROOT) and file_hash(path) == expected, "source identity: "+relative)
    specs = {spec["id"]: spec for spec in config["models"]}
    task_cache, rotations = {}, {}
    rebuilt, compact_rows = [], []
    for item in matrix:
        label = item["fit_id"]
        task_key = (item["family"], item["replicate"])
        if task_key not in task_cache:
            prefix = f"task_{task_key[0]}_{task_key[1]}"
            metadata = read(directory/f"{prefix}.json")
            with np.load(directory/f"{prefix}.npz", allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            checks.require(metadata["phase"] == phase and metadata["family"] == task_key[0]
                           and metadata["replicate"] == task_key[1] and metadata["config_sha256"] == object_hash(config), prefix+": metadata identity")
            checks.require(set(arrays) == set(metadata["array_sha256"]), prefix+": array registry")
            for name, array in arrays.items():
                checks.require(array.dtype == np.float64 and array.ndim == 2 and array.shape[1] == 1
                               and bool(np.isfinite(array).all()), prefix+": array shape/dtype "+name)
                checks.require(hashlib.sha256(array.tobytes()).hexdigest() == metadata["array_sha256"][name], prefix+": array hash "+name)
            for split in ("train", "validation"):
                checks.close(arrays[f"{split}_y"], arrays[f"{split}_clean"] + config["tasks"]["noise_std"]*arrays[f"{split}_unit_noise"], prefix+": noisy target "+split)
            checks.require(not np.intersect1d(arrays["train_x"], arrays["validation_x"]).size, prefix+": split overlap")
            task_cache[task_key] = metadata, arrays
            progress["tasks"] += 1
        metadata, arrays = task_cache[task_key]
        compact = read(directory/f"{label}.json")
        checks.require(all(compact.get(key) == value for key, value in item.items()), label+": compact identity")
        artifact = directory/f"{label}.pt"
        checks.require(compact["artifact_sha256"] == file_hash(artifact), label+": artifact hash")
        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
        checks.require(bundle["status"] == compact["status"] == "completed", label+": fit status")
        checks.require(all(bundle["identity"].get(key) == value for key, value in item.items())
                       and bundle["identity"]["task_sha256"] == object_hash(metadata), label+": bundle task identity")
        rotation_key = (*task_key, item["initialization"])
        if rotation_key in rotations:
            checks.require(rotations[rotation_key] == bundle["identity"]["rotation"], label+": shared initialization")
        rotations[rotation_key] = bundle["identity"]["rotation"]
        checks.require(bundle["rotation"] == bundle["identity"]["rotation"], label+": rotation identity")
        spec = specs[item["model_id"]]
        n, v = 3*item["labels"]//4, item["labels"]//4
        history = bundle["checkpoints"]
        checks.require(len(history) == compact["checkpoint_count"] and bool(history) and history[0]["step"] == 0, label+": checkpoint registry")
        validation_scores = []
        previous_time, previous_step = -1., -1
        for checkpoint in history:
            location = f"{label}/step{checkpoint['step']}"
            checks.require(math.isfinite(checkpoint["elapsed"]) and previous_time <= checkpoint["elapsed"] <= bundle["fit_seconds"]
                           and checkpoint["step"] > previous_step, location+": checkpoint chronology")
            previous_time, previous_step = checkpoint["elapsed"], checkpoint["step"]
            for split, count, field in (("train", n, "training_mse"), ("validation", v, "validation_mse")):
                prediction = numpy_prediction(checkpoint["state"], spec, arrays[f"{split}_x"][:count], checks, location)
                error = float(np.mean((prediction-arrays[f"{split}_y"][:count])**2))
                checks.close(error, checkpoint[field], location+": "+field)
                if split == "validation":
                    validation_scores.append(error)
            progress["checkpoints"] += 1
        checks.require(bundle["updates"] == history[-1]["step"] and bundle["updates"] <= config["training"]["max_updates"], label+": updates")
        checks.require(0 <= bundle["fit_seconds"] <= bundle["total_seconds"], label+": fitting time")
        checks.close(bundle["fit_seconds"], compact["fit_seconds"], label+": compact time")
        allowances = config["training"]["time_budgets_seconds"]
        checks.require(len(compact["rows"]) == len(allowances), label+": allowance count")
        for allowance, saved_row in zip(allowances, compact["rows"]):
            checks.require(all(saved_row.get(key) == value for key, value in item.items())
                           and saved_row["allowance"] == allowance and saved_row["status"] == "completed", label+": row identity")
            eligible = [i for i, row in enumerate(history) if row["elapsed"] <= allowance]
            checks.require(bool(eligible), label+": missing eligible checkpoint")
            # The exact selection must use saved scalar scores: NumPy/CUDA
            # roundoff can reorder an exact numerical tie after independent replay.
            selected = min(eligible, key=lambda i: (history[i]["validation_mse"], i))
            key = format(float(allowance), "g")
            checks.require(bundle["selected"][key] == selected, label+": selected earliest admissible minimum")
            checkpoint = history[selected]
            checks.require(saved_row["selected_step"] == checkpoint["step"] and saved_row["selected_seconds"] == checkpoint["elapsed"], label+": selected metadata")
            checks.close(saved_row["noisy_validation_mse"], validation_scores[selected], label+": selected validation score")
            prediction = numpy_prediction(checkpoint["state"], spec, arrays["grid_x"], checks, label+": clean grid")
            clean = arrays["grid_clean"]
            mse, variance = float(np.mean((prediction-clean)**2)), float(np.var(clean))
            checks.require(variance > 0, label+": nondegenerate integration truth")
            normalized = mse/variance
            for metric, value in (("clean_mse", mse), ("clean_variance", variance), ("normalized_mse", normalized)):
                checks.close(bundle["metrics"][key][metric], value, label+": saved "+metric)
            checks.close(saved_row["normalized_mse"], normalized, label+": compact clean score")
            rebuilt.append({**item, "allowance": allowance, "normalized_mse": normalized,
                            "noisy_validation_mse": validation_scores[selected]})
            compact_rows.append(saved_row)
        progress["fits"] += 1
        if progress["fits"] % 100 == 0:
            print(f"audit: {progress['fits']}/{len(matrix)} fits, {progress['checkpoints']} checkpoints", flush=True)
    checks.require(compact_rows == summary["rows"], "summary rows differ from immutable compact records")
    checks.require(len(task_cache) == len(config["tasks"]["families"])*config["tasks"]["replicates"][phase], "task count")
    result = aggregate(rebuilt, config, phase)
    result.update(phase=phase, summary_sha256=file_hash(directory/"summary.json"), phase_seconds=receipt["elapsed_seconds"])
    if phase == "pilot":
        rates = {}
        for spec in config["models"]:
            if spec["kind"] == "ridge":
                continue
            scores = []
            for rate in sorted(config["training"]["pilot_learning_rates"]):
                values = [np.mean([row["noisy_validation_mse"] for row in rebuilt
                                   if row["model_id"] == spec["id"] and row["family"] == family
                                   and row["learning_rate"] == rate and row["allowance"] == config["training"]["primary_time_seconds"]])
                          for family in config["tasks"]["families"]]
                scores.append((float(np.mean(values)), rate))
            rates[spec["id"]] = min(scores)[1]
        checks.require(rates == summary["selected_rates"], "independent pilot rate selection")
        result["selected_rates"] = rates
    if phase == "development":
        selected = {}
        for target, registry in (("selected_hybrid", "hybrid_candidates"), ("capacity_comparator", "capacity_candidates")):
            candidates = config["analysis"][registry]
            selected[target] = min(candidates, key=lambda model: result["models"][model]["balanced_error"])
            checks.require(selected[target] == summary["selected_models"][target], "independent development "+target)
        result["selected_models"] = selected
    return result


def compare_report(result: dict[str, Any], report_path: Path, config: Mapping[str, Any], checks: Checks) -> None:
    """Check published family effects, bootstrap intervals and gate independently."""
    report = read(report_path)
    phase = result["phase"]
    published = report["analyses"][phase]
    for model, values in result["models"].items():
        checks.close(values["balanced_error"], published["models"][model]["balanced_error"], model+": published balanced mean")
        for family, value in values["family_errors"].items():
            checks.close(value, published["models"][model]["family_errors"][family], model+": published family "+family)
    selected = report["selected_models"]
    comparators = [*config["analysis"]["equal_basis_comparators"], selected["capacity_comparator"]]
    contrasts = {}
    for comparator in comparators:
        replay = contrast(result["functions"], config, phase, selected["selected_hybrid"], comparator)
        saved = published["contrasts"][comparator]
        for key in ("relative_improvement", "interval", "smooth_regression", "improving_local_families"):
            checks.close(replay[key], saved[key], comparator+": published "+key)
        checks.require(replay["passed"] == saved["passed"], comparator+": published gate")
        contrasts[comparator] = replay
    gate = all(row["passed"] for row in contrasts.values())
    checks.require(gate == published["gate"]["passed"], "published overall scientific gate")
    result["contrasts"] = contrasts
    result["gate_passed"] = gate
    result["published_report_sha256"] = file_hash(report_path)


def main() -> None:
    """Write a new audit receipt on success or failure, never overwriting one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional published vm_n01_run_record.json for development/confirmation")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/vm_n01_v1.json")
    args = parser.parse_args()
    # Reject an existing report before doing work; failed receipts also remain.
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    checks = Checks()
    progress = {"fits": 0, "checkpoints": 0, "tasks": 0}
    receipt: dict[str, Any] = {"observer_sha256": file_hash(Path(__file__)), "phase_directory": str(args.phase_dir.resolve()),
                               "created_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "threads": 1,
                               "status": "failed", "findings": [], "progress": progress}
    try:
        config = read(args.config)
        receipt["config_sha256"] = object_hash(config)
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            result = audit_phase(args.phase_dir, config, checks, progress)
            if args.report:
                compare_report(result, args.report, config, checks)
        receipt["results"] = result
        receipt["status"] = "passed"
    except Exception as error:
        receipt["findings"].append(f"{type(error).__name__}: {error}")
    receipt.update(checks=checks.count, maximum_absolute_difference=checks.maximum_absolute_difference,
                   elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "results"}, indent=2))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
