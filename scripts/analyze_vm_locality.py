"""Independently audit locality-v2 artifacts and apply task-level decisions.

This observer never runs an optimizer or generates a target. NumPy reconstructs
every saved feature/readout state, and an augmented least-squares solve checks
the positive-ridge readouts independently of the trainer's centered solver.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.locality_protocol import (
    CONFIG_PATH, SOURCE_PATHS, checked_path, file_hash, object_hash, planned_runs,
    read_json, source_manifest, verify_confirmation_lock, verify_dependencies,
    verify_run_manifest, write_json,
)


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _close(actual: Any, expected: Any, label: str, *, atol: float = 2e-9,
           rtol: float = 1e-8) -> float:
    a, b = _array(actual), _array(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label}: shape mismatch or nonfinite values")
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: maximum discrepancy {difference}")
    return difference


def numpy_features(state: Mapping[str, Any], x: Any) -> np.ndarray:
    """Evaluate explicit float64 kernels or integer Fourier pairs independently."""
    theta = np.asarray(_array(x), dtype=np.float64).reshape(-1, 1)
    count = int(state["basis_count"])
    if state["basis_type"] == "fourier":
        if count <= 0 or count % 2:
            raise ValueError("Invalid Fourier feature count")
        phase = theta * np.arange(1, count//2+1, dtype=np.float64)
        result = np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
    elif state["basis_type"] == "circular":
        centers = np.asarray(_array(state["centers"]), dtype=np.float64).reshape(-1)
        kappas = np.asarray(_array(state["kappas"]), dtype=np.float64).reshape(-1)
        if centers.shape != (count,) or kappas.shape != (count,) or not np.all(kappas > 0):
            raise ValueError("Invalid circular geometry")
        result = np.exp(-2*kappas*np.sin((theta-centers)/2)**2)
    else:
        raise ValueError("Unknown saved basis type")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed features")
    return result


def numpy_prediction(state: Mapping[str, Any], x: Any) -> np.ndarray:
    """Reconstruct predictions without importing the model or trainer modules."""
    features = numpy_features(state, x)
    weight = np.asarray(_array(state["readout_weight"]), dtype=np.float64)
    bias = np.asarray(_array(state["readout_bias"]), dtype=np.float64)
    if weight.shape != (features.shape[1], 1) or bias.shape != (1,):
        raise ValueError("Invalid saved readout shape")
    return features@weight+bias


def independent_ridge(features: np.ndarray, target: np.ndarray,
                        penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve an augmented design with an explicitly unpenalized intercept."""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Locality v2 requires strictly positive fixed regularization")
    n, count = features.shape
    design = np.column_stack((features, np.ones(n)))
    ridge_rows = np.column_stack((math.sqrt(n*penalty)*np.eye(count), np.zeros(count)))
    coefficients = np.linalg.lstsq(np.vstack((design, ridge_rows)),
                                  np.vstack((target, np.zeros((count, 1)))), rcond=None)[0]
    return coefficients[:-1], coefficients[-1]


def replay_selections(history: Sequence[Mapping[str, Any]], caps: Sequence[float]) -> dict[str, Any]:
    """Select only fully completed checkpoints; an exact tie keeps the earliest."""
    if not history or history[0]["step"] != 0:
        raise ValueError("Missing initial checkpoint")
    previous_step, previous_time = -1, -1.
    for row in history:
        step, elapsed, score = row["step"], row["elapsed_seconds"], row["validation_mse"]
        if step <= previous_step or not np.isfinite(elapsed) or elapsed < previous_time or elapsed < 0 or not np.isfinite(score):
            raise ValueError("Nonfinite or nonmonotonic checkpoint history")
        started = row["checkpoint_start_elapsed_seconds"]
        if not np.isfinite(started) or not previous_time <= started <= elapsed:
            raise ValueError("Checkpoint start/end chronology changed")
        if row["admissible_time_budgets"] != [cap for cap in caps if elapsed <= cap]:
            raise ValueError("Checkpoint time eligibility differs from its completed timestamp")
        previous_step, previous_time = step, elapsed
    result = {}
    for cap in caps:
        eligible = [row for row in history if row["elapsed_seconds"] <= cap]
        result[str(float(cap))] = None if not eligible else min(eligible, key=lambda row:(row["validation_mse"],row["step"]))
    return result


def _state_schema(state: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    """Check explicit effective geometry against the serialized model leaves."""
    is_fourier = spec["id"].startswith("fourier")
    expected_type = "fourier" if is_fourier else "circular"
    count = spec["features"]
    geometry_count = 0 if spec["kind"] == "ridge" else count*(2 if spec["geometry"]=="both" else 1)
    optimized = geometry_count+(count+1 if spec["kind"]=="joint" else 0)
    if (state["kind"]!=spec["kind"] or state["basis_type"]!=expected_type
            or state["optimized_parameter_count"]!=optimized):
        raise ValueError("Checkpoint architecture or optimization count changed")
    leaves = {"readout_weight":state["readout_weight"],"readout_bias":state["readout_bias"],**state["basis_state"]}
    if any(_array(value).dtype!=np.float64 or not np.isfinite(_array(value)).all() for value in leaves.values()):
        raise ValueError("Checkpoint contains nonfinite or non-float64 model leaves")
    if is_fourier:
        if state["basis_state"]:
            raise ValueError("Fourier model has unexpected geometry state")
        return
    width_key = "bank.raw_kappa" if spec.get("geometry") in ("widths","both") else "bank.fixed_kappa"
    if set(state["basis_state"])!={"bank.centers",width_key}:
        raise ValueError("Saved basis leaves disagree with the geometry ablation")
    centers = _array(state["basis_state"]["bank.centers"])
    raw = _array(state["basis_state"][width_key])
    if centers.shape!=(count,1) or raw.shape!=(count,):
        raise ValueError("Invalid serialized geometry shape")
    widths = np.where(raw>20,raw,np.logaddexp(0,raw)) if width_key.endswith("raw_kappa") else raw
    _close(state["centers"],centers[:,0],"Explicit/serialized center parity",atol=0,rtol=0)
    _close(state["kappas"],widths,"Explicit/serialized width parity",atol=2e-14,rtol=2e-14)


def _state_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Require exact equality of selected/terminal state copies and their source."""
    if set(a)!=set(b):
        raise ValueError("Saved state schema changed")
    for key in a:
        if isinstance(a[key],Mapping):
            _state_equal(a[key],b[key])
        elif isinstance(a[key],(torch.Tensor,np.ndarray)):
            _close(a[key],b[key],"Saved state differs from checkpoint: "+key,atol=0,rtol=0)
        elif a[key]!=b[key]:
            raise ValueError("Saved state metadata differs from checkpoint")


def _metrics(prediction: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    error = prediction-arrays["grid_clean"]
    variance = float(np.var(arrays["grid_clean"]))
    if variance <= 0 or not np.isfinite(error).all():
        raise ValueError("Invalid clean variance or prediction")
    band = np.abs(arrays["grid_x"]) >= .9*math.pi
    values = {"clean_mse":float(np.mean(error**2)),"clean_variance":variance,
              "normalized_clean_mse":float(np.mean(error**2))/variance,
              "mean_absolute_error":float(np.mean(np.abs(error))),"max_absolute_error":float(np.max(np.abs(error))),
              "boundary_mse":float(np.mean(error[band]**2)) if band.any() else None,
              "boundary_points":int(band.sum())}
    if "grid_noisy" in arrays:
        values["noisy_rmse"] = float(np.sqrt(np.mean((prediction-arrays["grid_noisy"])**2)))
    return values


def _initial_geometry(config: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, np.ndarray]:
    # Initialization alone has no target content. Independent entropy/formula
    # replay verifies target-independent starts without calling a task generator.
    phases = {"pilot":1,"development":2,"confirmation":3}
    family = {"smooth":1,"narrow":2,"multimodal":3,"nonperiodic":4}[cell["family"]]
    entropy = [config["seeds"]["initialization"],2,phases[cell["phase"]],family,cell["replicate"],cell["initialization"]]
    rotation = np.random.default_rng(np.random.SeedSequence(entropy)).uniform(0,2*math.pi)
    spec = config["initialization"]
    centers = ((-math.pi+2*math.pi*np.arange(spec["kernels"])/spec["kernels"]+rotation+math.pi)%(2*math.pi)-math.pi)
    widths = ([spec["concentration"]] if cell["model_id"] not in ("fixed32","fixed48")
              else spec[cell["model_id"]+"_widths"])
    return {"centers":np.repeat(centers,len(widths)),"kappas":np.tile(widths,len(centers))}


def audit_fit(cell: Mapping[str, Any], summary: Mapping[str, Any], artifacts: Mapping[str, Any],
              arrays: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every checkpoint, positive-ridge optimum and selected time prefix."""
    spec = next(model for model in config["models"] if model["id"] == cell["model_id"])
    training = config["training"]
    caps = training["time_budgets_seconds"]
    if summary["model_id"] != cell["model_id"] or summary["kind"] != spec["kind"]:
        raise ValueError("Saved model/kind differs from planned cell")
    for key, expected in (("penalty",training["penalty"]),("dtype","float64"),("basis_count",spec["features"]),
                          ("parameter_count",spec["parameters"]),("max_updates",training["max_updates"]),
                          ("validation_every",training["validation_every"]),("time_budgets_seconds",caps),
                          ("primary_time_seconds",training["primary_time_seconds"]),("learning_rate",cell["learning_rate"])):
        if summary[key] != expected:
            raise ValueError(f"Saved {key} differs from protocol")
    n,v = 3*cell["total_labels"]//4,cell["total_labels"]//4
    if len(arrays["train_x"]) != n or len(arrays["validation_x"]) != v or summary["n_train"] != n or summary["n_validation"] != v:
        raise ValueError("Total-label accounting or split sizes differ")
    _close(summary["ridge_alpha"],n*training["penalty"],"Ridge alpha")
    history = summary["history"]
    selection = replay_selections(history,caps)
    steps = [row["step"] for row in history]
    completed = summary["updates_completed"]
    expected_steps = list(range(0,completed+1,training["validation_every"]))
    if expected_steps[-1] != completed:
        expected_steps.append(completed)
    if steps != expected_steps or completed > training["max_updates"]:
        raise ValueError("Missing or unplanned validation checkpoints")
    keys = {str(step) for step in steps}
    if set(artifacts["checkpoint_states"]) != keys or set(artifacts["checkpoint_predictions"]) != keys:
        raise ValueError("Missing or extra saved checkpoint states/predictions")
    updates = summary["update_history"]
    if [row["step"] for row in updates] != list(range(1,completed+1)):
        raise ValueError("Missing optimization update records")
    last_time = history[0]["elapsed_seconds"]
    for row in updates:
        scalars = [row[key] for key in ("objective","training_mse","readout_penalty","gradient_norm_before_clip","geometry_gradient_norm")]
        scalars.extend(row["parameter_gradient_norms"].values())
        if not np.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < last_time or any(not np.isfinite(value) or value<0 for value in scalars):
            raise ValueError("Nonfinite or nonmonotonic optimization records")
        _close(row["objective"],.5*row["training_mse"]+row["readout_penalty"],"Logged update objective")
        last_time = row["elapsed_seconds"]
    if updates and history[-1]["elapsed_seconds"] < last_time:
        raise ValueError("Final checkpoint predates its last update")
    for point in history:
        if point["step"] and updates[point["step"]-1]["elapsed_seconds"]>point["checkpoint_start_elapsed_seconds"]:
            raise ValueError("Checkpoint begins before its last optimizer update")
        if point["step"]<completed and updates[point["step"]]["elapsed_seconds"]<point["elapsed_seconds"]:
            raise ValueError("Next optimizer update predates completed checkpoint")
    elapsed = summary.get("fit_elapsed_seconds",summary.get("fit_seconds"))
    _close(elapsed,history[-1]["elapsed_seconds"],"Final timed checkpoint",atol=0,rtol=0)
    _close(summary["overshoot_seconds"],max(0.,elapsed-max(caps)),"Timed overshoot",atol=0,rtol=0)
    if spec["kind"] == "ridge":
        if completed != 0 or summary["stop_reason"] != "fixed_closed_form":
            raise ValueError("Fixed model performed optimizer updates")
    elif summary["stop_reason"] == "wall_time":
        if elapsed < max(caps) or any(row["elapsed_seconds"] >= max(caps) for row in history[:-1]):
            raise ValueError("Adaptive fit stopped outside the first exhausted time block")
    elif summary["stop_reason"] != "maximum_updates" or completed != training["max_updates"]:
        raise ValueError("Adaptive fit ended before either registered ceiling")

    max_prediction, max_readout, max_resolved_prediction, conditions, kappas = 0.,0.,0.,[],[]
    stationarity = []
    initial_signature = None
    for point in history:
        key = str(point["step"])
        state = artifacts["checkpoint_states"][key]
        if state["model_id"] != cell["model_id"] or state["basis_count"] != spec["features"] or state["fitted_parameter_count"] != spec["parameters"]:
            raise ValueError("Checkpoint architecture metadata changed")
        if not state["readout_matches_geometry"]:
            raise ValueError("Checkpoint readout was not refreshed at its geometry")
        _state_schema(state,spec)
        features = numpy_features(state,arrays["train_x"])
        weight,bias = _array(state["readout_weight"]),_array(state["readout_bias"])
        if point["step"] == 0 and state["basis_type"] == "circular":
            geometry = _initial_geometry(config,cell)
            _close(_array(state["centers"]).reshape(-1),geometry["centers"],"Target-independent initial centers",atol=1e-12,rtol=1e-12)
            _close(_array(state["kappas"]).reshape(-1),geometry["kappas"],"Declared initial widths",atol=1e-12,rtol=1e-12)
        if state["basis_type"]=="circular":
            initial=artifacts["checkpoint_states"]["0"]
            mode=spec.get("geometry","fixed")
            for field,frozen in (("centers",mode in ("fixed","widths")),("kappas",mode in ("fixed","centers"))):
                if frozen:
                    _close(state[field],initial[field],"Frozen geometry changed: "+field,atol=0,rtol=0)
        if spec["kind"] in ("ridge","vp") or point["step"] == 0:
            solved_weight,solved_bias = independent_ridge(features,arrays["train_y"],training["penalty"])
            max_readout = max(max_readout,_close(weight,solved_weight,"Independent positive-ridge weights"),
                               _close(bias,solved_bias,"Independent unpenalized intercept"))
            max_resolved_prediction=max(max_resolved_prediction,_close(features@weight+bias,features@solved_weight+solved_bias,"Independent solve prediction sensitivity"))
        for split in ("train","validation"):
            predicted = numpy_prediction(state,arrays[split+"_x"])
            saved = artifacts["checkpoint_predictions"][key][split]
            max_prediction = max(max_prediction,_close(saved,predicted,f"Checkpoint {key} {split} predictions"))
            _close(point[split+"_mse"],float(np.mean((_array(saved)-arrays[split+"_y"])**2)),f"Checkpoint {key} {split} MSE",atol=1e-11,rtol=1e-10)
        objective = .5*point["train_mse"]+.5*training["penalty"]*float(np.sum(weight**2))
        _close(point["objective"],objective,"Explicit readout-only penalty")
        centered = features-features.mean(axis=0)
        singular = np.linalg.svd(centered.T@centered+n*training["penalty"]*np.eye(features.shape[1]),compute_uv=False)
        condition = float(singular[0]/singular[-1])
        _close(point["readout_diagnostics"]["normal_matrix_condition_2"],condition,"Regularized normal conditioning",atol=1e-7,rtol=1e-7)
        conditions.append(condition)
        residual=features@weight+bias-arrays["train_y"]
        gradient=features.T@residual/n+training["penalty"]*weight
        stationarity.append({"step":point["step"],"readout_stationarity_l2":float(np.linalg.norm(gradient)),
            "intercept_stationarity_abs":float(abs(residual.mean())),"normal_matrix_condition_2":condition,
            "readout_norm":float(np.linalg.norm(weight)),"readout_matches_ridge_optimum_expected":spec["kind"]!="joint" or point["step"]==0})
        if state["basis_type"] == "circular":
            kappas.extend(_array(state["kappas"]).reshape(-1).tolist())
        if point["step"] == 0 and cell["model_id"] in ("fixed16","vp_centers","vp_widths","vp_both","joint_both"):
            initial_signature = np.concatenate((_array(state["centers"]).ravel(),_array(state["kappas"]).ravel(),weight.ravel(),bias.ravel()))

    _state_equal(artifacts["initial_state"],artifacts["checkpoint_states"]["0"])
    _state_equal(artifacts["final_state"],artifacts["checkpoint_states"][str(steps[-1])])

    timepoints = {}
    if set(summary["selections"]) != set(selection):
        raise ValueError("Missing or extra time allowances")
    for cap in caps:
        key = str(float(cap));point = selection[key];selected = summary["selections"][key]
        eligible = [row for row in history if row["elapsed_seconds"] <= cap]
        if selected["eligible_checkpoints"] != len(eligible) or selected["completed_updates_within_allowance"] != sum(row["elapsed_seconds"]<=cap for row in updates):
            raise ValueError("Incorrect eligible checkpoint/update count")
        if point is None:
            if selected["status"] != "budget_unavailable" or selected["selected_step"] is not None or key in artifacts["selected_states"]:
                raise ValueError("Unavailable time allowance was silently filled")
            timepoints[key] = {"available":False,"primary_error":None}
            continue
        if selected["status"] != "completed" or selected["selected_step"] != point["step"] or selected["best_validation_mse"] != point["validation_mse"]:
            raise ValueError("Time-prefix checkpoint selection differs from noisy-validation rule")
        if (selected["checkpoint_elapsed_seconds"] != point["elapsed_seconds"]
                or selected["last_admissible_checkpoint_step"] != eligible[-1]["step"]
                or selected["last_admissible_checkpoint_seconds"] != eligible[-1]["elapsed_seconds"]
                or selected["actual_trajectory_fit_seconds"] != elapsed
                or selected["allowance_seconds"] != cap):
            raise ValueError("Selected checkpoint time metadata changed")
        _close(selected["checkpoint_budget_utilization"],eligible[-1]["elapsed_seconds"]/cap,"Budget utilization")
        saved_state = artifacts["selected_states"][key]
        _state_equal(saved_state,artifacts["checkpoint_states"][str(point["step"])])
        for split in ("train","validation"):
            _close(artifacts["selected_predictions"][key][split],artifacts["checkpoint_predictions"][str(point["step"])][split],"Selected split prediction differs",atol=0,rtol=0)
        grid_prediction = numpy_prediction(saved_state,arrays["grid_x"])
        max_prediction = max(max_prediction,_close(artifacts["selected_predictions"][key]["grid"],grid_prediction,"Time-prefix grid prediction"))
        metrics = _metrics(_array(artifacts["selected_predictions"][key]["grid"]),arrays)
        for name,value in metrics.items():
            if value is not None:
                _close(selected["metrics"][name],value,"Time-prefix metric "+name,atol=1e-11,rtol=1e-10)
        _close(selected["primary_error"],metrics["normalized_clean_mse"],"Time-prefix primary error")
        timepoints[key] = {"available":True,"primary_error":metrics["normalized_clean_mse"],"metrics":metrics,
                           "selected_step":point["step"],"checkpoint_elapsed_seconds":point["elapsed_seconds"],
                           "completed_updates_within_allowance":selected["completed_updates_within_allowance"]}
    primary = str(float(training["primary_time_seconds"]))
    if not timepoints[primary]["available"] or summary["status"] != "completed":
        raise ValueError("Primary time allowance is incomplete")
    _close(summary["primary_error"],timepoints[primary]["primary_error"],"Summary primary error")
    if summary["selected_step"]!=selection[primary]["step"] or summary["best_validation_mse"]!=selection[primary]["validation_mse"]:
        raise ValueError("Primary selection summary differs")
    if summary["metrics"]!=summary["selections"][primary]["metrics"]:
        raise ValueError("Primary metric summary differs")
    return {"primary_error":timepoints[primary]["primary_error"],"timepoints":timepoints,
            "max_prediction_discrepancy":max_prediction,"max_ridge_weight_discrepancy":max_readout,
            "checkpoint_count":len(history),"updates_completed":completed,"fit_seconds":elapsed,
            "overshoot_seconds":summary["overshoot_seconds"],"normal_condition_max":max(conditions),
            "max_resolved_prediction_discrepancy":max_resolved_prediction,
            "sensitivity_scope":"Numerical solver replay discrepancy only; no target-noise perturbation study",
            "conditioning_by_checkpoint":stationarity,
            "concentration_min":min(kappas) if kappas else None,"concentration_max":max(kappas) if kappas else None,
            "initial_signature":initial_signature}


def _task_arrays(value: Mapping[str, Any], config: Mapping[str, Any], cell: Mapping[str, Any],
                   lock_hash: str | None) -> dict[str, np.ndarray]:
    """Read exact stored pools; never evaluate or regenerate a clean function."""
    if value["sha256"] != object_hash({key:item for key,item in value.items() if key!="sha256"}):
        raise ValueError("Task serialization hash mismatch")
    meta = value["metadata"]
    if meta["config_sha256"] != object_hash(config) or meta["phase"] != cell["phase"] or meta["family"] != cell["family"] or meta["replicate"] != cell["replicate"]:
        raise ValueError("Task belongs to another protocol cell")
    if meta["confirmation_lock_sha256"] != lock_hash:
        raise ValueError("Task is not bound to the invocation lock")
    if value["function_sha256"] != object_hash(value["function"]):
        raise ValueError("Target parameter hash mismatch")
    if value["label_budgets"] != config["tasks"]["label_budgets"] or value["noise_levels"] != config["tasks"]["noise_levels"]:
        raise ValueError("Task label/noise registry differs")
    pools = {name:np.asarray(array,dtype=np.float64) for name,array in value["arrays"].items()}
    required = {split+suffix for split in ("train","validation","grid") for suffix in ("_x","_clean","_unit_noise")}
    if set(pools) != required or set(value["array_sha256"]) != required or value["dtype"] != "float64":
        raise ValueError("Task source pools are incomplete")
    for name,array in pools.items():
        if array.ndim!=2 or array.shape[1]!=1 or not np.isfinite(array).all() or hashlib.sha256(array.tobytes()).hexdigest()!=value["array_sha256"][name]:
            raise ValueError("Invalid or changed task source pool: "+name)
    largest = max(config["tasks"]["label_budgets"])
    for split,count in (("train",3*largest//4),("validation",largest//4),("grid",config["tasks"]["grid_points"])):
        if any(len(pools[split+suffix])!=count for suffix in ("_x","_clean","_unit_noise")):
            raise ValueError("Task maximum pool sizes differ from total-label contract")
    phase_id={"pilot":1,"development":2,"confirmation":3}[cell["phase"]]
    family_id={"smooth":1,"narrow":2,"multimodal":3,"nonperiodic":4}[cell["family"]]
    roots=config["seeds"][cell["phase"]]
    streams={name:[roots["task" if index==0 else "observation"],2,phase_id,family_id,cell["replicate"],index]
             for name,index in {"function":0,"train_inputs":1,"train_noise":2,"validation_inputs":3,"validation_noise":4,"grid_noise":5}.items()}
    if meta["stream_components"]!=streams or meta["task_root"]!=roots["task"] or meta["observation_root"]!=roots["observation"]:
        raise ValueError("Task seed provenance changed")
    # Only observation streams are replayed. No clean target is evaluated here.
    for split in ("train","validation","grid"):
        count=len(pools[split+"_x"])
        rng=np.random.default_rng(np.random.SeedSequence(streams[split+"_noise"]))
        _close(pools[split+"_unit_noise"],rng.standard_normal((count,1)),"Independent observation noise stream",atol=0,rtol=0)
        if split!="grid":
            rng=np.random.default_rng(np.random.SeedSequence(streams[split+"_inputs"]))
            x=rng.uniform(-math.pi,math.pi,(count,1))
        else:
            x=(-math.pi+(np.arange(count,dtype=np.float64)+.5)*2*math.pi/count)[:,None]
        _close(pools[split+"_x"],x,"Independent input stream/grid",atol=2e-15,rtol=2e-15)
    n,v = 3*cell["total_labels"]//4,cell["total_labels"]//4;noise = cell["noise_std"]
    result = {"grid_x":pools["grid_x"],"grid_clean":pools["grid_clean"],
              "grid_noisy":pools["grid_clean"]+noise*pools["grid_unit_noise"]}
    for split,count in (("train",n),("validation",v)):
        result[split+"_x"] = pools[split+"_x"][:count]
        result[split+"_y"] = pools[split+"_clean"][:count]+noise*pools[split+"_unit_noise"][:count]
    return result


def aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], cap: float) -> dict[str, Any]:
    """Average initialization/noise/budgets within target, then targets/families."""
    phase=rows[0]["cell"]["phase"] if rows else "development"
    if phase=="pilot":
        raise ValueError("Pilot roles and learning-rate candidates are separate diagnostics, not population bundles")
    fields=("model_id","family","replicate","noise_std","total_labels","initialization")
    actual=[tuple(row["cell"][field] for field in fields) for row in rows]
    expected={(spec["id"],family,replicate,noise,budget,initialization)
        for spec in config["models"] for family in config["tasks"]["families"]
        for replicate in range(config["tasks"]["replicates"][phase])
        for noise in config["tasks"]["noise_levels"] for budget in config["tasks"]["label_budgets"]
        for initialization in range(config["tasks"]["initializations"][phase])}
    matrix_complete=bool(rows) and len(actual)==len(expected) and set(actual)==expected and all(row["cell"]["phase"]==phase for row in rows)
    grouped = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    detailed = defaultdict(list)
    unavailable=0
    for row in rows:
        if not row.get("audited"):
            continue
        point = row["timepoints"][str(float(cap))]
        cell = row["cell"]
        if not point["available"]:
            unavailable+=1
            continue
        error = point["primary_error"]
        grouped[cell["model_id"]][cell["family"]][cell["replicate"]].append(error)
        detailed[(cell["model_id"],cell["family"],cell["noise_std"],cell["total_labels"])].append(error)
    models = {}
    per_task=len(config["tasks"]["noise_levels"])*len(config["tasks"]["label_budgets"])*config["tasks"]["initializations"][phase]
    per_detail=config["tasks"]["replicates"][phase]*config["tasks"]["initializations"][phase]
    for model,families in grouped.items():
        tasks = {family:{str(task):float(np.mean(values)) for task,values in targets.items() if len(values)==per_task} for family,targets in families.items()}
        means = {family:float(np.mean(list(targets.values()))) for family,targets in tasks.items() if len(targets)==config["tasks"]["replicates"][phase]}
        primary = config["tasks"]["periodic_families"]
        models[model] = {"task_errors":tasks,"family_errors":means,
            "available_cells_by_target":{family:{str(task):len(values) for task,values in targets.items()} for family,targets in families.items()},
            "required_cells_per_target":per_task,
            "task_sd_by_family":{family:float(np.std(list(targets.values()),ddof=1)) if len(targets)>1 else None for family,targets in tasks.items()},
            "balanced_periodic_error":float(np.mean([means[family] for family in primary])) if all(family in means for family in primary) else None}
    return {"allowance_seconds":cap,"complete":matrix_complete and unavailable==0 and all(row.get("audited") for row in rows),
        "matrix_complete":matrix_complete,
        "unavailable_cells":unavailable,"models":models,"cells":[{"model_id":key[0],"family":key[1],
        "noise_std":key[2],"total_labels":key[3],"mean_error":float(np.mean(values)) if len(values)==per_detail else None,
        "observed_mean_error":float(np.mean(values)),"complete":len(values)==per_detail,
        "required_initialization_target_cells":per_detail,"observed_initialization_target_cells":len(values)}
        for key,values in sorted(detailed.items())]}


def choose_development(models: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    """Select the VP strategy and parameter-count comparator in config order."""
    vps = [spec for spec in config["models"] if spec["kind"]=="vp"]
    winner = min(vps,key=lambda spec:models[spec["id"]]["balanced_periodic_error"])
    controls = [spec for spec in config["models"] if spec["kind"]=="ridge" and spec["parameters"]==winner["parameters"]]
    comparator = min(controls,key=lambda spec:models[spec["id"]]["balanced_periodic_error"])
    return {"vp_model":winner["id"],"capacity_comparator":comparator["id"]}


def contrast(models: Mapping[str, Any], first: str, second: str,
               config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute ratio-of-balanced-errors and whole-function paired bootstrap."""
    a,b = models[first],models[second]
    rng = np.random.default_rng(np.random.SeedSequence([config["seeds"]["analysis"],2,0,0,7]))
    draws_a,draws_b = [],[];differences = {};family_improvement = {}
    for family in config["tasks"]["periodic_families"]:
        ids = sorted(a["task_errors"][family],key=int)
        if ids!=sorted(b["task_errors"][family],key=int):
            raise ValueError("Unpaired target bundles in primary contrast")
        x=np.array([a["task_errors"][family][key] for key in ids]);y=np.array([b["task_errors"][family][key] for key in ids])
        indices=rng.integers(0,len(ids),(config["analysis"]["bootstrap_resamples"],len(ids)))
        draws_a.append(x[indices].mean(axis=1));draws_b.append(y[indices].mean(axis=1))
        differences[family]=(x-y).tolist();family_improvement[family]=1-float(x.mean()/y.mean()) if y.mean()>0 else None
    denom=np.mean(draws_b,axis=0)
    confidence=config["analysis"]["bootstrap_confidence"]
    interval=None if np.any(denom<=0) else np.quantile(1-np.mean(draws_a,axis=0)/denom,[(1-confidence)/2,(1+confidence)/2]).tolist()
    improvement=1-a["balanced_periodic_error"]/b["balanced_periodic_error"] if b["balanced_periodic_error"]>0 else None
    gate=config["analysis"]["primary_gate"]
    checks={"relative_improvement":improvement is not None and improvement>=gate["both_comparisons_relative_improvement_min"],
            "bootstrap_lower_positive":interval is not None and interval[0]>gate["both_comparisons_lower_bootstrap_improvement_above"],
            "families_improving":sum(value is not None and value>0 for value in family_improvement.values())>=gate["each_comparison_improving_periodic_families_min"],
            "family_regression_limit":all(value is not None and value>=-gate["each_comparison_max_family_regression"] for value in family_improvement.values())}
    return {"model":first,"comparator":second,"relative_improvement":improvement,
            "family_improvements":family_improvement,"paired_target_differences":differences,
            "bootstrap":{"lower":None if interval is None else interval[0],"upper":None if interval is None else interval[1],
                "resamples":config["analysis"]["bootstrap_resamples"],"confidence":confidence,
                "seed_components":[config["seeds"]["analysis"],2,0,0,7],
                "cluster":"Whole target bundle within family; retain all label/noise/initialization pairings"},
            "checks":checks,"gate_passed":all(checks.values())}


@threadpool_limits.wrap(limits=1)
def analyze_run(manifest_path: Path, config: Mapping[str, Any], *, lock: Path | None = None) -> dict[str, Any]:
    """Audit an immutable run and retain every failure in a strict-JSON report."""
    manifest_path=Path(manifest_path).resolve();manifest=read_json(manifest_path);errors=[];rows=[];evidence={}
    if manifest["kind"]!="locality_run_v2" or manifest["config_hash"]!=object_hash(config):
        raise ValueError("Not a matching locality-v2 manifest")
    phase=manifest["phase"];lock_record=None
    if phase=="confirmation":
        if lock is None:
            candidates=[checked_path(path) for path,digest in manifest["prerequisite_hashes"].items() if digest==manifest["lock_hash"]]
            if len(candidates)!=1:
                raise ValueError("Confirmation analysis requires the exact invocation lock")
            lock=candidates[0]
        lock_record=verify_confirmation_lock(lock,config)
        if file_hash(Path(lock))!=manifest["lock_hash"] or lock_record["learning_rates"]!=manifest["learning_rates"]:
            raise ValueError("Confirmation lock differs from invocation")
    expected=planned_runs(config,phase,manifest.get("learning_rates"))
    if [row["cell"] for row in manifest["rows"]]!=expected or manifest["planned_count"]!=len(expected):
        errors.append("Run matrix is missing, duplicated, reordered or outside the protocol")
    try:
        manifest,_=verify_run_manifest(manifest_path,config,ROOT,phase)
        if manifest["source_hashes"]!=source_manifest() or set(manifest["source_hashes"])!=set(SOURCE_PATHS):
            raise ValueError("Current source differs from recorded source binding")
        for name,digest in manifest["source_hashes"].items():
            snapshot=manifest_path.parent/"sources"/name
            if file_hash(snapshot)!=digest:
                raise ValueError("Source snapshot changed: "+name)
            evidence[snapshot.relative_to(ROOT).as_posix()]=digest
        for dependencies in (manifest["evidence_hashes"],manifest["prerequisite_hashes"]):
            if dependencies:
                verify_dependencies(dependencies)
            evidence.update(dependencies)
        for name in manifest["prerequisite_hashes"]:
            dependency=read_json(checked_path(name))
            nested=dependency.get("evidence_hashes",{})
            if nested:
                verify_dependencies(nested)
                evidence.update(nested)
        config_path=manifest_path.parent/"config.json"
        if read_json(config_path)!=config:
            raise ValueError("Saved config copy differs")
        evidence[config_path.relative_to(ROOT).as_posix()]=file_hash(config_path)
    except Exception as error:
        errors.append(str(error))
    if not manifest.get("complete") or manifest["status"]!="completed" or not manifest.get("deadline_respected"):
        errors.append("Invocation is incomplete or outside its unchanged phase ceiling")
    evidence[manifest_path.relative_to(ROOT).as_posix()]=file_hash(manifest_path)
    task_cache={};initial_groups=defaultdict(list)
    for entry in manifest["rows"]:
        cell=entry["cell"];result={"cell":cell,"status":entry["status"],"audited":False}
        try:
            if entry["status"]!="completed":
                raise ValueError("Planned cell is "+entry["status"]+"; retained without survivor substitution")
            paths={name:checked_path(entry[name+"_path"]) for name in ("task","summary","artifact")}
            for name,path in paths.items():
                rel=path.relative_to(ROOT).as_posix();digest=file_hash(path)
                if manifest["evidence_hashes"].get(rel)!=digest or (name!="task" and entry[name+"_hash"]!=digest):
                    raise ValueError("Missing or changed "+name+" evidence binding")
            if str(paths["task"]) not in task_cache:
                task_cache[str(paths["task"])]=read_json(paths["task"])
            arrays=_task_arrays(task_cache[str(paths["task"])],config,cell,manifest["lock_hash"])
            summary=read_json(paths["summary"])
            artifacts=torch.load(paths["artifact"],map_location="cpu",weights_only=True)
            audited=audit_fit(cell,summary,artifacts,arrays,config)
            signature=audited.pop("initial_signature")
            if signature is not None:
                group=tuple(cell[key] for key in ("family","replicate","noise_std","total_labels","initialization"))
                initial_groups[group].append((cell["model_id"],signature))
            result.update(audited,audited=True)
        except Exception as error:
            result["audit_error"]=f"{type(error).__name__}: {error}";errors.append(cell["id"]+": "+result["audit_error"])
        rows.append(result)
    initial_max=0.
    for group,values in initial_groups.items():
        try:
            expected_initial={spec["id"] for spec in config["models"]}&{"fixed16","vp_centers","vp_widths","vp_both","joint_both"}
            if {name for name,_ in values}!=expected_initial:
                raise ValueError("Incomplete matched-initialization comparison")
            for _,signature in values[1:]:
                initial_max=max(initial_max,_close(signature,values[0][1],"Matched initial geometry/readout",atol=1e-11,rtol=1e-11))
        except Exception as error:
            errors.append(str(group)+": "+str(error))
    complete=not errors
    result={"kind":"locality_analysis_v2","protocol_id":config["protocol_id"],"phase":phase,"complete":complete,
            "config_hash":object_hash(config),"source_hashes":manifest["source_hashes"],"evidence_hashes":evidence,
            "input_manifest":manifest_path.relative_to(ROOT).as_posix(),"learning_rates":manifest.get("learning_rates"),
            "planned_count":len(expected),"audited_count":sum(row["audited"] for row in rows),"audit_errors":errors,
            "audit":{"all_passed":complete,"failed_checks":len(errors),"scope":"Source/config/evidence, exact matrix, observation streams, every saved checkpoint, temporal selection, readout and prediction replay"},
            "choices":None,"initial_readout_parity_max_difference":initial_max,"rows":rows,
            "methodological_gate_evaluated":False,"gate_passed":False,
            "scope":"Synthetic scalar mechanism study; wall-time selection may vary on rerun. Numerical replay is distinct from optimizer reproducibility."}
    if phase=="pilot":
        result["pilot_scope"]="Only stored-state/time-selection audit; this observer does not select pilot rates from grid scores"
    else:
        result["time_allowances"]={str(float(cap)):aggregate(rows,config,cap) for cap in config["training"]["time_budgets_seconds"]}
        if complete:
            primary=result["time_allowances"][str(float(config["training"]["primary_time_seconds"]))]["models"]
            choices=choose_development(primary,config) if phase=="development" else lock_record["choices"]
            result["choices"]=choices
            result["co_primary"]={control:contrast(primary,choices["vp_model"],control,config)
                                  for control in ("fixed16",choices["capacity_comparator"])}
            result["optimizer_contrast"]=contrast(primary,"vp_both","joint_both",config)
            result["methodological_gate_evaluated"]=True
            result["gate_passed"]=all(value["gate_passed"] for value in result["co_primary"].values())
            result["decision_scope"]="Confirmation decision" if phase=="confirmation" else "Development descriptive gate; selection must be frozen before confirmation"
    # Reject NaN/infinity or leaked ndarray/tensor values before returning.
    object_hash(result)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--config",type=Path,default=CONFIG_PATH)
    parser.add_argument("--lock",type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    result=analyze_run(args.input,read_json(args.config),lock=args.lock)
    write_json(args.output,result)
    print(f"Locality audit complete={result['complete']}; report={args.output}",flush=True)
    return 0 if result["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
