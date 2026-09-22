"""Independently audit saved VM methodology artifacts and apply locked decisions.

The auditor never fits a model or generates a task. It reconstructs predictions
from saved states with explicit equations and recomputes metrics, selection and
paired task-level bootstrap intervals from the complete declared run matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.methodology_protocol import SOURCE_PATHS, file_hash, object_hash, read_json, verify_confirmation_lock, write_json


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _assert_close(actual: Any, expected: Any, *, label: str, atol: float = 1e-10,
                  rtol: float = 1e-9) -> None:
    a, b = _array(actual), _array(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label}: nonfinite values or different shapes")
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: maximum difference {float(np.max(np.abs(a-b))):.9g}")


def reconstruct_prediction(model_id: str, state: Mapping[str, Any], x: Any,
                           *, gradient: bool) -> np.ndarray:
    """Rebuild predictions without importing the trainer or layer definitions.

    Gradient checkpoints use CPU float32 arithmetic to match training precision;
    parity across CPU/CUDA is checked at the declared engineering tolerance.
    Metrics themselves are recomputed from the saved predictions in float64.
    """
    dtype = torch.float32 if gradient else torch.float64
    theta = torch.as_tensor(_array(x).copy(), dtype=dtype)
    values = {name: torch.as_tensor(_array(value).copy(), dtype=dtype) for name, value in state.items()}
    if model_id == "polar":
        k = torch.logaddexp(values["raw_kappa"], torch.zeros_like(values["raw_kappa"]))
        result = k * torch.cos(theta + values["phase"]) + values["bias"]
    elif model_id == "cartesian":
        result = theta.cos() @ values["weight_cos"].T + theta.sin() @ values["weight_sin"].T + values["bias"]
    elif model_id == "reference":
        coefficients = values["coefficients"]
        result = theta.cos() * coefficients[0] + theta.sin() * coefficients[1] + coefficients[2]
    elif model_id in {"fourier_ridge", "fixed_vm_ridge"}:
        coefficients = values["coefficients"]
        features = _fixed_features(model_id, _array(theta))
        return features @ _array(coefficients[:-1]) + _array(coefficients[-1])
    elif model_id in {"learned_vm", "learned_geodesic"}:
        d = theta.unsqueeze(-2) - values["bank.centers"]
        k = torch.logaddexp(values["bank.raw_kappa"], torch.zeros_like(values["bank.raw_kappa"]))
        if model_id == "learned_vm":
            features = torch.exp((-2 * k * (d / 2).sin().square()).sum(-1))
        else:
            d = (d + math.pi) % (2 * math.pi) - math.pi
            features = torch.exp((-0.5 * k * d.square()).sum(-1))
        result = features @ values["readout.weight"].T + values["readout.bias"]
    elif model_id in {"sincos_mlp", "raw_angle_mlp"}:
        features = torch.cat((theta.cos(), theta.sin()), -1) if model_id == "sincos_mlp" else theta / math.pi
        hidden = torch.tanh(features @ values["hidden.weight"].T + values["hidden.bias"])
        result = hidden @ values["readout.weight"].T + values["readout.bias"]
    else:
        raise ValueError(f"Unknown saved model: {model_id}")
    return result.numpy().astype(np.float64)


def _fixed_features(model_id: str, x: np.ndarray) -> np.ndarray:
    # Use the recorded Torch float64 elementary-function backend: a rank-
    # deficient unregularized bank can amplify single-ulp sin/cos differences
    # into large coefficient/subspace differences. The independent stationarity
    # and minimum-norm checks below assess the exact fitted design matrix.
    theta = torch.as_tensor(np.asarray(x, dtype=np.float64).copy())
    if model_id == "fourier_ridge":
        angle = theta * torch.arange(1, 25, dtype=torch.float64)
        return torch.cat((angle.cos(), angle.sin()), dim=-1).numpy()
    centers = (-math.pi + 2 * math.pi * torch.arange(16, dtype=torch.float64) / 16).repeat_interleave(3)
    widths = torch.tensor([1., 4., 16.], dtype=torch.float64).repeat(16)
    return torch.exp(-2 * widths * ((theta - centers) / 2).sin().square()).numpy()


def _audit_linear_optimum(features: np.ndarray, y: np.ndarray, coefficients: Any,
                           penalty: float) -> dict[str, Any]:
    """Audit the recorded rounded design, separating replay from numerical accuracy.

    Absolute normal-equation residuals scale with the coefficient norm. A nearly
    rank-deficient bank can have enormous coefficients and cancellation despite
    backward-stable arithmetic. We therefore retain the absolute residual and
    use the explicitly reported normwise float64 roundoff criterion below.
    This is not a bound on prediction/coefficient forward error.
    """
    design = np.column_stack((features, np.ones(len(features))))
    coefficients = np.asarray(_array(coefficients), dtype=np.float64)
    residual = design @ coefficients - y
    gradient = design.T @ residual / len(design)
    gradient[:-1] += penalty * coefficients[:-1]
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    epsilon = np.finfo(np.float64).eps
    cutoff = epsilon * max(design.shape) * singular[0]
    rank = int(np.sum(singular > cutoff))
    coefficient_norm = float(np.linalg.norm(coefficients))
    gradient_scale = (singular[0] * (singular[0] * coefficient_norm + np.linalg.norm(y)) / len(design)
                      + penalty * np.linalg.norm(coefficients[:-1]))
    backward_error = float(np.linalg.norm(gradient) / max(gradient_scale, np.finfo(np.float64).tiny))
    # Dimension-scaled epsilon covers matrix products, SVD truncation and
    # accumulation, with a declared safety factor; it is not a learned gate.
    bound = float(64 * epsilon * max(design.shape))
    if not np.isfinite(gradient).all() or backward_error > bound:
        raise ValueError(f"Closed-form coefficients fail normwise stationarity check: {backward_error} > {bound}")

    replay_design, replay_y = design, np.asarray(y, dtype=np.float64)
    if penalty:
        regularizer = np.zeros((design.shape[1]-1,design.shape[1]))
        regularizer[:,:-1] = math.sqrt(len(design)*penalty) * np.eye(design.shape[1]-1)
        replay_design = np.concatenate((design,regularizer))
        replay_y = np.concatenate((replay_y,np.zeros((len(regularizer),y.shape[1]))))
    replayed = np.linalg.lstsq(replay_design,replay_y,rcond=None)[0]
    # Both the recorded runner and this observer use BLAS1 and the identical
    # rounded feature matrix. Exact coefficient replay is a procedural check,
    # independent of the scale-aware stationarity diagnostic above.
    _assert_close(coefficients,replayed,label="Closed-form coefficient replay",atol=0,rtol=0)
    projection = right[:rank].T @ (right[:rank] @ coefficients)
    projection_relative = float(np.linalg.norm(coefficients-projection)/max(coefficient_norm,np.finfo(np.float64).tiny))
    condition = float(singular[0]/singular[-1]) if singular[-1]>0 else None
    return {"design_shape":list(design.shape),"numerical_rank":rank,"rank_cutoff":float(cutoff),
            "singular_value_max":float(singular[0]),"singular_value_min":float(singular[-1]),
            "condition_number":condition,"effective_retained_condition_number":float(singular[0]/singular[rank-1]) if rank else None,
            "coefficient_max_absolute":float(np.max(np.abs(coefficients))),"coefficient_l2_norm":coefficient_norm,
            "absolute_stationarity_residual_max":float(np.max(np.abs(gradient))),
            "stationarity_scale":float(gradient_scale),"normwise_backward_error":backward_error,
            "float64_roundoff_bound":bound,"roundoff_bound_formula":"64 * float64_eps * max(n_observations,n_design_columns)",
            "minimum_norm_independent_svd_projection_relative_difference":projection_relative if penalty==0 else None,
            "coefficient_replay_max_difference":float(np.max(np.abs(coefficients-replayed))),
            "training_prediction_replay_max_difference":float(np.max(np.abs(design@coefficients-design@replayed))),
            "numerically_sensitive":condition is None or condition>1/math.sqrt(epsilon),
            "interpretation":"Exact same-backend coefficient replay and normwise backward-error check; no coefficient/prediction forward-accuracy guarantee. Independent SVD subspaces can be sensitive to decomposition roundoff."}


def recompute_metrics(prediction: Any, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compute scalar error summaries in float64, with no trainer helpers."""
    prediction = np.asarray(_array(prediction), dtype=np.float64)
    clean, noisy, x = arrays["grid_clean"], arrays["grid_noisy"], arrays["grid_x"]
    error = prediction - clean
    variance = float(np.var(clean))
    if variance <= 0 or not np.isfinite(error).all():
        raise ValueError("Invalid clean variance or prediction")
    boundary = np.abs(x) >= 0.9 * math.pi
    return {"clean_mse": float(np.mean(error ** 2)), "clean_variance": variance,
            "normalized_clean_mse": float(np.mean(error ** 2)) / variance,
            "noisy_rmse": float(np.sqrt(np.mean((prediction - noisy) ** 2))),
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "max_absolute_error": float(np.max(np.abs(error))),
            "boundary_mse": float(np.mean(error[boundary] ** 2)),
            "boundary_mean_absolute_error": float(np.mean(np.abs(error[boundary]))),
            "boundary_points": int(np.count_nonzero(boundary))}


def replay_validation_selection(history: Sequence[Mapping[str, Any]], min_delta: float,
                                 patience: int) -> tuple[int, float, int]:
    """Replay best-checkpoint selection and the separate min-delta patience anchor."""
    if not history or history[0]["step"] != 0:
        raise ValueError("Missing eligible initialization checkpoint")
    best, selected_step, stale = float(history[0]["validation_mse"]), 0, 0
    significant_best = best
    if not np.isfinite(best):
        raise ValueError("Nonfinite initial validation score")
    for index, row in enumerate(history):
        score = float(row["validation_mse"])
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation history")
        selected = index == 0 or score < best
        significant = index == 0 or score < significant_best - min_delta
        if row["selected"] != selected:
            raise ValueError("Saved checkpoint selection differs from noisy validation rule")
        if index:
            if row["step"] <= history[index-1]["step"]:
                raise ValueError("Validation steps are not strictly increasing")
            if selected:
                best, selected_step = score, row["step"]
            if significant:
                significant_best, stale = score, 0
            else:
                stale += 1
            if row.get("significant_improvement", significant) != significant:
                raise ValueError("Patience anchor differs from min-delta rule")
            if stale >= patience and index != len(history) - 1:
                raise ValueError("Training continued after declared patience was exhausted")
    return selected_step, best, stale


def select_ridge_lambda(history: Sequence[Mapping[str, Any]], tolerance: float = 1e-12) -> float:
    """Select the smallest lambda within tolerance of the global best score."""
    if not history or any(not np.isfinite(row["validation_mse"]) for row in history):
        raise ValueError("Incomplete or nonfinite ridge candidate scores")
    best = min(row["validation_mse"] for row in history)
    return min(row["lambda"] for row in history if row["validation_mse"] <= best + tolerance)


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in ("stage", "phase", "family", "replicate", "kind", "model",
                 "optimizer", "learning_rate", "penalty", "n_train", "initialization_index"))


def _expected_cells(config: Mapping[str, Any], stage: str, phase: str,
                     choices: Mapping[str, Any] | None) -> list[tuple[Any, ...]]:
    """Independently enumerate the canonical matrix, including every reference."""
    spec, rows = config[stage], []
    families = spec["pilot"]["conditions"] if stage == "A" and phase == "pilot" else spec["family_ids"]
    for family in families:
        for replicate in range(spec[phase]["task_pairs"]):
            common = dict(stage=stage, phase=phase, family=family, replicate=replicate)
            if stage == "A":
                for penalty in spec["explicit_amplitude_penalties"]:
                    rows.append(_identity(dict(common, model="reference", kind="reference", penalty=penalty)))
                    for optimizer, opt in spec["optimizers"].items():
                        rates = opt["pilot_learning_rates"] if phase == "pilot" else [choices["A"][optimizer]]
                        for rate in rates:
                            for coordinate in spec["models"]:
                                rows.append(_identity(dict(common, kind="gradient", model=coordinate,
                                                    penalty=penalty, optimizer=optimizer, learning_rate=rate)))
            else:
                sizes = [spec["pilot"]["train_observations"]] if phase == "pilot" else spec["train_observations"]
                for size in sizes:
                    for model in spec["models"]:
                        if model["kind"] == "ridge":
                            rows.append(_identity(dict(common, model=model["id"], kind="ridge", n_train=size)))
                        else:
                            rates = spec["optimizer"]["pilot_learning_rates"] if phase == "pilot" else [choices["B"][model["id"]]]
                            repeats = 1 if phase == "pilot" else spec[phase]["initialization_seeds"]
                            for index in range(repeats):
                                for rate in rates:
                                    rows.append(_identity(dict(common, kind="gradient", model=model["id"],
                                                n_train=size, initialization_index=index, learning_rate=rate)))
    return rows


def _audit_optimizer(summary: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    manifest = summary["optimizer_parameters"]
    names = [entry["name"] for entry in manifest]
    if len(names) != len(set(names)) or sum(entry["count"] for entry in manifest) != summary["parameter_count"]:
        raise ValueError("Optimizer manifest count or membership mismatch")
    groups = artifacts["optimizer_state"]["param_groups"]
    ids = [index for group in groups for index in group["params"]]
    if len(ids) != len(set(ids)) or len(ids) != len(names):
        raise ValueError("Optimizer state omits or duplicates parameters")
    if any(group["weight_decay"] != 0 or group["lr"] != summary["learning_rate"] for group in groups):
        raise ValueError("Native decay or learning rate differs from declared experiment")
    if {group["name"] for group in groups} != set(names):
        raise ValueError("Optimizer state membership differs from manifest")
    for group in groups:
        if summary["optimizer"] == "Adam":
            if tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8 or group["amsgrad"]:
                raise ValueError("Adam numerical configuration mismatch")
        elif group["momentum"] != 0:
            raise ValueError("SGD momentum must be zero")
    if len(summary["update_history"]) != summary["updates_completed"]:
        raise ValueError("Missing recorded updates")
    for row in summary["update_history"]:
        if not np.isfinite(row["gradient_norm_before_clip"]) or any(not np.isfinite(v) for v in row["parameter_gradient_norms"].values()):
            raise ValueError("Nonfinite stored gradient diagnostics")


def _audit_initial_state(row: Mapping[str, Any], artifacts: Mapping[str, Any],
                         task_manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if row["kind"] != "gradient":
        return
    state = artifacts["initial_state"]
    if row["stage"] == "A":
        p = task_manifest["function"]["parameters"]
        x = np.array([[-math.pi], [-0.61], [0.0], [1.23]], dtype=np.float64)
        expected = p["amplitude_before"] * np.cos(x + p["phase_before"])
        _assert_close(reconstruct_prediction(row["model"], state, x, gradient=True), expected,
                      label="Known prechange initial function", atol=1e-6, rtol=1e-5)
        return
    metadata = task_manifest["metadata"]
    root = config["seeds"][row["phase"]]["initialization"][row["initialization_index"]]
    components = [root, 2, config["B"]["family_ids"][row["family"]], row["replicate"], 4]
    seed = int(np.random.SeedSequence(components, spawn_key=(0,)).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63)-1)
    rng = np.random.default_rng(seed)
    rotation = rng.uniform(0, 2 * math.pi)
    centers = ((-math.pi + 2 * math.pi * np.arange(16) / 16 + rotation + math.pi) % (2 * math.pi) - math.pi)[:, None]
    readout = rng.normal(0, .25, (1, 16))
    starts = {}
    for model, inputs, hidden in (("sincos_mlp", 2, 12), ("raw_angle_mlp", 1, 16)):
        first, second = 1 / math.sqrt(inputs), 1 / math.sqrt(hidden)
        starts[model] = {"hidden.weight": rng.uniform(-first, first, (hidden, inputs)),
                         "hidden.bias": rng.uniform(-first, first, hidden),
                         "readout.weight": rng.uniform(-second, second, (1, hidden)),
                         "readout.bias": rng.uniform(-second, second, 1)}
    if row["model"] in starts:
        for name, expected in starts[row["model"]].items():
            _assert_close(state[name], expected.astype(np.float32), label=f"Initial {name}", atol=0, rtol=0)
    else:
        _assert_close(state["bank.centers"], centers.astype(np.float32), label="Initial kernel centers", atol=0, rtol=0)
        _assert_close(state["readout.weight"], readout.astype(np.float32), label="Shared signed readout", atol=0, rtol=0)
        _assert_close(state["readout.bias"], np.zeros(1), label="Initial readout bias", atol=0, rtol=0)
        effective = np.logaddexp(0, _array(state["bank.raw_kappa"]).astype(np.float64))
        _assert_close(effective, np.full((16, 1), 4.), label="Initial effective kernel widths", atol=1e-6)
    if metadata["replicate_id"] != row["replicate"]:
        raise ValueError("Initialization used the wrong paired task")


def audit_fit(row: Mapping[str, Any], artifacts: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
              task_manifest: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce a complete fit's saved predictions, metrics and selection."""
    summary, model = row["summary"], row["model"]
    linear_diagnostics = []
    if summary["status"] != "completed":
        raise ValueError("Incomplete fit cannot be audited as completed")
    if row["kind"] == "gradient":
        if summary["learning_rate"] != row["learning_rate"] or summary["training_dtype"] != "float32" or summary["metric_dtype"] != "float64":
            raise ValueError("Fit learning rate or numerical dtype differs from matrix")
        if row["stage"] == "A":
            requested = config["A"]["pilot"]["updates"] if row["phase"] == "pilot" else config["A"]["updates"]
            if summary["updates_requested"] != requested or summary["record_every"] != config["A"]["record_every"] or summary["penalty"] != row["penalty"]:
                raise ValueError("A update budget or explicit penalty differs from protocol")
        else:
            spec = config["B"]["optimizer"]
            requested = config["B"]["pilot"]["max_updates"] if row["phase"] == "pilot" else spec["max_updates"]
            if summary["max_updates"] != requested or summary["penalty"] != spec["readout_penalty"] or summary["gradient_clip"] != config["training_numeric"]["B_gradient_clip_norm"]:
                raise ValueError("B update budget, clipping or penalty differs from protocol")
            if any(summary[key] != spec[key] for key in ("batch_size", "validation_every", "patience_evaluations", "min_delta")):
                raise ValueError("B validation/stopping configuration differs from protocol")
    _audit_initial_state(row, artifacts, task_manifest, config)
    gradient = row["kind"] == "gradient"
    if gradient:
        _audit_optimizer(summary, artifacts)
    state = {"coefficients": artifacts["coefficients"]} if model == "reference" else artifacts["selected_state"]
    max_prediction_difference = 0.
    for split, saved in artifacts["predictions"].items():
        rebuilt = reconstruct_prediction(model, state, arrays[f"{split}_x"], gradient=gradient)
        # Ridge lambda=0 may be ill-conditioned; its saved predictions are still
        # used for exact metric replay, with float64 feature parity separately.
        _assert_close(rebuilt, saved, label=f"Selected {split} prediction", atol=1e-5 if gradient else 1e-7,
                      rtol=1e-4 if gradient else 1e-7)
        max_prediction_difference = max(max_prediction_difference, float(np.max(np.abs(rebuilt-_array(saved)))))
    metrics = recompute_metrics(artifacts["predictions"]["grid"], arrays)
    recorded_metrics = summary["final_metrics"] if row["stage"] == "A" and gradient else summary["metrics"]
    for key, value in metrics.items():
        _assert_close(recorded_metrics[key], value, label=f"Final metric {key}")

    primary_error = metrics["normalized_clean_mse"]
    if row["stage"] == "B":
        _assert_close(summary["primary_error"], primary_error, label="B normalized primary error")
    if row["stage"] == "A" and gradient:
        history = summary["history"]
        expected_steps = list(range(0, summary["updates_requested"] + 1, summary["record_every"]))
        if [item["step"] for item in history] != expected_steps or summary["updates_completed"] != summary["updates_requested"]:
            raise ValueError("A history or fixed update budget is incomplete")
        for item in history:
            step = str(item["step"])
            recorded = artifacts["recorded_predictions"][step]
            for split in ("train", "validation", "grid"):
                rebuilt = reconstruct_prediction(model, artifacts["recorded_states"][step], arrays[f"{split}_x"], gradient=True)
                _assert_close(rebuilt, recorded[split], label=f"A step {step} {split}", atol=1e-5, rtol=1e-4)
                target = arrays["grid_clean"] if split == "grid" else arrays[f"{split}_y"]
                expected_mse = float(np.mean((_array(recorded[split])-target) ** 2))
                _assert_close(item["clean_mse" if split == "grid" else f"{split}_mse"], expected_mse, label=f"A curve {step}/{split}")
        variance = metrics["clean_variance"]
        trapezoid = getattr(np, "trapezoid", np.trapz)
        primary_error = float(trapezoid([item["clean_mse"] for item in history], expected_steps) / (summary["updates_requested"]*variance))
        validation_area = float(trapezoid([item["validation_mse"] for item in history], expected_steps) / (summary["updates_requested"]*variance))
        _assert_close(summary["primary_error"], primary_error, label="A clean integrated primary error")
        _assert_close(summary["validation_curve_area_normalized"], validation_area, label="A pilot validation integrated error")
    elif row["stage"] == "B" and gradient:
        history = summary["history"]
        if [item["step"] for item in history] != list(range(0, summary["updates_completed"] + 1, summary["validation_every"])):
            raise ValueError("B checkpoint evaluations do not follow the fixed interval")
        selected, best, stale = replay_validation_selection(history, summary["min_delta"], summary["patience_evaluations"])
        if selected != summary["selected_step"] or summary["best_validation_mse"] != best:
            raise ValueError("Selected B checkpoint differs from replay")
        if summary["stop_reason"] == "patience":
            if stale < summary["patience_evaluations"]:
                raise ValueError("B stopped before declared patience")
        elif summary["updates_completed"] != summary["max_updates"]:
            raise ValueError("B stopped without patience or full budget")
        for item in history:
            step = str(item["step"])
            saved = artifacts["validation_predictions"][step]
            rebuilt = reconstruct_prediction(model, artifacts["checkpoint_states"][step], arrays["validation_x"], gradient=True)
            _assert_close(rebuilt, saved, label=f"B validation state {step}", atol=1e-5, rtol=1e-4)
            _assert_close(item["validation_mse"], float(np.mean((_array(saved)-arrays["validation_y"]) ** 2)), label=f"B validation score {step}")
        for name, value in artifacts["selected_state"].items():
            _assert_close(value, artifacts["checkpoint_states"][str(selected)][name], label="Selected state bytes", atol=0, rtol=0)
        components = [config["seeds"][row["phase"]]["initialization"][row["initialization_index"]], 2,
                      config["B"]["family_ids"][row["family"]], row["replicate"], 4]
        seed = int(np.random.SeedSequence(components, spawn_key=(1,)).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63)-1)
        if summary["batch_seed"] != seed:
            raise ValueError("Batch stream is not the prescribed independent child")
        rng = np.random.default_rng(seed)
        count = summary["batch_size"] * summary["max_updates"]
        expected = np.concatenate([rng.permutation(len(arrays["train_x"])) for _ in range(math.ceil(count/len(arrays["train_x"])))])[:count]
        _assert_close(artifacts["batch_indices"], expected.reshape(summary["max_updates"], summary["batch_size"]), label="Shared cyclic batches", atol=0, rtol=0)
    elif row["kind"] == "ridge":
        history = summary["history"]
        if sorted(item["lambda"] for item in history) != config["B"]["ridge"]["lambda_grid"] or summary["solves_completed"] != 4:
            raise ValueError("Ridge omitted prescribed lambda candidates")
        for item in history:
            coefficients = _array(artifacts["candidate_states"][str(item["lambda"])]).astype(np.float64)
            linear_diagnostics.append(dict(_audit_linear_optimum(_fixed_features(model, arrays["train_x"]), arrays["train_y"], coefficients, item["lambda"]),penalty=item["lambda"]))
            predicted = _fixed_features(model, arrays["validation_x"]) @ coefficients[:-1] + coefficients[-1]
            _assert_close(item["validation_mse"], float(np.mean((predicted-arrays["validation_y"])**2)), label="Ridge validation replay", atol=1e-7, rtol=1e-7)
            if item["alpha"] != len(arrays["train_x"]) * item["lambda"]:
                raise ValueError("Ridge alpha must be n_train times lambda")
        selected = select_ridge_lambda(history)
        if summary["selected_lambda"] != selected:
            raise ValueError("Ridge selection violates smallest-lambda tie rule")
        _assert_close(state["coefficients"], artifacts["candidate_states"][str(selected)], label="Selected ridge coefficients", atol=0, rtol=0)
    elif model == "reference":
        x = arrays["train_x"]
        # Rebuild the exact Torch float64 design used by the original solve.
        angles = torch.from_numpy(x.copy())
        features = torch.cat((angles.cos(),angles.sin()),dim=1).numpy()
        linear_diagnostics.append(dict(_audit_linear_optimum(features, arrays["train_y"], artifacts["coefficients"], row["penalty"]),penalty=row["penalty"]))
    return {"primary_error": primary_error, "metrics": metrics,
            "maximum_prediction_reconstruction_difference": max_prediction_difference,
            "selection_replayed": True,"linear_solver_diagnostics":linear_diagnostics}


def _audit_task(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
                 config: Mapping[str, Any], stage: str, phase: str,
                 family: str, replicate: int) -> None:
    """Rebuild observation/noise streams and target equations from saved parameters."""
    function, metadata = manifest["function"], manifest["metadata"]
    p, norm = function["parameters"], function["normalization"]
    if manifest["function_sha256"] != object_hash(function):
        raise ValueError("Function parameter hash mismatch")
    root = config["seeds"][phase]["observation"][replicate]
    family_id = 0 if stage == "A" else config["B"]["family_ids"][family]
    if metadata["task_root"] != config["seeds"][phase]["task"][replicate] or metadata["observation_root"] != root:
        raise ValueError("Task/observation roots are not paired by registry position")
    n = config["A"]["train_observations"] if stage == "A" else (config["B"]["pilot"]["train_observations"] if phase == "pilot" else max(config["B"]["train_observations"]))
    if arrays["train_x"].shape != (n,1) or arrays["validation_x"].shape != (config["data"]["validation_observations"],1):
        raise ValueError("Observation split sizes differ from protocol")
    points = config["data"]["clean_evaluation_grid"]["points"]
    grid = (-math.pi+(np.arange(points)+.5)*(2*math.pi/points))[:,None]
    _assert_close(arrays["grid_x"], grid, label="Clean midpoint grid", atol=0, rtol=0)
    noise = (.1*p["amplitude_after"]/math.sqrt(2) if stage == "A" else
             config["B"]["pilot_noise_std"] if phase == "pilot" else config["B"]["noise_std_by_replicate_parity"][replicate%2])
    _assert_close(manifest["noise_std"],noise,label="Declared observation noise")

    def target(theta: np.ndarray) -> np.ndarray:
        if stage == "A":
            raw = p["amplitude_after"]*np.cos(theta+p["phase_after"])+p["bias"]
        elif family == "smooth":
            raw = sum(a*np.cos(h*theta)+b*np.sin(h*theta) for h,a,b in zip(p["harmonics"],p["cos_coefficients"],p["sin_coefficients"]))
        elif family == "nonperiodic":
            raw = theta/math.pi+.2*np.sin(theta+p["phase"])
        else:
            raw = np.zeros_like(theta)
            for center,width,amplitude in zip(p["centers"],p["widths"],p["amplitudes"]):
                d = (theta-center+math.pi)%(2*math.pi)-math.pi
                profile = np.maximum(1-(d/width)**2,0)**3 if family == "narrow" else np.maximum(1-np.abs(d)/width,0)
                raw += amplitude*profile
        return (raw-norm["mean"])/norm["scale"]

    for split,split_id in (("train",1),("validation",2)):
        stream = [root,1 if stage=="A" else 2,family_id,replicate,split_id]
        if metadata["stream_components"][split] != stream:
            raise ValueError("Recorded observation stream components differ from protocol")
        rng = np.random.default_rng(np.random.SeedSequence(stream))
        x = rng.uniform(-math.pi,math.pi,arrays[f"{split}_x"].shape)
        _assert_close(arrays[f"{split}_x"],x,label=f"Independent {split} angles",atol=0,rtol=0)
        clean = target(x)
        _assert_close(arrays[f"{split}_y"],clean+noise*rng.standard_normal(x.shape),label=f"Independent {split} response")
        _assert_close(arrays[f"{split}_clean"],clean,label=f"Independent {split} clean target")
    clean = target(grid)
    _assert_close(arrays["grid_clean"],clean,label="Independent clean function")
    rng = np.random.default_rng(np.random.SeedSequence([root,1 if stage=="A" else 2,family_id,replicate,3]))
    _assert_close(arrays["grid_noisy"],clean+noise*rng.standard_normal(grid.shape),label="Separate test noise")
    if stage == "B":
        points = config["B"]["generator_normalization"]["grid_points"]
        generator_grid = (-math.pi+(np.arange(points)+.5)*(2*math.pi/points))[:,None]
        scaled = target(generator_grid)
        _assert_close(np.mean(scaled),0.,label="Generator mean",atol=1e-12)
        _assert_close(np.var(scaled),1.,label="Generator variance",atol=1e-12)


@threadpool_limits.wrap(limits=1)
def audit_directory(directory: Path) -> dict[str, Any]:
    """Check every matrix cell and archived dependency, retaining audit failures."""
    path = directory / "manifest.json"
    manifest = read_json(path)
    config, errors, rows = manifest["config"], [], []
    if manifest["config_hash"] != object_hash(config):
        errors.append("Manifest configuration hash mismatch")
    if set(manifest["source_hashes"]) != set(SOURCE_PATHS):
        errors.append("Incomplete source dependency manifest")
    if manifest["phase"] == "confirmation":
        lock = verify_confirmation_lock(Path(manifest["confirmation_lock"]), config)
        if lock["lock_sha256"] != manifest["confirmation_lock_sha256"]:
            raise ValueError("Confirmation analysis lock differs from generated tasks")
    for name, digest in manifest["source_hashes"].items():
        snapshot = directory / "sources" / name
        if not snapshot.is_file() or file_hash(snapshot) != digest:
            errors.append(f"Changed source snapshot: {name}")
    expected = Counter(_expected_cells(config, manifest["stage"], manifest["phase"], manifest["pilot_hyperparameters"]))
    actual = Counter(_identity(row) for row in manifest["runs"])
    if actual != expected:
        errors.append("Run matrix has missing, duplicate or unplanned cells")
    task_cache = {}
    for row in manifest["runs"]:
        evidence = {key: row[key] for key in row if key not in {"summary", "artifact_hashes"}}
        evidence["audited"] = False
        if row["status"] != "completed":
            evidence["audit_error"] = f"Planned run {row['status']}"
            errors.append(f"{row['run_id']}: {evidence['audit_error']}")
            rows.append(evidence)
            continue
        try:
            key = f"{row['family']}_{row['replicate']}"
            if key not in task_cache:
                task_dir = directory / "tasks" / key
                for name, digest in manifest["tasks"][key].items():
                    if file_hash(task_dir / name) != digest:
                        raise ValueError(f"Changed task artifact: {key}/{name}")
                task_manifest = read_json(task_dir / "manifest.json")
                with np.load(task_dir / "data.npz", allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                expected_arrays = {"train_x","train_y","train_clean","validation_x","validation_y","validation_clean","grid_x","grid_clean","grid_noisy"}
                if set(arrays) != expected_arrays or set(task_manifest["arrays"]) != expected_arrays:
                    raise ValueError("Missing or unexpected task arrays")
                for name, array in arrays.items():
                    if array.dtype != np.float64 or array.ndim != 2 or array.shape[1] != 1 or not np.isfinite(array).all():
                        raise ValueError(f"Invalid task-array dtype, shape or finite values: {name}")
                    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    if digest != task_manifest["arrays"][name]["sha256"] or list(array.shape) != task_manifest["arrays"][name]["shape"]:
                        raise ValueError(f"Task array manifest mismatch: {name}")
                _audit_task(arrays, task_manifest, config, manifest["stage"], manifest["phase"], row["family"], row["replicate"])
                task_cache[key] = arrays, task_manifest
            arrays, task_manifest = task_cache[key]
            arrays = {name: values[:row["n_train"]] if name.startswith("train_") and "n_train" in row else values
                      for name, values in arrays.items()}
            for name, digest in row["artifact_hashes"].items():
                if file_hash(directory / name) != digest:
                    raise ValueError(f"Changed fit artifact: {name}")
            fit_dir = directory / "fits" / row["run_id"]
            if read_json(fit_dir / "summary.json") != row["summary"]:
                raise ValueError("Summary differs from manifest copy")
            artifacts = torch.load(fit_dir / "artifacts.pt", map_location="cpu", weights_only=True)
            evidence.update(audit_fit(row, artifacts, arrays, task_manifest, config), audited=True,
                            noise_std=task_manifest["noise_std"], summary=row["summary"])
        except Exception as error:
            evidence["audit_error"] = f"{type(error).__name__}: {error}"
            errors.append(f"{row['run_id']}: {evidence['audit_error']}")
        rows.append(evidence)
    return {"stage": manifest["stage"], "phase": manifest["phase"], "config": config,
            "complete": not errors and manifest["status"] == "completed", "audit_errors": errors,
            "planned_cells": sum(expected.values()), "audited_cells": sum(row["audited"] for row in rows),
            "rows": rows, "manifest_path": str(path.resolve()), "manifest_sha256": file_hash(path),
            "recorded_invocation_seconds": manifest["time_seconds"],
            "observer_threadpools":threadpool_info(),
            "linear_solver_diagnostics":[{"run_id":row["run_id"],"model":row["model"],"family":row["family"],
                "replicate":row["replicate"],"n_train":row.get("n_train"),"candidates":row["linear_solver_diagnostics"]}
                for row in rows if row.get("linear_solver_diagnostics")],
            "observer_correction":"BLAS threads fixed at1 to replay original arithmetic; absolute stationarity replaced by recorded dimension-scaled normwise float64 backward error plus exact coefficient replay. Training and methodological gates unchanged."}


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def balanced_b(rows: Sequence[Mapping[str, Any]], families: Sequence[str]) -> dict[str, Any]:
    """Average initializations and sample sizes within task, then families equally."""
    grouped: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sizes: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("audited") and row["stage"] == "B":
            grouped[row["model"]][row["family"]][row["replicate"]].append(row["primary_error"])
            sizes[row["model"]][row["family"]][row["n_train"]].append(row["primary_error"])
    result = {}
    for model, family in grouped.items():
        tasks = {name: {str(task): _mean(values) for task, values in replicates.items()} for name, replicates in family.items()}
        family_means = {name: _mean(list(values.values())) for name, values in tasks.items()}
        result[model] = {"task_errors": tasks, "family_errors": family_means,
                         "per_size_family_errors": {name: {str(n): _mean(values) for n, values in ns.items()} for name, ns in sizes[model].items()},
                         "balanced_periodic_error": _mean([family_means[name] for name in families]) if all(name in family_means for name in families) else None}
        noise = {}
        for level in (.05,.20):
            subgroup = {name:{task:value for task,value in values.items() if int(task)%2 == (0 if level==.05 else 1)}
                        for name,values in tasks.items()}
            means = {name:_mean(list(values.values())) for name,values in subgroup.items() if values}
            noise[str(level)] = {"family_errors":means,
                "balanced_periodic_error":_mean([means[name] for name in families]) if all(name in means for name in families) else None,
                "task_counts":{name:len(values) for name,values in subgroup.items()},
                "interpretation":"Descriptive parity subgroups; noise is confounded with task identity, not a factorial noise effect"}
        result[model]["noise_subgroups"] = noise
    return result


def replay_pilot_rates(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Recompute complete-candidate noisy-validation pilot choices independently."""
    groups = list(config["A"]["optimizers"]) if stage=="A" else [item["id"] for item in config["B"]["models"] if item["kind"]=="gradient"]
    candidates, choices = {}, {}
    for group in groups:
        rates = config["A"]["optimizers"][group]["pilot_learning_rates"] if stage=="A" else config["B"]["optimizer"]["pilot_learning_rates"]
        entries = []
        for rate in rates:
            cells = [row for row in rows if row["kind"]=="gradient" and row["learning_rate"]==rate and row["optimizer" if stage=="A" else "model"]==group]
            eligible = len(cells)==(8 if stage=="A" else 4) and all(row.get("audited") for row in cells)
            score = _mean([row["summary"]["validation_curve_area_normalized" if stage=="A" else "best_validation_mse"] for row in cells]) if eligible else None
            entries.append({"learning_rate":rate,"eligible":eligible,"mean_score":score})
        finite = [entry for entry in entries if entry["eligible"]]
        if finite:
            best = min(entry["mean_score"] for entry in finite)
            choices[group] = min(entry["learning_rate"] for entry in finite if entry["mean_score"]<=best+1e-12)
        candidates[group] = entries
    return {"complete_selection":len(choices)==len(groups),"choices":choices,"candidates":candidates}


def _improvement(vm: float, comparator: float) -> float | None:
    return 1 - vm/comparator if comparator > 0 else None


def bootstrap_improvement(pairs: Mapping[str, tuple[Sequence[float], Sequence[float]]],
                           *, seed: int, resamples: int = 10000, confidence: float = .95,
                           stage: str) -> dict[str, Any]:
    """Resample paired whole-task bundles independently within each family."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1 if stage == "A" else 2, 0, 0, 5]))
    first, second = [], []
    for family, (vm, control) in pairs.items():
        a, b = np.asarray(vm, dtype=np.float64), np.asarray(control, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid paired task bundle for {family}")
        indices = rng.integers(0, len(a), size=(resamples, len(a)))
        first.append(a[indices].mean(axis=1))
        second.append(b[indices].mean(axis=1))
    a, b = np.mean(first, axis=0), np.mean(second, axis=0)
    if np.any(b <= 0):
        return {"lower": None, "upper": None, "reason": "Nonpositive resampled comparator error",
                "resamples": resamples, "confidence": confidence}
    improvement = 1 - a/b
    q = [(1-confidence)/2, (1+confidence)/2]
    low, high = np.quantile(improvement, q)
    return {"lower": float(low), "upper": float(high), "resamples": resamples,
            "confidence": confidence, "method": "percentile; whole paired task bundles"}


def analyze_a(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], complete: bool) -> dict[str, Any]:
    """Report every A cell and its fixed Adam/no-penalty phase-bearing contrast."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("audited") and row["stage"] == "A" and row["kind"] == "gradient":
            groups[(row["family"], row["optimizer"], row["penalty"], row["model"])].append(row)
    cells = [{"condition": key[0], "optimizer": key[1], "penalty": key[2], "model": key[3],
              "mean_primary_error": _mean([row["primary_error"] for row in values]),
              "task_errors": {str(row["replicate"]): row["primary_error"] for row in values},
              "sample_sd": float(np.std([row["primary_error"] for row in values], ddof=1)) if len(values)>1 else None}
             for key, values in sorted(groups.items())]
    primary = config["A"]["primary"]
    primary_rows = [row for rows in groups.values() for row in rows if row["family"] in primary["conditions"]
                    and row["optimizer"] == primary["optimizer"] and row["penalty"] == primary["penalty"]]
    bundles: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    conditions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in primary_rows:
        bundles[row["model"]][row["replicate"]].append(row["primary_error"])
        conditions[row["family"]][row["model"]].append(row["primary_error"])
    result: dict[str, Any] = {"cells": cells, "complete": complete, "gate_passed": False,
                             "interpretation": "incomplete; no complete success claim"}
    if not complete or set(bundles) != {"polar", "cartesian"}:
        return result
    ids = sorted(bundles["polar"])
    if ids != sorted(bundles["cartesian"]) or any(len(values) != len(primary["conditions"]) for model in bundles.values() for values in model.values()):
        result["complete"] = False
        return result
    polar = [_mean(bundles["polar"][i]) for i in ids]
    cartesian = [_mean(bundles["cartesian"][i]) for i in ids]
    improvement = _improvement(_mean(polar), _mean(cartesian))
    condition_improvements = {family: _improvement(_mean(values["polar"]), _mean(values["cartesian"])) for family, values in conditions.items()}
    bootstrap = bootstrap_improvement({"paired_all_primary_conditions": (polar, cartesian)}, seed=config["seeds"]["analysis"],
                          resamples=config["analysis"]["bootstrap"]["resamples"], confidence=config["analysis"]["bootstrap"]["confidence"], stage="A")
    gate = config["analysis"]["A_gate"]
    passed = improvement is not None and improvement >= gate["relative_improvement_min"] and all(value is not None and value > 0 for value in condition_improvements.values()) and bootstrap["lower"] is not None and bootstrap["lower"] > 0
    result.update(balanced_polar_error=_mean(polar), balanced_cartesian_error=_mean(cartesian),
                  paired_task_differences=[a-b for a,b in zip(polar,cartesian)],
                  relative_improvement=improvement, condition_improvements=condition_improvements,
                  bootstrap=bootstrap, gate_passed=passed,
                  interpretation="prespecified continuation gate passed" if passed else "prespecified continuation gate not met")
    return result


def analyze_b(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], complete: bool,
               comparator: str | None = None) -> dict[str, Any]:
    """Select the development comparator or assess the already locked comparator."""
    families = config["B"]["primary_gate_families"]
    models = balanced_b(rows, families)
    proper = config["B"]["proper_comparators"]
    result: dict[str, Any] = {"models": models, "complete": complete, "gate_passed": False,
                             "primary_comparator": comparator, "interpretation": "incomplete; no complete success claim"}
    if not complete or any(name not in models or models[name]["balanced_periodic_error"] is None for name in ["learned_vm", *proper]):
        return result
    if comparator is None:
        # The canonical rule supplies no special comparator tie tolerance.
        # Stable config order resolves an exact numerical tie deterministically.
        comparator = min(proper, key=lambda name: models[name]["balanced_periodic_error"])
    if comparator not in proper:
        raise ValueError("Comparator is not an eligible periodic control")
    vm, control = models["learned_vm"], models[comparator]
    improvement = _improvement(vm["balanced_periodic_error"], control["balanced_periodic_error"])
    family_improvements = {family: _improvement(vm["family_errors"][family], control["family_errors"][family]) for family in families}
    pairs = {}
    for family in families:
        ids = sorted(vm["task_errors"][family], key=int)
        if ids != sorted(control["task_errors"][family], key=int):
            result["complete"] = False
            return result
        pairs[family] = ([vm["task_errors"][family][i] for i in ids], [control["task_errors"][family][i] for i in ids])
    bootstrap = bootstrap_improvement(pairs, seed=config["seeds"]["analysis"],
                 resamples=config["analysis"]["bootstrap"]["resamples"], confidence=config["analysis"]["bootstrap"]["confidence"], stage="B")
    gate = config["analysis"]["B_gate"]
    passed = (improvement is not None and improvement >= gate["relative_improvement_min"]
              and sum(value is not None and value>0 for value in family_improvements.values()) >= gate["periodic_families_improving_min"]
              and all(value is not None and value >= -gate["periodic_family_relative_regression_max"] for value in family_improvements.values())
              and bootstrap["lower"] is not None and bootstrap["lower"] > 0)
    result.update(primary_comparator=comparator, relative_improvement=improvement,
                  family_improvements=family_improvements, bootstrap=bootstrap, gate_passed=passed,
                  comparisons={name: _improvement(vm["balanced_periodic_error"], values["balanced_periodic_error"]) for name,values in models.items() if name != "learned_vm"},
                  paired_task_differences={family:[a-b for a,b in zip(*values)] for family,values in pairs.items()},
                  interpretation="prespecified continuation gate passed" if passed else "prespecified continuation gate not met")
    return result


def analyze_directories(directories: Sequence[Path]) -> dict[str, Any]:
    """Audit same-phase A/B inputs, retain failures and return an honest report."""
    audits = [audit_directory(path) for path in directories]
    if not audits:
        raise ValueError("At least one saved stage is required")
    config = audits[0]["config"]
    if any(audit["config"] != config for audit in audits):
        raise ValueError("Cannot pool different experiment configurations")
    phases = {audit["phase"] for audit in audits}
    if len(phases) != 1 or len({audit["stage"] for audit in audits}) != len(audits):
        raise ValueError("Provide at most one A and one B directory from a single phase")
    phase = audits[0]["phase"]
    complete = all(audit["complete"] for audit in audits)
    result: dict[str, Any] = {"protocol_id":config["protocol_id"], "phase":phase,"complete":complete,
        "B_primary_comparator":None, "input_manifest_hashes":{audit["manifest_path"]:audit["manifest_sha256"] for audit in audits},
        "audits":[{key:value for key,value in audit.items() if key not in {"config","rows"}} for audit in audits],
        "claim_scope":"Synthetic mechanism tests; no novelty, universal superiority, powered population inference, or deployment claim",
        "curve_records":"Complete histories, states and diagnostic metrics remain in the hashed per-fit artifacts",
        "A":None,"B":None}
    for audit in audits:
        stage = audit["stage"]
        if phase == "pilot":
            # Pilot rows repeat learning rates; they are never pooled as if
            # independent tasks or used to select a primary comparator.
            result[stage] = {"complete":audit["complete"],"methodological_gate_evaluated":False,
                             "reason":"Disposable hyperparameter/feasibility pilot",
                             "pilot_selection_replay":replay_pilot_rates(audit["rows"],config,stage)}
        elif stage == "A":
            result["A"] = analyze_a(audit["rows"],config,audit["complete"])
        else:
            comparator = None
            if phase == "confirmation":
                manifest = read_json(Path(audit["manifest_path"]))
                comparator = verify_confirmation_lock(Path(manifest["confirmation_lock"]),config)["B_primary_comparator"]
            result["B"] = analyze_b(audit["rows"],config,audit["complete"],comparator)
            result["B_primary_comparator"] = result["B"]["primary_comparator"]
    if phase == "development":
        result["development_gate_status"] = "Descriptive only; confirmation decisions remain locked and separate"
    return result


def main() -> int:
    """Audit saved directories and write a new strict-JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze_directories(args.inputs)
    write_json(args.output,result)
    print(f"Independent audit complete={result['complete']}; report={args.output}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
