"""Independently replay saved VM-N02 evidence in NumPy, without fitting or draws.

Only artifact-integrity helpers are reused from the runner. Existing payloads
are loaded after their hashes pass. Historical unselected checkpoint states were
not retained, so their validation scores can be checked for selection consistency
but cannot be independently recomputed. No scientific pool is regenerated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_vm_n02_interpretation import read_json, sha256, verify_freeze, verify_lock, verify_stage, write_json


def numpy(value: Any) -> Any:
    """Detach saved tensors recursively; scientific calculations use NumPy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numpy(item) for item in value]
    return value


class Observer:
    """Accumulate all numerical checks without hiding later failures."""

    def __init__(self) -> None:
        self.checks = 0
        self.maximum_absolute_error = 0.0
        self.failures: list[dict[str, Any]] = []

    def check(self, label: str, actual: Any, expected: Any, tolerance: float = 1e-9) -> None:
        """Compare shape and finite entries with a fixed absolute tolerance."""
        self.checks += 1
        if actual is None or expected is None:
            ok, error = actual is None and expected is None, 0.0
        else:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            ok = a.shape == b.shape and np.array_equal(np.isfinite(a), np.isfinite(b))
            error = float(np.max(np.abs(a[np.isfinite(a)] - b[np.isfinite(b)]), initial=0)) if ok else math.inf
            ok = ok and error <= tolerance
        if math.isfinite(error):
            self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        if not ok:
            self.failures.append({"check": label, "maximum_absolute_error": error, "tolerance": tolerance})


def components(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Use the direct cosine equation, independently of the fitted sin² API."""
    return np.exp(state["kappa"] * (np.cos(x[:, None] - state["centers"]) - 1)) * state["weights"]


def response(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Return the sum of complete signed contributions plus intercept."""
    return components(state, x).sum(axis=1) + state["bias"]


def derivative(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Differentiate the independent direct-cosine representation analytically."""
    return (-components(state, x) * state["kappa"] * np.sin(x[:, None] - state["centers"])).sum(axis=1)


def mse(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean squared error in the preserved output scale."""
    return float(np.mean((left - right) ** 2))


def distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return shortest unsigned circular distance."""
    return np.abs(np.arctan2(np.sin(left - right), np.cos(left - right)))


def audit_peak(obs: Observer, label: str, unit: dict[str, Any], states: list[dict[str, Any]],
               metrics: dict[str, Any], details: dict[str, Any], arrays: Any, config: dict[str, Any]) -> None:
    """Rebuild curves, interventions, matching, recovery and exact identities."""
    rule = config["diagnostics"]
    x, xe, target = unit["curve"]["x"], unit["evaluation"]["x"], unit["evaluation"]["clean"]
    variance, rows, pair_metrics = float(np.var(target)), [], []
    sd = math.sqrt(variance)
    curves = [response(state, x) for state in states]
    obs.check(label + ".curves", arrays["curves"], curves)
    for split in ("train", "validation", "evaluation", "curve"):
        obs.check(label + ".generator_" + split, response(unit["truth"], unit[split]["x"]), unit[split]["clean"])
    for i, state in enumerate(states):
        prefix, saved = f"{label}.fit{i}", details["fits"][i]
        pred, comp, deriv = response(state, xe), components(state, xe), derivative(state, x)
        curve_comp = components(state, x)
        active = np.sqrt(np.mean(curve_comp ** 2, axis=0)) / sd >= rule["active_component_rms_over_target_sd"]
        eligible = active & (np.std(curve_comp, axis=0) / sd >= rule["phase_component_sd_over_target_sd"])
        select = int(np.argmax(np.mean(components(state, unit["validation"]["x"]) ** 2, axis=0)))
        nmse = mse(pred, target) / variance
        deletion = [mse(pred - comp[:, j], target) / variance - nmse for j in range(comp.shape[1])]
        row = {"clean_nmse": nmse, "clean_nrmse": math.sqrt(nmse),
               "noisy_rmse": math.sqrt(mse(pred, unit["evaluation"]["y"])),
               "primary_deletion_delta_nmse": deletion[select], "active_components": float(active.sum()),
               "phase_eligible_components": float(eligible.sum()), "unsplit_first_deletion_delta_nmse": deletion[0]}
        obs.check(prefix + ".selected_component", saved["validation_selected_component"], select)
        obs.check(prefix + ".all_deletions", saved["all_deletion_delta_nmse"], deletion)
        obs.check(prefix + ".derivative", arrays[f"derivative_{i}"], deriv)
        obs.check(prefix + ".components", arrays[f"components_{i}"], curve_comp)
        val_mse = mse(response(state, unit["validation"]["x"]), unit["validation"]["y"])
        history = state["validation_history"]
        best = min(history, key=lambda checkpoint: checkpoint["validation_mse"])
        obs.check(prefix + ".init_seed", state["init_seed"], config["initializations"][i])
        obs.check(prefix + ".checkpoint_schedule", [checkpoint["step"] for checkpoint in history],
                  list(range(0, config["peak"]["steps"] + 1, config["peak"]["validate_every"])))
        obs.check(prefix + ".selected_step", state["selected_step"], best["step"])
        obs.check(prefix + ".validation_history_min", state["validation_mse"], best["validation_mse"])
        obs.check(prefix + ".validation_replay", state["validation_mse"], val_mse)
        for name, offset in (("zero", 0), ("plus", rule["phase_offset_radians"]), ("minus", -rule["phase_offset_radians"])):
            changed = {**state, "centers": state["centers"].copy()}
            changed["centers"][select] += offset
            row[f"phase_{name}_delta_nmse"] = mse(response(changed, xe), target) / variance - nmse
        for factor in rule["width_factors"]:
            changed = {**state, "kappa": state["kappa"].copy()}
            changed["kappa"][select] *= factor
            row[f"width_{str(factor).replace('.', 'p')}_delta_nmse"] = mse(response(changed, xe), target) / variance - nmse
        geometries = {"permutation": {**state, **{k: state[k][::-1] for k in ("centers", "kappa", "weights")}},
                      "split": {**state, **{k: np.r_[state[k][:1], state[k]] for k in ("centers", "kappa", "weights")}},
                      "cancelling_pair": {**state, **{k: np.r_[state[k], state[k][:1], state[k][:1]] for k in ("centers", "kappa", "weights")}}}
        geometries["split"]["weights"][:2] = state["weights"][0] / 2
        geometries["cancelling_pair"]["weights"][-2:] = [.5, -.5]
        row["split_half_deletion_delta_nmse"] = mse(pred - components(geometries["split"], xe)[:, 0], target) / variance - nmse
        row["inserted_cancel_component_deletion_delta_nmse"] = mse(pred - components(geometries["cancelling_pair"], xe)[:, -2], target) / variance - nmse
        for name, changed in geometries.items():
            obs.check(prefix + "." + name, response(changed, x), curves[i], rule["identity_atol"])
            obs.check(prefix + "." + name + "_derivative", derivative(changed, x), deriv, rule["identity_atol"])
        rot = rule["coordinate_rotation_radians"]
        chord = np.exp(-.5 * state["kappa"] * ((np.cos(x[:, None]) - np.cos(state["centers"])) ** 2
                       + (np.sin(x[:, None]) - np.sin(state["centers"])) ** 2)) @ state["weights"] + state["bias"]
        for name, equivalent in (("wrap", response(state, x + 2 * math.pi)), ("chordal", chord),
                                 ("rotation", response({**state, "centers": state["centers"] + rot}, x + rot))):
            obs.check(prefix + "." + name, equivalent, curves[i], rule["identity_atol"])
        if unit["case"] == "isolated":
            center_error = float(distance(state["centers"], unit["truth"]["centers"])[0])
            width_error = float(abs(state["kappa"] / unit["truth"]["kappa"] - 1)[0])
            allowed = bool(eligible[0]) and state["weights"][0] > 0
            row.update(center_error_rad=center_error, kappa_relative_error=width_error, recovery_eligible=float(allowed),
                       recovered=float(allowed and center_error <= rule["center_error_radians_max"] and width_error <= rule["kappa_relative_error_max"] and row["clean_nrmse"] <= rule["quality_nrmse_max"]))
        for key, value in row.items():
            obs.check(prefix + "." + key, saved["metrics"][key], value)
        for key, value in saved["metrics"].items():
            if key.endswith(("identity_error", "derivative_error")):
                obs.check(prefix + ".saved_" + key, value, 0, rule["identity_atol"])
        rows.append(row)
    for pair_index, (i, j) in enumerate(itertools.combinations(range(len(states)), 2)):
        saved = details["initialization_pairs"][pair_index]
        obs.check(label + f".pair{pair_index}.curve", saved["curve_nrmse"], math.sqrt(mse(curves[i], curves[j])) / sd)
        lc, rc = components(states[i], x), components(states[j], x)
        cost = np.mean((lc[:, :, None] - rc[:, None, :]) ** 2, axis=0)
        rr = np.arange(cost.shape[0])
        cc = np.asarray(min(itertools.permutations(range(cost.shape[1])), key=lambda p: float(cost[rr, p].sum())))
        lrms, rrms = np.sqrt(np.mean(lc ** 2, axis=0)), np.sqrt(np.mean(rc ** 2, axis=0))
        la, ra = lrms / sd >= rule["active_component_rms_over_target_sd"], rrms / sd >= rule["active_component_rms_over_target_sd"]
        active = (la & (np.std(lc, axis=0) / sd >= rule["phase_component_sd_over_target_sd"]))[rr] & (ra & (np.std(rc, axis=0) / sd >= rule["phase_component_sd_over_target_sd"]))[cc]
        weight = (lrms[rr] + rrms[cc]) / 2
        total = weight[active].sum()
        match = {"active_pair_count": int(active.sum()), "unmatched_active_count": int(la.sum() + ra.sum() - 2 * active.sum()),
                 "unmatched_active_rms_mass": float((lrms[la].sum() + rrms[ra].sum() - lrms[rr][active].sum() - rrms[cc][active].sum()) / sd),
                 "center_distance": float((weight[active] * distance(states[i]["centers"][rr], states[j]["centers"][cc])[active]).sum() / total) if total else None,
                 "log_kappa_distance": float((weight[active] * np.abs(np.log(states[i]["kappa"][rr] / states[j]["kappa"][cc]))[active]).sum() / total) if total else None,
                 "signed_contribution_nrmse": math.sqrt(float(cost[rr, cc].mean())) / sd}
        for key, value in match.items():
            obs.check(label + f".pair{pair_index}." + key, saved["matching"][key], value)
        pair_metrics.append(match)
    for key in rows[0]:
        obs.check(label + ".unit_" + key, metrics[key], float(np.mean([row[key] for row in rows])))
    obs.check(label + ".curve_pair_nrmse", metrics["curve_pair_nrmse"], np.mean([math.sqrt(mse(curves[i], curves[j])) / sd for i, j in itertools.combinations(range(len(states)), 2)]))
    for key in pair_metrics[0]:
        values = [pair[key] for pair in pair_metrics if pair[key] is not None]
        obs.check(label + ".matched_" + key, metrics["matched_" + key], float(np.mean(values)) if values else None)


def audit_proxy(obs: Observer, label: str, unit: dict[str, Any], state: dict[str, Any],
                metrics: dict[str, Any], arrays: Any, config: dict[str, Any]) -> None:
    """Rebuild fixed prediction interventions, retrained arm and exact controls."""
    w, b, rule = state["weights"], state["bias"], config["diagnostics"]
    predict = lambda x, weights=w, bias=b: np.c_[np.cos(x), np.sin(x)] @ weights + bias
    aligned, shifted = (unit["evaluation"][name] for name in ("aligned", "shifted"))
    obs.check(label + ".paired_clock", aligned["x"][:, 1], shifted["x"][:, 1])
    obs.check(label + ".latent_shift", shifted["latent"] - aligned["latent"],
              np.full_like(aligned["latent"], config["proxy"]["shift_radians"]))
    obs.check(label + ".paired_sensor_noise", aligned["x"][:, 0] - aligned["latent"], shifted["x"][:, 0] - shifted["latent"])
    obs.check(label + ".paired_target_noise", aligned["y"] - aligned["clean"], shifted["y"] - shifted["clean"])
    valclock = unit["validation"]["x"][:, 1]
    reference = np.mean(w[1] * np.cos(valclock) + w[3] * np.sin(valclock))
    for regime in ("aligned", "shifted"):
        data = unit["evaluation"][regime]
        x, target = data["x"], data["clean"]
        pred, variance = predict(x), float(np.var(target))
        nmse = mse(pred, target) / variance
        row = {"clean_nmse": nmse, "clean_nrmse": math.sqrt(nmse), "noisy_rmse": math.sqrt(mse(pred, data["y"]))}
        clock = w[1] * np.cos(x[:, 1]) + w[3] * np.sin(x[:, 1])
        comparisons = {"delete": pred - clock, "reference": pred - clock + reference,
                       "sensor_retrained": predict(x[:, :1], state["sensor_weights"], state["sensor_bias"])}
        for name, offset in (("zero", 0), ("plus", rule["phase_offset_radians"]), ("minus", -rule["phase_offset_radians"])):
            changed = x.copy()
            changed[:, 1] += offset
            comparisons["phase_" + name] = predict(changed)
        for name, altered in comparisons.items():
            row[name + "_delta_nmse"] = mse(altered, target) / variance - nmse
            obs.check(label + "." + regime + ".array_" + name, arrays[regime + "_" + name], altered)
        row["full_minus_sensor_nmse"] = -row["sensor_retrained_delta_nmse"]
        for key, value in row.items():
            obs.check(label + "." + regime + "." + key, metrics[regime + "_" + key], value)
        rotation = rule["coordinate_rotation_radians"]
        rotated = np.r_[w[:2] * np.cos(rotation) - w[2:] * np.sin(rotation), w[:2] * np.sin(rotation) + w[2:] * np.cos(rotation)]
        for name, changed in (("zero", comparisons["phase_zero"]), ("wrap", predict(x + 2 * math.pi)), ("rotation", predict(x + rotation, rotated))):
            obs.check(label + "." + regime + "." + name, changed, pred, rule["identity_atol"])
        obs.check(label + "." + regime + ".prediction", arrays[regime + "_prediction"], pred)
    angle = arrays["attribution_x"]
    deriv = -w[1] * np.sin(angle) + w[3] * np.cos(angle)
    obs.check(label + ".derivative", arrays["derivative_clock"], deriv)
    amplitude, phase = np.hypot(w[:2], w[2:]), np.arctan2(-w[2:], w[:2])
    obs.check(label + ".polar_derivative", -amplitude[1] * np.sin(angle + phase[1]), deriv, rule["identity_atol"])
    for regime in ("aligned", "shifted"):
        data = unit["evaluation"][regime]
        obs.check(label + ".polar_" + regime, (amplitude * np.cos(data["x"] + phase)).sum(axis=1) + b, predict(data["x"]), rule["identity_atol"])
        truth = unit["truth"]
        obs.check(label + ".generator_" + regime, truth["bias"] + truth["amplitude"] * np.cos(data["latent"] - truth["center"]), data["clean"], rule["identity_atol"])
    obs.check(label + ".generator_direct_effect", metrics["generator_clock_direct_effect"], 0, rule["identity_atol"])
    # Verify both saved ridge optima algebraically, without refitting them.
    train = unit["train"]
    design = np.c_[np.cos(train["x"]), np.sin(train["x"])]
    for name, features, weights, bias in (("full", design, w, b), ("sensor", design[:, [0, 2]], state["sensor_weights"], state["sensor_bias"])):
        residual = features @ weights + bias - train["y"]
        obs.check(label + ".ridge_" + name, features.T @ residual / len(features) + config["proxy"]["ridge"] * weights, np.zeros_like(weights))
        obs.check(label + ".intercept_" + name, residual.mean(), 0)


def audit(output_dir: Path, stage: str, summary_path: Path | None = None) -> dict[str, Any]:
    """Verify artifact provenance, independently replay units, and bootstrap."""
    obs = Observer()
    config, source_hash = verify_freeze(output_dir)
    receipt = verify_stage(output_dir, stage, config, source_hash)
    if stage == "evaluation":
        lock_hash = verify_lock(output_dir, config, source_hash)
        if receipt["evaluation_lock_sha256"] != lock_hash:
            raise ValueError("Evaluation receipt does not bind the verified reserved lock")
    records = {case: [] for case in config["cases"]}
    for case in config["cases"]:
        for index in range(config["pools"][stage]["units"]):
            label = f"{case}_{index:03d}"
            directory = output_dir / stage / label
            payload = numpy(torch.load(directory / "payload.pt", map_location="cpu", weights_only=True))
            metrics, details = read_json(directory / "metrics.json"), read_json(directory / "details.json")
            with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
                if case == "proxy":
                    audit_proxy(obs, label, payload["unit"], payload["states"][0], metrics, arrays, config)
                else:
                    audit_peak(obs, label, payload["unit"], payload["states"], metrics, details, arrays, config)
            records[case].append(metrics)
    summary_path = summary_path or output_dir / f"analysis_{stage}" / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary["source_manifest_sha256"] != source_hash or summary["stage_receipt_sha256"] != sha256(output_dir / stage / "receipt.json"):
            raise ValueError("Analysis summary is not bound to verified source/stage evidence")
        for case, rows in records.items():
            indices = np.random.default_rng(config["diagnostics"]["bootstrap_seed"]).integers(
                0, len(rows), size=(config["diagnostics"]["bootstrap_draws"], len(rows)))
            for key, saved in summary["cases"][case].items():
                values = np.array([row.get(key) if row.get(key) is not None else np.nan for row in rows], dtype=float)
                complete = bool(np.isfinite(values).all())
                obs.check(case + ".bootstrap_" + key + ".count", saved["n_finite"], np.isfinite(values).sum())
                obs.check(case + ".bootstrap_" + key + ".units", saved["n_units"], len(rows))
                obs.check(case + ".bootstrap_" + key + ".complete", saved["complete"], complete)
                obs.check(case + ".bootstrap_" + key + ".mean", saved["mean"], float(values.mean()) if complete else None)
                obs.check(case + ".bootstrap_" + key + ".ci95", saved["ci95"], np.quantile(values[indices].mean(axis=1), [.025, .975]) if complete else None)
                obs.check(case + ".bootstrap_" + key + ".sd", saved["unit_sd"], (float(values.std(ddof=1)) if len(values) > 1 else 0.0) if complete else None)
    return {"status": "passed" if not obs.failures else "failed", "stage": stage, "checks": obs.checks,
            "maximum_absolute_error": obs.maximum_absolute_error, "failures": obs.failures,
            "observer_sha256": sha256(Path(__file__)), "source_manifest_sha256": source_hash,
            "stage_receipt_sha256": sha256(output_dir / stage / "receipt.json"),
            "summary_audited": str(summary_path) if summary_path.exists() else None,
            "limitations": ["Unselected historical checkpoint states are absent; only selected validation score and recorded earliest-minimum selection are replayed.",
                            "Artifact integrity reuses runner helpers; numerical reconstruction and bootstrap are independent NumPy calculations.",
                            "Randomization sensitivity and overlap metrics are not independently reconstructed; their aggregates are checked when a summary exists."]}


def main() -> int:
    """Run a read-only evidence audit and save a new external audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "development", "evaluation"), default="evaluation")
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    destination, root = args.audit_output.resolve(), args.output_dir.resolve()
    if destination.exists() or any(destination.is_relative_to(root / name) for name in ("pilot", "development", "evaluation", "frozen")):
        raise ValueError("Audit output must be new and outside frozen/stage artifact directories")
    try:
        result = audit(root, args.stage, args.summary)
    except Exception as error:
        result = {"status": "failed", "checks": 0, "maximum_absolute_error": None,
                  "failures": [{"error_type": type(error).__name__, "error": str(error)}]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
