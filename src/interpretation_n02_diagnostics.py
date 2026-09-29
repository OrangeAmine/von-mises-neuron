"""Versioned function-first interpretation diagnostics for the VM-N02 study.

No geometry here is a new model: evaluate the existing circular bank and
Cartesian sum APIs. Component allocation is explicitly representation dependent.
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.circular_layers_v2 import CartesianCircularLinearV1, PolarCircularLinearV2
from src.locality_models import circular_features, chordal_features


def cpu64(value: Any) -> Tensor:
    """Return detached CPU float64 values for reproducible analysis."""
    return torch.as_tensor(value, device="cpu", dtype=torch.float64).detach()


def contribution(state: dict[str, Any], x: Tensor, *, chordal: bool = False) -> Tensor:
    """Return signed complete contributions, including readout compensation."""
    feature_fn = chordal_features if chordal else circular_features
    return feature_fn(cpu64(x), cpu64(state["centers"]), cpu64(state["kappa"])) * cpu64(state["weights"])


def response(state: dict[str, Any], x: Tensor, *, chordal: bool = False) -> Tensor:
    """Return scalar response on each angle without modifying state."""
    return contribution(state, x, chordal=chordal).sum(-1) + cpu64(state["bias"])


def derivative(state: dict[str, Any], x: Tensor) -> Tensor:
    """Return signed d(predicted response)/d(angle), in response units/radian."""
    angle = cpu64(x).reshape(-1)
    return (-contribution(state, angle) * cpu64(state["kappa"])
            * (angle[:, None] - cpu64(state["centers"])).sin()).sum(-1)


def circular_distance(left: Any, right: Any) -> Tensor:
    """Return shortest unsigned angle differences, invariant to wrapping."""
    delta = cpu64(left) - cpu64(right)
    return torch.atan2(delta.sin(), delta.cos()).abs()


def fwhm(kappa: Any) -> list[float | None]:
    """Return peak-kernel half-height widths; None means no crossing."""
    return [2 * math.acos(1 - math.log(2) / float(k)) if k >= math.log(2) / 2
            else None for k in cpu64(kappa).reshape(-1)]


def _mse(a: Tensor, b: Tensor) -> float:
    return float((cpu64(a) - cpu64(b)).square().mean())


def _relative_change(a: Tensor, b: Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(cpu64(a)))
    if denominator <= 1e-14:
        return float("nan")
    return float(torch.linalg.vector_norm(cpu64(a) - cpu64(b))) / denominator


def _error(a: Tensor, b: Tensor) -> float:
    return float((cpu64(a) - cpu64(b)).abs().max())


def _plain(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def match_components(left: dict[str, Any], right: dict[str, Any], x: Tensor,
                     target_sd: float, active_threshold: float,
                     phase_threshold: float = 0.02) -> dict[str, Any]:
    """Match signed curves after whole-response comparison, not raw phases.

    Hungarian assignment includes every component. Only active-active matched
    pairs contribute to center and width summaries. Unmatched activity remains
    explicit instead of assigning a preferred phase to a negligible component.
    """
    lc, rc = contribution(left, x), contribution(right, x)
    lrms, rrms = lc.square().mean(0).sqrt(), rc.square().mean(0).sqrt()
    cost = (lc[:, :, None] - rc[:, None, :]).square().mean(0)
    rows, cols = linear_sum_assignment(cost.numpy())
    la, ra = lrms / target_sd >= active_threshold, rrms / target_sd >= active_threshold
    phase_left = la & (lc.std(0, unbiased=False) / target_sd >= phase_threshold)
    phase_right = ra & (rc.std(0, unbiased=False) / target_sd >= phase_threshold)
    active = phase_left[rows] & phase_right[cols]
    weight = (lrms[rows] + rrms[cols]) / 2
    center = circular_distance(cpu64(left["centers"])[rows], cpu64(right["centers"])[cols])
    width = (cpu64(left["kappa"])[rows] / cpu64(right["kappa"])[cols]).log().abs()
    total = float(weight[active].sum())
    return {"rows": rows.tolist(), "cols": cols.tolist(), "active_pairs": active.tolist(),
            "active_pair_count": int(active.sum()),
            "unmatched_active_count": int(la.sum() + ra.sum() - 2 * active.sum()),
            "unmatched_active_rms_mass": float(lrms[la].sum() + rrms[ra].sum()
                                                - lrms[rows][active].sum() - rrms[cols][active].sum()) / target_sd,
            "center_distance": float((weight[active] * center[active]).sum()) / total if total else None,
            "log_kappa_distance": float((weight[active] * width[active]).sum()) / total if total else None,
            "signed_contribution_nrmse": float(cost[rows, cols].mean().sqrt()) / target_sd,
            "pair_rms_weights": weight.tolist()}


def equivalent_states(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Construct exact permutation/split/cancellation representations."""
    s = {key: cpu64(state[key]).clone() for key in ("centers", "kappa", "weights", "bias")}
    perm = {k: (v.flip(0) if k != "bias" else v.clone()) for k, v in s.items()}
    split = {k: (torch.cat((v[:1], v)) if k != "bias" else v.clone()) for k, v in s.items()}
    split["weights"][:2] = s["weights"][0] / 2
    cancel = {k: (torch.cat((v, v[:1], v[:1])) if k != "bias" else v.clone()) for k, v in s.items()}
    cancel["weights"][-2:] = torch.tensor([0.5, -0.5], dtype=torch.float64)
    return {"permutation": perm, "split": split, "cancelling_pair": cancel}


def _overlap(state: dict[str, Any], x: Tensor, sd: float, threshold: float,
             phase_threshold: float) -> float:
    c = contribution(state, x)
    eligible = ((c.square().mean(0).sqrt() / sd >= threshold)
                & (c.std(0, unbiased=False) / sd >= phase_threshold))
    c = c[:, eligible]
    if c.shape[1] < 2:
        return 0.0
    centered = c - c.mean(0)
    norms = centered.square().sum(0).sqrt()
    corr = (centered.T @ centered / (norms[:, None] * norms[None, :])).abs()
    corr.fill_diagonal_(0)
    return float(corr.max())


def peak_diagnostics(states: list[dict[str, Any]], unit: dict[str, Any],
                     config: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any], dict[str, np.ndarray]]:
    """Compute every declared peak control, then aggregate within a function."""
    rule = config["diagnostics"]
    x = cpu64(unit["curve"]["x"])
    truth = cpu64(unit["curve"]["clean"])
    xe, ye = cpu64(unit["evaluation"]["x"]), cpu64(unit["evaluation"]["clean"])
    xv = cpu64(unit["validation"]["x"])
    variance = float(ye.var(unbiased=False))
    sd = math.sqrt(variance)
    active_threshold = rule["active_component_rms_over_target_sd"]
    curves = [response(s, x) for s in states]
    # Whole-function comparison precedes all assignments of individual kernels.
    pairs = [{"i": i, "j": j,
              "curve_nrmse": math.sqrt(_mse(curves[i], curves[j])) / sd}
             for i, j in itertools.combinations(range(len(states)), 2)]
    for pair in pairs:
        pair["matching"] = match_components(states[pair["i"]], states[pair["j"]], x, sd, active_threshold,
                                               rule["phase_component_sd_over_target_sd"])
    rows: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    arrays = {"x": x.numpy(), "truth": truth.numpy(), "curves": torch.stack(curves).numpy()}
    for i, state in enumerate(states):
        prediction, d = response(state, xe), derivative(state, x)
        nmse = _mse(prediction, ye) / variance
        comp = contribution(state, xe)
        active = contribution(state, x).square().mean(0).sqrt() / sd >= active_threshold
        phase_eligible = active & (contribution(state, x).std(0, unbiased=False) / sd >= rule["phase_component_sd_over_target_sd"])
        select = int(contribution(state, xv).square().mean(0).argmax())
        deletion = [_mse(prediction - comp[:, j], ye) / variance - nmse for j in range(comp.shape[1])]
        row = {"clean_nrmse": math.sqrt(nmse), "clean_nmse": nmse,
               "noisy_rmse": math.sqrt(_mse(prediction, unit["evaluation"]["y"])),
               "primary_deletion_delta_nmse": deletion[select], "active_components": float(active.sum()),
               "phase_eligible_components": float(phase_eligible.sum()),
               "maximum_component_overlap": _overlap(state, x, sd, active_threshold, rule["phase_component_sd_over_target_sd"]),
               "overlap_eligible_pairs": float(int(phase_eligible.sum()) * (int(phase_eligible.sum()) - 1) / 2),
               "parameter_count": float(state["parameter_count"]),
               "fit_seconds": float(state["seconds"]),
               "selected_step": float(state["selected_step"])}
        for name, offset in (("zero", 0.0), ("plus", rule["phase_offset_radians"]),
                             ("minus", -rule["phase_offset_radians"])):
            changed = copy.deepcopy(state)
            changed["centers"] = cpu64(state["centers"]).clone()
            changed["centers"][select] += offset
            result = response(changed, xe)
            row[f"phase_{name}_delta_nmse"] = _mse(result, ye) / variance - nmse
            if name == "zero":
                row["zero_identity_error"] = _error(result, prediction)
        for factor in rule["width_factors"]:
            changed = copy.deepcopy(state)
            changed["kappa"] = cpu64(state["kappa"]).clone()
            changed["kappa"][select] *= factor
            result = response(changed, xe)
            label = str(factor).replace(".", "p")
            row[f"width_{label}_delta_nmse"] = _mse(result, ye) / variance - nmse
            if factor == 1:
                row["width_identity_error"] = _error(result, prediction)
        row["wrap_identity_error"] = _error(response(state, x + 2 * math.pi), curves[i])
        row["chordal_identity_error"] = _error(response(state, x, chordal=True), curves[i])
        rotation = rule["coordinate_rotation_radians"]
        rotated = copy.deepcopy(state)
        rotated["centers"] = cpu64(state["centers"]) + rotation
        row["rotation_identity_error"] = _error(response(rotated, x + rotation), curves[i])
        equivalents = equivalent_states(state)
        for name, equivalent in equivalents.items():
            row[f"{name}_identity_error"] = _error(response(equivalent, x), curves[i])
            row[f"{name}_derivative_error"] = _error(derivative(equivalent, x), d)
        # Exact function equivalence does not preserve a single-component deletion.
        cancel = equivalents["cancelling_pair"]
        cancel_component = contribution(cancel, xe)[:, -2]
        row["inserted_cancel_component_deletion_delta_nmse"] = _mse(prediction - cancel_component, ye) / variance - nmse
        split_component = contribution(equivalents["split"], xe)[:, 0]
        row["split_half_deletion_delta_nmse"] = _mse(prediction - split_component, ye) / variance - nmse
        row["unsplit_first_deletion_delta_nmse"] = deletion[0]
        rng = np.random.default_rng(int(state["init_seed"]) + rule["randomization"]["peak_seed_addend"])
        randomized = copy.deepcopy(state)
        randomized["weights"] = torch.as_tensor(rng.normal(0, 1, comp.shape[1]), dtype=torch.float64)
        randomized["bias"] = torch.tensor(rng.normal(0, 1), dtype=torch.float64)
        geometry = copy.deepcopy(state)
        geometry["centers"] = torch.as_tensor(rng.uniform(0, 2 * math.pi, comp.shape[1]), dtype=torch.float64)
        geometry["kappa"] = torch.as_tensor(rng.uniform(*config["peak"]["initial_kappa_range"], comp.shape[1]), dtype=torch.float64)
        both = copy.deepcopy(geometry)
        both["weights"], both["bias"] = randomized["weights"], randomized["bias"]
        for name, altered in (("readout", randomized), ("geometry", geometry), ("both", both)):
            row[f"random_{name}_attribution_relative_l2"] = _relative_change(d, derivative(altered, x))
            arrays[f"derivative_random_{name}_{i}"] = derivative(altered, x).numpy()
        if unit["case"] == "isolated":
            center_error = float(circular_distance(state["centers"], unit["truth"]["centers"])[0])
            width_error = float((cpu64(state["kappa"]) / cpu64(unit["truth"]["kappa"]) - 1).abs()[0])
            eligible = bool(phase_eligible[0]) and float(cpu64(state["weights"])[0]) > 0
            row.update({"center_error_rad": center_error, "kappa_relative_error": width_error,
                        "recovery_eligible": float(eligible),
                        "recovered": float(eligible and center_error <= rule["center_error_radians_max"]
                                           and width_error <= rule["kappa_relative_error_max"]
                                           and row["clean_nrmse"] <= rule["quality_nrmse_max"])})
        rows.append(row)
        details.append({"metrics": row, "all_deletion_delta_nmse": deletion,
                        "validation_selected_component": select, "active": active.tolist(),
                        "phase_eligible": phase_eligible.tolist(),
                        "fwhm": fwhm(state["kappa"]), "state": _plain(state)})
        arrays[f"components_{i}"] = contribution(state, x).numpy()
        arrays[f"derivative_{i}"] = d.numpy()
    metrics = {key: float(np.max([r[key] for r in rows]) if key.endswith(("identity_error", "derivative_error"))
                          else np.mean([r[key] for r in rows])) for key in rows[0]}
    metrics["initialization_sd_clean_nrmse"] = float(np.std([r["clean_nrmse"] for r in rows], ddof=1))
    metrics["curve_pair_nrmse"] = float(np.mean([p["curve_nrmse"] for p in pairs]))
    for key in ("center_distance", "log_kappa_distance", "signed_contribution_nrmse", "active_pair_count", "unmatched_active_count", "unmatched_active_rms_mass"):
        valid = [p["matching"][key] for p in pairs if p["matching"][key] is not None]
        metrics[f"matched_{key}"] = float(np.mean(valid)) if valid else float("nan")
    metrics["matching_pairs_without_active_components"] = float(sum(p["matching"]["active_pair_count"] == 0 for p in pairs))
    return metrics, {"fits": details, "initialization_pairs": pairs, "target_sd": sd}, arrays


def cartesian_model(weights: Any, bias: Any) -> CartesianCircularLinearV1:
    """Load fitted coefficients into the unchanged existing model API."""
    w = cpu64(weights)
    # Avoid introducing a second scientific architecture or using ambient RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7702)
        model = CartesianCircularLinearV1(w.numel() // 2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight_cos.copy_(w[:w.numel() // 2][None])
        model.weight_sin.copy_(w[w.numel() // 2:][None])
        model.bias.copy_(cpu64(bias).reshape(1))
    return model


def proxy_response(state: dict[str, Any], x: Tensor, *, sensor_only: bool = False) -> Tensor:
    """Evaluate the fitted Cartesian API, optionally the separately retrained arm."""
    key = "sensor_" if sensor_only else ""
    model = cartesian_model(state[f"{key}weights"], state[f"{key}bias"])
    with torch.no_grad():
        return model(cpu64(x)[:, :1] if sensor_only else cpu64(x)).squeeze(-1)


def proxy_diagnostics(state: dict[str, Any], unit: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any], dict[str, np.ndarray]]:
    """Distinguish clock reliance, shifted association and zero direct role."""
    w, b = cpu64(state["weights"]), cpu64(state["bias"])
    rule = config["diagnostics"]
    val_clock = cpu64(unit["validation"]["x"])[:, 1]
    reference = (w[1] * val_clock.cos() + w[3] * val_clock.sin()).mean()
    metrics: dict[str, float] = {"parameter_count": float(state["parameter_count"]), "fit_seconds": float(state["seconds"])}
    arrays: dict[str, np.ndarray] = {}
    for regime in ("aligned", "shifted"):
        data = unit["evaluation"][regime]
        x, target = cpu64(data["x"]), cpu64(data["clean"])
        variance = float(target.var(unbiased=False))
        prediction = proxy_response(state, x)
        clock = w[1] * x[:, 1].cos() + w[3] * x[:, 1].sin()
        nmse = _mse(prediction, target) / variance
        metrics[f"{regime}_clean_nmse"] = nmse
        metrics[f"{regime}_clean_nrmse"] = math.sqrt(nmse)
        metrics[f"{regime}_noisy_rmse"] = math.sqrt(_mse(prediction, data["y"]))
        comparisons = {"delete": prediction - clock, "reference": prediction - clock + reference,
                       "sensor_retrained": proxy_response(state, x, sensor_only=True)}
        for label, amount in (("plus", rule["phase_offset_radians"]), ("minus", -rule["phase_offset_radians"]), ("zero", 0.0)):
            changed = x.clone()
            changed[:, 1] += amount
            comparisons[f"phase_{label}"] = proxy_response(state, changed)
        for name, altered in comparisons.items():
            metrics[f"{regime}_{name}_delta_nmse"] = _mse(altered, target) / variance - nmse
            arrays[f"{regime}_{name}"] = altered.numpy()
        metrics[f"{regime}_full_minus_sensor_nmse"] = -metrics[f"{regime}_sensor_retrained_delta_nmse"]
        metrics[f"{regime}_zero_identity_error"] = _error(prediction, comparisons["phase_zero"])
        metrics[f"{regime}_wrap_identity_error"] = _error(prediction, proxy_response(state, x + 2 * math.pi))
        rotation = rule["coordinate_rotation_radians"]
        # x'=x+d needs a'=a cos(d)-b sin(d), b'=a sin(d)+b cos(d).
        rotated = copy.deepcopy(state)
        rotated["weights"] = torch.cat((w[:2] * math.cos(rotation) - w[2:] * math.sin(rotation),
                                         w[:2] * math.sin(rotation) + w[2:] * math.cos(rotation)))
        metrics[f"{regime}_rotation_identity_error"] = _error(prediction, proxy_response(rotated, x + rotation))
        arrays[f"{regime}_truth"], arrays[f"{regime}_prediction"] = target.numpy(), prediction.numpy()
        arrays[f"{regime}_clock"] = x[:, 1].numpy()
    # Cartesian-to-polar coordinates are an exact representation of this response.
    amplitude = torch.sqrt(w[:2].square() + w[2:].square())
    phase = torch.atan2(-w[2:], w[:2])
    angle = torch.arange(config["samples"]["curve"], dtype=torch.float64) * (2 * math.pi / config["samples"]["curve"])
    derivative_clock = -w[1] * angle.sin() + w[3] * angle.cos()
    arrays["attribution_x"], arrays["derivative_clock"] = angle.numpy(), derivative_clock.numpy()
    rng = np.random.default_rng(rule["randomization"]["proxy_seed"])
    rand_amp = torch.as_tensor(rng.uniform(*rule["randomization"]["proxy_amplitude_range"], 2), dtype=torch.float64)
    rand_phase = torch.as_tensor(rng.uniform(0, 2 * math.pi, 2), dtype=torch.float64)
    for name, a, p in (("readout", rand_amp, phase), ("geometry", amplitude, rand_phase), ("both", rand_amp, rand_phase)):
        altered = -a[1] * (angle + p[1]).sin()
        metrics[f"random_{name}_attribution_relative_l2"] = _relative_change(derivative_clock, altered)
        arrays[f"derivative_random_{name}"] = altered.numpy()
    side = 32
    grid = torch.arange(side, dtype=torch.float64) * 2 * math.pi / side
    points = torch.cartesian_prod(grid, grid)
    pred = proxy_response(state, points)
    polar = (amplitude * (points + phase).cos()).sum(-1) + b
    metrics["polar_identity_error"] = _error(pred, polar)
    metrics["polar_derivative_error"] = _error(derivative_clock, -amplitude[1] * (angle + phase[1]).sin())
    latent = cpu64(unit["evaluation"]["aligned"]["latent"])
    generator_target = (cpu64(unit["truth"]["bias"]) + cpu64(unit["truth"]["amplitude"])
                        * (latent - cpu64(unit["truth"]["center"])).cos())
    metrics["generator_clock_direct_effect"] = _error(generator_target, cpu64(unit["evaluation"]["aligned"]["clean"]))
    # The SCM's surgical clock perturbation holds its target argument z fixed.
    # This structural zero is distinct from the observational regime shift in z.
    return (metrics, {"state": _plain(state), "validation_clock_reference": float(reference),
                     "clock_amplitude": float(amplitude[1]), "clock_preferred_angle": float((-phase[1]) % (2 * math.pi)),
                     "clock_phase_eligible": bool(amplitude[1] / math.sqrt(2) / float(cpu64(unit["evaluation"]["aligned"]["clean"]).std(unbiased=False))
                                                  >= max(rule["active_component_rms_over_target_sd"], rule["phase_component_sd_over_target_sd"])),
                     "direct_effect_status": "exact generator equation; clock absent conditional on fixed latent z",
                     "cartesian_weight_order": ["cos_sensor", "cos_clock", "sin_sensor", "sin_clock"]},
                     {**arrays, "response_surface": pred.reshape(side, side).numpy()})
