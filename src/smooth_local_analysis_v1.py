"""Pure, function-paired analysis for the prospective VM-N01 comparison.

No optimizer, target generator, or confirmation lock is accessed here. Every
declared cell remains visible, and missing/failed cells prevent a benefit claim.
The bootstrap resamples whole function bundles within each periodic family.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np


_COMPLETE = {"complete", "completed", "ok", "success"}


def _valid(row: Mapping[str, Any], metric: str = "normalized_mse") -> bool:
    value = row.get(metric)
    return (row.get("status", "complete") in _COMPLETE
            and isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _phase_rows(records: Sequence[Mapping[str, Any]], phase: str,
                allowance: float | None = None) -> list[Mapping[str, Any]]:
    return [row for row in records if row.get("phase") == phase
            and (allowance is None or row.get("allowance") == allowance)]


def _budgets(config: Mapping[str, Any], phase: str) -> list[int]:
    key = "pilot_label_budgets" if phase == "pilot" else "label_budgets"
    return list(config["tasks"][key])


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("model_id"), row.get("family"), row.get("replicate"),
            row.get("initialization"), row.get("labels"), row.get("allowance"))


def _expected(config: Mapping[str, Any], phase: str) -> set[tuple[Any, ...]]:
    axes = ([spec["id"] for spec in config["models"]], config["tasks"]["families"],
            range(config["tasks"]["replicates"][phase]),
            range(config["tasks"]["initializations"][phase]), _budgets(config, phase),
            config["training"]["time_budgets_seconds"])
    return set(product(*axes))


def _balanced_score(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                    phase: str, metric: str, families: Sequence[str]) -> float:
    expected = set(product(families, range(config["tasks"]["replicates"][phase]),
                           range(config["tasks"]["initializations"][phase]), _budgets(config, phase)))
    by_identity: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family") in families:
            by_identity[(row.get("family"), row.get("replicate"), row.get("initialization"), row.get("labels"))].append(row)
    if set(by_identity) != expected or any(len(value) != 1 or not _valid(value[0], metric) for value in by_identity.values()):
        raise ValueError(f"Incomplete or failed {phase} selection evidence for {metric}")
    family_values = []
    for family in families:
        function_values = []
        for replicate in range(config["tasks"]["replicates"][phase]):
            function_values.append(float(np.mean([float(row[0][metric]) for key, row in by_identity.items()
                                                  if key[:2] == (family, replicate)])))
        family_values.append(float(np.mean(function_values)))
    return float(np.mean(family_values))


def select_learning_rates(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, float]:
    """Choose each adaptive rate from balanced pilot noisy validation MSE only.

    All five families enter rate selection. A failed candidate is not silently
    dropped: incomplete evidence raises before making any selection.
    """
    rows = _phase_rows(records, "pilot", config["training"]["primary_time_seconds"])
    rates = sorted(config["training"]["pilot_learning_rates"])
    result: dict[str, float] = {}
    for spec in config["models"]:
        if spec["kind"] == "ridge":
            continue
        scored = []
        for rate in rates:
            candidate = [row for row in rows if row.get("model_id") == spec["id"] and row.get("learning_rate") == rate]
            score = _balanced_score(candidate, config, "pilot", "noisy_validation_mse", config["tasks"]["families"])
            scored.append((score, rate))
        result[spec["id"]] = float(min(scored)[1])
    return result


def select_models(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Choose hybrid optimizer and capacity control on balanced development error."""
    rows = _phase_rows(records, "development", config["training"]["primary_time_seconds"])
    families = config["tasks"]["periodic_families"]
    output: dict[str, Any] = {"scores": {}}
    for target, registry in (("selected_hybrid", "hybrid_candidates"), ("capacity_comparator", "capacity_candidates")):
        candidates = config["analysis"][registry]
        scores = [_balanced_score([row for row in rows if row.get("model_id") == model], config,
                                  "development", "normalized_mse", families) for model in candidates]
        output[target] = candidates[int(np.argmin(scores))]
        output["scores"].update(dict(zip(candidates, scores)))
    return output


def paired_contrast(functions: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                    candidate: str, comparator: str, phase: str = "confirmation") -> dict[str, Any]:
    """Compute the frozen contrast and stratified whole-function interval.

    Args:
        functions: Per-function means across labels and initializations, with
            model_id, family, replicate, allowance and normalized_mse fields.
        config: Prospective thresholds, analysis seed and replicate counts.
        candidate: Model whose relative reduction is tested.
        comparator: Reference model's error is the denominator.
        phase: Evidence phase, determining expected independent functions.

    Returns:
        Balanced errors, paired variation, percentile interval, family guards,
        all paired function rows and an explicit pass/fail decision.
    """
    families = list(config["tasks"]["periodic_families"])
    primary = config["training"]["primary_time_seconds"]
    count = config["tasks"]["replicates"][phase]
    chosen = [row for row in functions if row.get("model_id") in {candidate, comparator}
              and row.get("family") in families and row.get("allowance") == primary
              and row.get("phase", phase) == phase]
    keyed: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in chosen:
        keyed[(row["model_id"], row["family"], row["replicate"])].append(row)
    expected = set(product((candidate, comparator), families, range(count)))
    result: dict[str, Any] = {"candidate": candidate, "comparator": comparator,
                              "phase": phase, "allowance": primary, "passed": False}
    if set(keyed) != expected or any(len(rows) != 1 or not _valid(rows[0]) for rows in keyed.values()):
        return {**result, "complete": False, "reason": "missing_failed_or_duplicate_function_bundle"}
    arrays = {model: {family: np.asarray([keyed[(model, family, r)][0]["normalized_mse"]
                                        for r in range(count)], dtype=np.float64)
                      for family in families} for model in (candidate, comparator)}
    errors = {model: float(np.mean([np.mean(values) for values in arrays[model].values()]))
              for model in arrays}
    if errors[comparator] <= 0 or float(arrays[comparator]["smooth"].mean()) <= 0:
        return {**result, "complete": True, "reason": "nonpositive_reference_error", "errors": errors}
    relative = 1 - errors[candidate] / errors[comparator]
    n_bootstrap = int(config["analysis"]["bootstrap_resamples"])
    rng = np.random.default_rng(np.random.SeedSequence([config["seeds"]["analysis"], 101]))
    boot = {model: np.zeros(n_bootstrap, dtype=np.float64) for model in arrays}
    paired = []
    family_results = {}
    for family in families:
        a, b = arrays[candidate][family], arrays[comparator][family]
        # The SAME index resamples every paired model and the whole already
        # averaged label/initialization bundle for each independent function.
        indices = rng.integers(0, count, size=(n_bootstrap, count))
        for model in arrays:
            boot[model] += arrays[model][family][indices].mean(axis=1) / len(families)
        family_results[family] = {"candidate_error": float(a.mean()), "comparator_error": float(b.mean()),
                                  "relative_improvement": None if b.mean() <= 0 else float(1 - a.mean() / b.mean()),
                                  "paired_difference_mean": float((b-a).mean()), "paired_difference_sd": _sd((b-a).tolist())}
        for replicate in range(count):
            paired.append({"family": family, "replicate": replicate, "candidate_error": float(a[replicate]),
                           "comparator_error": float(b[replicate]), "error_reduction": float(b[replicate] - a[replicate]),
                           "relative_improvement": None if b[replicate] <= 0 else float(1-a[replicate]/b[replicate])})
    if np.any(boot[comparator] <= 0):
        return {**result, "complete": True, "reason": "nonpositive_bootstrap_reference", "errors": errors}
    improvement_samples = 1 - boot[candidate] / boot[comparator]
    alpha = 1 - float(config["analysis"]["confidence"])
    interval = np.quantile(improvement_samples, [alpha/2, 1-alpha/2]).tolist()
    smooth_regression = -family_results["smooth"]["relative_improvement"]
    improving_local = sum(family_results[name]["candidate_error"] < family_results[name]["comparator_error"]
                          for name in ("narrow", "multimodal", "mixed"))
    thresholds = config["analysis"]["gate"]
    checks = {"practical_effect": relative >= thresholds["each_comparison_min_relative_improvement"],
              "positive_lower_interval": interval[0] > thresholds["each_comparison_lower_interval_above"],
              "smooth_guard": smooth_regression <= thresholds["each_comparison_max_smooth_regression"],
              "local_families": improving_local >= thresholds["each_comparison_min_improving_local_families"]}
    return {**result, "complete": True, "errors": errors, "relative_improvement": relative,
            "interval": interval, "confidence": config["analysis"]["confidence"],
            "bootstrap_resamples": n_bootstrap, "bootstrap_unit": "whole_function_within_family",
            "smooth_regression": smooth_regression, "improving_local_families": int(improving_local),
            "families": family_results, "paired_functions": paired, "checks": checks,
            "passed": all(checks.values())}


def summarize(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
              selected_hybrid: str, capacity_comparator: str,
              phase: str = "confirmation") -> dict[str, Any]:
    """Report every declared result and enforce the fixed three-comparison gate.

    Development and confirmation contain one rate per model. Pilot rate
    selection is deliberately separate to avoid pooling distinct candidates.
    """
    if phase not in {"development", "confirmation"}:
        raise ValueError("Summary requires development or confirmation; use select_learning_rates for pilot")
    if selected_hybrid not in config["analysis"]["hybrid_candidates"]:
        raise ValueError("Selected hybrid is not a registered candidate")
    if capacity_comparator not in config["analysis"]["capacity_candidates"]:
        raise ValueError("Selected capacity control is not a registered candidate")
    rows = _phase_rows(records, phase)
    expected = _expected(config, phase)
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_identity(row)].append(row)
    missing, unexpected = expected - set(groups), set(groups) - expected
    duplicates = {key for key, values in groups.items() if len(values) != 1}
    failures = [row for row in rows if not _valid(row)]
    integrity = {"complete": not (missing or unexpected or duplicates or failures),
                 "expected_records": len(expected), "observed_records": len(rows),
                 "missing_count": len(missing), "unexpected_count": len(unexpected),
                 "duplicate_count": len(duplicates), "failed_count": len(failures),
                 "missing": [list(key) for key in sorted(missing)],
                 "unexpected": [list(key) for key in sorted(unexpected, key=str)],
                 "duplicates": [list(key) for key in sorted(duplicates, key=str)],
                 "failed_fit_ids": sorted({str(row.get("fit_id", "unknown")) for row in failures})}
    cells, functions = [], []
    families = config["tasks"]["families"]
    n_replicates = config["tasks"]["replicates"][phase]
    n_initializations = config["tasks"]["initializations"][phase]
    budgets = _budgets(config, phase)
    for spec, family, allowance in product(config["models"], families, config["training"]["time_budgets_seconds"]):
        model = spec["id"]
        for labels in budgets:
            valid_values, function_values = [], []
            failed = absent = 0
            for replicate in range(n_replicates):
                init_values = []
                for initialization in range(n_initializations):
                    candidates = groups.get((model, family, replicate, initialization, labels, allowance), [])
                    absent += int(not candidates)
                    if len(candidates) == 1 and _valid(candidates[0]):
                        init_values.append(float(candidates[0]["normalized_mse"]))
                    else:
                        failed += int(bool(candidates))
                valid_values.extend(init_values)
                if len(init_values) == n_initializations:
                    function_values.append(float(np.mean(init_values)))
            complete = len(function_values) == n_replicates
            cells.append({"phase": phase, "model_id": model, "family": family, "labels": labels,
                          "allowance": allowance, "parameters": spec["parameters"], "features": spec["features"],
                          "expected_fits": n_replicates*n_initializations, "complete_fits": len(valid_values),
                          "missing_fits": absent, "failed_fits": failed,
                          "status": "complete" if complete else "incomplete",
                          "normalized_mse": _mean(function_values) if complete else None,
                          "observed_normalized_mse": _mean(valid_values),
                          "function_sd": _sd(function_values) if complete else None})
        for replicate in range(n_replicates):
            values = []
            for labels, initialization in product(budgets, range(n_initializations)):
                candidates = groups.get((model, family, replicate, initialization, labels, allowance), [])
                if len(candidates) == 1 and _valid(candidates[0]):
                    values.append(float(candidates[0]["normalized_mse"]))
            complete = len(values) == len(budgets)*n_initializations
            functions.append({"phase": phase, "model_id": model, "family": family, "replicate": replicate,
                              "allowance": allowance, "status": "complete" if complete else "incomplete",
                              "expected_fits": len(budgets)*n_initializations, "complete_fits": len(values),
                              "normalized_mse": _mean(values) if complete else None})
    models = {}
    primary = config["training"]["primary_time_seconds"]
    for spec in config["models"]:
        family_errors = {}
        family_sd = {}
        for family in families:
            matching = [row for row in functions if row["model_id"] == spec["id"]
                        and row["family"] == family and row["allowance"] == primary]
            complete = all(_valid(row) for row in matching) and len(matching) == n_replicates
            values = [row["normalized_mse"] for row in matching] if complete else []
            family_errors[family], family_sd[family] = _mean(values), _sd(values)
        periodic = [family_errors[family] for family in config["tasks"]["periodic_families"]]
        models[spec["id"]] = {"balanced_error": None if any(value is None for value in periodic) else _mean(periodic),
                               "family_errors": family_errors, "family_function_sd": family_sd,
                               "parameters": spec["parameters"], "features": spec["features"]}
    comparators = [*config["analysis"]["equal_basis_comparators"], capacity_comparator]
    contrasts = {comparator: paired_contrast(functions, config, selected_hybrid, comparator, phase) for comparator in comparators}
    optimizer_contrasts = {"hybrid_joint_vs_vp": paired_contrast(functions, config, "hybrid_joint", "hybrid_vp", phase),
                           "adaptive_joint_vs_vp": paired_contrast(functions, config, "adaptive16_joint", "adaptive16_vp", phase)}
    gate = {"passed": integrity["complete"] and all(row["passed"] for row in contrasts.values()),
            "complete_evidence": integrity["complete"], "selected_hybrid": selected_hybrid,
            "capacity_comparator": capacity_comparator,
            "comparison_passed": {key: value["passed"] for key, value in contrasts.items()},
            "nonperiodic_in_gate": False}
    return {"phase": phase, "cells": cells, "functions": functions, "models": models,
            "contrasts": contrasts, "optimizer_contrasts": optimizer_contrasts,
            "integrity": integrity, "gate": gate}
