"""Render compact VM-N02 tables, claim verdicts and static scientific figures.

This observer does not generate functions, fit models, or select checkpoints.
All example plots use unit zero, while tables include every reserved unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_scorecard(summary: dict, config: dict, units: list[dict]) -> list[dict]:
    """Apply prospective per-claim criteria without replacing failed outcomes."""
    rules = config["diagnostics"]
    rows = []

    def claim(case: str, metric: str, threshold: float, relation: str, label: str,
              lower_bound: bool = False) -> None:
        item = summary["cases"][case][metric]
        value = item["ci95"][0] if lower_bound and item["complete"] else item["mean"]
        passed = value is not None and (value <= threshold if relation == "<=" else
                                       value > threshold if relation == ">" else value >= threshold)
        rows.append({"case": case, "claim": label, "metric": metric,
                     "mean": item["mean"], "ci95": item["ci95"], "threshold": threshold,
                     "criterion": ("lower CI " if lower_bound else "mean ") + relation,
                     "verdict": "survived" if passed else "failed" if item["complete"] else "not evaluable"})

    for case in ("isolated", "redundant"):
        claim(case, "clean_nrmse", rules["quality_nrmse_max"], "<=", "Accurate complete response")
        claim(case, "curve_pair_nrmse", rules["curve_pair_nrmse_max"], "<=", "Stable complete response across initializations")
        claim(case, "primary_deletion_delta_nmse", rules["intervention_normalized_mse_effect_min"], ">=", "Fitted component deletion worsens prediction error")
        for control in ("phase_plus", "phase_minus", "width_0p5", "width_2p0"):
            claim(case, f"{control}_delta_nmse", rules["intervention_normalized_mse_effect_min"], ">=", f"{control} reaches declared practical error increase")
    claim("redundant", "matched_center_distance", rules["matched_center_dispersion_radians"], "<=", "Matched center dispersion stays below descriptive instability flag")
    claim("redundant", "matched_log_kappa_distance", rules["matched_log_kappa_dispersion"], "<=", "Matched width dispersion stays below descriptive instability flag")
    claim("isolated", "recovered", rules["recovery_fraction_min"], ">=", "Isolated positive peak parameters recover under declared conditions")
    claim("proxy", "aligned_clean_nrmse", rules["quality_nrmse_max"], "<=", "Aligned proxy meets demanding prediction threshold")
    claim("proxy", "aligned_delete_delta_nmse", rules["intervention_normalized_mse_effect_min"], ">", "Clock reliance in aligned data", lower_bound=True)
    claim("proxy", "shifted_full_minus_sensor_nmse", rules["intervention_normalized_mse_effect_min"], ">", "Clock-associated prediction degrades relative to retrained sensor-only under shift", lower_bound=True)
    clauses = [row["verdict"] for row in rows[-2:]]
    rows.append({"case": "proxy", "claim": "Combined predictive-proxy reliance criterion",
                 "metric": "both paired lower confidence bounds", "mean": None, "ci95": None,
                 "threshold": rules["intervention_normalized_mse_effect_min"], "criterion": "both lower CI >",
                 "verdict": "survived" if all(v == "survived" for v in clauses) else
                            "not evaluable" if "not evaluable" in clauses else "failed"})
    for case in config["cases"]:
        for name in ("readout", "geometry", "both"):
            claim(case, f"random_{name}_attribution_relative_l2", rules["randomized_attribution_relative_l2_min"], ">=", f"Derivative depends on {name} randomization")
        expected = ([f"{regime}_{control}_identity_error" for regime in ("aligned", "shifted")
                     for control in ("zero", "wrap", "rotation")] + ["polar_identity_error", "polar_derivative_error", "generator_clock_direct_effect"]
                    if case == "proxy" else [f"{control}_identity_error" for control in
                    ("zero", "width", "wrap", "chordal", "rotation", "permutation", "split", "cancelling_pair")]
                    + [f"{control}_derivative_error" for control in ("permutation", "split", "cancelling_pair")])
        selected = [row for row in units if row["case"] == case]
        complete = (len(selected) == config["pools"]["evaluation"]["units"]
                    and len({row["unit"] for row in selected}) == len(selected))
        errors = []
        for row in selected:
            for key in expected:
                try:
                    value = float(row[key])
                    complete = complete and math.isfinite(value)
                    errors.append(value)
                except (KeyError, TypeError, ValueError):
                    complete = False
        maximum = max(errors) if complete else None
        rows.append({"case": case, "claim": "Exact representations, zero perturbations and coordinate controls",
                     "metric": "maximum_exact_control_error", "mean": maximum, "ci95": None,
                     "threshold": rules["identity_atol"], "criterion": "every control <=",
                     "verdict": "not evaluable" if maximum is None else
                                "survived" if maximum <= rules["identity_atol"] else "failed"})
    rows.extend([
        {"case": "redundant", "claim": "Unique component phase/width meaning follows from accurate predictions",
         "metric": "exact split and cancellation counterexamples", "mean": None, "ci95": None,
         "threshold": None, "criterion": "logical identifiability", "verdict": "refuted by exact counterexample"},
        {"case": "proxy", "claim": "Predictive clock reliance establishes a direct physical role",
         "metric": "generator equation conditional on latent oscillator", "mean": 0, "ci95": None,
         "threshold": None, "criterion": "direct recorded-clock intervention", "verdict": "refuted in this generator"},
        {"case": "redundant", "claim": "Matched centers/widths identify physical sources",
         "metric": "no unique generator-to-kernel correspondence", "mean": None, "ci95": None,
         "threshold": None, "criterion": "identifiability", "verdict": "unidentifiable"},
    ])
    return rows


def figures(run: Path, destination: Path, summary: dict) -> None:
    """Plot preselected first units and all-unit aggregate effects with intervals."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    palette = ["#236B8E", "#D17832", "#5E9345"]
    for j, case in enumerate(("isolated", "redundant")):
        a = np.load(run / "evaluation" / f"{case}_000" / "arrays.npz")
        ax = axes[0, j]
        ax.plot(a["x"], a["truth"], color="black", lw=2, label="Known target")
        for k, curve in enumerate(a["curves"]):
            ax.plot(a["x"], curve, color=palette[k], lw=1.25, ls=["-", "--", ":"][k], label=f"Initialization {k + 1}")
        ax.set(xlabel="Angle (radians)", ylabel="Response units", title=f"{case.title()}: complete response, unit 0")
        ax.legend(fontsize=8)
        bx = axes[1, j]
        if case == "isolated":
            bx.plot(a["x"], a["derivative_0"], color=palette[0], label="Fitted derivative")
            bx.plot(a["x"], a["derivative_random_readout_0"], color=palette[1], ls="--", label="Randomized readout")
            bx.plot(a["x"], a["derivative_random_geometry_0"], color=palette[2], ls=":", label="Randomized geometry")
            bx.set(ylabel="Response units / radian", title="Same attribution scale across randomizations")
        else:
            detail = json.loads((run / "evaluation" / f"{case}_000" / "details.json").read_text())
            matching = next(p["matching"] for p in detail["initialization_pairs"] if p["i"] == 0 and p["j"] == 1)
            assignment = dict(zip(matching["rows"], matching["cols"]))
            for k in range(a["components_0"].shape[1]):
                bx.plot(a["x"], a["components_0"][:, k], color=plt.cm.tab10(k), lw=1.4,
                        label=f"Component {k + 1}, init 1")
                bx.plot(a["x"], a["components_1"][:, assignment[k]], color=plt.cm.tab10(k), lw=1, ls=":")
            bx.set(ylabel="Signed component response", title="Curve-matched components; dotted = init 2")
        bx.set_xlabel("Angle (radians)")
        bx.legend(fontsize=8)
    proxy = summary["cases"]["proxy"]
    ax = axes[0, 2]
    labels = ["Aligned\nfull", "Shifted\nfull", "Shifted\nsensor-only\nretrained", "Shifted\n+π/2\nalignment"]
    keys = ["aligned_clean_nmse", "shifted_clean_nmse"]
    vals = [proxy[k]["mean"] for k in keys]
    # Paired error levels are read from the unit table, not added CI endpoints.
    with (run / "analysis_evaluation" / "units.csv").open(newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["case"] == "proxy"]
    vals.extend([np.mean([float(r["shifted_clean_nmse"]) + float(r[f"shifted_{name}_delta_nmse"]) for r in records])
                 for name in ("sensor_retrained", "phase_plus")])
    ax.bar(range(4), vals, color=[palette[0], "#B34F4D", palette[2], palette[1]])
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Normalized clean MSE", title="Calendar proxy: all 12 trajectories")
    ax.tick_params(axis="x", labelsize=9)
    bx = axes[1, 2]
    contrasts = [("Aligned deletion", "aligned_delete_delta_nmse"),
                 ("Shifted deletion", "shifted_delete_delta_nmse"),
                 ("Shifted reference", "shifted_reference_delta_nmse"),
                 ("Shifted +π/2", "shifted_phase_plus_delta_nmse"),
                 ("Shifted −π/2", "shifted_phase_minus_delta_nmse")]
    for index, (label, key) in enumerate(contrasts):
        v = proxy[key]
        bx.errorbar(v["mean"], index, xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt="o", color=palette[0], capsize=3)
    bx.axvline(0, color="gray", lw=1)
    bx.set(yticks=range(len(contrasts)), yticklabels=[v[0] for v in contrasts],
           xlabel="Paired Δ normalized MSE (95% interval)", title="Recorded-clock dependence in this generator")
    fig.suptitle("VM-N02: accurate functions do not guarantee unique or causal parameter meaning", fontsize=15)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "vm_n02_interpretation.png", dpi=180)
    fig.savefig(destination / "vm_n02_interpretation.svg")
    plt.close(fig)


def main() -> None:
    """Save derived claims and figures, retaining all evaluation-unit evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("docs"))
    args = parser.parse_args()
    run = args.output_dir
    summary = json.loads((run / "analysis_evaluation" / "summary.json").read_text())
    config = json.loads((run / "frozen" / "config.json").read_text())
    with (run / "analysis_evaluation" / "units.csv").open(newline="") as stream:
        units = list(csv.DictReader(stream))
    scorecard = build_scorecard(summary, config, units)
    (args.destination / "vm_n02_scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    with (args.destination / "vm_n02_scorecard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scorecard[0]))
        writer.writeheader()
        writer.writerows(scorecard)
    figures(run, args.destination / "figures", summary)


if __name__ == "__main__":
    main()
