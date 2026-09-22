"""Plot complete audited VM methodology evidence without fitting or generating data.

Curves show arithmetic means over designed synthetic tasks. Initialization seeds
are averaged within B tasks first. These figures do not treat observations,
seeds or grid points as independent experimental replications, and contain no
population-confidence bands or automatic methodological conclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COORDINATES = {"polar": ("Polar", "#007f86"), "cartesian": ("Cartesian", "#cb6527")}
B_MODELS = {
    "learned_vm": ("Learned VM", "#007f86", "o"),
    "fourier_ridge": ("Fourier ridge", "#314ea3", "s"),
    "fixed_vm_ridge": ("Fixed VM ridge", "#9556a4", "D"),
    "learned_geodesic": ("Learned geodesic", "#cb6527", "^"),
    "sincos_mlp": ("SinCos MLP", "#658e32", "v"),
    "raw_angle_mlp": ("Raw-angle MLP", "#6b7280", "P"),
}
FAMILY_NAMES = {
    "phase_only": "Phase change", "amplitude_only": "Amplitude change",
    "both": "Phase and amplitude change", "weak_phase": "Weak-signal phase change",
    "smooth": "Smooth periodic", "narrow": "Narrow periodic",
    "multimodal": "Multimodal periodic", "nonperiodic": "Nonperiodic negative control",
}


def file_hash(path: Path) -> str:
    """Return SHA256 of exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read finite JSON without importing experiment modules."""
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_audited_inputs(inputs: Sequence[Path], analysis_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require complete same-phase A/B audits bound to the exact input manifests.

    This function only reads saved JSON. It does not access task generators,
    model checkpoints, training code, GPU APIs or unused confirmation seeds.
    """
    analysis = read_json(analysis_path)
    if analysis.get("complete") is not True:
        raise ValueError("Reporting requires analysis.complete=true")
    if analysis.get("phase") not in ("development", "confirmation"):
        raise ValueError("Full-study figures require development or confirmation; pilots are disposable")
    if len(inputs) != 2:
        raise ValueError("Provide exactly one A directory and one B directory")
    expected_hashes = {str(Path(name).resolve()): value for name, value in analysis["input_manifest_hashes"].items()}
    actual_paths = {str((directory / "manifest.json").resolve()) for directory in inputs}
    if set(expected_hashes) != actual_paths:
        raise ValueError("Inputs differ from the analysis input manifest set")
    audits = analysis.get("audits", [])
    if len(audits) != 2 or {entry["stage"] for entry in audits} != {"A", "B"} or not all(entry.get("complete") for entry in audits):
        raise ValueError("Both complete A and B audits are required")
    manifests = {}
    for directory in inputs:
        path = directory / "manifest.json"
        if file_hash(path) != expected_hashes[str(path.resolve())]:
            raise ValueError(f"Input manifest changed after analysis: {path}")
        manifest = read_json(path)
        if manifest.get("status") != "completed" or any(row["status"] != "completed" for row in manifest["runs"]):
            raise ValueError("Incomplete or failed cells cannot be hidden in figures")
        if manifest["phase"] != analysis["phase"] or manifest["stage"] in manifests:
            raise ValueError("Inputs must have matching phases and distinct stages")
        manifests[manifest["stage"]] = manifest
    if set(manifests) != {"A", "B"} or manifests["A"]["config"] != manifests["B"]["config"]:
        raise ValueError("A/B input configurations must agree")
    return manifests, analysis


def prepare_a_curves(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Average normalized clean MSE curves over whole paired A task bundles."""
    spec = manifest["config"]["A"]
    count = spec[manifest["phase"]]["task_pairs"]
    groups: dict[tuple[str, str, float, str], dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for row in manifest["runs"]:
        if row["kind"] != "gradient":
            continue
        key = (row["family"], row["optimizer"], row["penalty"], row["model"])
        if row["replicate"] in groups[key]:
            raise ValueError(f"Repeated task in A curve group: {key}")
        groups[key][row["replicate"]] = row["summary"]["history"]
    steps = list(range(0, spec["updates"] + 1, spec["record_every"]))
    if steps[-1] != spec["updates"]:
        steps.append(spec["updates"])
    result = []
    for family in spec["family_ids"]:
        for optimizer in spec["optimizers"]:
            for penalty in spec["explicit_amplitude_penalties"]:
                for coordinate in spec["models"]:
                    key = (family, optimizer, penalty, coordinate)
                    tasks = groups.get(key, {})
                    if set(tasks) != set(range(count)):
                        raise ValueError(f"Missing or unplanned A task curves: {key}")
                    values = []
                    for replicate in range(count):
                        history = tasks[replicate]
                        if [item["step"] for item in history] != steps:
                            raise ValueError("A curves must retain the full prescribed recording grid")
                        normalized = [item["clean_mse"] / item["clean_variance"] for item in history]
                        if not np.isfinite(normalized).all() or np.any(np.asarray(normalized) < 0):
                            raise ValueError("Invalid A clean learning curve")
                        values.append(normalized)
                    result.append({"family": family, "optimizer": optimizer, "penalty": penalty,
                                   "model": coordinate, "steps": steps, "task_ids": list(range(count)),
                                   "task_curves": values, "mean_curve": np.mean(values, axis=0).tolist()})
    if len(groups) != len(result):
        raise ValueError("Unplanned A curve groups")
    return result


def prepare_b_curves(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Average initialization errors within each task, then tasks arithmetically."""
    spec = manifest["config"]["B"]
    count = spec[manifest["phase"]]["task_pairs"]
    sizes = spec["train_observations"]
    grouped: dict[tuple[str, str, int], dict[int, dict[int | None, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in manifest["runs"]:
        key = (row["family"], row["model"], row["n_train"])
        seeds = grouped[key][row["replicate"]]
        index = row["initialization_index"]
        if index in seeds:
            raise ValueError("Duplicate task/initialization/sample-size B result")
        error = row["summary"]["primary_error"]
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid B clean-function error")
        seeds[index] = error
    result = []
    for family in spec["family_ids"]:
        for model in spec["models"]:
            model_id = model["id"]
            expected_seeds = {None} if model["kind"] == "ridge" else set(range(spec[manifest["phase"]]["initialization_seeds"]))
            task_curves: list[list[float]] = [[] for _ in range(count)]
            for size in sizes:
                group = grouped.get((family, model_id, size), {})
                if set(group) != set(range(count)):
                    raise ValueError("Missing or unplanned B task results")
                for replicate in range(count):
                    if set(group[replicate]) != expected_seeds:
                        raise ValueError("Missing or unplanned B initialization results")
                    task_curves[replicate].append(float(np.mean(list(group[replicate].values()))))
            mean = np.mean(task_curves, axis=0).tolist()
            reported = analysis["B"]["models"][model_id]["per_size_family_errors"][family]
            if not np.allclose(mean, [reported[str(size)] for size in sizes], atol=1e-12, rtol=1e-12):
                raise ValueError("B plotting aggregates disagree with audited arithmetic means")
            result.append({"family": family, "model": model_id, "sample_sizes": sizes,
                           "task_ids": list(range(count)), "task_curves": task_curves,
                           "mean_curve": mean})
    expected_groups = len(spec["family_ids"]) * len(spec["models"]) * len(sizes)
    if len(grouped) != expected_groups:
        raise ValueError("Unplanned B result groups")
    return result


def _style_axes(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9, colors="#374151")
    axis.grid(axis="y", color="#e5e7eb", linewidth=.7)
    axis.set_axisbelow(True)


def plot_a(curves: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], path: Path,
           show_tasks: bool) -> None:
    """Render every A condition/optimizer/penalty/coordinate cell."""
    spec = manifest["config"]["A"]
    families, optimizers = list(spec["family_ids"]), list(spec["optimizers"])
    figure, axes = plt.subplots(len(families), len(optimizers), figsize=(12, 13), squeeze=False)
    for family_index, family in enumerate(families):
        for optimizer_index, optimizer in enumerate(optimizers):
            axis = axes[family_index, optimizer_index]
            for curve in curves:
                if curve["family"] != family or curve["optimizer"] != optimizer:
                    continue
                _, color = COORDINATES[curve["model"]]
                style = "-" if curve["penalty"] == 0 else "--"
                if show_tasks:
                    for values in curve["task_curves"]:
                        axis.plot(curve["steps"], values, color=color, linestyle=style, linewidth=.65, alpha=.16)
                axis.plot(curve["steps"], curve["mean_curve"], color=color, linestyle=style, linewidth=2.1)
            _style_axes(axis)
            axis.set_title(f"{FAMILY_NAMES[family]} · {optimizer}", loc="left", fontsize=11, pad=11)
            axis.set_xlim(0, spec["updates"])
            axis.set_ylim(bottom=0)
            axis.set_xlabel("Adaptation updates", fontsize=10)
            axis.set_ylabel("Clean MSE / clean variance", fontsize=10)
    handles = [Line2D([0], [0], color=color, linewidth=2.1, label=name) for name, color in COORDINATES.values()]
    handles += [Line2D([0], [0], color="#4b5563", linestyle="-" if penalty == 0 else "--", linewidth=2,
                       label=f"Amplitude penalty λ = {penalty:g}") for penalty in spec["explicit_amplitude_penalties"]]
    figure.suptitle(f"Stage A · {manifest['phase'].capitalize()} learning curves", x=.075, ha="left", fontsize=18, fontweight="bold", y=.981)
    figure.text(.075, .951, f"All four conditions and both optimizers · arithmetic means over {spec[manifest['phase']]['task_pairs']} paired synthetic tasks", fontsize=11, color="#4b5563")
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045), ncol=4, frameon=False, fontsize=10)
    caption = "Descriptive task averages; no population-confidence bands. Every recorded update is retained."
    if show_tasks:
        caption += " Faint curves show individual tasks."
    figure.text(.075, .018, caption, fontsize=9, color="#4b5563", wrap=True)
    figure.tight_layout(rect=(.035, .09, .995, .935), h_pad=2.3, w_pad=2.0)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_b(curves: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], path: Path,
           show_tasks: bool) -> str:
    """Render B sample-efficiency curves, keeping the negative control distinct."""
    spec = manifest["config"]["B"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    positive = all(value > 0 for curve in curves for values in curve["task_curves"] for value in values)
    scale = "log" if positive else "linear"
    for axis, family in zip(axes.flat, spec["family_ids"]):
        if family == "nonperiodic":
            axis.set_facecolor("#fff7ed")
        for curve in curves:
            if curve["family"] != family:
                continue
            label, color, marker = B_MODELS[curve["model"]]
            style = "--" if curve["model"] == "raw_angle_mlp" else "-"
            if show_tasks:
                for values in curve["task_curves"]:
                    axis.scatter(curve["sample_sizes"], values, color=color, marker=marker, s=17, alpha=.19, edgecolors="none")
            axis.plot(curve["sample_sizes"], curve["mean_curve"], color=color, marker=marker, markersize=5,
                      linewidth=2.0, linestyle=style, label=label)
        _style_axes(axis)
        axis.set_xscale("log", base=2)
        axis.set_xticks(spec["train_observations"], [str(value) for value in spec["train_observations"]])
        axis.set_xlim(26, 640)
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        axis.set_title(FAMILY_NAMES[family], loc="left", fontsize=12, pad=11)
        axis.set_xlabel("Training observations", fontsize=10)
        axis.set_ylabel("Clean MSE / clean variance" + (" (log scale)" if scale == "log" else ""), fontsize=10)
        if family == "nonperiodic":
            axis.text(.03, .035, "Excluded from periodic success gates", transform=axis.transAxes, fontsize=9, color="#9a3412")
    figure.suptitle(f"Stage B · {manifest['phase'].capitalize()} sample efficiency", x=.075, ha="left", fontsize=18, fontweight="bold", y=.978)
    figure.text(.075, .938, f"All six models · initialization means within tasks, then arithmetic means over {spec[manifest['phase']]['task_pairs']} tasks per family", fontsize=11, color="#4b5563")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .063), ncol=3, frameon=False, fontsize=10)
    caption = "Means are computed on the original error scale; no powered population inference or independent-seed claim."
    if show_tasks:
        caption += " Faint points show task means."
    figure.text(.075, .024, caption, fontsize=9, color="#4b5563", wrap=True)
    figure.tight_layout(rect=(.035, .15, .995, .92), h_pad=2.2, w_pad=2.6)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return scale


def create_figures(inputs: Sequence[Path], analysis_path: Path, output: Path,
                   *, show_task_points: bool = False) -> dict[str, Any]:
    """Create a new reporting directory from already audited evidence only."""
    manifests, analysis = load_audited_inputs(inputs, analysis_path)
    a_curves = prepare_a_curves(manifests["A"])
    b_curves = prepare_b_curves(manifests["B"], analysis)
    output.mkdir(parents=True, exist_ok=False)
    plot_data = {"phase": analysis["phase"], "A": a_curves, "B": b_curves,
                 "aggregation": "A: arithmetic task means of full normalized clean MSE curves; B: arithmetic initialization means within task/sample size, then arithmetic task means"}
    _write_json(output / "plot_data.json", plot_data)
    plot_a(a_curves, manifests["A"], output / "A_learning_curves.png", show_task_points)
    b_scale = plot_b(b_curves, manifests["B"], output / "B_sample_efficiency.png", show_task_points)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "phase": analysis["phase"],
              "source_script": str(Path(__file__).resolve()), "source_script_sha256": file_hash(Path(__file__)),
              "analysis_path": str(analysis_path.resolve()), "analysis_sha256": file_hash(analysis_path),
              "input_manifest_hashes": analysis["input_manifest_hashes"],
              "show_task_points": show_task_points, "B_y_scale": b_scale,
              "claim_scope": "Descriptive synthetic task averages; no powered population inference, independent-seed interpretation or automatic superiority conclusion",
              "aggregation": plot_data["aggregation"],
              "outputs": {name: file_hash(output / name) for name in ("plot_data.json", "A_learning_curves.png", "B_sample_efficiency.png")}}
    _write_json(output / "figure_manifest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Provide a reporting-only CLI; existing output directories are refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, type=Path, required=True, metavar=("A_DIR", "B_DIR"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--show-task-points", action="store_true", help="Show faint individual A task curves and B task means")
    args = parser.parse_args(argv)
    report = create_figures(args.inputs, args.analysis, args.output_dir, show_task_points=args.show_task_points)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
