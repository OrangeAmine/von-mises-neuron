"""Create compact VM-N01 tables and a static figure from immutable fit receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.smooth_local_analysis_v1 import summarize
from src.smooth_local_protocol_v1 import file_hash, load_config, verify_lock, write_json


def read(path: Path) -> Any:
    """Read UTF-8 evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Create a compact table with every declared cell retained."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime_rows(summary: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Report actual cost and selected updates alongside equal allowances."""
    output = []
    primary = config["training"]["primary_time_seconds"]
    for spec in config["models"]:
        rows = [r for r in summary["rows"] if r["model_id"] == spec["id"] and r["allowance"] == primary]
        times = np.array([r["fit_seconds"] for r in rows], dtype=float)
        steps = np.array([r["selected_step"] for r in rows if r["selected_step"] is not None], dtype=float)
        output.append({"phase": summary["phase"], "model_id": spec["id"], "fits": len(rows),
            "features": spec["features"], "parameters": spec["parameters"],
            "fit_seconds_mean": float(times.mean()), "fit_seconds_sd": float(times.std(ddof=1)),
            "fit_seconds_min": float(times.min()), "fit_seconds_max": float(times.max()),
            "selected_step_mean": float(steps.mean()) if len(steps) else None,
            "selected_step_min": float(steps.min()) if len(steps) else None,
            "selected_step_max": float(steps.max()) if len(steps) else None})
    return output


def plot_summary(path: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    """Plot family errors and paired locked-comparator intervals."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    selected = summary["gate"]["selected_hybrid"]
    capacity = summary["gate"]["capacity_comparator"]
    ids = ["fourier16", "hybrid_fixed16", capacity, "adaptive16_joint", selected]
    families = config["tasks"]["periodic_families"]
    x = np.arange(len(families))
    width = .15
    colors = ["#475569", "#94a3b8", "#b45309", "#0d9488", "#4338ca"]
    for index, (model, color) in enumerate(zip(ids, colors)):
        values = [summary["models"][model]["family_errors"][f] for f in families]
        axes[0].bar(x+(index-2)*width, values, width, label=model, color=color)
    axes[0].set_xticks(x, families)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized clean MSE (log scale; lower is better)")
    axes[0].set_title("Confirmation families | 64 / 256 total labels")
    axes[0].legend(fontsize=8, frameon=False)
    for index, (comparator, contrast) in enumerate(summary["contrasts"].items()):
        value = 100*contrast["relative_improvement"]
        low, high = np.array(contrast["interval"])*100
        axes[1].errorbar(value, index, xerr=[[value-low], [high-value]], fmt="o", capsize=4, color="#4338ca")
    axes[1].set_yticks(range(len(summary["contrasts"])), list(summary["contrasts"]))
    axes[1].axvline(0, color="#64748b", linewidth=1)
    axes[1].axvline(10, color="#b45309", linestyle="--", linewidth=1, label="10% practical threshold")
    axes[1].set_xlabel(f"{selected}: error reduction (%)")
    axes[1].set_title("Paired 95% intervals | whole functions")
    axes[1].legend(fontsize=8, frameon=False)
    figure.suptitle(f"VM-N01: benefit gate {'PASS' if summary['gate']['passed'] else 'FAIL'} | 6 functions/family, 2 initializations", fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Write new reporting artifacts; never fit or tune models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    config = load_config()
    lock = verify_lock(args.lock, config)
    phases = {phase: read(args.study_dir / phase / "summary.json") for phase in ("pilot", "development", "confirmation")}
    receipts = {phase: read(args.study_dir / phase / "receipt.json") for phase in phases}
    for phase in phases:
        if file_hash(args.study_dir / phase / "summary.json") != receipts[phase]["summary_sha256"]:
            raise ValueError("Phase summary changed after terminal receipt")
    selected = lock["selected_models"]
    analyses = {phase: summarize(phases[phase]["rows"], config, selected["selected_hybrid"], selected["capacity_comparator"], phase)
                for phase in ("development", "confirmation")}
    runtime = [row for phase in phases.values() for row in runtime_rows(phase, config)]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "vm_n01_cells.csv", [r for a in analyses.values() for r in a["cells"]])
    write_csv(output / "vm_n01_function_results.csv", [r for a in analyses.values() for r in a["functions"]])
    write_csv(output / "vm_n01_pilot_results.csv", phases["pilot"]["rows"])
    write_csv(output / "vm_n01_runtime.csv", runtime)
    engineering = read(args.study_dir / "engineering_standard/results.json")
    compact = {"protocol": config, "confirmation_lock_sha256": file_hash(args.lock),
        "inherited_archive_sha256": "18191869b3383ee437e33de713b4aaed6a7f103e9ae88c5581ce8ea8a44f31b5",
        "source_identity": lock["sources"], "selected_rates": lock["selected_rates"],
        "selected_models": selected, "feasibility": lock["feasibility"], "phase_receipts": receipts,
        "environment": read(args.study_dir / "confirmation/plan.json")["environment"],
        "engineering": engineering,
        "analyses": {phase: {key: value for key, value in analysis.items() if key not in ("cells", "functions")}
                     for phase, analysis in analyses.items()},
        "runtime": runtime,
        "phase_timings": {phase: summary["timing"] for phase, summary in phases.items()},
        "artifact_scope": "Exact arrays and checkpoints remain in ignored results; CSVs retain all compact declared results."}
    write_json(output / "vm_n01_run_record.json", compact)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure_path = figure_dir / "vm_n01_confirmation.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    if analyses["confirmation"]["integrity"]["complete"]:
        plot_summary(figure_path, analyses["confirmation"], config)
    print(json.dumps(analyses["confirmation"]["gate"], indent=2))


if __name__ == "__main__":
    main()
