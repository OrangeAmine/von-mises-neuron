"""Export descriptive N07 evidence and a figure from the locked run artifacts.

This reporting-only exporter was added after confirmation. All selection,
estimands and gates are implemented in the prospectively frozen runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.benchmark_standard_dataset import compute_benchmark_metrics
from scripts.run_calendar_trust_v1 import audit, digest, verify_lock, write_json
from src.calendar_trust_v1 import delayed_mix


def export(study: Path, evidence: Path, figure: Path) -> None:
    """Replay evidence, retain room-level records and plot preset diagnostics."""
    if evidence.exists() or figure.exists():
        raise FileExistsError("refusing to overwrite an existing report artifact")
    lock = verify_lock(study)
    replay = audit(study)
    config = lock["config"]
    confirm = json.loads((study / "confirmation/summary.json").read_text())
    development = json.loads((study / "development/summary.json").read_text())
    engineering = json.loads((study / "engineering/results.json").read_text())
    eta = confirm["selected_eta"]
    rows = confirm["records"][str(eta)]
    # Replay engineering metrics too; checksum changes are never silently accepted.
    for name, expected in engineering["artifacts_sha256"].items():
        if digest(study / "engineering" / name) != expected:
            raise ValueError(f"changed engineering artifact: {name}")
    with np.load(study / "engineering/predictions.npz") as arrays:
        for arm, record in engineering["acceptance"].items():
            computed = compute_benchmark_metrics(torch.as_tensor(arrays[arm]), torch.as_tensor(arrays["truth"]),
                                                 engineering["normalization"])
            if computed != record["metrics"]:
                raise AssertionError(f"engineering metric replay differs: {arm}")
        mixed = delayed_mix(arrays["experts"], arrays["truth"], arrays["issue_ns"], arrays["target_ns"],
                            arrays["target_ns"], arrays["initial_loss"], int(arrays["initialization_cutoff_ns"]),
                            1., eta, config["feedback_half_life_updates"])
        np.testing.assert_array_equal(mixed.prediction, arrays["AdaptiveMixture"])
    aggregates = []
    arms = ("sensor", "seasonal", "fixed", "adaptive")
    for regime in config["regimes"]:
        for arm in arms:
            selected = [r for r in rows if r["regime"] == regime and r["arm"] == arm]
            row = {"regime": regime, "arm": arm, "rooms": len(selected), "endpoints": {}}
            for step in [config["primary_step"], *config["diagnostic_steps"]]:
                endpoint = [r["endpoints"][str(step)] for r in selected]
                row["endpoints"][str(step)] = {
                    "room_mean_rmse_ppm": float(np.mean([e["rmse"] for e in endpoint])),
                    "room_sd_rmse_ppm": float(np.std([e["rmse"] for e in endpoint], ddof=1)),
                    "room_mean_mae_ppm": float(np.mean([e["mae"] for e in endpoint])),
                    "room_mean_normalized_rmse": float(np.mean([e["rmse"]/r["normalization_std"] for e, r in zip(endpoint, selected)]))}
            for portion in ("early_48h", "later"):
                row[portion] = {"room_mean_rmse_ppm": float(np.mean([r[portion]["rmse"] for r in selected])),
                                "room_mean_mae_ppm": float(np.mean([r[portion]["mae"] for r in selected]))}
            aggregates.append(row)
    weights = {}
    for regime in config["regimes"]:
        per_room = []
        for seed in config["confirmation_seeds"]:
            with np.load(study / f"confirmation/room_{seed}_{regime}_eta_{eta:g}.npz") as data:
                days = (data["issue_ns"] - data["test_start_ns"]) // (24 * 3_600_000_000_000)
                values = data["weights"][:, config["primary_step"]-1, 1]
                per_room.append([float(values[days == day].mean()) for day in range(14)])
        weights[regime] = {"per_room_daily_means": per_room, "mean": np.mean(per_room, axis=0).tolist(),
                           "sd": np.std(per_room, axis=0, ddof=1).tolist()}
    preservation = json.loads((ROOT / "docs/integration/preservation_20260929.json").read_text())
    imported = [r["path"] for r in preservation["imports"]]
    changed = subprocess.check_output(["git", "diff", lock["code_revision"], "--name-only", "--", *imported], cwd=ROOT, text=True)
    notes_path = ROOT / "VM_exploration.md"
    notes_ok = digest(notes_path) == preservation["owner_notes_sha256"] if notes_path.exists() else None
    if changed.strip() or notes_ok is False:
        raise AssertionError("historical delivery or owner notes changed")
    receipt = {"material_passport": "ANALYZED; calculations replayed by the same agent",
               "study": config["study"], "selected_eta": eta, "analysis": confirm["analysis"],
               "engineering_acceptance": engineering["acceptance"], "all_new_mixtures_passed": engineering["all_new_mixtures_passed"],
               "development_selection_scores": development["selection_scores"], "aggregate_diagnostics": aggregates,
               "calendar_weight_diagnostics": weights, "room_records": rows,
               "room_fit_receipts": json.loads((study / "confirmation/manifest.json").read_text())["rooms"],
               "pytest_receipt": (study / "engineering/pytest.log").read_text().splitlines()[-1],
               "replay": replay, "engineering_replay": "all five metric sets and adaptive forecasts matched exactly",
               "preservation": {"imported_files_unchanged_since_integration": len(imported), "owner_notes_unchanged": notes_ok,
                                "comparison_revision": lock["code_revision"],
                                "note": "four teaching files already differed from original imports in the integration commit; null notes status means absent untracked file"},
               "runtime_seconds": {"development": development["seconds"], "confirmation": confirm["seconds"], "engineering_including_tests": engineering["seconds"]},
               "environment": lock["environment"], "source_revision": lock["code_revision"],
               "source_sha256": lock["source_sha256"], "report_exporter_sha256": digest(Path(__file__)),
               "evidence_sha256": {name: digest(study / name) for name in ("lock.json", "source_snapshot.zip", "development/summary.json",
                    "engineering/results.json", "engineering/pytest.log", "confirmation/summary.json", "audit.json")}}
    evidence.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#53677b", "#b87225", "#8c74ab", "#087f82"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for axis, regime in zip(axes[:2], config["regimes"]):
        selected = [r for r in aggregates if r["regime"] == regime]
        means = [r["endpoints"]["4"]["room_mean_rmse_ppm"] for r in selected]
        deviations = [r["endpoints"]["4"]["room_sd_rmse_ppm"] for r in selected]
        axis.bar(np.arange(4), means, yerr=deviations, capsize=3, color=colors)
        axis.set_xticks(np.arange(4), ["Sensor", "Seasonal", "50:50", "Adaptive"], rotation=25)
        axis.set_title("Stable routine" if regime == "stable" else "Routine shifted by +3 hours")
        axis.set_ylabel("One-hour RMSE (ppm)\nRoom mean ± room SD")
        axis.spines[["top", "right"]].set_visible(False)
    for regime, color in zip(config["regimes"], ("#b87225", "#087f82")):
        mean, sd = np.array(weights[regime]["mean"]), np.array(weights[regime]["sd"])
        days = np.arange(14) + 1
        axes[2].plot(days, mean, color=color, label=regime.capitalize(), linewidth=2)
        axes[2].fill_between(days, np.maximum(0, mean-sd), np.minimum(1, mean+sd), color=color, alpha=.15)
    axes[2].axhline(.5, color="#999999", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Test day", ylabel="Calendar weight at one-hour horizon", ylim=(0, 1), title="Received errors change calendar trust")
    axes[2].legend(frameon=False)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("N07 confirmation: 12 paired synthetic rooms • fixed eta = 8", fontsize=14)
    fig.text(.5, .01, "Scientific gate passed; standard base-VM regression gate failed. Shading is room SD, not a confidence interval.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    receipt["figure_sha256"] = digest(figure)
    write_json(evidence, receipt)
    print(json.dumps({"analysis": receipt["analysis"], "aggregate_diagnostics": aggregates,
                      "calendar_weight_means": {k: v["mean"] for k, v in weights.items()}}, indent=2))


def main() -> None:
    """Export into caller-supplied fresh artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    export(args.study, args.evidence, args.figure)


if __name__ == "__main__":
    main()
