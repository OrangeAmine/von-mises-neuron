"""Plot saved final forecasts without fitting, tuning, or changing scores."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def plot_final(run_dir: Path, output: Path) -> None:
    """Plot all final horizon errors and the first archived forecast origin.

    Args:
        run_dir: Completed practical final evaluation directory.
        output: New PNG path; existing artifacts are never overwritten.
    """
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["phase"] != "final":
        raise ValueError("A completed final run is required")
    fold = run_dir / manifest["test_starts"][0]
    report = json.loads((fold / "results.json").read_text())
    selected = manifest["selection"]["selected_family"]
    names = list(dict.fromkeys([selected, "Seasonal24", "Seasonal168", "Persistence"]))
    colors = ["#147d70", "#d47727", "#8976b5", "#8b949e"]
    labels = {"RidgeResidual": "Selected ridge", "Seasonal24": "Daily seasonal",
              "Seasonal168": "Weekly seasonal", "Persistence": "Persistence"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    for name, color in zip(names, colors):
        metrics = report["families"][name]
        axes[0].plot(range(1, 25), metrics["per_horizon_rmse"], color=color,
                     linewidth=2, label=f"{labels.get(name, name)} ({metrics['rmse']:.3f})")
    origins = report["families"][selected]["n_targets"] // 24
    axes[0].set(title=f"Error across all {origins} final forecast origins", xlabel="Hours ahead",
                ylabel="RMSE (kW)", xlim=(1, 24))
    axes[0].set_xticks([1, 6, 12, 18, 24])
    axes[0].legend(title="Forecast (pooled RMSE, kW)", frameon=False, fontsize=9)
    for name in [selected, "Seasonal24"]:
        suffix = "_ensemble" if name.startswith("Residual") else ""
        with np.load(fold / f"{name}{suffix}.npz", allow_pickle=False) as archive:
            times, prediction, target = archive["target_times"][0], archive["prediction"][0], archive["target"][0]
        axes[1].plot(times, prediction, color=colors[names.index(name)], linewidth=2,
                     label=labels.get(name, name))
    axes[1].plot(times, target, color="#172b4d", linewidth=2, label="Observed")
    origin = str(times[0] - np.timedelta64(1, "h"))[:16].replace("T", " ")
    axes[1].set(title=f"First reserved origin: {origin}", xlabel="Target availability time",
                ylabel="Hourly mean power (kW)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    best_simple = min(report["families"][name]["rmse"] for name in ["Seasonal24", "Seasonal168", "Persistence"])
    gain = 100 * (1 - report["families"][selected]["rmse"] / best_simple)
    fig.suptitle(f"Household power: {gain:.1f}% lower final RMSE than the best simple control",
                 fontsize=14, fontweight="bold")
    fig.supxlabel("One household; 24-hour forecasts overlap. First origin shown by rule, not selected for appearance.",
                  fontsize=9, color="#526170")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_final(args.run_dir, args.output)
