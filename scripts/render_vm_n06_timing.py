"""Render saved N06 teaching samples with correct piecewise-constant source steps.

Display-only successor to the frozen analyzer's line-connected source plot.
Does not generate or alter any trajectory, policy, forecast or reported metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source: Path, output: Path) -> None:
    """Plot saved source steps, sampled states and applied duty."""
    demo = json.loads(source.read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, row in demo['equal_error'].items():
        axes[0].step(demo['hours'], np.array(row['source_forecast']) / 1000,
                     where='post', label=name.replace('_', ' '))
        axes[1].plot(np.arange(97) / 4, row['states'], label=name.replace('_', ' '))
        axes[2].step(demo['hours'], row['actions'], where='post', label=name.replace('_', ' '))
    axes[0].step(demo['hours'], np.array(demo['source']) / 1000, where='post',
                 color='black', linestyle=':', label='Actual source')
    axes[0].set_ylabel('Source (1000 ppm m3/h)')
    axes[1].axhline(900, color='black', linestyle=':')
    axes[1].set_ylabel('Controlled CO2 (ppm)')
    axes[2].set(ylabel='Applied duty (0/1)', xlabel='UTC hour', xlim=(5, 16))
    axes[0].legend()
    fig.suptitle('Equal source errors, separate controlled outcomes (authored teaching case)')
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
