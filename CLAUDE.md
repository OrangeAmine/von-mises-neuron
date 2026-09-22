# CLAUDE.md - Project Instructions

> Historical project guidance. `AGENTS.md` and `docs/project_state.md` supersede
> outdated map, environment and mathematical claims below. In particular,
> novelty is unestablished, and composed-model periodicity requires checking
> projections and residuals. Use the current benchmark's explicit gates.

## Project Overview

Von Mises neuron research project. Implements a novel neural network layer
(`VonMisesLinear`) and forecaster (`VonMisesForecaster`) that natively
preserve 2π-periodicity in the input, based on the von Mises distribution.

Core idea: `s = exp(κ · cos(θ + μ))` — the weight sits *outside* the cosine,
so the neuron's output is strictly 2π-periodic in θ (unlike standard neurons
that break periodicity by multiplying inside the nonlinearity).

## Languages

- **Primary:** Python

## Frameworks & Tools

- **ML Framework:** PyTorch (core dependency)
- **Data:** NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn

## Development Commands

- Run benchmark: `python scripts/benchmark_standard_dataset.py`
- Run tests: `pytest tests/`

## Project Structure

```
src/
  von_mises_neuron.py   # VonMisesLinear layer + VonMisesForecaster model
  preprocessing.py      # Z-score normalization, Savitzky-Golay smoothing
  sequences.py          # Sliding window (X, y) sequence creation
  split.py              # Chronological train/val/test splitting
scripts/
  benchmark_standard_dataset.py   # Synthetic signal benchmark
tests/
  test_von_mises_neuron.py        # Unit tests for VM layer and forecaster
```

## Coding Conventions

- Use `pathlib.Path` over `os.path` for file system operations
- Include inline comments explaining *why*, not *what*, for non-obvious logic
- Write Google-style docstrings for new modules and functions you create
- Keep data loading, model definitions, training loops, and evaluation in separate modules
- Store experiment results in organized directories with timestamps or run IDs

## Environment

- Platform: Windows
- **GPU:** Models must use GPU by default when CUDA is available
  - Auto-detect via `torch.cuda.is_available()` and use it when present
- Use virtual environments (venv or conda) for dependency isolation
- Maintain `requirements.txt` for reproducibility

## AI Assistant Best Practices

### Role: Research Engineer

- Prioritize correct tensor shapes, robust data loading, and modular code
- Always verify dimension alignment across layers (input → hidden → output)
- Validate DataLoader outputs (batch shape, dtype, device) before training

### Accuracy First

- Correctness above all else — mistakes erode trust
- If the user is wrong, correct them respectfully rather than appeasing
- Double-check numerical results, shape transformations, and index operations

### Flag Speculation

- Explicitly mark any predictions or speculative reasoning with **[Speculation]** or similar
- Distinguish between established facts, empirical observations, and hypotheses

### White-Box Implementer

- When asked to implement a model, do not just import a library black-box
- Explain the math of specific layers via inline comments in the code
- Include the relevant equations as comments before each computational block

## Mandatory Benchmark

Any new or modified model must pass the synthetic benchmark before being considered valid:
- The benchmark generates a deterministic sinusoidal signal and trains/evaluates the VonMisesForecaster
- The model must meet the acceptance thresholds defined in the script (RMSE, MAE, R2, MAPE)
- Exit code 0 = passed, exit code 1 = failed
- If a model change causes the benchmark to fail, the change must be fixed before merging
