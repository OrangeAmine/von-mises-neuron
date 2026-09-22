# Project instructions

## Purpose and ownership

Treat this repository as an ongoing research engineering project exploring
Von Mises inspired neural layers for circular inputs and time-series forecasting.
Carry work through implementation, appropriate validation, and an honest report
of the evidence. Preserve working experiments and the user's research notes.

Read `docs/project_state.md` for the latest audited handoff. Read the relevant
source and tests before changing behavior. `CLAUDE.md` contains earlier project
guidance; this file updates its incomplete project map and validation guidance.
Design documents describe intentions, not proof of implementation or success.
Update the state document after material architectural or experimental changes.

## Repository map

- `src/von_mises_neuron.py`: `VonMisesLinear`, base VM encoder + LSTM forecaster,
  and pooling/log-output/LayerNorm ablations.
- `src/temporal.py`: angular timestamp features, frozen-kappa bank,
  time-conditioned phases, depthwise temporal convolutions, temporal forecaster.
- `src/vm_variants.py`: deep VM pooling model and VM attention models.
- `src/baselines.py`: LSTM, Transformer, and hand-crafted feature baselines.
- `src/sincos.py`: architecture-matched Cartesian sine/cosine control.
- `src/circular.py`, `src/angular_data.py`: circular losses/metrics and synthetic
  wind, protein, and robotics data.
- `src/data_loading.py`: real occupancy, power-consumption, and Jena loaders.
- `src/preprocessing.py`, `src/split.py`, `src/sequences.py`: preprocessing and
  chronological window generation.
- `scripts/`: synthetic/real benchmarks, ablation, architecture exploration,
  circular experiments, stress tests, interpretability, and presentation tools.
  `controlled_temporal_comparison.py` runs paired synthetic controls;
  `controlled_real_comparison.py` uses contiguous hourly occupancy windows;
  `diagnose_occupancy.py` provides cached-data and persistence diagnostics.
- `tests/`: layer, model, circular, temporal, and data-pipeline tests.
- `docs/`: reports, design proposals, and historical implementation plans.
- `data/raw/`, `results/`: local ignored data and generated artifacts. A fresh
  clone does not include them. Never assume saved results travel with the code.
- `VM_exploration.md`: user research notes; untracked at the initial handoff.

## Environment and commands

Use Python and PyTorch. Prefer an isolated environment and maintain
`requirements.txt` when dependencies change. Do not replace or modify global
Python installations as part of routine work.

On the audited Windows machine, `py -3.11` has the working ML environment;
bare `python` selects Python 3.14 without PyTorch. Verify rather than assume
these machine-specific details on another host. CUDA is the default whenever
`torch.cuda.is_available()` is true; otherwise use CPU. Move models and tensors
consistently with `.to(device)` and store fixed tensors as registered buffers.

Run from the repository root:

```powershell
py -3.11 -m pytest tests/ -q
py -3.11 scripts/benchmark_standard_dataset.py --output-dir results/<run-id>/benchmark
py -3.11 scripts/benchmark_real_datasets.py --help
```

Use the selected environment's `python` instead of `py -3.11` on other hosts.
Check each script's arguments before launching it. `--skip-baselines` on the
synthetic benchmark currently runs both base VM and TemporalVM. Use a new output
directory to avoid overwriting historical evidence. Presentation scripts import
`python-pptx`, which is currently absent from `requirements.txt`.

## Validation and experiment discipline

- Run relevant tests for changes, and the full suite for model/pipeline changes.
  Documentation-only edits do not require training.
- Every new or modified model must pass the standard synthetic benchmark before
  being called validated. Existing acceptance thresholds: RMSE <= 35, MAE <= 25,
  R2 >= 0.20, MAPE <= 5 percent. Do not weaken thresholds to obtain a pass.
- For temporal model changes, also inspect its absolute thresholds and the
  comparison with base VM: at most 10 percent worse RMSE/MAE/MAPE, and at most
  0.10 lower R2 under the current comparator. The standard script now enforces
  both VM models' absolute checks and temporal regression in `all_passed` and
  its exit status. SinCos has absolute gates; other standard baselines have
  finite-only gates. Inspect per-model checks rather than conflating them.
  The controlled runner separates core validation from exploratory ablation
  failures; completion alone does not validate every ablation.
- Validate tensor shapes `(batch, lookback, features)` and `(batch, horizon)`,
  dtype, device, finite outputs, and gradient flow. Test claimed periodicity at
  the boundary where it is meant to hold, including residual paths.
- Split chronologically before fitting normalization and windowing. Fit sensor
  normalization on training data only. Temporal input order is sensor columns
  first, raw angular time columns last; do not z-score those angles.
- Keep timestamp phase origins consistent across splits. Review smoothing and
  imputation for access to observations after the prediction origin.
  Custom angular periods now use a fixed epoch unless a shared origin is
  supplied. Calendar angles use local wall time; annual encoding uses the
  actual Gregorian year length. Hourly means are available at bin end, not
  their left-edge labels. Reject gap-crossing windows in new real-data runs.
- Record seed, configuration, split, target, units, metrics, parameter counts,
  runtime, environment, and code revision. Use multiple seeds for comparative
  claims and report variability. Keep inputs and training budgets comparable.
- Distinguish fresh reproductions, historical saved results, and hypotheses.
  Synthetic domain generators are not real-world domain validation. Test-set
  tuning and repeated selection on the same benchmark limit generalization.
- Preserve meaningful metrics/configuration summaries in tracked documentation
  because `results/` is ignored. Do not commit large raw data or generated decks
  as a substitute for reproducible experiment records.

## Mathematical guardrails

- The core layer uses `z_j = sum_i softplus(raw_kappa_ji) * cos(x_i + mu_ji)`
  and outputs `z + bias` or `exp(z) + bias`. The preferred input angle is `-mu`
  modulo `2*pi`. `raw_kappa` is an unconstrained softplus parameter, not log(kappa).
- Core VM layers are invariant under a `2*pi` shift in each angular input.
  Affine input projections and raw-input residuals can break that guarantee in
  composed models. The temporal encoder's default sensor residual does so;
  distinguish sensor periodicity from periodicity in the time-angle pathway.
- By expanding the cosine, a log-output VM layer is a linear map of per-input
  sine/cosine features with a polar parameterization of its coefficients.
  Include a matched sine/cosine baseline when testing what VM contributes.
- In the current log-output frozen-kappa bank, kappa scales amplitude, not
  frequency or peak width. Daily/weekly periods come from timestamp features.
  Do not claim that frozen concentration bands force temporal decomposition.
- The unnormalized exponential kernel is not a probability density without
  normalization, and adding bias removes its general positivity guarantee.
  Exponential outputs and their gradients can overflow; cosine being bounded
  does not make all parameter-dependent outputs or gradients uniformly bounded.
- Do not infer novelty, general superiority, or LSTM replacement from names,
  design prose, or one successful run. Label untested ideas as hypotheses.

## Implementation style

Use `pathlib.Path`, clear type hints, and Google-style docstrings for new
functions/modules. Explain equations and non-obvious design choices with short
comments. Keep data loading, model definitions, training, and evaluation
separate; reuse existing helpers where practical. Avoid unrelated refactors
during experiments. Report what changed, how it was checked, and remaining
limitations without presenting intended outcomes as observed results.
