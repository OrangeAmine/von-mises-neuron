# VM methodology v1: implementation and usage

Use the [canonical plan](vm_methodology_plan.md) and
[configuration](../configs/vm_methodology_v1.json) together. The current evidence
and limitations are in [the results report](vm_methodology_results.md).
Stage C, center/width follow-ups and other optional extensions require a separate
protocol; these commands do not authorize expanding the matrix.

Run commands from the repository root. This machine uses `py -3.11`; bare
`python` selects a different installation. Training defaults to CUDA when
available and remains serialized. Do not modify global Python installations.
Use `py -3.11 -m pytest tests/ -q` for the full suite.
## Layers and explicit conversion

[Layer source](../src/circular_layers_v2.py) and
[numerical tests](../tests/test_circular_layers_v2.py) define these public APIs:

- `PolarCircularLinearV2(in_features, out_features, *, response="sum",`
  `init_kappa=1.0, init_policy="constant", implementation="factored", ...)`
  accepts `response="sum"|"exp"`, `init_policy="constant"|"unit_variance"`,
  and `implementation="factored"|"reference"`. Inputs end in `in_features`;
  optional `phase_shift` ends in `(out_features, 1)`.
- `PolarCircularLinearV2.from_legacy(layer)` and
  `CartesianCircularLinearV1.from_polar(layer)` create independent states without
  random draws. Legacy checkpoints are converted explicitly, never overwritten.
  The Cartesian layer also has `from_legacy(layer)`.
- `amplitude_regularizer(layer_or_layers, coefficient)` implements matched
  effective-amplitude regularization; phase/raw-parameter decay is different.
- `PeakCircularKernelBankV1(in_features, num_kernels, *, centers, concentrations,`
  `learn_centers, learn_concentrations, width_sharing="per_edge")` requires explicit
  tensors. It returns features, with a separate signed readout in the trainer.
  Fixed geometry is buffered; the VM and chordal-Gaussian forms are equivalent.
- `DirectCalendarPhaseV1.from_legacy_encoder(encoder)` collapses the actual bank
  and phase projections. `forward(angles)` returns a tuple of phase tensors.

The sum response is not a log-density; exponential responses can overflow.
Conversion parity does not establish equal optimization trajectories or speed.
[Forecaster adapters](../src/methodology_forecasters.py) expose
`PolarForecasterV2.from_legacy(model)` and
`DirectCalendarForecasterV1.from_legacy(model)`. They are engineering controls,
outside the A/B arms; see [adapter tests](../tests/test_methodology_forecasters.py)
and [standard-gate tests](../tests/test_validate_methodology_forecasters.py).

## Tasks, fitting and artifacts

[Task source](../src/methodology_tasks.py) and
[tests](../tests/test_methodology_tasks.py) provide
`generate_task(config, stage, phase, family, replicate_id=0, *, confirmation_lock=None, root=None)`.
It returns `TaskDataV1` with float64 `(N, 1)` arrays. `train_prefix(n)` preserves
nested samples and shared validation/grid arrays. `as_arrays()` exposes trainer
names; `initialization_bundle(config, task, initialization_index=0)` and
`batch_seed(...)` supply dedicated shared streams. Model identity never chooses
observations or batch order.

[Trainer source](../src/methodology_training.py) and
[tests](../tests/test_methodology_training.py) provide:

- `make_a_model(coordinate, amplitude_before, phase_before, *, device=None)`;
  `train_a(model, data, *, optimizer_name, learning_rate, penalty, ...)` records
  the full fixed-update curve. `closed_form_a(data, penalty)` is diagnostic.
- `make_b_model(model_id, initialization_bundle, *, device=None)`;
  `train_b(model, data, *, learning_rate, batch_seed=None, batch_indices=None, ...)`
  selects minimum noisy-validation MSE, including initialization. `min_delta`
  controls a separate patience anchor. `fit_ridge(model_id, data, *, lambdas=...)`
  solves all prescribed candidates and selects on noisy validation.

Fit calls return `{"summary": ..., "artifacts": ...}`, including failure status,
histories, predictions and initial/final/selected states. Training is float32;
metrics and closed-form solves are float64. The runner saves strict JSON and
`artifacts.pt`, loadable with `torch.load(..., map_location="cpu", weights_only=True)`.
Use the runner for protocol work; direct fit APIs do not enforce the entire
pilot/development/confirmation workflow.

## Runner and reporting commands

All argument forms below were checked with their actual `--help` entry points.
Uppercase path names are placeholders. The runner's optional `--config CONFIG`
goes **before** its subcommand; its default is the canonical JSON.

```text
py -3.11 scripts/run_vm_methodology.py run --stage A --phase pilot --output-dir NEW_A_PILOT
py -3.11 scripts/run_vm_methodology.py select-pilots --pilots A_PILOT B_PILOT --output SELECTION_JSON
py -3.11 scripts/run_vm_methodology.py feasibility --pilots A_PILOT B_PILOT --output FEASIBILITY_JSON
py -3.11 scripts/run_vm_methodology.py run --stage A --phase development --output-dir NEW_A_DEV --selection SELECTION_JSON --feasibility FEASIBILITY_JSON
py -3.11 scripts/analyze_vm_methodology.py --inputs A_DEV B_DEV --output DEVELOPMENT_ANALYSIS_JSON
py -3.11 scripts/run_vm_methodology.py lock --selection SELECTION_JSON --development-analysis DEVELOPMENT_ANALYSIS_JSON --feasibility FEASIBILITY_JSON --output LOCK_JSON
py -3.11 scripts/run_vm_methodology.py run --stage A --phase confirmation --output-dir NEW_A_CONFIRM --lock LOCK_JSON
py -3.11 scripts/analyze_vm_methodology.py --inputs A_DIR B_DIR --output ANALYSIS_JSON
py -3.11 scripts/plot_vm_methodology.py --inputs A_DIR B_DIR --analysis ANALYSIS_JSON --output-dir NEW_FIGURES
py -3.11 scripts/summarize_vm_methodology.py --inputs A_DIR B_DIR --analysis ANALYSIS_JSON --output SECONDARY_JSON
```

Run each training stage separately with `--stage B` for its paired B invocation.
Plotting optionally accepts `--show-task-points`; the secondary summary adds no
gates. Both require complete audited inputs bound to `ANALYSIS_JSON`.
[Runner tests](../tests/test_run_vm_methodology.py) and
[independent audit tests](../tests/test_analyze_vm_methodology.py) cover enforcement.
## Preservation and confirmation boundaries

Every output directory/report must be new. `run --resume` uses the original
invocation directory and original absolute deadline; it does not renew time or
retry failed/interrupted fits. Running manifests use hash-verified summary
references; final manifests restore full summaries. Detailed per-fit evidence
remains saved throughout, and corruption is rejected before resuming.

The canonical JSON intentionally retains its historical design status and null
learning-rate/comparator fields. Current resolved choices live in the pilot
selection and `results/vm_methodology_v1/study_20260908/confirmation_lock.json`:
A Adam 0.1, SGD 0.3; B Adam 0.01 for every gradient arm; comparator `fixed_vm_ridge`.
The lock binds source, configuration, generator, development and analysis evidence.
Reserved functions cannot be generated or restored without a verified lock.
Once accessed, confirmation seeds/functions cannot support a new tuned-confirmatory
claim; changes require a new protocol and unused seeds. Never silently retry or
select favorable surviving runs.

`results/` and generated data are ignored: a fresh clone does not contain the
lock, fitted states, exact arrays or reports. Preserve the matching artifact
directories and hashes for replay; tracked documentation alone cannot reproduce
saved checkpoints. Keep frozen forecasting evidence and owner notes intact.
