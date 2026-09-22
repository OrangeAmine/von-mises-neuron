# Controlled comparison plan, 2026-09-08

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan and run under delegated project orchestration
- Origin Date: 2026-09-08
- Verification Status: UNVERIFIED (plan; results recorded separately)
- Version Label: controlled_comparison_v1

## Question and scope

Does the temporal phase pathway contribute useful predictive information beyond
the existing VM encoder and a Cartesian sine/cosine encoder? Which parts of the
temporal architecture account for its behavior? This is an exploratory local
engineering experiment, not a novelty claim or a confirmatory study.

The user authorized autonomous implementation and experiment selection from the
audited project state. No external data upload or paid compute is involved.

## Fixed protocol before training

- Synthetic generator: existing 90-day CO2 series, data seed 42.
- Historical calendar protocol: `--calendar-clock local` (Europe/Paris, default).
  Prospective fixed-offset protocol: `--calendar-clock fixed` (UTC+01:00,
  preserving winter fitting angles and elapsed generator phases). Keep separate
  output directories and record the choice; never silently replace the historical
  arm. The rationale and frozen-checkpoint sensitivity are in
  `docs/synthetic_clock_audit.md`. This option was added after the original run.
- Chronological 80/10/10 split; train-only normalization; no smoothing.
- Inputs: 72 hourly observations; output: next 12 CO2 observations in ppm.
- Training seeds: 42, 43, 44. Dataset stays fixed, so variability describes
  initialization/training, not variation across sampled datasets.
- Existing optimizer and budget: Adam, learning rate 0.001, weight decay 0.0001,
  batch 64, maximum 50 epochs, patience 15, gradient clip 1, existing scheduler.
- Save actual epochs, validation loss, parameter counts, runtime, environment,
  source hashes, revision, split boundaries and normalization statistics.
- Select checkpoints on validation loss only. No hyperparameter search or
  retries selected for favorable test outcomes.

## Comparisons

1. Base VM versus SinCos: identical inputs, architecture, parameter count and
   initial function. Copy the VM's non-encoder weights and convert polar
   coefficients to Cartesian. Reset the training RNG after model construction
   for equal minibatch/dropout random streams. Adam and weight decay act in
   different coordinates: the comparison includes their interaction with the
   parameterization.
2. TemporalVM full versus no time conditioning, all-one fixed amplitudes,
   learnable amplitudes, no convolution bridges, and mean pooling. These are
   component removals or substitutions with the same training budget; report
   their differing parameter counts. Convolution removal also removes the
   existing residual bridge, so it does not isolate convolution alone.
   Before the first controlled run, the initialization protocol was refined:
   copy every shared tensor from the full temporal reference into each ablation.
   For unit amplitudes, absorb the original amplitudes into the phase projection
   columns, giving the same initial phase function. The fixed-scale comparison
   therefore probes parameterization and optimization, not distinct frequencies.
3. Base VM and SinCos with the same appended raw calendar angles as TemporalVM,
   to distinguish extra timestamp information from phase conditioning.
4. Persistence and daily seasonal persistence, plus the existing hand-crafted
   linear comparator, establish simple reference performance.

Sensor inputs here are real-valued measurements, not physical angles. Their
z-scored values enter trigonometric features as an architectural choice. The
temporal residual deliberately retains their nonperiodic information; no
end-to-end sensor periodicity is required for this forecasting task. True
circular-input tasks require a separate protocol.

## Reporting and decisions

Report every seed and mean/sample standard deviation for RMSE, MAE, R2 and MAPE.
Use paired seed differences descriptively, without p-values or generalization
claims from three seeds. Retain failed variants in the report. Every new model
is checked against unchanged synthetic thresholds (RMSE <=35, MAE <=25,
R2 >=0.20, MAPE <=5%). Temporal full also faces the existing base-VM regression
limits: 10% error and 0.10 absolute R2. Failed ablations remain unvalidated,
without weakening gates or tuning against test results.

Inspect real occupancy data next using the same target/split as its existing
benchmark, persistence/seasonal controls and split distributions. Other real
datasets follow only after checking regular sampling and missing-data handling.
The synthetic protocol will not be revised in response to those test metrics.

Run outputs go into new directories under `results/direction_2026-09-08/`.
Monitor process completion and per-model progress with a 30-minute ceiling per
synthetic invocation; retain partial results if a run fails. Tracked Markdown
reports preserve meaningful results because `results/` is ignored.
