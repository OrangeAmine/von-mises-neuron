# Circular locality v2: prospective protocol

## Material Passport

- Origin: academic-research-suite / experiment-agent, implementation and execution.
- Date: 2026-09-21.
- Authority: owner requested saving the review and starting the proposed question.
- Contract: this document and [vm_locality_v2.json](../configs/vm_locality_v2.json).
- Status at drafting: no v2 training or confirmation functions accessed.
- This is a local prospective protocol, not an externally registered study.

## Question and scope

When does adapting circular basis locations and widths improve prediction under
equal label and computation budgets? The [literature review](vm_literature_review.md)
motivates variable projection and fixed circular/Fourier controls. V1's negative
results remain unchanged. This protocol introduces fresh task streams and new
modules, without editing frozen v1 implementations or confirmation evidence.

The scope is real-valued regression on a scalar angular input. Stage A, relative
angle interactions, uncertain circular outputs, forecasting and large/deep
architectures are not included. No new forecaster adapter is introduced; scalar
methodology results do not establish standard forecasting validation. Existing
forecaster acceptance thresholds remain unchanged.

## Tasks and total label budgets

Use the four declared v1 mathematical families (smooth, narrow, multimodal,
nonperiodic) with entirely fresh random streams. Each target is standardized by
its generator on 65,536 deterministic midpoints; this is a declared synthetic
generator, not preprocessing fitted on test labels. Evaluation uses 4,096 clean
midpoints, inaccessible to optimization, checkpoint or pilot-rate selection.

Total labeled budgets are 64, 128 and 512, including validation labels. Split
each budget 75/25: 48/16, 96/32 and 384/128 training/validation. Separate nested
input/noise streams prevent any label switching from validation to training
across budgets. All models share exact observations, splits and initialization
streams. Both noise standard deviations, 0.05 and 0.2, are crossed with every
target. They may share unit-noise draws scaled by the respective noise level.
Neither noise levels, sample sizes nor initializations are independent targets.

Pilot has one target per family and one initialization. Development has three
targets per family and two initializations; confirmation has eight and two.
The primary population is the three periodic generator families. Nonperiodic
targets diagnose misspecification; all ten models are periodic, so this arm is
not a comparison with an unrestricted nonperiodic model.

## Models and numerical objective

Sixteen equally spaced centers receive one shared seed-controlled global
rotation. Initial concentrations are four. Fixed16 and all four adaptive arms
therefore begin with identical geometry and the exact same ridge readout.

| Model | Features | Fitted scalar parameters | Adam parameters |
| --- | ---: | ---: | ---: |
| fixed16 | 16 | 17 | 0 |
| vp_centers | 16 | 33 | 16 |
| vp_widths | 16 | 33 | 16 |
| vp_both | 16 | 49 | 32 |
| joint_both | 16 | 49 | 49 |
| fixed32, fourier32 | 32 | 33 | 0 |
| fixed48, fourier48 | 48 | 49 | 0 |
| fourier16 | 16 | 17 | 0 |

Fixed32 uses widths 4/16 at the same 16 centers; fixed48 uses 1/4/16. These
are nested expansions of the fixed16 dictionary. Fourier controls use complete
sine/cosine pairs at integer harmonics 1..8, 1..16 or 1..24. Fourier outputs with
isotropic ridge do not depend on the initialization rotation; duplicated rows
preserve pairing and are not counted as independent evidence.

Every arm minimizes J = mean((Phi*w+b-y)^2)/2 + 0.001*||w||^2/2.
The intercept is unpenalized. All arithmetic and fitting use float64. Center
features and targets and solve the positive-ridge system without an inverse;
alpha=n_train*0.001. Independently verify against augmented least squares.
Concentrations use softplus of unconstrained raw parameters, with no clipping.

Variable projection solves the readout at every geometry state and differentiates
the reduced objective. Only designated geometry enters Adam. Joint Adam updates
geometry/readout/bias together after the common exact initialization; it receives
no later closed-form refit. Adam uses betas .9/.999, epsilon1e-8, no decay, no
gradient clipping, no early stopping, maximum6,000 updates and validation every
10 updates including step zero. Positive lambda is fixed prospectively, not
selected from v1 confirmation. The same coefficient penalty is not identical
function-space regularization across dictionaries; feature scale/correlation
and counts still differ.

## Compute allowances and selection

Use CUDA if available, one serialized worker, one CPU/BLAS thread. Prewarm the
device before timing fits. Each model receives the same maximum2-second fit
allowance. Report checkpoints admissible by0.25,0.75 and2seconds; primary is2.
Timing includes data transfer, model/optimizer setup, initial readout fitting,
all update/reduced solves, noisy validation and checkpoint state copies.
Synchronize device work before start/checkpoint timestamps. Post-fit clean
scoring, artifact serialization and disk I/O are outside this model-compute
allowance and separately measured within the phase ceiling.

Select minimum noisy-validation MSE among checkpoints whose entire evaluation
and state capture finished within each allowance. Exact ties choose the earlier
checkpoint. A late checkpoint remains recorded but is ineligible. Stop after
the first completed block past the maximum allowance. Retain overshoot and the
final state; a failed fit or no eligible checkpoint is explicit, never replaced
by a favorable survivor. Fixed fits finish after their one solution; do not burn
idle time. Equal allowance means equal available computation, not equal actual
use. Report actual time, update/checkpoint counts, optimizer/fitted parameters,
overshoot and any update ceiling. Wall-time reruns need not select the same
step; stored-state numerical replay is a different reproducibility property.

## Pilot, development and confirmation

Pilot sweeps rates .01/.03/.1 for each of the four adaptive models on total128
labels, all families/noise levels, one initialization. Fixed controls run once.
A separate max-shape timing panel uses total512 labels and rate .03 for adaptive
models, all families/noise levels; its scores cannot choose rates. This produces
224 records (128 adaptive,96 fixed). Select each adaptive model's rate using
equal family/noise mean noisy-validation MSE at the2-second allowance from the
128-label sweep only. All candidate records must complete; exact ties choose
the smaller rate. No grid score selects pilot rates.

Pilot ceiling10minutes, development45minutes, confirmation90minutes. Before
each larger phase, project adaptive fits at their full fit cap plus the maximum
observed adaptive block overshoot; fixed fits at the p90 fixed fit time from the
maximum-size pilot panel; all fits with observed p90 non-fit overhead; and add
60 seconds. This is a measured feasibility estimate, not a worst-case bound.
The maximum-size pilot panel must be included in timings.
Progress uses small atomic status files and immutable per-fit journal records;
the full registered matrix is written at start and finish. A terminal receipt
checks the deadline after the full final matrix is flushed and hashed; only that
receipt's own small metadata write is outside its timestamp. Before reserved
generation, also measure a complete read-only lock verification and add its
cost for all 32 confirmation target bundles to the feasibility projection.
This additional guard cost is absent from pilot timings. It must fit the same
90-minute ceiling, rather than expanding it.
If the projection fails, retain the finding; do not silently reduce the matrix
or change caps. Any revision precedes a new lock and is documented.

Development runs all1,440 records (576 adaptive,864 fixed). Select one VP
strategy by its balanced primary clean-grid error. Choose a fixed comparator
from fixed32/fourier32 when centers-only or widths-only wins, or fixed48/fourier48
when both wins. Choose the lowest balanced error; exact ties follow config order.
This separates an equal-basis contrast against fixed16 from an equal-fitted-
parameter contrast. Neither is a claim that all aspects of capacity are equal.
The isolated optimizer contrast is vp_both versus joint_both, regardless of
which VP strategy is selected. All development outcomes remain visible.

Confirmation runs all3,840 records (1,536 adaptive,2,304 fixed), not just the
selected models. Before any reserved target generation or restoration, require
a verified lock binding canonical configuration, all relevant source bytes,
complete pilot/development evidence, rates/comparators, feasibility, full tests
and an independent numerical/protocol review. No scientific code changes after
the lock; no confirmation-driven choices, retries or new scoring variants.

## Analysis and evidence

For each fit use clean grid MSE divided by grid variance. Average initializations,
noise levels and total-label budgets within each target; targets within each
family; and give equal weight to the three periodic families. Whole target
bundles are the bootstrap units, independently resampled within families,
preserving every paired noise/budget/initialization observation. Use10,000
resamples, a fixed analysis seed and percentile95% intervals.

The selected VP strategy has two co-primary contrasts, against fixed16 and the
selected equal-parameter comparator. A positive continuation finding requires,
in BOTH contrasts, >=10% aggregate error improvement, lower bootstrap improvement
bound>0, at least two periodic families improving and no family regressing>10%.
This is a bounded decision rule, not a powered population inference claim.
Every missing/failed primary record prevents a complete success claim.

Report every model/family/noise/label/time cell, including negative controls and
failures. Secondary diagnostics include geometry, ridge conditioning, coefficient
and prediction sensitivity, actual compute and vp_both/joint_both differences.
Label post-hoc analyses explicitly; they cannot replace failed primary gates.
Generator families were chosen after observing v1, so new seeds provide fresh
functions within known families, not unseen-domain validation.

Save exact arrays, initial/final/all checkpoint states, noisy validation
predictions, selected clean predictions, every candidate, source/config hashes,
environment, runtime and failures in fresh results directories. Do not overwrite
existing runs. Preserve compact quantitative records in nonignored documentation.
