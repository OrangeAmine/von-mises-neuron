# VM-N01: prospective smooth-plus-local comparison

## Material Passport

- Origin: academic-research-suite / experiment-agent, plan and run modes.
- Date: 2026-09-22. Owner explicitly authorized VM-N01 and routine design choices.
- Status at initial freeze: no VM-N01 models fitted or target pools generated.
- Contract: this document and `configs/vm_n01_v1.json`; local prospective freeze,
  not external preregistration. V1/v2 studies and their failed gates stay frozen.

## Question and estimand

Can a low-order Fourier backbone preserve smooth-target performance while
adaptive circular kernels improve narrow, multimodal and mixed targets under
comparable label and computation allowances? Implement
`b + sum_m(a_m*cos(m*theta)+d_m*sin(m*theta)) + sum_j(w_j*exp(kappa_j*(cos(theta-c_j)-1)))`.
Harmonics are positive integers; widths use positive softplus concentrations.
Signed readouts and an unpenalized intercept are fitted. Broad/local roles and
component identifiability are hypotheses, not conclusions of this model form.

The v2 smooth regression of 47.58%, joint-Adam advantage, and competitive larger
Fourier bases are constraints motivating this comparison. No historical fit is
rerun. This task does not launch VM-N02 or forecasting/control follow-ups.

## Fixed design and information budget

The JSON defines eleven models, all reported. The hybrid has three Fourier
harmonics plus ten kernels: 16 features and 37 fitted parameters with adaptive
centers/widths. Fourier16 and hybrid_fixed16 are the two equal-basis controls;
the best development result among Fourier36, fixed36 and hybrid_fixed36 is the
equal-fitted-parameter control (37 parameters). Kernel-only adaptive models have
49 parameters and are descriptive, not parameter-matched hybrid controls.
Fixed16 is also reported to retain the motivating historical type of reference.
Both hybrid and kernel-only arms include joint Adam and variable projection.
The unrestricted diagnostic uses theta/pi plus 15 ordinary line Gaussian RBFs
at equally spaced centers from -pi to pi, width .45; it is not periodic.

Use fresh independent function pools: five families, 1/2/6 functions per family
in pilot/development/confirmation and 1/1/2 initializations. Confirmation has 24
independent periodic functions; replication within a function is not independent
evidence. All families and their equations are fixed in JSON. Mixed targets add
the raw smooth generator and two signed compact cubic bumps before normalization.
Centers are uniform on the circle, all generator choices independent unless
pairing is explicitly declared. Compact cubic bump is `(1-(wrap(theta-c)/w)^2)^3`
inside its support and zero outside. Triangular peak is
`max(1-abs(wrap(theta-c))/w,0)`.

Each function is standardized on 8,192 deterministic midpoint angles as part of
the synthetic generator definition, not fitted preprocessing. Integration uses
2,048 different midpoints. Total observational budgets are 64=48+16 and
256=192+64 training+validation labels. Pilot uses only 256, including max-shape
timing. Independent training and validation streams grow by prefixes without
switching roles; shared unit-noise draws have SD .1. All models, rates and
initializations reuse identical labels. Unique label counts are 1,280 pilot,
2,560 development and 7,680 confirmation: 11,520 total, including all tuning
and validation. The clean deterministic grid is integration truth, not available
to fitting or pilot-rate selection. Development clean scores select the hybrid
optimizer and capacity comparator; reserved clean truth is accessed only after
source/config/evidence lock. Reusing familiar generator families limits this to
within-generator confirmation.

SeedSequence entropy is `[phase_seed,101,family_index,replicate,stream_id]`;
function/inputs/noise use distinct stream IDs. Initialization uses its separate
root with phase index, family, replicate, initialization. These roots differ
from v1/v2. No reserved target is generated before a verified confirmation lock.

## Fitting and computation

Raw sin/cos and peak-normalized kernels have no feature standardization. Every
fit minimizes `.5*MSE + .5*.001*sum(weight^2)`; the intercept and geometry have
no penalties. The coefficient penalty is not equal function-space regularization
across bases; counts, scale and correlations differ. Use the frozen positive
ridge API for fixed fits, initial readouts, and differentiable VP solves.
Joint Adam starts at the same ridge solution as VP and receives no later refit.
Adaptive geometry starts at equally spaced centers with a shared seeded global
rotation, kappa4. Fixed multiscale banks repeat centers at kappas1,4,16.

Float64, CUDA when available, one serialized fitting worker and one CPU/BLAS
thread. No feature/model selection beyond the declared candidates. Pilot selects
one rate per adaptive model from .01/.05 by equal-family noisy validation MSE.
Adam betas(.9,.999), epsilon1e-8, no native weight decay, clipping or early stop.
Maximum3,000 updates; validation at step0 and every10 updates. Capture complete
CPU states and validation before synchronized timestamps. Select the earliest
minimum validation MSE completed by .25 or1second; primary is1second. Initialization,
transfer, ridge solves, optimization, validation and capture count toward the
allowance. Late checkpoints and failed states remain recorded but are ineligible.
Fixed fits stop after their single solution; unused allowance is not wasted.
Equal allowance is not equal actual time. Post-fit clean scoring and disk I/O
are outside fitting time and inside phase time. Report actual time, updates,
parameter/basis counts, overshoot and failed/incomplete attempts.

Phase ceilings are4/10/30minutes, below v2's10/45/90minute ceilings. Planned fit
counts are75/220/1320. Measure pilot feasibility before development and again
before confirmation using the conservative expression in JSON; include measured
lock verification. A failed projection or ceiling stops the phase, preserves
evidence and does not authorize expanding the budget or silent reruns.

## Selection, uncertainty and fixed benefit gate

Choose hybrid_joint or hybrid_vp using lowest balanced development normalized
clean MSE, and choose the capacity control similarly. Ties use JSON order.
Run and publish every model in confirmation, not just the selected pair.
For each function, average both label budgets and initializations. Give each
periodic family equal weight. Resample whole paired function bundles separately
within families10,000 times for percentile95% improvement intervals using the
fixed analysis seed. Do not treat grid points, labels or initializations as
independent functions. Small function counts do not support powered population
or unseen-domain claims; percentile intervals are descriptive.

For EACH of the three contrasts, require at least10% relative aggregate error
reduction, lower paired interval strictly above0, smooth-family error no more
than10% higher, and improvement in at least two of narrow/multimodal/mixed.
All must pass; no comparator can override another failure. No complete success
claim with missing/failed fits. Nonperiodic is diagnostic and excluded from the
benefit gate. Report optimizer contrasts, all model/family/label/allowance cells,
function-level paired variability and runtime even when the gate fails.

## Engineering and evidence gates

Before formal fitting: equation/wrap/ridge/VP-gradient/state-replay checks,
float32/float64 finite gradients and CPU/CUDA parity; full repository suite;
and the unchanged standard synthetic data/training/metric/absolute gates for a
narrow per-sensor hybrid-feature LSTM adapter. M3/K10 features feed Linear32,
LayerNorm, dropout.1, LSTM64 and head12, seed42,90days,72lookback,50epoch maximum.
Run a fresh base VM alongside it; do not tune adapter using test outcomes.
This certifies only the adapter's integration if it passes; static fitted bases
do not inherit forecaster validation and engineering success is no benefit gate.

Freeze scientific source bytes before pilot. Bind config, sources, selections,
pilot/development evidence, required checks, review and feasibility before any
confirmation generation. Retain each attempt, arrays and checkpoints locally in
ignored `results/vm_n01_v1/`. Durable docs contain compact tables, figure,
provenance, observed failures, limits and completion record. Verify all130
inherited hashes at delivery except the authorized project-state update.

## Context receipt

Source snapshot SHA256 `18191869b3383ee437e33de713b4aaed6a7f103e9ae88c5581ce8ea8a44f31b5`:
130 files verified and overlaid into this isolated worktree, recorded separately
in `vm_n01_inherited_baseline.json`. Original checkout and source context remain
read-only. The research-workspace-context validator passed for the supplied
`D:/Desktop/Claude/Hybrid_models_forecasting`, revision
`5f48fadda23f3132ecece500370b0c124e408ef4`, dirty state, context manifest
`49aed2956b668ffdbf00c531a4f4967515724ebd9747132b1618eb934f0af337`.
Selected learning owners: Chapter5 correction-prediction discussion and
`docs/learning/labs/model_selection.md`; these motivate a reference plus
correction and fresh locked evaluation. No external scientific payloads were
required or replayed; no unavailable resource was substituted.
