# VM-N05: phase-conditioned uncertainty with delayed feedback

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent (roles performed inline)
- Origin Mode: authorized implementation, prospective plan, run and descriptive validation
- Origin Date: 2026-09-26
- Verification Status at specification: no reserved N05 trajectories generated or inspected
- Version Label: vm_n05_v1

## Question and frozen design

Can phase conditioning improve interval score around a fixed simple forecast?
The primary candidate is a **fixed-geometry VM residual-scale regression**,
compared with fixed Fourier residual-scale regression. This is not a test of
learned kernel centers/widths or a new point forecaster. Persistence is selected
in advance for simplicity and eligibility, not by ranking N04 confirmation.
N04's scientific failure and unvalidated neural adapters remain unchanged.

[The configuration](../configs/vm_n05_v1.json) specifies eight fresh room seeds,
disjoint from every N04 stage, and four paired conditions. Use the unchanged N04
first-order generator at tau/event-duration ratio 1, regular attendance, hourly
issuance, a 15-minute grid, and one-step observation delay. Predict noisy ppm at
15 minutes, one hour (primary) and four hours. Conditions are stable, +3-hour
evaluation-only routine shift, evaluation-only noise SD 30 instead of 10 ppm,
and the shift plus noise plus N04's three two-hour sensor outages. The noise
extension multiplies only evaluation measurement noise, keeping its paired
standardized draws. Hidden outage truth is for scoring only and never arrives.
Room draws are independent; the four interventions within a room are paired.
This small familiar-generator design is not real-building validation.

## Timeline and information

Days are zero-based elapsed UTC days from 2026-01-05. Days [0,14) are reserved
for point training/warm-up; persistence has zero fitted parameters and needs no
training labels or normalization. Days [14,28) supply scale-fit origins; only
targets with event AND actual arrival strictly before day 28 may fit scale.
Scale factors then freeze. Days [28,42) supply calibration origins. Static pools
freeze after arrivals at the first evaluation issuance (day 42 exactly).
Evaluation origins span [42,56), with a complete four-hour future grid.
Historical origin features may cross a partition because they use earlier
arrived observations; labels never cross a fit cutoff. A scale-origin residual
still unavailable at day 28 is permanently excluded from scale fitting and
never moved into the separate calibration pool. A calibration-origin residual
unmatured at day 42 is excluded from static pools but may join rolling pools
when it actually arrives. No evaluation label changes a scale coefficient.

At each issue time, **process all arrivals first, then issue every horizon**.
For any residual from u to u+h, require u < t, target <= t and actual arrival
<= t. Same-time arrivals are allowed. Select pools by sorted target time, then
issue time; input row ordering cannot change them. Non-arriving outage packets
never supply residual labels. Nonfinite arrived labels are excluded and counted;
nonfinite offline scoring truth fails the run rather than dropping hard rows.
No future-value filtering determines issuance eligibility.

Use six-hour causal histories with the inherited maximum forward-fill age of
three hours and a preserved regular grid. The point forecast is the latest
arrived reading, identical bit-for-bit across all six arms. Conditional arms
share four origin-known inputs: one-hour change / 50 ppm, its absolute value,
missing fraction in the history, and latest reading age / 12 steps; plus the
known future target's UTC daily phase. No future sensor or regime labels enter
features. Feature units are fixed in advance; no sensor normalization is fitted.

## Methods, scales and small samples

All arms use the same horizon-specific label eligibility. Static uses every
calibration error mature at day 42; static-matched uses its latest 168 errors.
Rolling uses the latest 168 mature calibration/evaluation errors. The matched
static control isolates pool size more closely, though elapsed time spans still
vary under missingness. Static versus full rolling alone confounds updating and
pool size. Neither static arm spends scale-fitting labels; conditional versus
unconditional comparisons additionally differ in scale label use. The primary
VM/Fourier comparison and hour-bin comparison have identical information budgets.

Each conditional arm fits log(max(abs(error),1 ppm)) by positive ridge
regression, penalty 0.01 on the mean squared objective with an unpenalized
intercept. All have 24 clock features and the four common inputs: Fourier uses
harmonics 1..12 (sine and cosine); hour-bin uses 24 one-hot hourly bins; VM uses
24 equally spaced centers and exp(8*(cos(theta-center)-1)). Only linear weights
are fitted. There is one fixed configuration per arm, no label-based tuning,
no validation search and no calibration-score-based model selection. Equal
feature counts do not imply equal function-space regularization.

Exponentiate the fitted log scale after bounding it to log(1)..log(1000) ppm.
This bounds the **scale factor**, not final widths. Record floor/ceiling events.
If fewer than 60 eligible scale labels, or fewer than 8 labels in the target
hour, or nonfinite inference covariates, use the clipped global median absolute
scale-fit error (1 ppm if no labels). Do not fit on nonfinite covariates. This
common sparse-hour rule applies to all conditional arms, including smooth bases.
If conditional rolling calibration has fewer than 20 residuals, use the matching
unscaled rolling interval. Global tiny pools return an infinite radius when the
order-statistic rank exceeds n, including n=0; flag and retain these intervals.
Do not silently substitute a finite cap or omit an infinite score.

For available scores s=abs(error)/scale, k=ceil((n+1)*(1-alpha)), alpha=0.1.
Use the actual kth ordered observation (one-based, no interpolation), or infinity
when k>n. Bounds are point +/- scale*kth_score. Unscaled methods set scale=1.
Use the same frozen factor function for previous scores and the new query.
There is no truncation of the final radius, bounds or physical support.
Positive finite scales do not imply exchangeability or conditional coverage.

## Metrics, inference and useful effect

Score exactly matched origins in ppm. Primary is the mean one-hour interval
score: width + 20*lower miss distance + 20*upper miss distance. Average within
room/scenario, equally across four scenarios, then equally across eight rooms.
The useful-effect gate requires VM score at least 5% below Fourier, a paired
95% room-bootstrap improvement lower bound >0, at most 10% regression in every
scenario, and VM overall coverage >=85% in every scenario. A failed conjunct
means the overall gate fails. No substitute comparator or horizon is allowed.
The coverage guard is an empirical usability check, not a 90% coverage theorem.

Report all arms/horizons: score, coverage, width, lower and upper miss rates,
point MAE/RMSE and counts, including six four-hour target-clock phase bins and
origin-known recent-rise (>25 ppm in past hour), recent-fall (<-25), otherwise
quiet, and stale-sensor (latest age > normal one-step delay) diagnostics. Empty
strata remain explicit. These secondary cuts are descriptive, without discoveries
selected from their intervals or a claim of simultaneous confidence coverage.

Primary uncertainty resamples eight independent **complete fitted room pipelines**
with all scenarios/horizons/arms paired, 5,000 percentile draws, seed 95005.
Each room independently supplies scale-fit and calibration labels, so this
captures joint trajectory and fitted-calibration variability across rooms;
it does not isolate calibration variance from room heterogeneity or estimate
uncertainty conditional on one room. Retain room effects and their sample SD.
Separately report a conditional paired two-day evaluation-block bootstrap,
holding each fitted scale AND issued interval fixed and all eight rooms present.
Carry all scenarios/arms together within each sampled block. That second
interval measures evaluation-path sampling only: it omits factor/quantile
refitting and cannot substitute for calibration-method uncertainty. Overlapping
rows are never independent replication units. Eight rooms and seven blocks
per room give limited precision; no population power claim is made.

## Compute, locking, validation and artifacts

Use Python 3.11, one BLAS thread, torch CUDA float64 for the small ridge solves
when available (otherwise CPU), NumPy for ledger/order statistics. Maximum 72
conditional scale fits (8 rooms x 3 horizons x 3 arms); paired scenarios reuse
those factors because their earlier partitions are identical. No point training.
One nonreserved fixture seed checks feasibility, capped at 120 seconds. Multiply
its one-room/four-scenario runtime by eight and 1.5 before accepting the 600-second
formal cap. Analysis cap 120 seconds. Keep partial failures and do not silently
retry formal fits. A failed or infeasible study is a completed negative result.

Freeze exact config/protocol/generator/new scientific source hashes before any
reserved trajectory generation. Engineering fixtures/tests may precede the lock;
their seeds cannot be reserved. Formal access requires a completed passing
engineering receipt, including the repository full suite for this pipeline.
No standard forecast benchmark is needed: persistence and N04 models are
unchanged, and scalar interval-scale fits are not point-model acceptance.

New versioned modules, runner, compact tables/figure and source-bound JSON belong
to N05. Full rows/scales stay in ignored results/vm_n05/. Preserve inherited
sources, owner notes and source locks. Export a documented interval table with
UTC issue/target/actual observation availability, horizon, unchanged point,
bounds, units, calibration cutoff/count, and source/config IDs; absent arrivals
are null and offline truth is explicitly separate. N06 is an optional consumer,
not a runtime dependency or a task to modify. No controller or real-data claim.

## Context and theory boundary

The supplied Chapter 5 companion (lessons 13–14) and sensor-timing lab inform
the ledger and interpretation. Their rise-coverage repair did not establish
superiority to persistence or decision gains. Exact read-only file identities
are in [the context receipt](vm_n05_context_receipt.json).
The [conformal tutorial](https://arxiv.org/abs/2107.07511) distinguishes coverage
guarantees from their assumptions. [Adaptive conformal inference](https://arxiv.org/abs/2106.00170)
uses a different update algorithm; its guarantees are not claimed here.
