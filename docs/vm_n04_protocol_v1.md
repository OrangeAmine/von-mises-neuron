# VM-N04: prospective circular forecasting protocol v1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan, followed by authorized implementation and execution
- Origin Date: 2026-09-22
- Verification Status: prospective; no VM-N04 pilot or reserved trajectories inspected
- Version Label: vm_n04_v1

## Question and finite design

Does adapting circular kernel centers and widths improve concentration forecasting
after a sharp daily forcing passes through first-order dynamics? The primary
comparison is learned kernels against **both** Fourier and fixed circular kernels.
Calendar information versus sensor history is a separate contrast. Static locality
and the sibling occupancy-probability study do not establish either result.

The configuration is [vm_n04_v1.json](../configs/vm_n04_v1.json). Cross three
response-time/event-duration ratios (0.25, 1, 4), regular/irregular attendance,
stable/+3-hour test-only routine shift, and complete/one bounded sensor-outage
condition: **24 test regimes**. Dynamics are dC/dt=(Ceq-C)/tau with a 420 ppm
baseline and room-specific pulse amplitude 300–600 ppm. Daily events last two
hours; room arrival phase is drawn between 08:00 and 10:00. The exact update
`C_next = Ceq + (C-Ceq)*exp(-dt/tau)` applies to piecewise-constant forcing on
15-minute grid intervals. Irregular attendance has daily probability 0.75,
bounded half-hour arrival jitter and bounded amplitude variation. These are
synthetic equilibrium concentrations, not a calibrated building mass balance.
Steady-state gain is held fixed across tau, so an equivalent source rate would
vary with tau; this isolates smoothing, not an isolated ventilation intervention.

All clocks use a fixed UTC daily cycle starting 2026-01-05. No DST change is in
the scientific design. A distinct diagnostic deliberately offsets **calendar
inputs only** by +60 minutes and verifies restoration using the known offset;
this never replaces a primary result. Real routine shifts alter physical forcing
while clocks remain correct. Independent room seeds also draw independent daily
attendance/noise streams. The same draws are paired across regime interventions.

## Information, labels and forecast definition

Each room has 56 days, split chronologically into 28 training, 14 validation and
14 test days before normalization/windowing. Windows remain within each split.
Fit normalization only to training measurements available by its cutoff. Inputs
contain 24 grid points (six hours) of observed concentration, observed/missing
mask and age. A six-hour history is shorter than the slowest eight-hour response
time; any advantage is conditional on that history budget.

Observations have independent Gaussian noise SD 10 ppm and arrive one 15-minute
step after their event. At each origin, both event and arrival must be no later
than issuance. Forward-fill uses only arrived observations, with maximum age
12 steps (three hours); reject origins with unresolved history. Keep the regular
grid through outages; never compress row distance. A structural timestamp gap
rejects the full history/target span. Report all exclusion reasons/counts.

Outages remove two hours of sensor delivery on zero-based test days 2, 6 and 10
around the event. They do not delete rows or remove scoring truth: the simulator
retains a separate noisy target stream for offline scoring during outages.
Outage membership and realized attendance are diagnostic labels, never inputs.
Complete aggregates, if used, are available only after bin end and the last
required sample arrival; timing helper tests cover this even though primary
measurements are point samples.

Issue forecasts hourly. Predict every 15-minute endpoint from 15 minutes through
four hours; the primary endpoint is **one hour** (step 4), secondary endpoints
15 minutes and four hours (steps 1 and 16). A whole-horizon average is explicitly
separate. The primary target is the noisy concentration in ppm; latent clean
concentration is a secondary diagnostic. Preserve origin/target UTC identities.
Every method uses the same eligible origins within a regime, including outages.
Hourly issuance covers only one of four 15-minute origin phases; timing claims
are conditional on that issuance schedule. Split-contained histories exclude
the first six hours after the test boundary, so no immediate-shift claim is made.

Use all eligible hourly training/validation origins (approximately 660/325).
Each has 16 labels, substantially overlapping in event time; record both window
counts and unique target timestamps. Validation labels count toward the disclosed
information budget. Labels near a fitting split cutoff must already have arrived.
There is no online adaptation or access to test labels for fitting or selection.

## Models and selection

Four direct residual forecasters share an identical flattened sensor MLP, two
32-wide GELU layers and 16-output residual head around the latest available
concentration. Vary only the calendar branch: none; 16 Fourier harmonics (32
features); 32 fixed peak-normalized circular kernels; or 32 kernels with learned
centers and softplus concentrations, initially kappa=32. Calendar branches use
the same future daily angles and a shared-across-horizons linear readout. Kernel
features are `exp(kappa*(cos(theta-center)-1))`. Kappa controls locality here.
The sensor path is not periodic. Registered buffers keep fixed geometry and
integer harmonics on the model device.

Seed model construction identically to pair sensor weights; pair calendar
readout initialization and reset minibatch RNG after construction. Equal feature
counts do not equalize functional capacity: learned geometry adds 64 fitted
parameters. Report exact fitted parameter counts and time for every arm. No
architectural search, VP optimization, or claim of equal wall-clock cost.

References: latest available sensor persistence; training-only daily-bin mean
(seasonal); and multioutput sensor-only residual ridge with an unpenalized
intercept and penalty selected from {0.0001,0.01,1} by validation whole-horizon
MSE. Seasonal uses the same known future clock; persistence/ridge do not. No
future measured sensors or realized attendance enters any model. No oracle is
needed.

All neural arms get Adam, 80 epochs, batches of at most 256, weight decay 1e-4,
gradient clip 1, no scheduler or early stop. Select minimum whole-horizon
validation MSE over epoch 0 and every fifth epoch; ties prefer the earlier epoch.
Pilot compares learning rates .003 and .01 and selects **one common rate** by
mean validation normalized MSE equally across the four arms and six training
cells; ties select the smaller rate. Test errors do not select rate or checkpoint.
Development confirms execution and reports all outcomes; it makes no further
model, seed, metric or budget choice.

## Replication, runtime and reserved access

Fresh disjoint room seeds: one pilot room, two development rooms and six reserved
confirmation rooms, as listed in the JSON. Pilot has one initialization per
rate; development/confirmation have two paired initializations nested inside
each room. Initialization seeds are reused across independent rooms. Because
shift/outage affect only test, fit once per response/attendance training cell
then evaluate all four test variants. Neural fit counts: 48 pilot, 96 development,
288 confirmation. Deterministic references fit once per training cell.

Caps include generation, fitting, evaluation and artifact writes: 600 seconds
pilot, 900 development and 1800 confirmation. Run CUDA serially when available,
CPU/BLAS threads=1, no cloud jobs. Pilot records measured total and per-fit time.
Project later stages from observed pilot cost with a conservative 1.5 multiplier;
do not enter a stage whose projected cost exceeds its cap. A deadline check
before/within fitting stops the stage and retains completed, failed and unrun
records; do not rerun a failed formal fit. A failed/infeasible study is a valid
bounded delivery, not permission to increase caps or prune hard regimes.

Freeze source/config/protocol SHA256s before pilot. Confirmation additionally
requires completed pilot/development receipts, selected common rate, measured
feasibility and completed engineering checks. Locks bind exact source bytes and
prior stage summaries. Reserved seeds are never used for debug fixtures.

## Analysis and gates

For each room and each regime, average the two initialization-specific endpoint
RMSEs, normalized by its training observed-sensor SD. Average equally across
24 regimes and then rooms. Primary effect is `1 - mean(learned)/mean(control)`.
Resample the **six rooms**, carrying every regime and initialization together,
10,000 paired bootstrap draws with seed 94004, for percentile 95% intervals.
Overlapping windows, horizon labels and initialization draws are not independent
replicates. Six rooms give limited precision; there is no population power claim.

The scientific gate requires >=5% aggregate improvement against **each** control,
each paired interval lower bound >0, and at most 10% mean normalized-RMSE
regression in **each** of the 24 regimes against each control. Both comparisons
must pass; no alternative arm can replace learned kernels. These are conjunctive
continuation criteria, not multiple unadjusted significance discoveries. Report
paired calendar-versus-none skill and skill against ridge/persistence/seasonal,
all horizon endpoint RMSE/MAE, whole-horizon errors, boundary (within one hour
of actual event onset/end), event and outage diagnostics, and initialization
variability nested by room. Empty subgroups remain explicitly empty.

## Engineering validation and reporting

Test the exact recurrence, regular grid, split scopes, delayed arrivals, bin
completion, forward-fill limits, origin/target identities, unavailable-value
mutations, periodicity of the calendar pathway and CPU/CUDA shape/dtype/gradient
behavior. Run the full suite for model/pipeline changes. Run the unchanged
standard synthetic benchmark and new adapters using its unchanged data,
training budget and absolute thresholds; apply base-VM temporal regression
checks as well. Retain failed engineering arms and label them unvalidated;
execution or scientific comparison does not override those gates.

Keep arrays/checkpoints in `results/vm_n04/`; keep compact CSV/JSON, one figure,
scientific report and state update tracked. Record environment, source identities,
actual training/inference/runtime and measured memory. Do not alter inherited
v1/v2 sources/configs/results, the original checkout or owner notes. Stop at the
declared synthetic answer. Real-data replication, uncertainty, control and
deployment require separate scope.
