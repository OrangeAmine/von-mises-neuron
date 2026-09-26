# N06 first-delivery contract

Frozen before new development/evaluation generation on 2026-09-26. The dispatch
authorizes this delivery and supersedes the proposal's unstarted status. This is
a synthetic educational/application study; no building calibration, energy,
health, comfort, monetary or adaptive-locality superiority claim is intended.

## Physical and information contract

N04 used `dC/dt=(Ceq-C)/tau`, holding equilibrium amplitude fixed across response
times. That is an uncontrolled smoothing experiment. N06 instead holds external
generation S independent of ventilation and uses
`V*dC/dt = S + Q*(Cout-C)`. C is ppm, V m3, Q m3/h, and S ppm*m3/h
(person generation in m3 CO2/h times 1e6). Q=45+u*405 normally; u is binary.
For each constant 15-minute interval, a=Q/V and
`Cnext=Cout+(C-Cout)*exp(-a*dt)+(S/V)*[-expm1(-a*dt)/a]`.
At Q=0 the exact limit is `Cnext=C+S*dt/V`. Both equilibrium `Cout+S/Q` and
decay rate vary with airflow. No source depends on a policy action.

The JSON fixes all quantities and units. UTC timestamps label left interval
boundaries; state C[k] exists before interval source S[k] acts. Sensor C[k]+noise[k]
arrives at k+1. Initial concentration is known. The command issued at k affects
interval k+delay; earlier queued commands cannot be revoked. All policies start
from the same C0 with zero pending commands. There is no ramp/minimum-on limit;
50 ppm command hysteresis is explicit. Binary capacity and delay are enforced.
Every rollout evolves its own state, sensor values, histories and command queue.
Paired policies share exogenous attendance and additive sensor-noise draws, not
their concentration observations. No outage or sensor smoothing is assumed.

The realistic arms only see arrived measurements and known calendar. Training
source labels are assumed available from an ideal historical attendance counter:
14 days per independent room, 1344 interval labels. This is a strong synthetic
assumption; concentration sensors alone do not supply those labels. There is no
live or future source measurement. Oracle alone receives realized future source,
but has the same delayed noisy concentration sensor. It uses the same heuristic,
not an optimized controller or a mathematical bound.

## Forecasts and decision rule

Seasonal: 96 quarter-hour means. VM: 16 fixed peak-normalized circular kernels
`exp(16*(cos(theta-center)-1))`, uniform centers, ridge linear readout. Fourier:
8 sine/cosine harmonic pairs with the same 17 fitted coefficients including
intercept. Geometry is fixed, not learned. Both use mean-square ridge 0.001 with
unpenalized intercept, raw UTC phase and training labels only. Predictions are
clipped below at zero source. Equal ridge penalties do not equalize function
space penalties. All three share exactly the same training information; no
forecast hyperparameter search. The fit uses float64 CUDA when available.

Reactive uses the latest arrived concentration. Anticipatory arms retain that
feedback guard and add a threshold on the maximum forecast concentration over
the next 2 hours. Starting at the latest arrived observation, they propagate
through already applied/queued actions and predicted source, assuming no NEW
future command (base flow after the queue). If either signal is >=900-margin,
request on; if both are <900-margin-50, request off; otherwise retain the previous
request. Reactive ignores the forecast. Equal-policy and zero-lookahead controls
must be identical. Forecast signals are recomputed from each controlled history.

The predicted source is action-independent. A no-new-command concentration
forecast is a conditional counterfactual from the current controlled state and
pending queue; it is not the uncontrolled trajectory from the initial state.
Action-conditioned forecasts use an explicit alternative action sequence.
Only source errors are scored in closed-loop comparisons. An additional calendar
adapter on the unchanged standard synthetic concentration dataset reports the
repository's unmodified absolute and base-VM regression checks. It is a diagnostic
transfer of the same calendar estimator, not closed-loop acceptance. Failed gates
remain failures; no model is called validated from software tests or this study.
The verified inherited base metrics may be reused without retraining unchanged
models. N04 neural adapters remain unvalidated and its scientific gate failed.

## Bounded design, selection and evaluation

Two fresh development rooms and six fresh evaluation rooms, each with 14 training
days followed by 3 scoring days. Days within a room and four paired regimes are
not independent replicates. Draw people count and routine phase once per room;
draw daily attendance and bounded jitter independently. Training and scoring
use independent RNG streams. The late-routine intervention shifts scoring source
by 3 hours, without shifting clocks or training. Capacity and delay regimes use
the same source/noise stream. Evaluation seeds are not debug fixtures.

Every arm gets the same three margin candidates {0,100,200} ppm. Select each arm's
margin separately for each of two objectives on development only, averaging
rooms and regimes equally; ties choose smaller margins. Each model is fitted
once per room, each arm/margin evaluated once per regime; two objectives rescore
the same runs. The table retains all candidates, including poor ones. Selection
then freezes before evaluation. No new architecture, controller or forecast-arm
selection is allowed after evaluation. Caps: 180 seconds for development plus
evaluation, and 120 seconds for the standard calendar diagnostic; serial runs.
Stop and preserve partial records if exceeded, with no implicit retry or pruning.

Objective J = hours above 900 ppm + lambda*(additional ventilation m3/1000).
Lambda is 0.02 for exceedance priority and 2 for effort priority. No claim that
these weights represent human preferences. Primary comparison is VM versus each
of reactive, seasonal and Fourier at the development-selected primary objective.
A practical-benefit gate requires >=5% lower mean J, positive paired 95% bootstrap
improvement lower bound, and <=10% worse J in every regime, against each control.
The oracle is descriptive. Bootstrap 2000 whole-room resamples, fixed seed; six
rooms limit precision. Report paired room differences and all regimes, nulls and
negative results. No inference to buildings or universal controller performance.

Metrics integrate the exact constant-interval curve: exceedance hours and
integrated severity ppm*h. Also report added-air m3, total-air m3, on hours,
activations and binary switches. Plot every achieved trade-off. Exact attained
mean-effort matches (within 1e-9 m3 numerical tolerance) may be identified; absent
matches remain failed. No interpolation/extrapolation supports a matched claim.

## Three teaching cases and delivery

Deterministic authored case: 216000 ppm*m3/h source at 08:00–10:00, initial 600 ppm,
one-hour actuator delay. One stipulated forecast misses 08:00–09:00, the other
09:00–10:00; source MAE/RMSE are identical. Each drives its own full dynamics.
Report resulting effort honestly, without assuming it matches. A second case
varies zero/limited/normal capacity and delay, and includes a poorly used oracle
with only 15-minute lookahead. A third changes objective weight across actually
achieved operating points. No hypothetical calibrated intervals; N05 is unused.

An offline local HTML viewer of deterministic precomputed runs supplies controls,
reset, linked state/action plots, metrics and worked explanations following
predict -> manipulate -> observe -> explain. It does not rerun/tune the study.
All interactive views must be inspected. Tests cover exact mass balance against
independent integration, zero flow, causal availability and action ordering,
future mutation, pairing, target semantics, delay/capacity, identity and metrics.
Run the full suite for new model/pipeline code. Preserve all frozen dependency
files except the explicitly authorized project-state addition; track compact
results/figures and write large arrays only to `results/vm_n06/`.
