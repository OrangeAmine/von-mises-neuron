# VM methodology work plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: experiment design under the owner's approved direction
- Origin Date: 2026-09-08
- Verification Status: DESIGNED; no proposed training experiments executed
- Version Label: vm_methodology_v1

## Objective and decisions

Determine whether polar optimization, learned circular locality, or explicit
angular interactions provide useful benefits beyond matched trigonometric
controls. Separate those mechanisms before combining them in a forecaster.
The owner approved this direction; this deliverable specifies the next work.
It does not report implementation or experimental success.

The canonical design is this document plus
[`configs/vm_methodology_v1.json`](../configs/vm_methodology_v1.json).
The [engineering proposal](vm_methodology_engineering_proposal.md) supplies API
details; this plan controls experiment scope if the proposal lists extensions.
The [assessment](vm_methodology_direction.md) explains the motivating evidence.

| Stage | Question | Required output | Next decision |
| --- | --- | --- | --- |
| E: numerical foundation | Can we preserve the old function while making computation and initialization explicit? | Versioned layers, conversion/parity tests, runtime/memory measurements | Proceed when engineering checks pass; speedup is not assumed |
| A: optimization | Do polar coordinates help adaptation on exactly the same function class? | Paired learning curves, matched penalties, coordinate-origin diagnostics | Retain a polar-specific claim only if its prespecified contrast survives confirmation |
| B: locality | Do learned circular kernels improve learning from limited observations? | Capacity-controlled learning curves against strong Fourier/kernel/MLP baselines | Confirm a restricted useful regime, or retain the simpler control |
| C: interactions | Do explicit relative-angle terms explain an unresolved limitation? | A separately frozen interaction protocol | Deferred until A/B identify the relevant question |

A and B can both proceed after E; a negative A result does not invalidate B.
Circular probability heads, deeper stacks, attention, real-data forecasting,
automatic architecture searches and claims of novelty are outside v1.

## E — numerical foundation and interpretation controls

Use new source modules and explicit converters. Keep `src/von_mises_neuron.py`,
`src/temporal.py`, `src/sincos.py`, the frozen forecasting runners and old
checkpoints unchanged. The current hash-checked forecasts must still load.

Planned components in `src/circular_layers_v2.py`:

- `PolarCircularLinearV2`: sum or exponential response, stable positive
  concentration initialization, `constant` or `unit_variance` initialization,
  and reference/factored execution. Legacy conversion copies the actual old
  parameters, including old initialization effects; it does not reinitialize.
- A matched Cartesian layer/control and explicit amplitude regularizer.
- `PeakCircularKernelBankV1`: start with one angular input and independent
  fixed/learned center and width switches; a separate signed linear readout.
- `DirectCalendarPhaseV1`: exact conversion of the legacy bank plus projections
  to direct sine/cosine phase maps. This is a representation/compute control;
  temporal forecasting training is not an A/B experimental arm.

For the circular kernel use
`log(phi) = -2*kappa*sin((theta-center)/2)^2` and `phi=exp(log(phi))`.
The center is the preferred angle. Do not add bias inside the kernel or silently
clip widths. Fixed tensors must be buffers. Log response is not a log-density,
and peak normalization is not probability normalization.

An essential exact identity is:

`phi(theta) = exp(-kappa*||u(theta)-u(center)||^2/2)`,
`u(a)=(cos(a),sin(a))`.

Thus the proposed peak-normalized VM is a Gaussian RBF on the unit-circle
embedding. Use that as an output/gradient parity check, never as an independent
competitor or evidence of novelty. The geodesic-distance feature in B is a
different response; no covariance-kernel positive-definiteness claim is made.

E acceptance requires:

1. Positive finite concentration requests are faithfully initialized, including
   0.001; invalid values fail explicitly. Large values use a stable inverse
   softplus. Unit-variance initialization scales by `sqrt(2/in_features)`;
   the variance statement is over independent initial phases, not a promise
   about trained activations.
2. Converted/reference/factored outputs and gradients agree in CPU float64
   (`atol=rtol=1e-10`) and CPU/CUDA float32 (`atol=1e-5, rtol=1e-4`) on declared
   finite cases. Exercise bias, leading dimensions, shared dynamic phase,
   near-zero/ordinary concentrations, log/exp modes and gradcheck. Extreme exp
   cases are explicit numerical-failure diagnostics, not silently clipped passes.
3. Check input/phase/raw-width/modulation/bias gradients, RNG preservation,
   independent state ownership, serialization, parameter counts and buffer
   device movement. Test per-input 2pi periodicity separately from global
   rotation and from raw-sensor residual paths.
4. Verify exact temporal conversion and VM/chordal-kernel equivalence. Equality
   of parameterization-specific gradients is expected only for the factored
   implementation retaining the same parameters; Cartesian conversion and
   temporal collapse do not promise identical optimization trajectories.
5. Profile reference/factored forward+backward with CPU threads fixed and CUDA
   synchronized, 20 warm-ups and 100 measured iterations, batch/time/input/output
   shapes `(32,72,8,64)`, `(32,72,64,64)`, `(8,168,32,128)`.
   Report median/p90 latency, peak allocated GPU memory and hardware. OOM is a
   retained measurement outcome. No speed threshold substitutes for correctness.
6. Run the full repository suite. New forecaster adapters must also receive the
   unchanged standard synthetic gates (RMSE <= 35, MAE <= 25, R2 >= 0.20, MAPE <= 5%).
   Temporal adapters additionally receive the existing <= 10% error regression
   and <= 0.10 R2 loss checks against fresh base VM. A failed adapter cannot be
   called a validated forecaster; methodology claims are evaluated separately.

## Shared experiment contract

All A/B targets are synthetic scalar functions. No LSTM, learned input affine
projection, LayerNorm or dropout is used. Training arrays and evaluation targets
are shared across compared models. Inputs are true angles in radians, except
the declared raw-angle reference. Report units and transformations explicitly.

Seed roles are separate, with no overlapping root seed values:

| Role | Task/function | Observations/noise | Model initialization |
| --- | --- | --- | --- |
| Disposable pilot | 101 | 201 | 301 |
| Development | 1101, 1102 | 2101, 2102 | 3101, 3102 |
| Confirmation registry | 5101–5108 | 6101–6108 | 7101, 7102 |

Pair the task and observation root seeds by position, not their Cartesian
product. A uses all eight confirmation pairs; B uses the first six. Derive
streams with `SeedSequence([root_seed, stage_id, family_id, replicate_id,
split_id])`, using stable integer IDs in the config. Use dedicated task,
training-observation, validation-observation, test-noise and initialization
streams. Model identity must not alter shared observations or batch order.
For A, use stochastic family ID 0 for every condition, so a paired task bundle
shares its pre-change phase, input angles and standard-normal noise draws;
condition IDs select only amplitude/phase transformations and noise scaling.

For B, train sizes 32/128/512 are nested prefixes of the same sampled sequence.
Validation uses 256 independent observations. A uses 128 training observations
and 256 validation observations. The common clean evaluation grid has 4096
midpoint angles in `[-pi,pi)`. Independent test noise is generated separately
for the secondary noisy-response metric. This grid is numerical integration
support, not 4096 independent experimental replications.

Training uses float32, with float64 metric accumulation and closed-form solves.
Adam uses betas (0.9, 0.999), eps 1e-8 and no AMSGrad. A uses no gradient clipping;
B clips total gradient norm at 1. CUDA determinism settings and library versions
must be recorded; do not promise bitwise reproducibility across devices.

Do not materialize or score confirmation functions until a lock artifact exists.
That lock records source/config/generator hashes, pilot choices, development
results, the chosen B primary comparator and all metric/selection rules.
Statistical scripts may be exercised on disposable pilot data first.

The confirmation seed registry is known to the code; this is a procedural
holdout, not cryptographic secrecy. A confirmation API must reject missing or
changed locks and refuse overwriting existing outputs. Configuration changes
after confirmation access require a new version and unused seed registry.

## A — same function, different optimization coordinates

Models have exactly three learned scalars:

- Polar: `kappa*cos(theta+phase)+bias`, with positive softplus concentration.
- Cartesian: `a*cos(theta)+b*sin(theta)+bias`, initialized by exact conversion.

Start from the known pre-change function, bias 0, rather than training a
backbone. This is a controlled warm-start adaptation experiment. Draw absolute
phase uniformly for each task; apply one of the fixed changes below.

| Condition | Amplitude before→after | Phase change |
| --- | --- | --- |
| Phase only | 1→1 | pi/2 |
| Amplitude only | 0.25→1 | 0 |
| Both | 0.25→1 | pi/2 |
| Weak signal phase | 0.02→0.02 | pi/2 |

Noise SD is 0.1 times the post-change clean signal SD. Do not standardize A's
targets: absolute amplitude is the manipulated quantity. Noise is sampled once
per observation. Reset optimizer state at the change for both representations.

Cross Adam/SGD (SGD momentum 0) with penalty 0/0.001. Disable native weight decay
on all parameters. The explicit penalty is `lambda*kappa^2/2`, or
`lambda*(a^2+b^2)/2`; bias is unpenalized. The data objective is half mean squared
error. Use the same learning rate within each optimizer's paired polar and
Cartesian comparison, without separate phase learning rates.

The pilot considers Adam learning rates 0.001/0.01/0.1 and SGD 0.003/0.03/0.3.
Choose one rate per optimizer by mean normalized noisy-validation curve area
across both representations, both penalties, and pilot phase/amplitude cases;
smaller rate breaks a 1e-12 tie. Rates then remain fixed. Confirmation does not
select a favorable optimizer or regularization cell.

A pilot rate is eligible only if every planned paired fit at that rate completes
with finite state/metrics. An incomplete rate cannot win on its surviving runs.
If no complete rate remains, revise the design before development; do not
silently invent another rate after inspecting confirmation.

Run 200 full-batch updates, record step 0 and every 10 steps, and use the entire
curve. There is no early stopping or test-selected checkpoint. Full-batch A is
deterministic after function/data selection; **do not replicate it under nominal
training seeds**. Fit a closed-form Cartesian least-squares reference to the
same training data as a diagnostic optimization endpoint, not another sampled
training arm. Report the penalized reference where applicable too.

Primary endpoint: clean-grid MSE integrated over update steps, divided by 200
and the post-change clean variance. Use trapezoidal integration at the fixed
recording steps. Secondary endpoints: final MSE, coefficient/amplitude error,
wrapped phase error and wall time. Phase error is undefined when predicted
amplitude is below `max(0.001,0.05*true_amplitude)`; retain the undefined count.

The primary contrast is **polar versus Cartesian, Adam, no penalty, phase-only
and both-change conditions**, balanced equally. Other cells test specificity
and optimizer dependence. They cannot replace a failed primary contrast.

Paired diagnostics include adding 2pi to the parameter phase and rotating input
coordinates and both target/model centers together by pi/3. They do not increase
sample size and use stored-state transformations, not additional full fits.
They establish functional/objective identities, not optimizer-origin robustness.
Rotated retraining is a separately budgeted future extension. Test historical
phase decay only through labeled objective/gradient coordinate-dependence
diagnostics, outside the primary training matrix.

A continuation criterion: at least 10% lower primary normalized curve area,
improvement in both phase-bearing conditions, no hidden numerical failures,
and a paired task-level 95% bootstrap interval whose improvement lower bound is
positive. If the interval crosses zero, label the result inconclusive even if
the point estimate is promising. Report all other conditions. This is a
prespecified engineering research decision, not a power or universality claim.

## B — circular locality and sample efficiency

The primary models each have 49 trainable scalars. Fixed feature count and
computation also vary and must be reported; parameter equality is not equality
of function class or computational budget.

| Family | Fixed configuration | Trainable count |
| --- | --- | ---: |
| Learned VM | 16 learned centers, 16 positive widths, 16 signed readout weights+bias | 49 |
| Fourier ridge | Integer harmonics 1–24; sine/cosine readout+bias | 49 |
| Fixed VM ridge | 16 evenly spaced centers × widths {1,4,16}; 48 readout weights+bias | 49 |
| Learned geodesic features | 16 centers/widths/weights+bias; `exp(-kappa*wrap(theta-c)^2/2)` | 49 |
| SinCos MLP | `[cos(theta),sin(theta)]` → tanh width 12 → scalar | 49 |
| Raw-angle MLP | `theta/pi` → tanh width 16 → scalar | 49 |

Learned VM/geodesic models share initial centers `-pi+2*pi*j/16`, a common
uniform [0,2pi) grid rotation per initialization seed, widths 4 and signed readout draws
`Normal(0,1/sqrt(16))`, with bias 0. Fixed VM uses the unrotated grid and has no
initialization-seed repeats. For each MLP affine layer, draw each weight and bias
independently from `Uniform(-1/sqrt(fan_in),1/sqrt(fan_in))`, from a dedicated
recorded initialization stream. Initial functions are not claimed to match across different
representations. All parameters receive zero native weight decay. Gradient
models receive the explicit readout penalty `0.0001*sum(weights^2)/2`; input,
center, width and hidden parameters are not included in that penalty.

Ridge models select lambda from 0/0.0001/0.01/1 using validation MSE, with smaller
lambda breaking a 1e-12 tie. The objective is half mean squared error plus half
lambda times squared readout norm, excluding bias. Accordingly sklearn-style
ridge alpha is `n_train*lambda`; lambda 0 uses an SVD minimum-norm solution.
Count all four solves. Closed-form baselines intentionally receive strong,
cheap optimization; a single poorly tuned linear baseline is insufficient.

Use four target families with equal scoring weight:

1. **Smooth periodic:** choose task-seed-random coefficients for integer
   harmonics 1–3, each coefficient `Normal(0,1/harmonic^2)`.
2. **Narrow periodic:** draw one or two compact polynomial bumps with equal
   probability. Amplitudes are independent uniform [0.5,1.5]. For each,
   `b(d,w)=(1-(d/w)^2)^3` if `abs(d)<w`, else 0, with wrapped angular distance,
   centers uniform on the circle and widths uniform in [0.15,0.45] radians.
3. **Multimodal periodic:** draw three, four or five signed triangular bumps with
   equal probability and independent equiprobable signs. Use wrapped
   distances, widths uniform in [0.2,0.8], centers uniform, signed amplitudes
   with magnitude uniform in [0.5,1.5]. Do not require separated centers.
4. **Nonperiodic negative control:** phase uniform [0,2pi),
   `theta/pi + 0.2*sin(theta+phase)` on
   `[-pi,pi)`. Its endpoint mismatch is intentional; periodic models should not
   be expected to reproduce it uniformly. Report it separately from success gates.

Center and scale each clean B function to zero mean and unit variance on a fixed
65536-point generator grid, as part of the declared data-generating process.
This oracle scale is not estimated from evaluation outcomes to choose models.
Apply no additional training target normalization: the generator-defined scale
is the scale used by every model, penalty and stopping tolerance. Noise SD alternates
0.05/0.20 by task replicate parity; report those subgroups, without claiming a
factorial noise effect. Pilot noise is 0.10.

Training/validation inputs are uniform on the circle/domain. Boundary stress is
a paired coordinate-rotation diagnostic of each periodic task: shift coordinates,
target centers/phases and model initialization consistently, including peaks
crossing the branch cut. Use mapped stored states and reevaluation, not additional
training. This does not establish optimization robustness to a rotated origin.
It is not an independent task family. Missing-arc
sampling and real temporal phase drift are extensions, not v1 primary cells.

All gradient models use Adam, batch 64, at most 600 updates, validation at step 0
and every 20 updates, early stopping after five evaluations without a 1e-6 MSE
improvement. Select the best validation checkpoint, including initialization.
Use a deterministic shuffled cyclic training stream shared across models for
each task/initialization seed. The B pilot uses 128 training/256 validation
observations and 100 updates. It compares learning rates 0.001/0.01, chooses one
per gradient family by the mean best eligible noisy-validation checkpoint MSE
across all four pilot task families, and breaks 1e-12 ties toward the smaller rate.
Each rate must have complete finite results on every planned pilot task; otherwise
it is ineligible. If no rate remains, revise before development. Then freeze rates and
stopping rules; no development-driven retuning is part of v1.

Primary endpoint: mean clean-function squared error over the 4096-point grid,
divided by the clean function's evaluation-grid variance. Report the complete
32/128/512 learning curve. Secondary outputs: noisy-response RMSE, mean/max
absolute error, boundary error, learned center/width distributions, fraction of
features below 1e-12, gradient norms, update count, runtime and peak GPU memory.
A finite but stalled fit is not concealed as a successful learning outcome.

Select the strongest alternative periodic control on development, among Fourier
ridge, fixed VM ridge, geodesic features and SinCos MLP. Selection uses mean
primary error, balanced equally across the three periodic families, task
replicates, initialization seeds and sample sizes. All six models still run
on confirmation. The raw-angle reference is reported, particularly on the
negative control, but is not used to make a weak primary periodic comparator.

Freeze that comparator before confirmation. B continuation requires at least 10%
lower task-balanced primary error than it, improvement in at least two of three
periodic families, no periodic family more than 10% worse, and a paired task-level
95% bootstrap interval with positive improvement lower bound. Report results
against every other control too; do not claim superiority to all if any beats
VM. The nonperiodic failure regime is never averaged away into a circular win.

The fixed VM control is still a VM/chordal-kernel representation; beating it
tests learned locations/widths, not the existence of circular features. If a
fixed kernel bank succeeds while learned VM does not, report that distinction
without relabeling a failed primary adaptation/locality comparison as a pass.

If B passes, the immediate next diagnostic is the 2×2 learned/fixed center and
width ablation with 16 features and a shared starting function/readout. Those
models have different trainable counts; label them mechanism ablations, not
the 49-parameter primary comparison. Lock a new budget and unused task seeds
before execution. If B fails, preserve the negative result; do not automatically
increase kernel count or tune concentrations on confirmation tasks.

## Aggregation, uncertainty and failure discipline

Average initialization seeds and sample-size endpoints within a task before
inferential aggregation. Average tasks within family, then families equally.
For A, pair task IDs across conditions and resample whole task bundles. For B,
resample whole paired task bundles within family, preserving seed and sample-size
dependence. Use 10000 bootstrap resamples with analysis seed 9001 and report 95%
percentile intervals. These small designed task samples can yield wide intervals;
there is no claim of a powered confirmatory population study. Replication on
new task families would still be required for broader methodological claims.

Use the ratio of balanced aggregate errors for percent improvement; do not
average unstable per-task percentage ratios. Report arithmetic task/family
errors, paired differences and all learning curves alongside the ratio.
Undefined phase metrics remain null with reasons; JSON must reject NaN/Infinity.

Record every planned run before launch. Nonfinite states/losses/gradients,
unexpected exceptions, timeouts and unfinished runs remain explicit. Do not
select a favorable surviving subset or silently retry a failed seed. Any
missing primary paired run prevents a complete success gate. A timeout is an
incomplete study, not a negative methodological finding. Repeated observations,
grid points, seeds and learning-curve points are not independent tasks.

## Compute budget and launch order

Counts below include all pilot learning-rate alternatives. A has no redundant
training-seed repetitions; ridge configs have no initialization-seed repetitions.

| Invocation | Gradient fits | Ridge configurations | Notes |
| --- | ---: | ---: | --- |
| A pilot | 48 | diagnostic references | 2 conditions × 2 coordinates × 2 optimizers × 2 penalties × 3 rates; 40 updates |
| B pilot | 32 | 8 | 4 targets × 4 gradient models × 2 rates; 100 updates |
| A development | 64 | diagnostic references | 4 conditions × 2 task pairs × 8 arms; 200 updates |
| B development | 192 | 48 | 4 targets × 2 tasks × 3 sizes; 2 initialization seeds for 4 gradient models |
| A confirmation | 256 | diagnostic references | 4 conditions × 8 task pairs × 8 arms |
| B confirmation | 576 | 144 | 4 targets × 6 tasks × 3 sizes; 2 initialization seeds for 4 gradient models |
| **Total A/B primary plan** | **1,168** | **200** | Each B ridge configuration has 4 candidate solves: 800 solves total |

E microbenchmarks, unit tests and standard forecaster validation are separate
from those training counts. A additionally has 84 unique diagnostic closed-form
solves: 4 pilot, 16 development, 64 confirmation; do not count repeated optimizer
arms as new reference solves. Pilot fits are disposable and not evidence for the
methodological claims. Start with E, then A/B pilots; finish pilot selection
and code tests before development. Freeze the confirmation lock after
development; evaluate both primary studies once even if development is
unpromising, provided correctness/runtime checks allow completion. Development
screening must not hide negative confirmations.

CUDA is the default when available. Use one training worker and no concurrent
GPU experiments. Independent agents may work on code, data and audits in
parallel, then serialize training. Hard ceilings per invocation: each pilot 10
minutes; A development/confirmation 15 minutes each; B development/confirmation 45
minutes each. Keep progress visible within 60 seconds. After pilots, extrapolate
using measured median and p90 fit times; if the declared full matrix will not
fit, revise and re-lock the design before accessing confirmation. Do not silently
cut seeds, updates or datasets to fit a deadline. Maximum declared A/B stage
ceilings total 140 minutes; this is a bound, not a runtime prediction.

## Implementation work packages and handoff

All paths below are planned unless already present. No listed command should
be advertised as runnable until its implementation and `--help` check exist.

| ID / owner role | Planned files | Acceptance / dependency |
| --- | --- | --- |
| E1 / layer engineer | `src/circular_layers_v2.py`, `tests/test_circular_layers_v2.py` | Numerical contracts, conversions, freeze modes and counts; independent review |
| E2 / optimization engineer | `src/methodology_training.py`, trainer tests | Explicit penalties/groups, full-batch A, bounded B, validation-only selection, failure records; depends E1 |
| D1 / data engineer | `src/methodology_tasks.py`, task tests | Independent seeded streams, nested samples, exact target formulas/rotations, no held-out tuning |
| E3 / audit engineer | `scripts/benchmark_circular_layers_v2.py` | Output/gradient parity plus actual latency/memory; standard adapter checks |
| R1 / orchestrator | `scripts/run_vm_methodology.py` | Config-driven manifests, pilot/dev/confirm separation, resume without rerunning completed fits, lock verification; depends E1/E2/D1 |
| R2 / independent auditor | `scripts/analyze_vm_methodology.py`, analysis tests | Recompute curves/metrics, replay selection/gates, paired aggregation/bootstrap; no training |
| H1 / orchestrator | `docs/vm_methodology_results.md`, project state | Track concise metrics, failed hypotheses, source/data hashes and next justified question |

Results root: `results/vm_methodology_v1/<unique-run-id>/`. Each run records
protocol/code/raw/generated-data hashes, model class/config and exact count,
initial/final states, stream seeds, hyperparameters, complete training and
validation histories, predictions/targets, timing/device/environment, failure
status and analysis version. Generated data and outputs stay ignored; tracked
reports preserve essential evidence. Never overwrite older snapshots or notes.

During implementation the numerical, data and trainer packages may be delegated
with exclusive file ownership. Integrate and independently review them before
launching a study. The immediate next work is **E1 and D1**, followed by E2/E3
and the bounded pilots. No application-use-case clarification is required for
this methodological program.

## C — reserved interaction follow-up

Do not launch a C grid from this plan. First decide from A/B whether a specific
interaction limitation remains. A candidate follow-up compares additive angular
features with explicit integer harmonics of `theta_1-theta_2`, a matched
trigonometric interaction control, and local joint kernels. Include additive
negative controls. Separate per-input 2pi periodicity from common-rotation
invariance, and distinguish extra information/features from VM parameterization.
It requires a new frozen task definition and unused confirmation seeds.
