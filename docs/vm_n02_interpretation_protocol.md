# VM-N02 v1: prospective interpretation protocol

## Material Passport

- Origin: academic-research-suite / experiment-agent; authorized implementation and execution.
- Scope: synthetic diagnostic study using existing CircularBasisV2 and CartesianCircularLinearV1 APIs.
- Status at creation: prospective; no pilot, development or reserved data generated.
- Baseline: a7431a926c751868b11a3918c6cdd29be6540a32; verified snapshot receipt in vm_n02_baseline_receipt.json.
- Machine-readable contract: configs/vm_n02_interpretation_v1.json. Hash this document and config before pilot; bind source and results before reserved access.

## Question and estimands

Do learned phases and widths identify stable predictive structure? Separate exact
representation identities, parameter identifiability, prediction dependence and
physical causality. No architecture superiority, forecast performance or real
domain claim is tested. Existing v1/v2 results provide context only. No old raw
payload is required or regenerated. Preserve all outcomes, including failures.

Three cases, each with 2 pilot, 3 development and 12 reserved independent units.
Disjoint seed namespaces generate fresh functions/trajectories and observation
noise. These are within-generator replications, not unseen-domain validation.
The two peak cases use three initializations of the SAME data per function;
initializations and angle samples are not independent scientific replicates.
The proxy uses a unique regularized linear solve, so optimizer initialization
is irrelevant and will not be counted as replication.

## Generators, information and fitting

1. **Identifiable isolated peak:** f(theta)=b+a exp(kappa(cos(theta-c)-1)),
   c uniform on the circle; a in [0.8,1.4], kappa in [3,9], b in [-0.3,0.3].
   Full-circle uniform observations, known single positive peak family and
   nonzero amplitude/concentration make continuous noiseless recovery possible
   up to c modulo 2*pi. Flat signals, partial support, multiple components or
   signed mixtures do not inherit this identification argument. Fit one existing
   circular kernel plus linear readout/intercept. Initial center is the maximum
   noisy training observation plus uniform [-0.5,0.5] radians; initialization
   uses training data, not true parameters. This is a deliberately favorable
   local recovery experiment, not a guarantee of global optimization.
2. **Redundant basis:** independently generate TWO positive peaks with the same
   ranges, first center uniform and separation uniform [1.4,3.8]. Fit FOUR
   kernels, initialized equidistantly with a uniform global rotation, kappa
   uniform [2,8]. Signed readout permits overlap/cancellation. Individual
   components need not correspond one-to-one to generator peaks.
3. **Calendar proxy:** latent oscillator q covers eight cycles with a random
   origin; recorded clock t=q; latent direction z=q+epsilon,
   epsilon~N(0,0.15^2); noisy sensor s=z+eta,
   eta~N(0,0.65^2); target b+a cos(z-c)+noise. Clock and target share the latent
   process in the observational generator. The target equation takes z, not
   clock, as its direct argument: a surgical change to recorded clock holding z
   fixed leaves truth unchanged. At shifted evaluation z=q+pi/2+epsilon, with
   paired sensor/target noise. Clock's marginal support stays circular, but its
   association changes. Independent train/validation/evaluation trajectories
   use separate noise streams; no temporal forecasting or overlapping windows.
   Fit the existing Cartesian sum model on sensor/clock cosine and sine features
   via positive ridge, unpenalized intercept. Also RETRAIN a sensor-only model
   using the identical training rows and penalty. No target-derived evaluation
   adjustment is fit. The known +pi/2 alignment is a diagnostic oracle control.

256 training and 128 noisy validation labels per unit; 512 evaluation labels.
Peak evaluation inputs are a full-circle grid with independent noise; response
curves use 1,024 full-circle points. Target units are arbitrary response units;
phase is radians. Peak errors use clean truth for mechanism diagnosis and also
retain noisy error. Clean evaluation target variance normalizes errors but is
never supplied to optimization/checkpoint selection. Proxy errors use the clean
paired targets. All samples and clean/noisy arrays remain locally reproducible.

Peak fits use joint Adam, fixed lr .03, 600 full-batch updates and 1e-5 readout
ridge, no geometry penalty. Initial readout uses the existing positive-ridge
solver. Select earliest minimum NOISY validation MSE at step 0 and every 25
updates through 600. No hyperparameter search, early stopping, retries or
post-evaluation checkpoint selection. Both learning settings and thresholds
are fixed before pilot; development is a diagnostic rehearsal, not selection.

## Diagnostic order and matching

First compare complete response functions across all unordered initialization
pairs: RMSE divided by clean target SD on the same grid. Then compare components.
Hungarian matching minimizes squared differences of SIGNED entire component
contribution curves w_j K_j; permutation itself has no meaning. A component is
active only if its curve RMS / target SD >= .02. Phase/width matching additionally
requires its centered curve SD / target SD >= .02 in both fits; raw activity
alone cannot identify a nearly constant component. Match all components first,
then report active-active pairs, inactive/unmatched counts, and weights equal to
the mean contribution RMS of each pair. Weighted center distances use shortest
circular differences; width dispersion is absolute log(kappa1/kappa2). Report
contribution mismatch and overlap (maximum absolute centered curve correlation
among phase-eligible components, with eligible-pair count; zero if no pair)
alongside matched parameters. Record unmatched active RMS mass and pair counts;
ties use scipy's deterministic row-ordered assignment on the recorded matrix.
These diagnostics do not confer identifiability
on a redundant basis. Negative readout makes a positive kernel center a trough
contribution, not a preferred positive output phase.

Recovery eligibility requires an isolated active positive component with enough
centered variation, but EVERY planned fit stays in the denominator: center error
<= .1 rad and relative kappa error <= .2, with clean NRMSE <= .1. Recovery requires
at least .9 of fits (averaged first within function). Report intervals rather
than using parameter pictures from selected successful fits. Width in this peak
kernel is concentration/locality; FWHM=2 acos(1-log(2)/kappa) when kappa>=log(2)/2,
otherwise no half-height crossing. Sum/log kappa is amplitude; it is never a
learned frequency band. cos(theta+mu) prefers -mu, while cos(theta-c) centers c.

## Prespecified interventions and explanation controls

- Delete the ENTIRE fitted component w_j K_j, including its readout amplitude.
  Record every component; the primary deletion chooses largest curve RMS using
  VALIDATION inputs, with smallest-index ties. Compare original, deletion, and
  center offsets +/-pi/2 for that same component, plus zero offset. Also multiply
  that component's concentration by .5, 1 and 2; factor 1 is an identity control.
- Exact identities: permutation, 2*pi wrapping, circular-to-chordal RBF,
  splitting one component into two identical kernels with half weights, and
  inserting a duplicate cancelling pair (+0.5,-0.5). The latter can make a
  single deletion appear important without changing the response. This is an
  algebraic diagnostic, not an independently fitted scientific replicate.
- Calendar: remove the complete clock contribution; replace its genuinely
  time-varying output by its noisy-validation-input mean; rotate clock +/-pi/2;
  zero shift; wrap by 2*pi. Do not call fixed parameter replacement time freezing.
  Distinguish fixed-prediction perturbations from sensor-only RETRAINING.
- A simultaneous coordinate-origin change theta->theta+.7, centers->centers+.7
  (and Cartesian coefficient rotation) must preserve output. Clock-only rotation
  with fixed parameters is a different diagnostic. Wrapping by 2*pi is a third
  property. Marginal circular support is preserved by rotations, while proxy
  joint support/conditional association may change; cancellation/deletion and
  geometry offsets change model state outside the fitted parameter distribution.
- Attribution display: signed derivative of predicted response with respect to
  angle (proxy: clock partial derivative), response units/radian. Compare fitted
  derivative with separately randomized readout, randomized geometry (uniform
  phases, widths drawn from initial range), and both. Keep target/output scale
  fixed, do not restandardize each display. Report relative L2 changes. Exact
  equivalent representations must preserve full-function derivative even when
  component allocation changes. Invariance there is DESIRABLE; sensitivity to
  randomized parameters is necessary here but not proof of causal correctness.

One randomization draw per fit/type is fixed, not a distributional guarantee:
peak RNG seed = initialization seed + 9071, independent N(0,1) weights and bias,
then uniform phases and initial-range widths. Proxy uses seed 8802, uniform
amplitudes [.2,1.8] and phases [0,2*pi] in its equivalent polar coordinates;
"readout" randomizes amplitude and "geometry" randomizes phase. Proxy widths do
not exist. A derivative norm <=1e-14 is unevaluable, never a randomization pass.
Scores average initializations within functions and functions equally. A
prediction derivative cannot detect an additive bias change; bias is not given
an attribution claim. Bootstrap counts and missing controls remain explicit.

Primary prediction gate: mean clean NRMSE <= .1; curve stability <= .05.
Descriptive instability flags: weighted matched center > .2 radians or log-width
dispersion > .3. Attribution sensitivity requires relative L2 change >= .2 for
each declared randomization. Exact/zero controls require max abs error <= 1e-10
in float64. Practical intervention effect is paired change in normalized MSE
>= .1; report sign and interval including reversals. Proxy reliance claim requires
positive lower interval beyond .1 for aligned ablation damage and shifted
full-minus-sensor-only error. The causal implication is rejected by the generator
counterfactual, irrespective of prediction or attribution quality. No global
success gate combines these distinct claims.

The .1 prediction threshold is retained for all cases. An analytical design
check (before data generation) gives a population-optimal linear proxy NRMSE
about .1461 with the declared sensor/latent noises. Thus proxy quality is
expected to fail this deliberately demanding threshold even if reliance is
clear; no threshold is relaxed to engineer a pass.

## Uncertainty, execution and freeze

Average fit metrics and pair metrics within each function, except exact-control
errors, which retain the worst error across all fits and must pass in EVERY unit.
Bootstrap the 12
whole independent functions/trajectories with replacement, 5,000 draws, fixed
seed, paired effects computed BEFORE resampling. Report mean and percentile
95% intervals, initialization SD separately, and individual unit tables.
Intervals are small-sample descriptive precision measures, not simultaneous
familywise tests or evidence of physical causality. Do not treat visual overlap
or baseline SD as a test. Preserve failed/nonfinite/missing outcomes explicitly;
no survivor-only aggregate can satisfy a claim.

CUDA if available; float64, one CPU thread, serialized fits. Check competing GPU
work before timed stages and record observed contention. Pilot cap 180 seconds,
development 240 seconds, reserved stage 900 seconds (22 minutes total ceilings,
not an intended runtime). Before reserved access, require measured pilot
runtime * (12/2) * 1.5 <= 900 seconds AND development runtime * (12/3) * 1.5 <=900.
Caps include generation, diagnostics and saving. Failed feasibility blocks
reserved generation; do not prune cases or relax caps. A deadline abort retains
completed and failed receipts. No cloud workload. Evaluation output directory
must be new; source/config/protocol and completed precursor hashes are frozen
in a lock before generating any reserved unit.

Engineering checks: equations, finite gradients, dtype/device/shape contracts,
matching permutations, inactive components, exact representations, attribution
finite differences and all intervention identities. Full suite for new pipeline.
Existing scientific model implementations stay byte-identical; no new forecaster
is introduced, so existing historical forecaster benchmark results are context,
not new validation of this study. Run unchanged standard synthetic gates as an
additional repository integration check, retaining failures and all thresholds.

Deliver runner, protocol/config, source lock, complete compact metric tables,
figures, claim scorecard, preserved arrays/states, replay audit and handoff.
