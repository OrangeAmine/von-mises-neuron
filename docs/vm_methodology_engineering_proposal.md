# Versioned VM methodology engineering proposal

Status: design only, 2026-09-08. No proposed classes or training results below
exist yet. This is an engineering input to the canonical methodology plan, not
a replacement for its experiment definitions. Preserve the frozen forecasting
implementation and artifacts. The completed algebra audit establishes only the
two identities recorded in `results/methodology_2026-09-08/algebra_audit.json`.

## 1. A successor layer with explicit numerical semantics

Implement new modules, for example `src/circular_layers_v2.py`, rather than
changing `VonMisesLinear` or silently migrating old checkpoints.

```python
PolarCircularLinearV2(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    response: Literal["sum", "exp"] = "sum",
    init_kappa: float = 1.0,
    init_policy: Literal["constant", "unit_variance"] = "constant",
    implementation: Literal["factored", "reference"] = "factored",
)

forward(x: Tensor, phase_shift: Tensor | None = None) -> Tensor
from_legacy(reference: VonMisesLinear) -> PolarCircularLinearV2
```

Store `phase` and `raw_kappa`, with `kappa=softplus(raw_kappa)`. Reject nonpositive
or nonfinite requested concentration and invalid dimensions. Initialize with
`raw = kappa + log(-expm1(-kappa))`, including very small positive concentration;
do not clamp all values below 0.05 to approximately 0.049. Validate representability
in the chosen dtype. Record requested and effective initial concentration.

The policies change initialization only, not the forward equation:

- `constant`: every edge starts at `init_kappa`.
- `unit_variance`: every edge starts at `init_kappa*sqrt(2/in_features)`.
  Under independent uniform initial phases, the pre-bias sum then has variance
  `init_kappa**2` over phase initialization. This does not assert stationary
  activations or gradients after learning, LayerNorm or stacking.

The experiment configuration must name its policy explicitly. Legacy conversion
copies effective parameters without applying either new initializer, preserves
dtype/device/mode, introduces no aliasing, and consumes no RNG draws. Legacy
`log_output=True` maps to `response="sum"`; its output includes additive bias
and is not a normalized log-density. Legacy state keys remain accepted only by
the explicit conversion API, never guessed during ordinary checkpoint loading.

Use `preferred_centers() = (-phase) modulo 2*pi` for interpretation. Retain the
phase convention `cos(x+phase)` internally for exact legacy conversion. Physical
interpretation applies directly to observed angular inputs; hidden activations
treated as angles do not acquire physical directions merely from their names.

## 2. Factored execution and the direct temporal control

Let `A=kappa*cos(phase)` and `B=-kappa*sin(phase)`. Calculate
`z0=linear(cos(x),A)+linear(sin(x),B)` and
`q=linear(sin(x),A)-linear(cos(x),B)`.
For the existing per-output shared phase offset, calculate
`z=cos(delta)*z0-sin(delta)*q`; otherwise use `z0`. Apply the declared response
and output bias afterward. Coefficients are derived inside forward so gradients
still optimize polar parameters. Do not cache detached derived coefficients
during training. This removes the large `(..., outputs, inputs)` activation;
the computational advantage must still be measured, especially at small sizes.

The shared-offset shape is explicitly `(..., out_features, 1)`. Reject general
per-edge offsets in this API: they do not admit the same shared rotation. Preserve
supported leading dimensions and document broadcasting deliberately.

Add a separate `DirectCalendarPhaseV1(time_size, hidden_size, num_layers)` control,
with an affine map of `[cos(theta), sin(theta)]` for each sensor layer, and an
explicit `from_legacy_encoder` conversion. For bank coefficient matrices `A,B`,
bank bias `b`, and projection `W,c`, its weights are `[W@A, W@B]` and bias `W@b+c`.
Copy every non-bank branch from the same reference encoder. This reproduces the
initial function, not optimizer trajectories, factorization priors or gradients
with respect to parameters that no longer exist.

Capacity accounting, with bank width H, T time angles and L phase projections:

| Component | Trainable parameter count |
| --- | ---: |
| Fixed-amplitude legacy bank plus projections | HT + H + L(H² + H) |
| Direct independent affine phase maps | L(2TH + H) |
| Polar or Cartesian core, input D, output O | 2DO + O with output bias |

At H=64, T=2, L=2, the first two counts are **8,512 versus 640**. A learnable
per-bank-neuron amplitude adds H to the first count. Do not call these temporal
models parameter matched. Report equal-input/equal-function initialization and
their different counts separately. A width sweep, if authorized, is a separate
capacity experiment and must not replace this exact control.

The bank span is at most 2T before bias; nonlinear sensor-phase modulation can
still produce richer time dependence. A direct map therefore removes redundant
representation before modulation, not all time interactions. Projection bias
and a shared shift of each sensor neuron's phases also have an exact offset
ambiguity. Initially retain that ambiguity for clean conversion; a gauge-fixed
follow-up may absorb the constant into the sensor phases and omit phase-map bias.

## 3. Function-space regularization with auditable optimizer groups

Create experiment-owned optimizer groups without changing the frozen trainer.
Disable optimizer weight decay for the compared circular encoders in both polar
and Cartesian models. Keep backbone/readout treatment identical. The first
comparison has no encoder penalty; the second adds exactly

`lambda/2 * mean_over_edges(kappa**2)`

or the Cartesian equivalent

`lambda/2 * mean_over_edges(weight_cos**2 + weight_sin**2)`.

Define explicitly whether each layer gets equal weight; recommend the sum of
layerwise means for this package. Exclude phase, raw concentration and encoder
bias from default decay. List parameter names/counts in the manifest and reject
missing or duplicate optimizer membership. Test equality of the penalty at
converted states; do not expect equal coordinate gradients or learning paths.

The existing raw-parameter decay favors phase zero and raw concentration zero
(effective kappa=log(2)); it is not amplitude shrinkage. Near zero amplitude,
phase is poorly identified and its gradient vanishes. These are optimization
properties to measure, not automatic defects or guaranteed advantages of polar
coordinates. Do not wrap the trainable phase in-place each optimizer step:
that silently changes its relationship to optimizer momentum/state.

## 4. Local circular kernels as a separate representation experiment

```python
PeakCircularKernelBankV1(
    in_features: int,
    num_kernels: int,
    *,
    centers: Tensor,                  # (kernels, angular_inputs), in radians
    concentrations: Tensor,           # follows declared sharing
    learn_centers: bool,
    learn_concentrations: bool,
    width_sharing: Literal["per_edge", "per_kernel", "global"] = "per_edge",
)
```

The feature for kernel m is

`phi_m(theta) = exp(sum_i kappa_mi * (cos(theta_i-center_mi)-1))`.

Compute its log as `-2*sum_i kappa_mi*sin((theta_i-center_mi)/2)**2`, which avoids
cancellation near the peak and prevents a positive exponent from ordinary
roundoff. Use stable softplus initialization for learned positive widths;
fixed centers/widths are buffers, not unused parameters. Validate supplied
initial tensors rather than drawing undocumented centers. Fixed centers must
come from the preregistered grid/random procedure, or training inputs only.

Attach a separate unrestricted signed `Linear(num_kernels, output_size)` readout.
Do not add bias inside the kernel. This separates local features from signed
prediction amplitudes. With D angles, M kernels and O outputs, learned centers
contribute MD parameters; learned widths contribute MD, M or 1 by sharing policy;
the signed readout contributes MO+O. Fixed tensors still consume feature capacity
and compute, so report feature count, trainable count and runtime together.

Cross fixed/learned centers with fixed/learned widths while preserving initial
centers, widths and readout. This tests adaptation rather than different starts.
Begin localization comparisons in one angular dimension: multiplying marginal
kernels in D dimensions also adds joint selectivity, so a multivariate gain
cannot automatically be attributed to peak sharpness alone.

The peak-normalized VM and a chordal Gaussian are **the same kernel**:

`exp(kappa*(cos(theta-center)-1))`
`= exp(-kappa*||u(theta)-u(center)||²/2)`, with `u(a)=(cos(a),sin(a))`.

For multiple angles, sum the weighted chordal distances across inputs. Include
this identity in output/input/center/width gradient parity tests; treat it as an
equivalent implementation control, not an independent competing family or
novelty claim. The kernel is peak normalized, not integral normalized; it is
not a probability density. Large width/dimension can underflow to zero and
produce negligible gradients. Width learning can still become ill-conditioned;
finite output alone does not establish stable training. Do not silently clamp
widths or outputs. Any concentration bounds/penalties need explicit versioned
configuration and comparison.

## 5. Bounded engineering packages and acceptance

1. **Numerical successor and adapters.** Test requested concentration recovery,
   rejected invalid inputs, fan-in variance formula, state conversion, no shared
   parameters/RNG disturbance, dtype/device movement and registered buffers.
2. **Factored execution.** Compare reference/factored sums and exponential
   outputs, with and without dynamic phases, across leading shapes and CPU/CUDA
   float32/float64. Check finite input, phase, raw-width, dynamic-offset and bias
   gradients; use float64 gradcheck away from numerical extremes. Preserve
   per-angle 2*pi invariance. Record numerical tolerances by dtype.
3. **Temporal conversion.** Test every phase projection and complete deterministic
   forecaster output against its reference; clone all remaining states and RNG.
   Check time-path periodicity separately from the raw-sensor residual path.
4. **Regularization and kernel controls.** Verify exact effective penalty
   equivalence, optimizer membership, freeze modes/counts, signed-readout
   behavior, kernel center maxima, periodic boundaries and chordal parity.
5. **Integration and gates.** Run the full suite for model/pipeline additions.
   Put new forecaster adapters through unchanged standard synthetic RMSE<=35,
   MAE<=25, R2>=0.20, MAPE<=5% gates before calling them validated. Temporal
   successors additionally require <=10% error regression and <=0.10 R2 loss
   against fresh base VM. Preserve and report failed candidates; do not weaken
   gates. These forecasting checks certify repository integration, not a
   localized circular-methodology hypothesis.

All new model checkpoints require versioned class/configuration identifiers and
complete source manifests. Existing forecasting artifacts remain usable under
their original hashes. No conversion should overwrite them. No training is
authorized or performed by this design document itself; the parent methodology
plan controls datasets, training budgets, comparisons and stopping decisions.
