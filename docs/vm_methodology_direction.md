# VM methodology assessment — 2026-09-08

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: mathematical assessment and proposed experiment direction
- Verification Status: ANALYZED; identities derived from source, initialization probed locally
- Scope: methodological hypotheses, not demonstrated superiority or novelty

The owner redirected the project from useful forecasting to pure methodology.
The central question is now: **what does a VM parameterization or circular
kernel contribute beyond matched trigonometric features?** Preserve the completed
forecasting evidence; do not use power performance as the sole criterion for
studying circular representations.

## What the existing layer represents

For the actual code convention, preferred input direction is `-mu` modulo2pi:

`z_j = sum_i kappa_ji cos(theta_i + mu_ji)`

Expanding cosine gives `z = A cos(theta) + B sin(theta)`, where
`A = kappa*cos(mu)` and `B = -kappa*sin(mu)`. Hence a log-output VM layer and
the matched SinCos layer have the same function class, apart from the exact
zero-amplitude boundary: finite softplus parameters cannot produce zero.
Applying the same exponential and bias preserves that equivalence. Deep stacks
remain nonlinear in their original inputs; the identity applies layer by layer.

Exact angular wrap consistency is useful, but ordinary sine/cosine encoding
also provides it. It is not invariance to arbitrary physical rotations, nor a
guarantee of circular-output equivariance. Affine input projections and raw
residual pathways require separate end-to-end checks.

In log output, kappa changes amplitude, not harmonic frequency or relative peak
width. Exponentiation creates localization and multiplicative interactions
between input dimensions. The current forecasters predominantly use log output,
so their results do not constitute a strong test of localized circular kernels.
One additive log layer cannot directly express a relative-angle interaction
such as `cos(theta_1-theta_2)`; later nonlinear layers can create interactions.

The controlled comparisons found no consistent VM advantage over the exact
SinCos control. They also exposed a synthetic clock mismatch that confounded
an apparent temporal-convolution benefit. These constrain claims about the
tested configurations, not every possible use of circular kernels.

## New concrete findings from source

1. **Regularization is not matched in function space.** Current Adam weight
   decay acts on phases and raw softplus parameters. Phase penalties depend
   on the arbitrary angular origin and even the chosen2pi representative.
   Decay toward raw concentration zero pulls kappa toward log(2), not zero.
   Cartesian decay instead penalizes `A^2+B^2 = kappa^2`. Existing paired
   experiments therefore compare both optimization coordinates and priors.
   Start a clean comparison with encoder weight decay disabled, then use an
   explicit matched amplitude penalty. Keep backbone settings matched.
2. **The temporal bank is algebraically redundant for representation.** The
   log-output bank followed immediately by its linear phase projection equals
   a direct affine map of input sine/cosine features. Two time angles provide
   at most four nonconstant feature directions, regardless of bank width.
   The subsequent sensor phase modulation is nonlinear and is not removed by
   this observation. A collapsed direct map is the essential next control.
   Factorized optimization may still differ and needs separate measurement.
3. **Initialization has reproducible edge-case defects.** Requested kappa0.001,
   0.01 and0.05 all initialize to0.04858735; kappa1000 raises OverflowError.
   The core uses a low-value floor and unstable inverse softplus. Validate
   positive finite inputs and use `k + log(-expm1(-k))`. Under independent
   uniform phases, log-response variance is `sum(kappa_i^2)/2`; initialization
   should address input width explicitly and be tested with/without LayerNorm.
4. **The forward can avoid the broadcast activation tensor.** Compute input
   sine/cosine once and use matrix products with coefficients derived from the
   original polar parameters. For a shared per-output phase shift delta,
   `z_delta = cos(delta)*z_0 - sin(delta)*q`, with
   `q = sum_i kappa_i sin(theta_i+mu_i)`. This preserves the mathematical
   function and gradients without a batch*time*output*input cosine tensor.
   Numerical parity and measured runtime/memory are separate claims.
5. **Interpretability and stability need narrower claims.** Raw phase is not a
   causal explanation; it is unidentifiable as amplitude approaches zero and
   can trade off against a time-projection bias. Exp-output gradients are not
   uniformly bounded over learned concentration. Stacked cosines also mean
   the source's literal `exp(exp(z))` explanation is inaccurate. Keep frozen
   source snapshots intact; make corrections in a versioned successor.

## What could be gained

- **Learning dynamics:** explicit strength/phase coordinates might improve
  adaptation when phase changes while strength stays similar. This is an
  untested optimization hypothesis, not extra expressiveness. Near-zero
  amplitudes may instead impede phase learning.
- **Localized circular features:** investigate
  `g(theta)=exp(kappa*(cos(theta-c)-1))`, with separate signed readout weights.
  This peak-normalized kernel has peak1 and a nonpositive exponent; concentration
  controls angular width without increasing peak height. It is not a normalized
  probability density, can underflow, and does not bound every gradient for
  arbitrary kappa. Sparse/local combinations avoid assuming every sensor must
  simultaneously match a single high-dimensional prototype.
- **Circular uncertainty:** for genuinely angular targets, a normalized VM
  likelihood can learn mean direction and concentration. This is a probabilistic
  output-head direction, distinct from inserting VM hidden layers. Multimodal
  targets require an appropriate mixture or richer circular distribution.
- **Structured angular interactions:** integer harmonics and relative-angle
  terms can represent multiple peaks and joint phase relationships while
  preserving intended2pi periodicity. They require matched Fourier controls;
  arbitrary learned real frequencies do not guarantee that same periodicity.

These ideas have established connections to circular statistics and Fourier
features. This assessment is not an exhaustive novelty search.

## Recommended next experiments, not yet executed

Use a small readout first so an LSTM does not obscure the layer's contribution.
Separate three questions:

1. Does polar optimization help on the **same function class**? Match initial
   functions, data/batches, explicit regularization and training budgets; vary
   strength, phase drift, noise and data quantity. Track phase/amplitude learning,
   error, variability and convergence. Include Cartesian and polar controls.
2. Do learned localized kernels improve **sample efficiency or robustness**?
   Compare stable VM kernels with matched Fourier/harmonic features, circular
   distance kernels and a sine/cosine MLP. Include smooth sinusoids, narrow and
   multimodal peaks, branch-cut crossings, and nonperiodic negative controls.
   Use diverse target generators so VM-shaped targets do not predetermine a win.
3. Do **relative-angle interactions** add value? Compare additive features with
   explicit interactions under matched capacity, and test the claimed rotation
   transformation law at the appropriate boundary.

Reserve independent generator seeds/tasks before tuning; report learning curves,
multiple training seeds, compute/memory and failure regimes. More parameters or
one successful forecasting run cannot answer these methodological questions.
Apply existing standard benchmarks as engineering regression checks when models
change, while keeping the methodological tests specific to the new claims.

## Sources and local evidence

- [Core implementation](../src/von_mises_neuron.py)
- [Exact SinCos control](../src/sincos.py)
- [Temporal implementation](../src/temporal.py)
- [Controlled results](controlled_comparison_results.md)
- [SciPy VM definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html):
  density `exp(kappa*cos(theta-c))/(2*pi*I0(kappa))`; normalization matters for
  probability semantics.
- [Tancik et al., NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html):
  Fourier feature mappings can change the learning of high-frequency functions
  in low-dimensional regression. This is relevant precedent, not evidence of
  a VM advantage in this project.

No model behavior was changed and no new training study was launched for this
assessment. The initialization probe was read-only. The reproducible
`scripts/audit_vm_methodology.py` diagnostic passed with seed 7, CPU float64:
maximum factored-forward difference 2.66e-15, gradient difference 5.33e-15
(log/exp, inputs/phases/raw concentrations/modulation/bias), and collapsed
temporal-projection difference 2.11e-15. These checks corroborate the derived
identities at sampled values; they are not training or speed benchmarks.
Full provenance is in `results/methodology_2026-09-08/algebra_audit.json`.
`vm_response_shapes.png` in that directory illustrates amplitude versus width
analytically; it is not an experimental result.
