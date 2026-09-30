# VM learning playground: offline companion

VM-N03 · Teaching material · 2026-09-22

This companion explains three lessons about circular layers. The interactive
examples illustrate equations and selection effects; they do not train a research
model or produce new scientific evidence. The saved v1 and locality-v2 findings
remain unchanged. All angles in the lessons are **radians**; illustrative response
values have arbitrary units unless a metric is explicitly labeled dimensionless.

[Open the playground](../../learning/vm_playground/index.html) ·
[Open the print-friendly HTML guide](../../learning/vm_playground/guide.html)

## Open, learn, reset

Open `learning/vm_playground/index.html` directly in a current browser. The
playground and HTML guide use local files, with no external fonts, packages,
accounts or model downloads. Keep the repository's relative directory layout so
that the research-source links work. Alternatively, from the repository root:

```powershell
py -3.11 -m http.server 8765 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8765/learning/vm_playground/index.html> or
<http://127.0.0.1:8765/learning/vm_playground/guide.html>. On another host, use its
Python executable in place of `py -3.11`. Stop the server with Ctrl+C. Printing the
HTML guide is optional; it remains readable offline without printing.

For each lesson, **predict** before opening the explanation, **manipulate** one
control at a time, **observe** the plot and numerical values, then **explain** what
changed in your own words. Use that lesson's reset button to return to its original
settings. Identical controls give identical examples. Changing a display filter
never refits or edits the historical study.

## 1. Same function, different coordinates

**Predict.** With amplitude κ fixed, what happens to the preferred input angle
when the phase μ increases? Does doubling κ make a cosine peak narrower?

**Manipulate.** Change amplitude and phase, compare polar and Cartesian response
curves, then compare an angle θ with θ + 2π. Try a very small amplitude and zero.

The one-input sum response is

\[
f(\theta)=\kappa\cos(\theta+\mu)
=\underbrace{\kappa\cos\mu}_{a}\cos\theta
+\underbrace{(-\kappa\sin\mu)}_{b}\sin\theta.
\]

This is the cosine addition formula, an **exact identity**, not a fit. For κ > 0,
the maximum occurs at θ = −μ modulo 2π. The inverse coordinates are
κ = √(a² + b²) and μ = atan2(−b, a), modulo 2π. Both descriptions represent the
same response, although their optimizer trajectories need not match. The polar
phase derivative is −κ sin(θ + μ), so its magnitude shrinks with κ.

**Worked example.** Let κ = 2 and μ = π/3. Then a = 1 and b = −√3. At θ = π/6,
the polar form is 2 cos(π/2) = 0; the Cartesian form is
√3/2 − √3/2 = 0. The preferred input is −π/3, equivalently 5π/3. Adding 2π to
either θ or μ leaves every response unchanged, up to floating-point rounding.

**Observe.** Changing κ scales the sum response vertically. Dividing a nonzero
response by κ recovers the same cosine shape. Changing μ shifts it horizontally.
At κ = 0, a = b = 0 and every phase produces the same flat response: there is no
unique preferred angle. Near zero, tiny coefficient changes can move the inferred
phase substantially while barely changing predictions. Do not interpret such a
phase as stable structure. Exact zero is a useful teaching boundary; the research
layers' finite unconstrained parameters produce strictly positive κ through
softplus in exact arithmetic. A raw softplus parameter is not log(κ).

**Explain the boundary of the claim.** For multiple inputs, the core sum layer is
Σᵢ[aᵢ cos(xᵢ) + bᵢ sin(xᵢ)] plus an optional bias. It is 2π-periodic in each input.
An affine projection before this layer or a raw-input residual around it can break
that guarantee for the complete model. A physical direction in radians has a
physical wrap; a standardized sensor value passed to cosine merely inherits a
mathematical 2π wrap in standardized units. Its preferred value is not automatically
a time-of-day, compass direction or physical mechanism. The historical
`log_output` name denotes the sum response plus bias; it is not the logarithm of
the final biased exponential output.

Sources: [versioned circular layers](../../src/circular_layers_v2.py),
[legacy core implementation](../../src/von_mises_neuron.py), and
[repository mathematical guardrails](../../AGENTS.md).

<details>
<summary>Self-check: μ = π/2, κ = 3. What are a, b and the preferred angle?</summary>

a = 0, b = −3, and the preferred input is −π/2 modulo 2π, or 3π/2.
The response is −3 sin θ. A phase of +π/2 is not the preferred input under the
cos(θ + μ) convention. At κ = 0, no unique preferred angle remains.

</details>

## 2. Amplitude versus locality

**Predict.** If the maximum of a kernel is fixed at one, what can increasing κ
change? Does it change the period?

**Manipulate.** Compare the sum response with the peak-normalized exponential
kernel while keeping the center fixed. Increase κ and inspect values away from
the center. Then change the integer harmonic separately.

Write the center as c = −μ. For one angular input,

\[
s(\theta)=\kappa\cos(\theta-c),\qquad
K(\theta,c)=\exp\{\kappa[\cos(\theta-c)-1]\}.
\]

In s, κ changes amplitude, not the normalized cosine shape. In K, the peak stays
at one and values away from the center decrease as κ grows: this is locality.
At κ = 0, s is zero and K is the constant one. For κ > 0, the kernel's minimum at
the antipode is exp(−2κ). The teaching kernel is peak-normalized;
the legacy exp(s) output has peak exp(κ), and an added bias changes it again.

**Worked example.** At κ = 2 and θ − c = π/2, s = 0 and K = exp(−2) ≈ 0.1353.
At the center K = 1; at the opposite angle K = exp(−4) ≈ 0.0183. Raising κ lowers
the off-center values without moving the center or changing the period.

The Gaussian connection is exact. Define the unit-circle embedding
u(θ) = (cos θ, sin θ). Then

\[
\begin{aligned}
\|u(\theta)-u(c)\|^2
&=2-2\cos(\theta-c),\\
K(\theta,c)
&=\exp\{-\tfrac\kappa2\|u(\theta)-u(c)\|^2\}.
\end{aligned}
\]

This is a Gaussian radial basis evaluated at points on the unit circle, with
Euclidean embedding width σ = 1/√κ when κ > 0. Chord distance is dimensionless
for a unit circle; it is not the same as unwrapped angular distance. The local
approximation K ≈ exp(−κ(θ − c)²/2) uses cos δ ≈ 1 − δ²/2 and is only an
approximation near the center. σ is not an exact angular standard deviation.

**Observe the independent roles.** In cos(hθ), integer harmonic h changes the
fundamental period to 2π/h while preserving 2π wrap consistency. The corresponding
kernel exp{κ[cos(h(θ − c)) − 1]} has h equally high peaks around the original
circle. With h > 1 it is a single Gaussian basis on the *h-fold embedding*
uₕ(θ) = (cos(hθ), sin(hθ)); several original angles map to the same embedded
point. It is not one isolated peak on the original circle. Kappa alone never
sets h. For a clock angle θ(t) = 2πt/T, the physical cycle T comes from the chosen
timestamp feature, and harmonic h gives T/h. Frozen concentration bands do not
create daily and weekly periods by themselves.

**Explain normalization.** Peak one does not mean integral one. The circular
probability density would be exp(κ cos(θ − c)) / [2πI₀(κ)], where I₀ is the
modified Bessel function. The lesson uses an unnormalized basis, not a density;
no probability is inferred from its height. A weighted sum of such bases with a
bias is a regression function and may be negative. The bounded peak-normalized
basis also does not establish bounded gradients or safe exponentials for every
other architecture and parameterization.

Source: `PeakCircularKernelBankV1.log_features` and `chordal_features` in
[the versioned implementation](../../src/circular_layers_v2.py).

<details>
<summary>Self-check: What changes when κ goes from 1 to 4? What changes when h goes from 1 to 2?</summary>

The sum amplitude increases fourfold. The peak-normalized kernel stays at height
one, suppresses off-center responses more strongly, and has embedding width
1/2 instead of 1. Changing h to 2 creates two peaks over a full 2π turn and halves
the fundamental period. It preserves the original 2π wrap, unlike an arbitrary
noninteger frequency.

</details>

## 3. A gain with a failed gate

### Read the saved evidence first

**Predict.** Can an average error reduction coexist with a failed continuation
rule? Which additional facts would you need before calling one method better?

**Manipulate.** Inspect the historical family, total-label, noise and compute
views. These controls select saved summaries, not new runs. Read the model names:
`vp_both` adapts centers and widths while solving its ridge readout at each geometry
state; `joint_both` updates geometry and readout together using Adam. `fixed16`
has the same basis count as the adaptive models. `fixed48` has the same fitted
scalar count as `vp_both`, but more fixed basis functions.

The compact [run record](../vm_locality_v2_run_record.json) and
[cell CSV](../vm_locality_v2_cells.csv) are the historical sources, explained in the
[v2 report](../vm_locality_v2_results.md). The CSV contains 720 confirmation cells
covering every model, family, noise level, label budget and time allowance. Its
`normalized_clean_mse` is clean-grid mean squared error divided by clean target
variance: **dimensionless, lower is better**. Noise SD is measured in standardized
target units. The nonperiodic family is diagnostic and excluded from the balanced
periodic primary score. All v2 model arms are periodic.

Total labels include training and validation: 64 = 48 + 16, 128 = 96 + 32, and
512 = 384 + 128. The primary allowance is two seconds per fit; the other saved
allowances are 0.25 and 0.75 seconds. A common allowance is not equal actual
runtime: fixed controls finish their ridge solve much earlier. Only checkpoints
completed within an allowance are eligible. These definitions come from the
[prospective protocol](../vm_locality_v2_protocol.md) and
[runtime analysis](../vm_locality_v2_results.md#runtime-and-numerical-diagnostics).

**Observe the locked decision.** At the primary allowance, `vp_both` reduced
balanced error by **54.54% against fixed16** and **35.85% against fixed48**.
But its smooth-family error was **47.58% higher than fixed16**, breaching the
unchanged **10%** family regression limit. Both co-primary contrasts had to pass,
so the overall gate failed. The complete rule also required sufficient aggregate
improvement, a positive lower bootstrap improvement bound and improvement in at
least two periodic families.
Source: [locked confirmation decisions](../vm_locality_v2_results.md#locked-confirmation-decisions)
and [the frozen rule](../vm_locality_v2_protocol.md#analysis-and-evidence).

Why can both be true? Each target first averages its paired budgets, noise and
initializations; targets are then averaged within family; the three periodic
families receive equal weight. A large absolute reduction in the narrow or
multimodal error can outweigh a small absolute increase in already-small smooth
error. A family-relative guard asks a different question from that aggregate.
The report's smooth errors are about 0.003304 versus 0.002239: an absolute increase
of about 0.001065 can still be a large percentage increase. The gate was not
rewritten after seeing this outcome.
Source: [same locked decision table and explanation](../vm_locality_v2_results.md#locked-confirmation-decisions).

**Compare the alternatives.** Variable projection's balanced error was **15.05%
higher than joint Adam's**, and all three periodic family means favored joint
Adam. Solving one part exactly does not guarantee the best full method within
fixed wall-clock time; joint Adam completed more updates in this setup. This
observed comparison does not establish update count as the sole cause. At **512
total labels**, larger Fourier controls won the narrow and multimodal comparisons
described in the report. Their weaker small-sample cells explain why they can
win there without having the best overall mean. At **0.25 seconds**, `vp_both`
was worse than fixed48; more computation changed that comparison.
Source: [all model outcomes](../vm_locality_v2_results.md#all-model-outcomes-and-conditional-answer)
and [runtime table](../vm_locality_v2_results.md#runtime-and-numerical-diagnostics).

**Keep v1 visible.** Both v1 primary confirmation gates also failed: polar
optimization's curve error was **62.94% higher than Cartesian**, and learned
locality error was **65.09% higher than its locked fixed-VM comparator**.
V2 changes label accounting, regularization, initialization and compute budgeting;
it is a different comparison and does not reverse v1. These are bounded synthetic
results, not real-world forecasting validation.
Sources: [v1 report](../vm_methodology_results.md) and
[v2 limitations](../vm_locality_v2_results.md#interpretation-limitations-and-preservation).

<details>
<summary>Self-check: If the fixed48 contrast passes, may it replace the failed fixed16 contrast?</summary>

No. Both were required by the locked co-primary rule. A favorable aggregate,
subgroup, alternative optimizer or comparator cannot retrospectively replace the
failed contrast. Reporting the conditional benefit and the failed overall gate
together is the faithful result.

</details>

### Then run the separate selection-bias exercise

**Predict.** If every candidate has the same true error, what happens to the
best development score as you inspect more candidates?

**Manipulate.** Increase the candidate count and compare the selected development
MSE with its independent reserved MSE. Compare small and larger development
samples. Reset returns the same synthetic exercise; it does not create a fresh
reserved research sample.

Every candidate's synthetic residuals are standard normal, so every candidate has
population MSE **1 in dimensionless units**. Candidate sets are nested at
1, 4, 16 and 64. The selected candidate minimizes its development sample's mean
squared residual. Development sizes are 16 or 128; its reserved score uses a
separate sample of 256 residuals. The display averages 200 repetitions using the
fixed teaching seed **20260922**. These numbers define the simulation; they are
not estimates from locality v2 or the external model-selection lesson.

For candidate j, write Dⱼ = mean(e²) on development data and Rⱼ for its independent
reserved score. Selection gives j* = argmin Dⱼ. In every repetition, adding a
candidate to a nested set cannot increase min Dⱼ. But independence gives
E[Rⱼ* | development results] = 1. More candidates can therefore improve the
reported development winner without improving its population error. Finite
reserved averages fluctuate around one; a single realization need not equal it.
Larger development samples generally reduce this selection optimism, without
removing the need for independent evaluation.

**Explain the limit.** Real candidate errors are often correlated, candidates may
have different true quality, and researchers may adapt their search after seeing
results. This deliberately simple exercise isolates selection among equally good
candidates. It does not diagnose the cause of the v2 gate failure, estimate bias
in that study, or measure a generalization gap for the VM models. Repeatedly
tuning against reserved outcomes would make them part of development; a reset
button cannot restore statistical independence.

<details>
<summary>Self-check: The winning development MSE fell while reserved MSE stayed near one. Did the candidate improve?</summary>

No improvement in population quality was built into this exercise. Selection
found a favorable development fluctuation. The independent reserved score tests
the selected candidate without reusing the fluctuation that selected it.

</details>

## Evidence guide and reading map

| Label | What it establishes here | What it does not establish |
| --- | --- | --- |
| **Exact identity** | Polar/Cartesian expansion; 2π wrap for the core function; Gaussian-on-the-circle equality. | Faster training, identifiable learned phases, or a superior model. |
| **Engineering validation** | Focused numerical, reset, source-integrity and browser checks show that this teaching delivery implements its intended behavior. | New scientific confirmation or the validity of an unstated learning claim. |
| **Empirical finding** | Saved v1/v2 outcomes under their stated synthetic tasks, controls, budgets and analysis rules. | Novelty, unseen-domain validation or general forecasting superiority. |
| **Open hypothesis** | For example, separating broad Fourier structure from adaptive local corrections might help. | A demonstrated result; this playground does not run that experiment. |

Read [the core equations](../../src/circular_layers_v2.py) for identities,
[the v1 report](../vm_methodology_results.md) for the earlier negative findings,
[the v2 protocol](../vm_locality_v2_protocol.md) for the prospective contract,
and [the v2 report](../vm_locality_v2_results.md) for decisions and limitations.
The [run record](../vm_locality_v2_run_record.json) and
[cell CSV](../vm_locality_v2_cells.csv) support inspection without ignored training
arrays. Full arrays and checkpoints are not shipped with this teaching resource.
The [VM-N03 proposal](../proposed_tasks_from_forecasting.md#vm-n03--vm-learning-playground-and-reading-companion)
records its origin; this delivery implements that task only. The user's
[exploratory notes](../../VM_exploration.md) retain their historical status and
are not substituted for audited conclusions.

The selection exercise is a teaching simulation. The research summaries are a
fixed historical record. Neither is a new model benchmark. This resource can
check its own calculations and present evidence honestly; it cannot establish
that learners acquired the concepts without a separate learning evaluation.
