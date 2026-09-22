# VM methodology v1: implementation and evidence

The approved program completed all **1,452 planned records** without failed,
missing or retried fits: 1,168 gradient fits, 200 B ridge configurations
(800 candidate solves) and 84 A diagnostic solves. **Both primary confirmation
gates failed.** Polar A
curve error was 62.94% higher than Cartesian; learned VM B error was 65.09%
higher than the locked fixed-VM comparator. Paired improvement intervals were
entirely negative. These are bounded synthetic findings, not universal claims.

Numerical/gradient parity and unchanged standard forecaster acceptance gates
passed. Factoring reduced GPU memory but slowed the three measured GPU layer
profiles; CPU profiles improved. The full suite passed **1,018 tests**. Versioned
implementations, explicit converters, audited runners, figures and reproducible
records are available; Stage C and optional follow-ups were not launched.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: approved implementation, execution and independent auditing
- Protocol: `docs/vm_methodology_plan.md` + `configs/vm_methodology_v1.json`
- Date: 2026-09-08
- Verification status: engineering, pilots, development and confirmation audited
- Scope: synthetic mechanism tests; no novelty or general-superiority claim

## Current state

The owner authorized the complete approved program. Implementation uses new
modules and explicit conversion APIs. Numerical and task packages were delegated
with exclusive file ownership; the trainer and independent analysis package are
separate. Confirmation remains inaccessible without a source/config/result-bound
lock. Both confirmation studies completed with unchanged locked sources. No
reserved function was generated before the lock was created.

The original canonical configuration is preserved, including unresolved learning
rates and primary comparator. Resolved choices are separate pilot and lock
artifacts, selected by the prescribed rules. No optional Stage C or center/width
follow-up is included in v1.

## Engineering evidence

Environment: Python 3.11.4, PyTorch 2.5.1+cu121, CUDA 12.1, NVIDIA GeForce RTX
3060 Laptop GPU. No packages or global Python installations were changed.

All numerical profile cases passed the approved output/gradient tolerances:
CPU float64 `atol=rtol=1e-10`; CPU/CUDA float32 `atol=1e-5, rtol=1e-4`.
Cases cross sum/exponential outputs, bias, shared dynamic phases and requested
concentrations 0.001/0.4. Extreme exponential concentration 1000 retains its
nonfinite output as a numerical failure regime, without clipping.

The prescribed profiles use one CPU thread, float32 sum response with dynamic
phase, 20 warm-ups and 100 synchronized measured forward/backward iterations.
These are single-machine measurements during local implementation work, not
isolated-hardware or cross-platform performance estimates.

| Shape (batch,time,input,output) | CPU reference/factored median ms | GPU reference/factored median ms | GPU reference/factored peak allocated bytes |
| --- | ---: | ---: | ---: |
| 32,72,8,64 | 30.2146 / 5.9173 | 1.0864 / 3.2055 | 46,615,040 / 23,176,704 |
| 32,72,64,64 | 134.4006 / 7.9132 | 3.3900 / 3.7642 | 207,013,888 / 24,839,680 |
| 8,168,32,128 | 71.1451 / 5.1477 | 2.5506 / 3.2080 | 136,096,256 / 24,569,344 |

The factored implementation saved GPU memory but was slower on GPU at these
shapes. CPU median latency improved. Numerical equivalence does not imply
faster execution. All p90 values, source hashes and environment are preserved
in `results/vm_methodology_v1/implementation_20260908/engineering_profile.json`.

Fresh unchanged standard synthetic validation, seed 42, 90 days, 72-hour inputs,
12-hour outputs, fixed maximum 50 epochs and historical local-calendar condition:

| Model | Parameters | RMSE ppm | MAE ppm | R2 | MAPE % | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh base VM | 43,148 | 5.357273 | 4.150812 | 0.978959 | 0.691626 | Absolute pass |
| PolarForecasterV2 | 43,148 | 5.357191 | 4.150651 | 0.978959 | 0.691601 | Absolute pass |
| DirectCalendarForecasterV1 | 44,620 | 5.187714 | 4.050339 | 0.980270 | 0.674325 | Absolute and temporal regression pass |

The base result reproduces the historical reference; converted models can follow
different floating-point/optimization trajectories. The direct calendar adapter
is a different temporal architecture from base VM. This is repository integration evidence, not a test
of learned localization, general forecasting superiority or novelty.

## Review and preservation

Independent review identified pre-training corrections to interruption evidence,
resumption deadlines, recorded initialization/batch streams and ridge tolerance
ties. B checkpoint selection also separates absolute best validation MSE from
the 1e-6 patience-improvement tolerance. Corrections precede A/B training.

The original working tree is intentional and remains uncommitted atop `90dc9f2`.
Preexisting file hashes and initial git status are saved in the implementation
artifact directory. The owner's `VM_exploration.md`, historical model sources,
power-study protocols/results and cached data are preserved.

## Methodological evidence

The prescribed pilots completed all 80 gradient fits, 8 B ridge configurations
(32 candidate solves) and 4 A diagnostic solves. A invocation runtime was
15.7051 seconds and B 26.4417 seconds. Independent replay selected A Adam 0.1
and SGD 0.3, and B Adam 0.01 for every gradient arm. No rate lacked its full
paired pilot matrix. These are disposable hyperparameter observations.

The original pilot audit failed four fixed-VM ridge records. Diagnosis found
auditor BLAS threads 12 versus fitting threads 1, and rank-deficient unregularized
designs with condition estimates 4.4e16–5.6e16 and coefficients up to 2.94e10.
Changing matrix-reduction order caused cancellation differences up to 1.34e-5.
With matched BLAS1, selected predictions, candidate validation scores and replayed
least-squares coefficients match exactly. Absolute normal-equation gradients
around 1.85e-6–8.78e-6 coexist with normwise backward errors below 6e-17.
The corrected auditor preserves these diagnostics, checks a stated float64
roundoff bound and exact same-design coefficient/prediction replay, and does not
claim accurate coefficients merely from reproducibility. Corrupted-coefficient
tests prevent the scaled check from accepting altered fits. The original failed
audit and both numerical diagnosis artifacts remain intact; no fit was repeated.
Ill-conditioned fixed-kernel results remain a disclosed numerical limitation.

Initial pooled all-rate p90 extrapolation estimated B confirmation at 46.49
minutes, exceeding the unchanged 45-minute ceiling. The full matrix uses the
validation-selected rate for each model, with exactly 144 fits per model. An
independently reviewed, fixed-composition projection retains all four pilot
family timings for each selected arm and estimates 38.63 minutes at the median,
41.20 at p90, including conservative ridge and invocation overhead. The 3.80-minute
margin is estimated from sparse timings; p90 is not a runtime guarantee. Both
estimates remain reported. No rate was chosen using runtime, and no seed, update,
candidate or diagnostic was removed. The hard invocation ceiling still applies.

Only the analysis and orchestration observers changed after the pilots; exact
source differences are recorded in `pilot_selection_audited.json`. Independent
AST review confirmed fit dispatch and execution routines unchanged. Numerical,
model, generator, trainer and canonical plan/config hashes match the originals.
An unused logging optimization candidate was archived without adoption or GPU
benchmarking. The full suite after corrections passed **1,009 tests in 17.13 s**.

A development completed all 64 gradient fits and 16 diagnostic solves in 97.08
seconds. Independent artifact replay passed. Under the declared primary Adam,
zero-penalty comparison, balanced normalized curve area was 0.0752534 for polar
and 0.0454218 for Cartesian: polar was 65.68% worse. The paired improvement
interval was [-68.09%, -63.28%]. Both primary conditions favored Cartesian.
This is development evidence; the protocol still requires reserved confirmation.

B development completed all 192 gradient fits and 48 ridge configurations
(192 solves) in 1,374.53 seconds. All 240 records pass independent auditing,
including 101,300 recorded updates and 5,257 checkpoints. The prescribed
development rule selected fixed VM ridge: balanced periodic error 0.0628917,
versus learned VM 0.122183, geodesic 0.121942, SinCos MLP 0.170490 and Fourier
ridge 0.192868. Learned VM was 94.27% worse than the selected comparator, with
paired improvement interval [-258.14%, -42.25%]. Smooth-family improvement was
16.11%, while narrow and multimodal errors were 225.64% and 3.63% worse.
The nonperiodic control favored the raw-angle MLP (0.00314278 versus learned
VM 0.104934). Development does not establish a confirmation result. The
independent audit replayed all 192 ridge coefficients exactly, while retaining
the numerical limitation of selected unregularized fixed-bank coefficients as
large as 4.83e10. The full suite before locking passed **1,018 tests in 21.91 s**.

The growing development manifest exposed
quadratic serialization overhead from repeatedly duplicating complete histories.
A read-only benchmark on the completed A development manifest measured median
pretty/compact JSON serialization at 355.43/104.92 ms; compact formatting alone
would not adequately bound confirmation overhead. A storage-only repair retains
immutable per-fit histories and uses hash-verified summary references during
progress, restoring full summaries in final manifests. The B development
process retained its original source snapshot throughout. The repair passed 33 runner
and integration tests and 17 independent probes of interruption, resumption,
source differences and lock enforcement. A no-training 720-row simulation with
600-update-sized copied logs reduced full/lightweight median serialization from
9.737 seconds / 176.94 MB to 0.01884 seconds / 0.495 MB. Conservative p90 cost
for 1,440 maximum-size progress writes plus one full final serialization was
41.54 seconds, excluding disk I/O and payload construction. Replaying pilot
selection under this observer version produced the identical rates, with no
training reruns. Both prior selection artifacts remain preserved.
No optimizer, model, task, stopping rule, budget or gate changes with this repair.

A follow-up measured actual atomic writes including payload construction,
serialization and disk I/O. Fifteen 720-row simulated progress writes had
median 17.625 ms and p90 21.239 ms; the representative full final write took
10.349 s. Every roundtrip retained exact values. Conservatively charging
1,441 maximum-size progress writes and one full final write adds 40.954 s.
Adding that entire amount to the pilot projection (double-counting some
already included overhead) gives 41.886 minutes, with 3.114 minutes below the
unchanged 45-minute ceiling. The measurements ran alongside development;
they establish measured feasibility, not a promise about future task runtime.
Artifacts are in `implementation_20260908/atomic_storage_benchmark_20260908_01/`.

The final development timing assessment retains all 192 observed gradient-fit
times. A coarse model-pooled p90 estimate including overhead is 45.638 minutes.
Accounting for the canonical model/family/sample-size frequencies instead uses
48 strata with four observations each, projecting 12 confirmation fits per
stratum. The resulting p90 estimate is 42.454 minutes including ridge/guard and
atomic-storage overhead. Scaling every observed fit to 600 updates before the
same aggregation gives a conservative sensitivity of 44.907 minutes (53 of 192
development fits stopped early). Both calculations retain every observation;
neither is a calibrated probability bound. The hard 45-minute ceiling remains.

Both complete development analyses and the comparator were frozen before
confirmation. Lock SHA256:
`077e786046909cd5966732a66bd56c53a715e5bb738173a17782a387e9ed9afd`.
It binds 13 source/config files and 876 evidence dependencies. The full timing
and independent-audit reports are embedded in the hashed feasibility envelope;
the original manifests, per-fit evidence and pre-lock observer bridge are bound
directly. A confirmation completed first and B followed on the single GPU
worker. Both completed studies are reported below.

## Locked confirmation: A

All 256 gradient fits and 64 diagnostic references completed in 286.86 seconds.
All eight reserved task pairs, four conditions, optimizers and penalties
are retained. The independent artifact replay passed; there were no retries
or failed fits. A separate auditor verified all 51,200 updates, 5,376 saved
curve checkpoints, 640 fit-artifact hashes and 64 task-artifact hashes, with
exact current and archived source equality and no pre-lock exemptions.
The primary Adam/zero-penalty gate **failed**.

Balanced normalized clean-curve area was **0.07865080** for polar
and **0.04826925** for Cartesian: polar error was **62.94% higher**.
The paired 95% bootstrap improvement interval was [-68.11%, -57.56%]
using 10,000 whole-task resamples, seed 9001. Phase-only error was 31.38%
higher and combined-change error 126.90% higher. This bounded adaptation
experiment does not support a polar optimization advantage under its primary
settings. It does not imply different representational capacity.

The table preserves every declared condition/optimizer/penalty cell. Values
are mean normalized curve area ± sample SD across eight tasks; the SD is
descriptive variability, not an independent-seed standard error. Improvement
uses the ratio of arithmetic means. Positive means polar has lower error.

| Condition | Optimizer | Amplitude penalty | Polar mean ± SD | Cartesian mean ± SD | Polar improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| phase_only | Adam | 0 | 0.084924 ± 0.004247 | 0.064639 ± 0.002789 | -31.38% |
| phase_only | Adam | 0.001 | 0.085023 ± 0.004261 | 0.064626 ± 0.002776 | -31.56% |
| phase_only | SGD | 0 | 0.067083 ± 0.002934 | 0.055062 ± 0.001290 | -21.83% |
| phase_only | SGD | 0.001 | 0.067190 ± 0.002965 | 0.055111 ± 0.001305 | -21.92% |
| amplitude_only | Adam | 0 | 0.025166 ± 0.000222 | 0.018032 ± 0.000800 | -39.57% |
| amplitude_only | Adam | 0.001 | 0.025135 ± 0.000214 | 0.017985 ± 0.000804 | -39.76% |
| amplitude_only | SGD | 0 | 0.080034 ± 0.006540 | 0.015435 ± 0.000423 | -418.51% |
| amplitude_only | SGD | 0.001 | 0.080190 ± 0.006562 | 0.015460 ± 0.000429 | -418.69% |
| both | Adam | 0 | 0.072378 ± 0.007362 | 0.031899 ± 0.000382 | -126.90% |
| both | Adam | 0.001 | 0.072422 ± 0.007354 | 0.031878 ± 0.000388 | -127.19% |
| both | SGD | 0 | 0.202584 ± 0.019654 | 0.029484 ± 0.000686 | -587.10% |
| both | SGD | 0.001 | 0.202841 ± 0.019695 | 0.029547 ± 0.000714 | -586.50% |
| weak_phase | Adam | 0 | 0.908147 ± 0.012711 | 1.365749 ± 0.047151 | 33.51% |
| weak_phase | Adam | 0.001 | 0.908276 ± 0.012728 | 1.365315 ± 0.046897 | 33.47% |
| weak_phase | SGD | 0 | 1.992760 ± 0.016938 | 0.055062 ± 0.001290 | -3519.11% |
| weak_phase | SGD | 0.001 | 1.992737 ± 0.016937 | 0.055111 ± 0.001305 | -3515.84% |

Weak-signal Adam behavior is a secondary specificity result. It cannot replace
the primary conditions. Finite but stalled polar SGD weak-signal fits remain
visible in the full curves and recorded updates; completion is not convergence.

## Locked confirmation: B

All 576 gradient fits and 144 ridge configurations (576 candidate solves)
completed in 1,870.76 seconds (31.18 minutes), within the unchanged 45-minute
ceiling. The complete independent replay passed, with zero failed or missing
records. A separate auditor verified 294,180 updates, 15,285 validation
checkpoints, 1,440 fit-artifact hashes and 48 task-artifact hashes. Locked sources
match both current files and archived copies without pre-lock exemptions.

The primary continuation gate **failed**. Learned VM balanced error was
**0.17861792**, versus **0.10819105** for the locked
fixed VM ridge comparator: learned VM error was **65.09% higher**.
The paired 95% bootstrap improvement interval was [-98.11%, -33.98%],
using 10,000 whole-task resamples within each periodic family, seed 9001.
Only smooth tasks improved (24.33%); narrow and multimodal errors were 58.28%
and 110.22% higher. Thus the aggregate-improvement, two-family-improvement,
regression-limit and positive-interval requirements were not met.

Task/family-balanced normalized clean MSE is below. Family entries are mean
± sample SD across six task bundles, after averaging initialization seeds and
sample sizes within each task. The nonperiodic column is not included in the
balanced periodic endpoint. Smaller values are better.

| Model (49 parameters each) | Periodic balanced | Smooth mean ± SD | Narrow mean ± SD | Multimodal mean ± SD | Nonperiodic mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_vm | 0.178618 | 0.002023 ± 0.001508 | 0.435381 ± 0.221517 | 0.098450 ± 0.112753 | 0.112390 ± 0.019816 |
| fixed_vm_ridge | 0.108191 | 0.002673 ± 0.002090 | 0.275068 ± 0.120607 | 0.046832 ± 0.031489 | 0.093066 ± 0.025350 |
| fourier_ridge | 0.187449 | 0.148353 ± 0.043094 | 0.255510 ± 0.112756 | 0.158483 ± 0.021475 | 0.231691 ± 0.063699 |
| learned_geodesic | 0.179095 | 0.002016 ± 0.001557 | 0.437395 ± 0.220268 | 0.097876 ± 0.110775 | 0.113306 ± 0.020090 |
| sincos_mlp | 0.160246 | 0.002273 ± 0.001722 | 0.345741 ± 0.313950 | 0.132724 ± 0.177602 | 0.124555 ± 0.015568 |
| raw_angle_mlp | 0.397231 | 0.034005 ± 0.046802 | 0.772932 ± 0.156421 | 0.384758 ± 0.148887 | 0.002760 ± 0.001466 |

Learned VM's balanced error was 4.71% lower than Fourier ridge and 0.27% lower
than geodesic, but 11.46% higher than SinCos MLP. These secondary comparisons
do not establish superiority to all controls. The raw-angle MLP had
nonperiodic error 0.00275971 versus learned VM 0.11239042, preserving the
expected failure regime of imposing circular structure on nonperiodic data.

Complete sample-efficiency endpoints (arithmetic task means of normalized
clean MSE; all models and all sizes retained):

| Family | Model | 32 training observations | 128 | 512 |
| --- | --- | ---: | ---: | ---: |
| smooth | learned_vm | 0.004078 | 0.001528 | 0.000462 |
| smooth | fixed_vm_ridge | 0.005107 | 0.002130 | 0.000782 |
| smooth | fourier_ridge | 0.430846 | 0.012423 | 0.001791 |
| smooth | learned_geodesic | 0.004124 | 0.001474 | 0.000449 |
| smooth | sincos_mlp | 0.003923 | 0.001796 | 0.001099 |
| smooth | raw_angle_mlp | 0.045019 | 0.025802 | 0.031193 |
| narrow | learned_vm | 0.632987 | 0.358734 | 0.314422 |
| narrow | fixed_vm_ridge | 0.633616 | 0.159653 | 0.031935 |
| narrow | fourier_ridge | 0.711502 | 0.047612 | 0.007418 |
| narrow | learned_geodesic | 0.636491 | 0.360086 | 0.315607 |
| narrow | sincos_mlp | 0.546055 | 0.252567 | 0.238602 |
| narrow | raw_angle_mlp | 0.754619 | 0.822217 | 0.741960 |
| multimodal | learned_vm | 0.172080 | 0.062676 | 0.060594 |
| multimodal | fixed_vm_ridge | 0.100815 | 0.028621 | 0.011060 |
| multimodal | fourier_ridge | 0.455729 | 0.016725 | 0.002994 |
| multimodal | learned_geodesic | 0.170374 | 0.063119 | 0.060135 |
| multimodal | sincos_mlp | 0.226634 | 0.072920 | 0.098619 |
| multimodal | raw_angle_mlp | 0.401098 | 0.366571 | 0.386604 |
| nonperiodic | learned_vm | 0.144910 | 0.095998 | 0.096264 |
| nonperiodic | fixed_vm_ridge | 0.139097 | 0.094005 | 0.046096 |
| nonperiodic | fourier_ridge | 0.598001 | 0.065963 | 0.031110 |
| nonperiodic | learned_geodesic | 0.145425 | 0.097080 | 0.097414 |
| nonperiodic | sincos_mlp | 0.166980 | 0.105783 | 0.100901 |
| nonperiodic | raw_angle_mlp | 0.003118 | 0.002379 | 0.002782 |

At 512 observations, Fourier ridge performs particularly well on narrow and
multimodal tasks. Learned VM's complete curves show its limitations; the
average across sizes is not a claim that one method wins at every size.

## Secondary confirmation diagnostics

These are descriptive summaries, not additional gates. Error summaries average
initializations and sizes within task, then tasks equally. Boundary MSE uses
the outer 10% of the angular domain. MaxAE below is the task-balanced mean of
per-fit maximum absolute errors, not a population worst-case bound.

| Family | Model | Noisy RMSE | MAE | Mean MaxAE | Boundary MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| smooth | learned_vm | 0.132759 | 0.029968 | 0.082952 | 0.001323 |
| smooth | fixed_vm_ridge | 0.134291 | 0.034814 | 0.102533 | 0.001428 |
| smooth | fourier_ridge | 0.319272 | 0.183932 | 0.828334 | 0.207312 |
| smooth | learned_geodesic | 0.132657 | 0.029620 | 0.083768 | 0.001323 |
| smooth | sincos_mlp | 0.134089 | 0.034021 | 0.089301 | 0.001320 |
| smooth | raw_angle_mlp | 0.215827 | 0.110136 | 0.547967 | 0.170369 |
| narrow | learned_vm | 0.644115 | 0.305309 | 3.169995 | 0.072563 |
| narrow | fixed_vm_ridge | 0.468779 | 0.236122 | 2.193680 | 0.073556 |
| narrow | fourier_ridge | 0.407159 | 0.167322 | 2.012787 | 0.048605 |
| narrow | learned_geodesic | 0.646060 | 0.307994 | 3.174615 | 0.072073 |
| narrow | sincos_mlp | 0.522387 | 0.238745 | 2.699823 | 0.036051 |
| narrow | raw_angle_mlp | 0.881946 | 0.443909 | 4.286535 | 0.162896 |
| multimodal | learned_vm | 0.308535 | 0.177541 | 1.099840 | 0.046654 |
| multimodal | fixed_vm_ridge | 0.236707 | 0.128497 | 0.747256 | 0.027276 |
| multimodal | fourier_ridge | 0.332844 | 0.186159 | 1.093219 | 0.125905 |
| multimodal | learned_geodesic | 0.308737 | 0.178147 | 1.097800 | 0.045765 |
| multimodal | sincos_mlp | 0.331542 | 0.193261 | 1.169267 | 0.018999 |
| multimodal | raw_angle_mlp | 0.625037 | 0.440232 | 1.987533 | 0.236195 |
| nonperiodic | learned_vm | 0.362499 | 0.149471 | 2.021713 | 0.955982 |
| nonperiodic | fixed_vm_ridge | 0.330644 | 0.147917 | 2.164389 | 0.763373 |
| nonperiodic | fourier_ridge | 0.430114 | 0.250785 | 2.319497 | 0.821144 |
| nonperiodic | learned_geodesic | 0.363854 | 0.151710 | 2.021007 | 0.962322 |
| nonperiodic | sincos_mlp | 0.377145 | 0.178494 | 1.986822 | 1.028284 |
| nonperiodic | raw_angle_mlp | 0.139979 | 0.037924 | 0.150585 | 0.008715 |

Gradient-fit behavior (144 fits per model; runtime/memory and update counts are
correlated engineering descriptions, not independent experimental samples):

| Model | Median updates | Patience stops | Budget stops | Selected initialization | Mean fit runtime s | Max allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_vm | 600 | 51 | 93 | 1 | 3.195 | 64.829 |
| learned_geodesic | 600 | 52 | 92 | 1 | 3.117 | 64.829 |
| sincos_mlp | 600 | 37 | 107 | 3 | 3.132 | 65.274 |
| raw_angle_mlp | 600 | 37 | 107 | 1 | 3.094 | 64.829 |

Learned geometry below pools 16 kernels per fit across 36 fits in each
model/family group. Kappa min/median/max are descriptive kernel distributions;
kernels are not independent replications. The center resultant is averaged over
fits; it describes within-fit concentration of locations, not causal directions.
The feature threshold is 1e-12, not a claim of floating-point underflow.

| Model | Family | Kappa min / median / max | Mean center resultant | Max grid feature fraction <1e-12 | Mean pre-clip gradient norm |
| --- | --- | ---: | ---: | ---: | ---: |
| learned_vm | smooth | 2.953 / 3.825 / 7.578 | 0.0724 | 0.000000 | 0.0888 |
| learned_vm | narrow | 2.520 / 5.678 / 10.818 | 0.1986 | 0.000000 | 0.1374 |
| learned_vm | multimodal | 0.336 / 6.104 / 10.899 | 0.1779 | 0.000000 | 0.0853 |
| learned_vm | nonperiodic | 2.679 / 4.286 / 8.986 | 0.2636 | 0.000000 | 0.0933 |
| learned_geodesic | smooth | 2.911 / 3.809 / 5.757 | 0.0693 | 0.000870 | 0.0920 |
| learned_geodesic | narrow | 2.534 / 5.679 / 10.877 | 0.1967 | 0.164688 | 0.1372 |
| learned_geodesic | multimodal | 0.436 / 6.109 / 10.977 | 0.1730 | 0.133575 | 0.0852 |
| learned_geodesic | nonperiodic | 2.091 / 4.320 / 9.781 | 0.2609 | 0.084824 | 0.0935 |

A recorded phase-error counts were 5376 defined and 0 undefined.
Final endpoint errors remain distinct from the adaptation-curve primary
endpoint. For the primary Adam/zero-penalty conditions, final normalized
clean MSE averaged across tasks is:

| Condition | Polar | Cartesian | Closed-form diagnostic |
| --- | ---: | ---: | ---: |
| phase_only | 0.00021190 | 0.00021226 | 0.00021225 |
| both | 0.00021177 | 0.00021208 | 0.00021225 |

The full secondary JSON retains every fit/task row, per-parameter gradient
summary, center/width value, phase-null reason, stop reason and diagnostic metric.
Exact per-update gradient norms remain in the original fit histories.

## Complete execution ledger

| Invocation | Gradient fits | B ridge configurations | A references | Runtime s | Ceiling min |
| --- | ---: | ---: | ---: | ---: | ---: |
| A pilot | 48 | 0 | 4 | 15.705 | 10 |
| B pilot | 32 | 8 | 0 | 26.442 | 10 |
| A development | 64 | 0 | 16 | 97.076 | 15 |
| B development | 192 | 48 | 0 | 1374.527 | 45 |
| A confirmation | 256 | 0 | 64 | 286.857 | 15 |
| B confirmation | 576 | 144 | 0 | 1870.759 | 45 |

Total: **1,168 gradient fits, 200 B ridge configurations / 800 candidate solves,
84 A diagnostic solves; 1,452 complete records**. No fit was retried, omitted
or selected for surviving a failure. Numerical-audit and timing-estimate
failures described above are retained. Actual B confirmation runtime was lower
than the conservative measured projections; no budget or gate was changed.

## Interpretation and limits

The approved program is complete. Both primary continuation gates failed.
A does not support a polar-coordinate adaptation advantage under the tested
optimizer protocol. B does not support learned VM locality over the locked
fixed-kernel comparator. A successful fixed VM bank supports that control's
utility here; it does not rescue the failed learned-locality claim. Stage C
and optional center/width follow-ups were not launched.

The sum-output polar layer remains exactly a sine/cosine map with polar
coefficient coordinates. Matched amplitude regularization was used; historical
phase/raw-concentration decay was only diagnosed. Peak-normalized VM kernels
are exactly Gaussian RBFs on the unit-circle embedding, so these results provide
no evidence of a new kernel family. Numerical parity, runtime behavior and
methodological evidence are distinct findings.

The small designed samples (eight A task bundles; six B tasks per family)
and their bootstrap intervals are not a powered population study. All B
models have 49 trainable parameters, but learned banks have 16 kernels and
the fixed bank has 48; parameter matching does not make their basis counts or
optimization procedures identical. Results depend on the prescribed rates,
validation rules and bounded training budgets. No confirmation-driven retuning
was performed. The 32/128/512 curves also use 256 validation observations each
and therefore are not total-label-budget comparisons. Noise parity is confounded
with task identity; subgroup results are descriptive.

Selected unregularized fixed-bank coefficients reached 7.93e10 in confirmation.
All 576 candidate coefficient replays matched exactly; worst normwise backward
error was at most 5.74e-15. This establishes replay consistency and the stated
roundoff checks, not accurate or robust coefficients in ill-conditioned systems.
The original failed audit, numerical diagnoses and corrected audit remain saved.

## Reproducible artifacts

See the [usage guide](vm_methodology_usage.md) for verified command forms and
[compact machine-readable run record](vm_methodology_run_record.json) for source
hashes, configuration reference, resolved rates/comparator, all primary cells,
task variability, complete sample curves, bootstrap outputs and engineering data.
The canonical JSON intentionally retains its original design-status fields;
actual completion and resolved choices live in the locked evidence and report.

Seed roots: pilot task/observation/initialization 101/201/301; development
1101–1102 / 2101–2102 / 3101–3102; confirmation A 5101–5108 / 6101–6108
and B the first six paired roots with initializations 7101/7102. Analysis seed
is 9001. Dedicated SeedSequence components preserve stage/family/replicate/split
separation and paired initialization/batch streams. A has three parameters;
every B model has 49. Domains are radians on [-pi, pi), with a 4,096-midpoint
clean evaluation grid. B targets use generator-defined standardization.

Figures (local ignored artifacts, visually inspected):

- [A confirmation learning curves](../results/vm_methodology_v1/study_20260908/confirmation_figures/A_learning_curves.png)
- [B confirmation sample-efficiency curves](../results/vm_methodology_v1/study_20260908/confirmation_figures/B_sample_efficiency.png)
- [Independent confirmation analysis](../results/vm_methodology_v1/study_20260908/confirmation_analysis.json)
- [Full secondary diagnostics](../results/vm_methodology_v1/study_20260908/confirmation_secondary.json)

Full initial/final/selected states, arrays, per-update histories, candidates,
snapshots and audit reports stay under the unique ignored results directories.
A fresh clone lacks those files; the compact record is not a substitute for
checkpoint archives. Confirmation functions are now consumed. Any subsequent
methodological selection requires a new protocol and unused task seeds.

## Final independent preservation audit

The final program audit passed the six canonical matrices, all 1,452 records,
deadlines, serialized workflow, source bindings and all 876 lock dependencies.
The initial inventory covered 85 preexisting files; only the requested
`docs/project_state.md` handoff changed. `VM_exploration.md`, canonical plan/config,
existing source and earlier research documents retain their original hashes.
No historical result directory was written by this program.

A pre-program inference provenance file binds five historical final artifacts,
including the selected RidgeResidual coefficient archive; all five hashes still
match. For the other 25 historical checkpoint/coefficient artifacts, the audit
records current hashes, readability/finiteness and pre-program modification times,
but found no prior saved digest. Independent pre/post byte identity therefore
cannot be established for those 25 from the available baseline evidence.
The absence of a baseline digest is an audit limitation, not an observed change.

Independent final audit and historical supplement are saved as
`implementation_20260908/independent_final_program_audit.json` and
`implementation_20260908/independent_historical_preservation_supplement.json`
under the methodology results root. The working tree remains uncommitted;
intentional earlier changes were preserved.
