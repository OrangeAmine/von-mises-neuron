# Proposed tasks informed by the forecasting research workspace

Created: 2026-09-22. Updated: 2026-09-29.
Status: **VM-N01 through VM-N06 complete and integrated into the main checkout**.
See the [research account](research_account_n01_n06.md) and
[integration record](integration_20260929.md). N01, N04, N05 and N06 retain failed
overall scientific gates; N02 has claim-specific outcomes; N03 is teaching
implementation. Historical dispatch records below are preserved. The calendar-
trust and uncertainty/decision-contract follow-ups have not been launched.

## Task dispatch — 22 September

The user requested new external tasks for N01 and N03, then N02 and N04. All were submitted as
separate Codex tasks in managed worktrees of the Von Mises Neuron project:

- N01: **VM-N01 — Smooth backbone with local VM corrections**;
  setup identifier `client-new-thread:4992305a-0de2-4449-8078-dc21b01a423e`.
- N03: **VM-N03 — Von Mises learning playground and companion**;
  setup identifier `client-new-thread:dadb1ddc-4928-4938-8b00-a0579506d5ec`.
- N02: **VM-N02 — Test learned phase and width interpretations**;
  setup identifier `client-new-thread:3eb4e2b1-ce72-4ece-a8d5-c3d0bf7b9644`.
- N04: **VM-N04 — Circular forecasting under changing routines**;
  setup identifier `client-new-thread:64a464b3-a569-4845-b89e-8028dfd06840`.

Each task receives the same verified 130-file working-source snapshot under
`results/task_launches/20260922T112135Z/`, with archive SHA256
`18191869b3383ee437e33de713b4aaed6a7f103e9ae88c5581ce8ea8a44f31b5`.
This includes the uncommitted research implementation and compact documentation
absent from the default branch. The prompts require contained extraction into
each task's own worktree and preservation of the original checkout. The archive
excludes raw data, historical result payloads, and Git internals.

N01 is authorized to implement and run the bounded prospective comparison with
the required model checks and honest reporting. N03 is authorized to implement
and verify the three teaching lessons and offline guide without research
campaigns. Their own tasks now own execution progress and outcomes.

N02 is authorized to implement and execute the bounded known-mechanism
interpretation study, including prospective comparison rules, component and
phase controls, parameter/response stability, paired uncertainty, and a candid
claim scorecard. The snapshot and all 130 entries were reverified before its
dispatch; only this proposal's dispatch record differed in the source checkout.
The research implementation is now committed at
`a7431a926c751868b11a3918c6cdd29be6540a32`. No N01/N03 result was assumed or
required to launch N02.

N04 is authorized to implement and execute the bounded synthetic forecasting
comparison: first-order dynamics, changed routines, one sensor-outage condition,
matched calendar branches and simple references, causal availability checks,
required model gates and trajectory-level paired reporting. The same archive
and 130 entries were verified before dispatch; only the dispatch document
differed from current source. Real-data replication, uncertainty calibration
and control experiments are outside this first delivery. No result from the
other new tasks is assumed or required.

## Task dispatch — 26 September

The user requested separate external tasks for N05 and N06:

- N05: **VM-N05 — Phase-conditioned uncertainty with delayed feedback**;
  setup identifier `client-new-thread:a7e21210-20ca-4153-beae-56cbd8221d19`.
- N06: **VM-N06 — Forecast timing and simulated decisions**;
  setup identifier `client-new-thread:3c725692-83dc-47c8-91f2-d1f92d64f3f1`.

N04's completed implementation was located in the managed worktree
`C:\Users\Amine\.codex\worktrees\9c11\von-mises-neuron`, on commit
`843bf96f289d6217b038ecbc2355eef4bf8b5e87`. Its report and handoff were read:
the scientific gate failed, its four neural adapters remain unvalidated after
base-VM regression failures, and residual ridge passed both engineering gates.
These are inherited reported outcomes, not a new replay or reclassification.

Both new task prompts receive a verified 156-file dependency snapshot of that
committed tree plus this original checkout's proposal and owner notes:
`results/task_launches/20260926T182459Z/vm_n04_dependency_snapshot.zip`.
Archive SHA256:
`acc928bcc98bd9f7e38fcf496a26da757247344fd77fa3bc312064a90e5e31d7`.
The sibling manifest records every file's hash and origin. Historical arrays
and checkpoints are excluded and may be read only from N04's named artifact
location after verifying the relevant identities.

N05 is authorized to implement and execute the bounded delayed-feedback
calibration comparison, retaining fixed point forecasts, fresh evaluation
trajectories and the original models' eligibility distinctions. N06 is
authorized to implement and evaluate the synthetic dynamic decision
demonstrator with matched policy constraints and explicit forecast/action
semantics. N06 can proceed with point forecasts; importing N05 intervals is
optional and must not require waiting or inventing calibration outcomes.
Neither task reopens N04's consumed confirmation data as fresh evidence or
authorizes field deployment. Their own tasks own progress and completion.

## Recommendation and evidence

Prioritize one architectural question, one interpretation question, and a small
learning resource. Keep forecasting, uncertainty, and decision experiments as
subsequent application branches. The central methodological question remains:
**what does adaptive circular locality contribute beyond appropriate fixed and
Fourier features, and under which conditions?**

The current [locality v2 report](vm_locality_v2_results.md) supports a conditional
answer. Adaptive centers and widths reduced balanced error by 54.54% against
fixed16 and 35.85% against fixed48, but increased smooth-family error by 47.58%
against fixed16. The overall gate failed. Joint Adam outperformed variable
projection, and larger Fourier bases were stronger in well-sampled cells.
These findings motivate a smooth-plus-local model, not another unrestricted
search for a generally superior VM neuron.

The other project's [controlled occupancy study][occupancy] demonstrates why
calendar information must be evaluated under changes in attendance. Its
[interpretability lessons][interpretation] separate predictive reliance from
physical meaning. Its [forecasting lessons][forecasting] show why uncertainty
must use errors available at forecast issuance and why overall coverage can
conceal failures during rises. The [playground roadmap][playgrounds] supplies a
useful teaching pattern: predict, manipulate, observe, explain.

These are reported findings and teaching methods inspected for this proposal;
their underlying experimental artifacts were not independently replayed here.

| ID | Proposed task | First deliverable | Priority/dependency |
|---|---|---|---|
| VM-N01 | Preserve smooth structure while learning local circular corrections | Bounded global-plus-local basis comparison | First scientific task |
| VM-N02 | Test what learned phases and widths actually explain | Functional interpretation scorecard | Can start with existing evidence; fresh data for new claims |
| VM-N03 | Build a VM learning playground and companion | Three interactive lessons and an evidence guide | First educational task; independent of new training |
| VM-N04 | Test circular features under dynamics and changing routines | Controlled time-series robustness benchmark | Application branch; can use existing bases before N01 finishes |
| VM-N05 | Calibrate uncertainty using phase and recent conditions | Delayed residual-calibration comparison | Needs a declared forecast task and predictions, e.g. N04 |
| VM-N06 | Test whether forecast timing improves a simulated decision | Small dynamic decision demonstrator | Later; N04, plus N05 only if intervals are used |

## VM-N01 — Smooth backbone with localized circular corrections

**Question:** can a low-order Fourier backbone retain smooth-target performance
while a learned circular residual captures narrow or multimodal structure?

**Why now:** this addresses the specific v2 failure. The forecasting workspace's
[correction-prediction lesson][forecasting] motivates separating a strong
reference from the correction being learned. Its [selection-bias lab][selection]
motivates a small declared candidate set and fresh evaluation.

**First delivery:** implement a versioned scalar-angle model of the form

\[
f(\theta)=b+\sum_{m=1}^{M}[a_m\cos(m\theta)+d_m\sin(m\theta)]
+\sum_j w_j\exp\{\kappa_j[\cos(\theta-c_j)-1]\}.
\]

The Fourier component represents broad structure; the kernel component is a
candidate local correction. This decomposition is a hypothesis, not a guarantee
that fitting will assign those roles or identify individual components uniquely.

Compare Fourier-only, fixed-kernel-only, adaptive-kernel-only, Fourier plus fixed
kernels, and Fourier plus adaptive kernels. Include joint Adam as a serious
adaptive comparator because v2 favored it; do not default to variable projection
as the presumed improvement. Reuse existing layer/solver APIs through new modules
without changing frozen implementations.

Use new smooth, narrow, multimodal, and mixed smooth-plus-local functions. Add a
nonperiodic diagnostic with an unrestricted comparator. Separate equal-basis and
equal-fitted-parameter contrasts. Count validation and tuning labels, declare
regularization/feature scaling, bound candidate searches, and report actual cost
alongside common compute allowances. An equal coefficient penalty alone does not
establish equal function-space regularization.

**Done when:** a prospective protocol, implementation, required tests, and a
complete result table answer whether the correction improves the declared
comparison without unacceptable smooth-family regression. Freeze a practical
effect threshold and family guard before reserved access. A failed benefit gate
is a completed scientific task. New seeds from familiar generators provide
within-generator confirmation, not unseen-domain validation.

## VM-N02 — Falsifiable phase and width interpretation

**Question:** do learned parameters describe stable predictive structure, or are
similar predictions compatible with different parameter pictures?

**Connection:** the other project's [Chapter 6 companion][interpretation] uses
gate freezing, attribution stability, and a predictive proxy to distinguish
model reliance from a physical mechanism. Translate that logic to circular
layers rather than treating a phase histogram as an explanation.

**First delivery:** construct three small known-mechanism cases: one identifiable
isolated peak, an overcomplete redundant basis, and a calendar-correlated proxy
whose association changes at evaluation. Report prediction error, response-curve
stability, and intervention effects separately.

- Compare complete fitted response curves across initializations before matching
  individual kernels. Kernel permutations, overlap, and readout compensation can
  make center/width comparisons misleading.
- For the original `cos(theta + mu)` convention, the preferred input is `-mu`.
  Distinguish this from the explicit center `c` in `cos(theta - c)`. Do not assign
  a meaningful phase to negligible-amplitude components.
- Ablate a complete component contribution or replace a time-varying modulation
  with a validation-derived reference; compare with declared phase-offset
  controls. Constant learned phases cannot be meaningfully “time frozen.”
- Keep evaluation-time interventions separate from retraining without the
  component. Describe whether each perturbation changes input support.
- Check any attribution display against parameter randomization and an exact
  equivalent representation where available. Passing these checks supports a
  bounded interpretation, not causality.

**Done when:** a scorecard states which phase/width claims survive the controls,
with paired uncertainty at the function or trajectory level. Existing artifacts
can supply exploratory diagnostics; prospective interpretation claims need new
held-out cases. The current `scripts/interpretability_analysis.py` concerns
preferred standardized sensor values, so it is not already a calendar-phase
interpretability study.

Methodological background: [Adebayo et al., Sanity Checks for Saliency Maps][saliency]
supports testing explanations beyond visual plausibility; its image experiments
do not themselves validate a circular-layer explanation.

## VM-N03 — VM learning playground and reading companion

**Question:** can the learner explain when a VM layer is equivalent to a simpler
representation, when localization changes the model, and why a successful average
does not settle a scientific claim?

**First delivery:** three deterministic, resettable lessons following the
[forecasting playground pattern][playgrounds], with a short offline guide:

1. **Same function, different coordinates.** Manipulate amplitude and phase and
   display the exactly equivalent sine/cosine coefficients and response. Include
   the preferred-angle sign, the near-zero amplitude case, and wrap consistency.
2. **Amplitude versus locality.** Contrast the log/sum response with the
   peak-normalized exponential kernel. Show that kappa scales amplitude in the
   former and changes concentration in the latter; periods come from the input
   angles or integer harmonics. Demonstrate the chordal Gaussian identity.
3. **A gain with a failed gate.** Explore the saved v2 family/label/compute
   summaries and a separate small synthetic exercise about selection bias.
   Let the learner predict how an aggregate improvement can coexist with a
   smooth-family failure. Every historical number links to its report.

Add an evidence guide with four explicit labels: exact identity, engineering
validation, empirical finding, and open hypothesis. Include worked equations,
self-checks, and links to the v1/v2 negative findings. Historical exploratory
notes remain intact and are distinguished from audited conclusions.

**Done when:** lessons work without private data or large training runs, reset
reproducibly, show units and assumptions, and pass focused numerical/UI checks.
Use a small local interface; a shared framework or full Forecast Lab port is not
a prerequisite. Teaching examples remain separate from new research evidence.

## VM-N04 — Circular forecasting under dynamics and routine changes

**Question:** does the observed benefit on localized scalar functions survive
physical smoothing, sensor history, and a changed relation between clock and
signal?

**Connection:** combine the [controlled occupancy study][occupancy] with the
[first-order-system explainer][physics]. A sharp attendance pattern need not
produce an equally sharp concentration trace: the dynamics integrate its effect.
This is the proposed bridge from static circular approximation to forecasting.

**First delivery:** a controlled time-series generator with separately specified
periodic forcing, first-order response time, observation noise, and availability.
Vary response time relative to event duration, arrival-phase shifts, irregular
attendance, and one bounded sensor-outage condition. Separate routine shifts
from timestamp/clock defects. Use new trajectories and train/test shift rules.

Hold the sensor backbone and training budget fixed while varying the calendar
branch: none, Fourier, fixed circular kernels, and learned circular kernels.
Include sensor-only residual ridge, persistence, and a training-derived seasonal
reference. Every calendar-enabled arm gets the same known future clock inputs;
no arm receives realized future attendance or sensors. A known-law oracle, if
shown, is a separately labeled diagnostic.

Report errors by horizon and regime, boundary behavior, and actual runtime.
Use independent generated trajectories as replication units; overlapping windows
are not independent experiments. Incorporate the [sensor-timing lab][timing]:
event time, arrival time, bin completion, contiguous windows, and mutation of
observations unavailable at the origin.

**Done when:** the benchmark identifies the conditions under which a circular
branch adds value to matched-information references. It need not show a win.
Only then consider one real-data replication with an explicit new data/split
contract. Do not reuse the consumed power final period as a fresh test. The
forecasting workspace's occupancy-probability result is motivation, not evidence
of improved CO2 or power forecasting here.

## VM-N05 — Phase-conditioned uncertainty with delayed feedback

**Question:** can circular features improve the timing of uncertainty estimates,
even if they do not improve the point forecast?

**Connection:** [Chapter 5][forecasting] reports that overall coverage concealed
undercoverage during rises, and that repairing coverage did not automatically
make the predictor preferable to persistence.

**First delivery:** keep point predictions fixed, initially using the strongest
simple reference. Compare global static, delayed rolling, fixed Fourier/hour-bin,
and VM-conditioned residual-scale calibration. Give all conditional methods the
same causal recent-change inputs and equal tuning opportunities, so a benefit
cannot be attributed merely to extra information.

Separate scale fitting from calibration and evaluation. At issuance time `t`,
an error from origin `u` and horizon `h` is eligible only after its target and
measurement have become available. Known target calendar phase is allowed;
future target-derived regime labels are not predictors. Bound widths and define
sparse-phase fallback behavior before evaluation.

**Done when:** an aligned comparison reports interval score, width, and empirical
coverage by horizon, phase, and prespecified origin-known conditions, with paired
block/trajectory uncertainty. Improved nominal coverage alone is insufficient.
Do not claim distribution-free conditional coverage for this proposed method.

[Gibbs and Candès' adaptive conformal work][aci] is relevant background for online
coverage under shift. Its algorithm and long-run guarantee are distinct from
ordinary rolling residual quantiles and do not automatically transfer to this
phase-conditioned, delayed-feedback proposal.

## VM-N06 — Forecast timing and a simulated decision

**Question:** does a forecast's useful lead time change a controllable outcome
under a declared objective?

**First delivery:** extend the N04 first-order process into a small synthetic
ventilation teaching scenario. Compare a current-observation reactive policy,
a seasonal-reference anticipatory policy, and a VM-informed anticipatory policy
with identical plant, actuator limits, delay, and tuning budget. Include an
explicit perfect-information oracle as a bound.

Follow the [Decision Lab proposal][playgrounds]: state evolves after actions;
distinguish no-action forecasts from action-conditioned predictions; keep
forecast accuracy and policy outcomes separate. Report exceedance duration and
ventilation effort, retaining unsuccessful matches on the trade-off curve.
Airflow effort is not measured energy saving. Do not feed action-modified truth
back into a comparison as though all policies encountered an unchanged target.

**Done when:** the demonstrator explains when phase accuracy, uncertainty, or
lead time changes the simulated decision, including cases where it cannot help.
This is a later educational/application branch, not field validation. No live
stream, calibrated building, or deployment should be assumed.

## Sequence and shared completion rules

Start with **N01 and N03**: a focused scientific extension and a companion that
makes the existing evidence understandable. N02 can begin with bounded
exploratory use of existing outputs and accompany N01's new cases. N04 has now
been explicitly adopted in the dispatch record above. N05 and N06 were
subsequently adopted on 26 September. These application extensions have their
own bounded contracts and do not revise the completed pure-methodology program.

For adopted experiments, write the smallest protocol that fixes the question,
comparison, practical effect, information/label budget, selection rule, compute
cap, and new held-out pool. Preserve all outcomes and report variability across
appropriate independent units. Do not add an architecture merely because a
reference project supports it.

Apply this repository's required tests and unchanged standard synthetic model
gates. Model/pipeline changes require the full suite; applicable temporal models
also retain the base-VM regression checks. A static-basis result does not itself
validate a forecasting adapter. Documentation and teaching-only work require
only relevant checks. Preserve the frozen v1/v2 source and evidence identities,
historical power results, and `VM_exploration.md`.

## Source-context receipt

- Receiving project: `D:\Desktop\Claude\von-mises-neuron`; inspected HEAD
  `90dc9f252f3d5eaf8dc06877ebde33ac9623f0e1`, with existing working-tree changes.
- User-supplied context: `D:\Desktop\Claude\Hybrid_models_forecasting`.
- Used its [research-workspace-context skill][context-skill], task class
  `direction`, then the named learning/study owners. The validator succeeded
  using `py -3.11`; context HEAD was
  `5f48fadda23f3132ecece500370b0c124e408ef4`, checkout state `dirty`.
- Context manifest SHA256:
  `49aed2956b668ffdbf00c531a4f4967515724ebd9747132b1618eb934f0af337`.
  This validates navigation, not scientific artifact contents. HEAD alone does
  not identify all inspected working-tree bytes.
- Owners selected: the synthetic-occupancy study; current learning companions
  and 22 September learning refresh; playground roadmap; this project's state,
  methodology direction, controlled comparisons, and locality v2 report.
  `ev-p6-pilot` / Paper 6 interpretation routing was consulted only as background
  for intervention/statistical design, not as a prospective completed result.
- No raw data, optional-local experiment payloads, or generated thesis binaries
  were required. Their availability/integrity was not established. Required
  navigation sources were available; no missing resource was substituted.
- The sibling project's frozen papers, experiments, and thesis were not changed.
  New proposals here do not inherit authorization to resume those workstreams.
- Primary web pages below were checked on 2026-09-22 for the two methodological
  connections; this was not a systematic literature or novelty review.

[occupancy]: ../../Hybrid_models_forecasting/docs/research/synthetic_occupancy_2026-09-16/milestone_6.md
[interpretation]: ../../Hybrid_models_forecasting/docs/learning/chapter6_reading_companion.md
[forecasting]: ../../Hybrid_models_forecasting/docs/learning/chapter5_reading_companion.md
[playgrounds]: ../../Hybrid_models_forecasting/docs/learning/playgrounds/roadmap.md
[selection]: ../../Hybrid_models_forecasting/docs/learning/labs/model_selection.md
[physics]: ../../Hybrid_models_forecasting/docs/learning/explainers/04a-first-order-systems.md
[timing]: ../../Hybrid_models_forecasting/docs/learning/labs/sensor_timing.md
[context-skill]: ../../Hybrid_models_forecasting/.agents/skills/research-workspace-context/SKILL.md
[saliency]: https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
[aci]: https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
