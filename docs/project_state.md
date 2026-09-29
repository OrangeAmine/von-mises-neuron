# Project state: ownership handoff

## Completed integration — N01–N06 (2026-09-29)

All six separately delivered tasks are integrated into this checkout. Original
source trees and 8,250 local result files were preserved with hashes; 153 unique
delivery files were imported, with N04 dependencies included once. See the
[integration record](integration_20260929.md) for provenance and current checks.
Raw results remain ignored local files; compact tracked records travel with a
fresh clone. Owner research notes and earlier frozen experiments are preserved.

The [consolidated research account](research_account_n01_n06.md) is the current
entry point. Completion is distinct from scientific success:

| Task | Completed outcome |
| --- | --- |
| [N01: smooth/local hybrid](vm_n01_results.md) | Overall gate failed: balanced gains but 11.97% smooth regression against the fixed hybrid. Separate forecasting adapter passed standard absolute acceptance. |
| [N02: interpretation](vm_n02_interpretation_results.md) | Claim-specific outcomes, no overall gate. Isolated recovery succeeded; redundant stability and several other claims failed. Reliance is not causality. |
| [N03: teaching](learning/vm_n03_completion.md) | Three lessons complete; integration adds a separate findings explorer and updates both reading companions. No learner-outcome claim. |
| [N04: changing routines](vm_n04_report_v1.md) | Overall gate failed. Four neural adapters failed base-VM regression acceptance and remain unvalidated; residual ridge passed. |
| [N05: uncertainty](vm_n05_report_v1.md) | Overall gate failed: room-level interval crosses zero and noisy coverage fails. Same persistence point forecast for all arms; fixed VM geometry. |
| [N06: decision timing](vm_n06_report_v1.md) | Overall gate failed versus matched Fourier. Source forecasting and objective-dependent decisions; new neural calendar adapters remain diagnostic/unvalidated. |

Open the [playground](../learning/vm_playground/index.html),
[findings explorer](../learning/vm_playground/research.html),
[companion](../learning/vm_playground/guide.html#new-findings), or
[N06 demonstrator](vm_n06_demo/index.html). View changes never revise a gate.

Fresh integration checks passed **1,802 Python tests**, **12 JavaScript checks**,
both pinned evidence exporters, preservation checks and desktop/mobile browser
inspection. See the integration record for scope and receipts; these counts are
not additional scientific replications.

No scientific campaign was rerun during integration. Adaptive trust in calendar
information and reconciliation of N05/N06 targets remain deferred for discussion.
No general VM superiority, field validation, causal interpretation or energy
saving is established. Sections below retain the earlier handoff and historical
test counts.

## Completed follow-up — circular locality v2 (2026-09-21)

The owner requested saving the literature review and starting the proposed
question: **When does adapting circular basis locations and widths improve
prediction under equal label and computation budgets?** The review is saved in
[vm_literature_review.md](vm_literature_review.md), with primary links, scope and
access limitations. The prospective contract is
[vm_locality_v2_protocol.md](vm_locality_v2_protocol.md) plus
[vm_locality_v2.json](../configs/vm_locality_v2.json).

Implementation is complete in separate `locality_*` modules: differentiable
positive-ridge variable projection, independently frozen/trainable centers and
widths, fresh nested task pools with total-label accounting, timed fitting and
source-bound stage controls. The artifact observer independently reconstructs
every checkpoint and performs function-level paired analysis. Focused packages
passed 244 tests (142 numerical/task/trainer/runner, 59 protocol and 43 observer).
Tiny CUDA parity checks passed. Cross-review corrected interrupted-run handling,
progress-I/O cost, finalization timing and evidence-lock semantics before formal
training. The full repository suite passed **1,262 tests in 86.37 seconds**;
all ten actual trainer-to-observer CPU integrations passed and independent
numerical/protocol reviews passed. Scientific sources were frozen before the
224-record serialized CUDA pilot under
`results/vm_locality_v2/study_20260921/pilot`. **All 224 records completed in
273.67 seconds**, and independent replay passed all 4,197 checkpoints. The
prescribed noisy-validation rule selected learning rate **0.1 for all four
adaptive models**. Measured base projections are **1,378.95 seconds** for
development and **3,577.21 seconds** for confirmation; both pass their unchanged
ceilings. **All 1,440 development records completed in 1,247.57 seconds** in the
sibling `development` directory; its artifact audit passed all 19,313 checkpoints.
The rule selected **vp_both versus fixed48**, with fixed16 as the other co-primary
control. Development aggregate improvements were 57.34% and 34.78%, respectively,
but smooth-family error regressed 12.23% versus fixed16, so the overall
development gate failed. Joint Adam had lower aggregate error than vp_both.

The new confirmation lock SHA256 is
`5dfcf1b76b34c9b0eabb5cdce75e3541f3ae237281c4d4c57c17c4d3dda1b498`.
A full read-only verification took 18.31 seconds; charging it for all 32 reserved
target bundles projects total confirmation time **4,163.12 seconds**, below the
unchanged 5,400-second ceiling. All required proofs passed before any reserved
generation. The serialized **3,840-record confirmation phase completed in
3,921.49 seconds**, within the unchanged ceiling, under
`results/vm_locality_v2/study_20260921/confirmation`. All **5,504 pilot/development/
confirmation records completed without failed or retried fits**. Confirmation
replay passed **52,301 checkpoints**; a separate independent aggregation audit
passed **95,830 checks** with no findings. No source, rate, comparator, budget or
gate changed during confirmation.

**The locked overall scientific gate failed.** vp_both reduced balanced
normalized clean error **54.54% against fixed16** (paired 95% interval 43.80% to
68.13%) and **35.85% against fixed48** (18.67% to 55.04%). However, smooth-family
error rose **47.58% against fixed16**, exceeding the unchanged 10% family limit;
the passing fixed48 contrast cannot override this. vp_both error was **15.05%
higher than joint Adam's**, with an improvement interval of -25.35% to -8.84%.
Adaptation helped against the fixed circular banks on narrow/multimodal cells,
but larger Fourier bases performed better at 512 labels. Fixed16 remained well
suited to smooth targets. These are conditional synthetic findings, not general
superiority or a variable-projection advantage.

See [vm_locality_v2_results.md](vm_locality_v2_results.md) for the final report,
[vm_locality_v2_run_record.json](vm_locality_v2_run_record.json) for complete
compact statistics/provenance, [all 720 cells](vm_locality_v2_cells.csv), and the
[confirmation figure](figures/vm_locality_v2_confirmation.png). Exact arrays,
checkpoints and audit scripts remain in ignored local `results/` directories.

The new design separates equal-basis and equal-fitted-parameter contrasts,
crosses noise with target identity, and selects only checkpoints completed
within a shared compute allowance. Pilots determine rates and feasibility;
development selects the VP strategy and parameter-matched comparator; a new
verified lock is required for reserved functions. Stage A follow-ups, Stage C,
probabilistic outputs and forecasting are outside this question.

V1 remains frozen and complete with failed primary gates. No historical source,
checkpoint, canonical v1 configuration or owner note has been changed. A fresh
110-file preservation inventory is in
`results/vm_locality_v2/implementation_20260921/preexisting_files_sha256.json`.
Its post-run verification passed, with only the authorized handoff changed.
The original v1 lock also passed fresh verification of all 13 source files and
876 development dependencies. V2 scientific files still match their
pre-full-suite hashes. No Stage C or forecasting follow-up was launched.

## Completed approved objective — pure VM methodology

### Final audited experiment outcome — 2026-09-08

The approved [methodology plan](vm_methodology_plan.md) and
[canonical JSON](../configs/vm_methodology_v1.json) were carried through
implementation, numerical/engineering validation, disposable pilots, development
and locked confirmation. All **1,452 records completed**: **1,168 gradient fits,
200 B ridge configurations / 800 candidate solves, and 84 A references**. No fit
was retried, omitted or selected for surviving a failure. CUDA experiments ran
serially on the RTX 3060 Laptop GPU using Python 3.11.4 / PyTorch 2.5.1+cu121.
No global installation changes were made. The full suite passed **1,018 tests**.

Both primary continuation gates failed on confirmation:

- **A — polar optimization:** normalized clean-curve area 0.07865080 versus
  Cartesian 0.04826925; polar error **62.94% higher**. The paired 95% bootstrap
  improvement interval was [-68.11%, -57.56%]. All eight paired primary task
  differences favored Cartesian. Weak-signal Adam improved by 33.51%, a secondary
  specificity result; weak-signal polar SGD stalled. Endpoints near the
  closed-form solution do not erase slower adaptation over the full curve.
- **B — learned locality:** learned VM error 0.17861792 versus the locked fixed
  VM ridge comparator 0.10819105; learned VM error **65.09% higher**, interval
  [-98.11%, -33.98%]. Smooth improved 24.33%; narrow and multimodal regressed
  58.28% and 110.22%. All six models, 32/128/512 sample sizes, two initializations
  and six tasks per family remain reported. The nonperiodic control favored
  raw-angle MLP (0.00275971 versus learned VM 0.11239042).

These are bounded synthetic mechanism tests, not novelty, universal superiority
or deployment evidence. A sum-output VM layer is a Cartesian sine/cosine map in
polar coordinates, and peak-normalized VM kernels equal chordal Gaussian RBFs.
Parameter matching does not equalize basis counts or optimization: B has 49
parameters per model, 16 learned kernels versus 48 fixed kernels. Validation
uses 256 observations even at 32 training observations. Task samples are small.
Selected unregularized fixed-bank coefficients reached 7.93e10; exact replay and
backward-error checks do not establish accurate or robust coefficients.

Stage C and optional center/width ablations **were not launched**. The v1 program
is complete; there is no approved automatic extension. Confirmation functions
are now consumed. Further methodological selection requires a new protocol and
unused task seeds. The operational-forecasting request remains superseded.

### Implementation and engineering evidence

New modules provide versioned polar/Cartesian/kernel/calendar layers, explicit
forecaster conversion, seeded tasks, bounded fitting, immutable fit evidence,
source-bound locks and independent equation-based auditing. Historical model
behavior and forecasting protocols were not changed. Numerical checks passed
on CPU float64/float32 and CUDA float32. Factored layers used less allocated GPU
memory but were slower at all three measured GPU shapes; CPU profiles improved.
Fresh unchanged standard gates passed: PolarForecasterV2 RMSE 5.35719 ppm and
DirectCalendarForecasterV1 5.18771 ppm versus base VM 5.35727 ppm. Both absolute
and temporal-regression requirements passed. These are engineering results.

Independent audits covered source/array/state hashes, exact planned matrices,
every retained candidate/validation decision, paired streams, task-balanced
aggregation and 10,000-resample bootstrap intervals with analysis seed 9001.
The original four fixed-ridge pilot audit failures remain saved. The corrected
auditor matches fitting BLAS threads and distinguishes ill-conditioned float64
cancellation from reproducibility, with exact coefficient/prediction replay.
No fit was rerun following the correction.

Pilot selections: A Adam 0.1 / SGD 0.3; B Adam 0.01 for every gradient model.
Development selected fixed VM ridge by the prescribed balanced periodic rule.
The canonical JSON retains its original design labels/nulls intentionally;
actual choices and completion are recorded separately. Lock SHA256:
`077e786046909cd5966732a66bd56c53a715e5bb738173a17782a387e9ed9afd`.
It binds 13 source/config files and 876 evidence dependencies. All confirmation
source bytes match the lock without exemptions; pre-lock observer changes are
documented and hashed separately.

Growing progress manifests exposed avoidable storage overhead. A reviewed
storage-only repair preserved full per-fit evidence and terminal summaries,
using hash-verified references during progress. All original timing estimates,
including over-ceiling estimates, remain saved. Final measured projection was
42.454 minutes, with full-update sensitivity 44.907 against the unchanged
45-minute B ceiling. Actual A/B confirmation took 286.86 / 1,870.76 seconds
(4.78 / 31.18 minutes). No deadline, gate or budget was relaxed.

### Durable handoff and preservation

- [Complete results and limitations](vm_methodology_results.md): all primary
  cells, variability, sample curves, secondary diagnostics and execution ledger.
- [Compact machine-readable record](vm_methodology_run_record.json): metrics,
  source/config hashes, resolved choices, environment and engineering data.
- [Usage guide](vm_methodology_usage.md): verified APIs/CLI and archive rules.
- [Implementation audit](vm_methodology_implementation_audit.md): independent
  numerical, execution and storage reviews.

Full local artifacts are in `results/vm_methodology_v1/implementation_20260908/`
and `results/vm_methodology_v1/study_20260908/`. These directories are ignored;
a fresh clone lacks the lock, exact arrays, checkpoints and full histories.
Preserve those archives for replay. The original working tree remains
uncommitted atop `90dc9f2`. The 85-file preexisting hash inventory found only this
state document changed; owner notes, canonical design/config, existing source
and earlier research documentation remain intact. No historical results were
overwritten and the completed power study was not rerun. Historical evidence
below is retained as the previous project direction.

The final independent ledger/provenance audit passed all six matrices and 876
lock dependencies. Five historical final artifacts, including the selected
RidgeResidual coefficient archive, match pre-program provenance hashes. The
other 25 historical checkpoint/coefficient artifacts have preserved current
hash/readability/mtime evidence but no located baseline digest; independent
pre/post byte identity cannot be proved for those from the available baseline.
This is a stated audit limitation, not an observed modification.

## Completed prior direction — useful real-world forecasting

The owner selected **useful real-world forecasting** after the control phase.
The first practical evaluation is complete: **residual ridge is the selected
reference**, with final RMSE **0.78274 kW versus 0.93386 kW** for the best simple
control (daily seasonal), a **16.18% improvement** on the reserved period.
See [the results](practical_forecasting_results.md),
[frozen protocol](practical_forecasting_protocol.md) and
[data contract](practical_forecasting_data_contract.md).

### Completed practical phase

- Built causal cache-only hourly power preparation: right-edge availability,
  at least 45 valid minute readings per sensor/hour, no imputation, train-only
  normalization, complete 168-hour inputs and 24-hour targets.
- Added persistence-residual wrappers, epoch-zero validation checkpoint
  selection, ridge and seasonal controls, fixed multi-period selection, and
  guarded final evaluation. Saved model coefficients/states and full provenance.
- Ran the unchanged standard synthetic gates. Residual LSTM and SinCos passed
  absolute gates. Residual TemporalVM passed absolute gates but failed the
  existing base-VM regression gates; it was excluded from power trials. The
  validator correctly exited 1 and retained that failure.
- Completed 18 neural fits and three ridge fits across development periods
  beginning 2008-02-01, 2008-08-01 and 2009-02-01. Seeds 42/43/44, maximum 50
  epochs, fixed equal-weight ensembles; no best-seed selection. All fits finite.
  Ridge's pooled RMSE 0.64793 improves 18.57% on weekly seasonal 0.79567.
  LSTM's lower pooled 0.62949 does not qualify: its summer regression exceeds
  the fixed 10% allowance. SinCos fails that same condition.
- Selected ridge before scoring the reserved 2010-10-29 to 2010-11-26 block.
  Refit only ridge, selected alpha 1000 on validation, and scored once against
  all simple controls. Final RMSE passes the prespecified 5% improvement gate.
  Final MAE 0.58158 kW, R2 0.13532, WAPE 48.53%; errors remain substantial.
- Independent audits reproduced all saved metrics, common timestamps/targets,
  ensembles and checkpoint selection; verified source/data hashes and separately
  reconstructed all training normalization statistics from raw rows.
- Added an offline CSV forecast command. Replaying the first final origin using
  saved ridge coefficients/normalization matched all 24 timestamps exactly and
  predictions within 4.44e-16 kW. Inputs must already follow the hourly coverage
  contract; the CSV interface cannot verify minute counts.
- Final full suite: **533 passed in 25.10 seconds**. Development/final commands
  exited 0; runtime 326.55/16.38 seconds. No global installations or raw-data/user
  note changes. Declared the directly used threadpoolctl dependency.

### Next work requires application context

Architecture identity is secondary to operational benefit. Keep residual ridge
as the current reference, with persistence and seasonal forecasts always present.
Do not retry or tune against the now-inspected reserved block and call that fresh
validation. The current frozen protocol and implementation hashes must remain
intact for artifact reproduction; future changes need a new protocol/run ID.

The owner has been asked for the actual quantity to predict, required horizon,
and the decision the forecast supports. That answer determines data access,
appropriate error costs, acceptable missing-data behavior and deployment scope.
Pending that answer, no live integration or architecture expansion is assumed.
Final ridge predictions include 0.976% negative power values, and its MAPE is
worse than daily seasonal. Summer development MAE is worse than persistence.
Physical constraints, uncertainty and a decision-specific loss need explicit
validation on new evidence. One historical household and overlapping forecasts
are insufficient for a production-readiness or general-superiority claim.

Local ignored artifacts: `results/practical_forecasting_2026-09-08/`, including
`development`, `final`, independent audit JSONs, and `inference_example` with
forecast/provenance/parity. Tracked source entry points are
`scripts/practical_forecasting.py`, `scripts/validate_residual_forecasters.py`,
`scripts/audit_practical_results.py`, and `scripts/predict_practical_forecast.py`.
All work remains uncommitted atop `90dc9f2`; the previous handoff follows.

## Latest orchestration handoff — 2026-09-08

The reliability/control phase is complete. Three delegated implementation/review
agents and the orchestrator repaired evaluation, added matched controls, and
completed a full standard benchmark, 54 controlled training fits, and 24
checkpoint-only clock sensitivity evaluations. The original audit below is
preserved as historical evidence; its benchmark-gate and timestamp bugs have
been repaired. All work remains in the working tree atop `90dc9f2`.
`VM_exploration.md` was preserved.

Final verification: `py -3.11 -m pytest tests/ -q` completed with **380 passed in
7.85 seconds**; `git diff --check` passed. The standard benchmark and both
controlled training invocations exited 0 under their documented gate semantics.
Pooling's exploratory failures remain explicit rather than reclassified as
success. The final clock-choice CLI change was covered by the full test suite;
no model/training behavior changed after the recorded fits.

### Implemented and checked

- Standard benchmark now requires finite metrics for every model, unchanged
  absolute thresholds for base VM, TemporalVM and SinCos, and TemporalVM's
  relative error/absolute R2 regression gates. JSON, terminal status and exit
  status agree; nonfinite values are saved as null with failure reasons.
- `src/sincos.py` provides parameter/architecture-matched Cartesian controls.
  Conversion preserves the initial function, other weights, dtype/device and
  RNG state; the standard configuration has exactly 43,148 parameters.
- Custom timestamp periods now share a fixed epoch by default, with an optional
  shared origin. Daily/weekly angles retain local clock semantics; annual
  angles use fractional positions in actual 365/366-day Gregorian years.
  Tests cover split continuity, DST, leap years and boundaries.
- Temporal ablations expose disabled time conditioning and positive learnable
  amplitudes. Defaults preserve old model states, random draws and outputs.
  Amplitude/frequency and path-specific periodicity claims were corrected.
- Controlled runners save source snapshots/hashes, data/split metadata,
  normalization, actual epoch histories, checkpoints and aligned predictions.
  Failure and cancellation reporting retain partial evidence.
- Controlled synthetic runs expose `--calendar-clock local|fixed`. Local
  preserves the original robustness condition; fixed aligns with elapsed-time
  generation. The choice is explicit in manifests; the standard default stays
  unchanged for historical reproduction.
- Real occupancy windows now explicitly reject gaps in the new comparison
  runner. Historical loader/benchmark behavior remains available for provenance.
- A README and linked historical-document caveats identify current guidance.

### Fresh evidence

The full standard benchmark passed, and all five earlier models reproduced
their audited metrics exactly. New SinCos RMSE is 5.4556 ppm in that standard
seed-42 run; see the exact saved/report value rather than rounded comparisons.
All controlled fits used seeds 42/43/44 and maximum 50 epochs, with validation
checkpoint selection. They reset the training RNG after construction, so their
seed-42 values intentionally differ from the historical standard run.

| Condition / model | RMSE mean ± sample SD (ppm) | Interpretation |
| --- | ---: | --- |
| Synthetic original clock, base VM | 5.4362 ± 0.2728 | Core absolute gates pass |
| Synthetic original clock, SinCos | 5.5124 ± 0.1542 | Same 43,148 parameters; seed ordering reverses |
| Synthetic original clock, TemporalVM | 5.3551 ± 0.0280 | All three base-regression gates pass; MAE/MAPE tradeoffs |
| Synthetic original clock, hand-crafted | 5.0610 ± 0.0486 | Best average in this clock condition |
| Synthetic aligned clock, TemporalVM | 4.9888 ± 0.0585 | Same trained checkpoints, no retraining |
| Synthetic aligned clock, SinCosTime | 4.9673 ± 0.0555 | Simpler matched-information control is competitive |
| Synthetic aligned clock, NoConv | 4.8979 ± 0.0935 | Reverses original bridge-removal disadvantage |
| Occupancy, base VM | 734.2275 ± 12.1453 | Far worse than persistence |
| Occupancy, TemporalVM | 735.9881 ± 3.2891 | No useful improvement on this split |
| Occupancy, persistence | 233.4200 (deterministic) | Essential control; still negative R2 |

The synthetic generator uses elapsed hours but its local calendar inputs cross
DST in the test split. Fixing the calendar to UTC+01:00 preserves all training,
validation, sensor and target tensors. All 24 original checkpoint predictions
reproduced exactly before correction. The large original NoConv disadvantage
was therefore partly sensitivity to this clock mismatch: it does **not**
establish that convolution/residual bridges improve aligned forecasting.
Aligned calendar information is useful in this constructed signal, but a unique
phase-conditioning or polar-parameterization advantage remains unproven.
Scale ablations show no clear additional benefit. Pooling fails absolute gates
in seeds 43 and 44 under both clocks; LSTM replacement is not supported.

Occupancy contains 34 dropped hourly bins; 58/213 legacy training windows cross
gaps. The new common contiguous sets have 155/23/23 windows. Every recurrent fit
selected epoch 1, with mean-like forecasts around 620–660 ppm against pooled test
targets averaging 1338.55 ppm. This is observed prediction behavior, not proof
of its cause. All seven trained models lose substantially to persistence.
The 23 overlapping test windows cover only 28 unique future hours.

Jena/power cache audits also found gaps; Jena has duplicate timestamps and
arithmetic averaging of degree-valued wind direction. No large real-data neural
reruns were launched before specifying those preprocessing policies. The cache
uses two Jena forward-filled wind values and zero backward-filled cells; there
is no evidence of actual backfill leakage in this cache.

### Durable evidence and reproduction

- [Controlled protocol](controlled_comparison_plan.md)
- [Full results, paired differences and limitations](controlled_comparison_results.md)
- [Clock mismatch and executed sensitivity](synthetic_clock_audit.md)
- [Occupancy diagnostics](occupancy_diagnostics.md)
- [Jena/power readiness audit](real_data_readiness.md)
- [Repository commands and mathematical scope](../README.md)

Local ignored evidence is under `results/direction_2026-09-08/`: `standard`,
`controlled_synthetic`, `controlled_occupancy`, `synthetic_clock_sensitivity`,
diagnostic JSONs, logs and figures. Fresh source snapshots accompany the training
runs, including the uncommitted implementations. No dependencies were installed
or global Python changes made. Independent reviews checked checkpoints,
predictions, common targets, source hashes and metric recomputation.

These are exploratory results on fixed datasets, not independent confirmation
or novelty evidence. Sample SD is across three training seeds, not across
independent datasets or forecast windows. The clock sensitivity was motivated
by source inspection after the first run; both conditions remain reported.

### Research decision at the earlier handoff (resolved above)

The owner needs to choose the primary objective for the next research phase:
useful real-world forecasting or isolating VM's contribution on physical
circular inputs. The input representation, targets and success criteria differ.
No further architecture expansion is justified by the present evidence alone.

For either path, retain matched SinCos and simple forecast controls, explicit
clock semantics, and complete failure reporting. The immediate forecasting
comparison would freeze fresh dataset realizations/chronological origins for
SinCosTime, TemporalVM, NoConv and NoTime, then address occupancy distribution
shift and validation behavior. The circular path should define actual angular
inputs/targets and matched circular-metric controls before experiments.

## Historical initial audit (preserved below)

Audited 2026-09-08 against commit `90dc9f2` (2026-03-13,
`feat: integrate TemporalVonMisesForecaster into benchmark with regression check`).
Scope: local source, tests, design documents, saved artifacts, and fresh synthetic
validation. This is not a literature review or an assessment of research novelty.

## Assessment

This is a functioning experimental codebase with multiple architectures and
substantial tests. The latest implemented direction is the Temporal VM Encoder:
calendar angles condition sensor-neuron phases, with optional temporal
convolutions and an LSTM backbone. It is already implemented, not merely proposed.
The next stage should establish what this mechanism contributes under controlled
comparisons, rather than assume the design's multi-scale claims are demonstrated.

The initial working tree had only one untracked file, `VM_exploration.md`.
It has been preserved. No model or training code was changed during this audit.
`AGENTS.md` now provides the ongoing project instructions.

## Implemented scope

| Area | Current implementation |
| --- | --- |
| Core neuron | Positive kappa through softplus; learned phases; log/exp outputs; optional external phase shifts; parameter inspection |
| Base forecaster | Per-timestep VM encoder, LayerNorm/dropout, LSTM and forecast head; pooling and other ablations |
| Temporal extension | Raw calendar angles; frozen-kappa bank; phase projections; sensor VM layers; depthwise conv/residual bridges; LSTM or mean pooling |
| Alternative architectures | Deep VM pooling with convolution/residual ablations; VM multi-head attention |
| Comparators | LSTM, Transformer, hand-crafted periodic feature model |
| Data and evaluation | Chronological splitting, train-only normalization, windows, three real-data loaders, synthetic circular domains, circular metrics |
| Artifacts | Benchmark JSONs, ablations, architecture exploration, stress tests, interpretability figures, two PowerPoint decks |

The V2 notes select the temporal encoder approach. The existing VM attention
model has saved experiments, despite older proposal text saying it had not yet
been benchmarked. The proposed explicit phase-time attention extension is not
the current implemented temporal model.

## Fresh verification

- `py -3.11 -m pytest tests/ -q`: **197 passed in 2.53 seconds**.
- Working interpreter: Python 3.11.4, PyTorch 2.5.1+cu121, NumPy 1.26.4,
  pandas 2.3.3, SciPy 1.17.0, scikit-learn 1.8.0, pytest 9.0.2.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, CUDA runtime 12.1.
- Bare `python` points to Python 3.14.6 with no PyTorch. There is no project venv
  in this checkout. No dependency installation was needed for the audit.
- Full benchmark command:
  `py -3.11 -u scripts/benchmark_standard_dataset.py --output-dir results/takeover_2026-09-08/benchmark`.
  Completed with exit code 0. Independently checked finite metrics for every
  model, absolute thresholds for both VM models, and temporal regression limits:
  **all passed**. The base and temporal metrics reproduce the saved run exactly.
- Environment metadata: `results/takeover_2026-09-08/environment.json` (local,
  ignored). Dependencies remain broadly constrained rather than locked.

Fresh run, seed 42, 50 maximum epochs, 90 synthetic days, 72-hour lookback,
12-hour forecast horizon:

| Model | Parameters | RMSE | MAE | R2 | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base VM | 43,148 | 5.3573 | 4.1508 | 0.97896 | 0.6916 |
| TemporalVM | 52,492 | 5.3350 | 4.1897 | 0.97913 | 0.6978 |
| LSTM | 43,668 | 6.1786 | 4.8660 | 0.97201 | 0.8119 |
| Transformer | 44,956 | 6.4793 | 5.1535 | 0.96922 | 0.8605 |
| Hand-crafted | 6,924 | 5.0836 | 3.9104 | 0.98105 | 0.6505 |

Full output: `results/takeover_2026-09-08/benchmark/benchmark_results.json`.
Independent acceptance checks: `results/takeover_2026-09-08/verification.json`.
The hand-crafted comparator remains best on this synthetic signal. TemporalVM's
small RMSE gain accompanies slightly worse MAE/MAPE, so this run supports parity
with base VM rather than a broad improvement. Historical real-data, circular,
ablation, and stress experiments were inspected but not retrained in this audit.

## Historical evidence available locally

These are saved runs, primarily seed 42, and are not multi-seed conclusions.
All paths below are relative to the repository root and ignored by Git.

| Experiment / source | Observation | Interpretation |
| --- | --- | --- |
| `results/benchmark/benchmark_results.json` | Base VM RMSE 5.3573; TemporalVM 5.3350; 43,148 vs 52,492 parameters | Temporal model matches base on this synthetic task; roughly 0.42% RMSE improvement with roughly 22% more parameters |
| `results/presentation_figures/metrics_summary.csv` | Synthetic hand-crafted RMSE 5.0836, VM 5.3573, LSTM 6.1786, Transformer 6.4793 | VM beats these neural comparators in this run, but not the hand-crafted comparator |
| `results/real_benchmark/real_benchmark_results.json` | Jena RMSE: VM 2.4303, LSTM 2.3523, Transformer 2.3007, hand-crafted 2.4230 | No VM advantage on this real-data run; TemporalVM is absent |
| Presentation metrics CSV above | Occupancy R2 is strongly negative for every model (VM -10.8471) | This setup needs investigation before using it to support forecasting claims |
| `results/ablation/ablation_results.json` | Base VM RMSE 5.3573; no LSTM 36.8512; one VM layer 5.1618 | Base architecture depends heavily on temporal modeling; more VM depth is not automatically better |
| Same ablation JSON | Exp-output stacking RMSE 5.1780 and `diverged=false` | Predicted instability in design comments was not observed in this configuration; numerical risks still require testing |
| `results/architecture_exploration/exploration_results.json` | DeepVM-4 RMSE 6.0609; VM-Attn-4L 5.3178 with 135,628 parameters | LSTM-free variants exist and have been tried; results do not establish an efficient replacement |
| `results/circular_domains/circular_domain_results.json` | Wind circular MAE: VM+LSTM 31.58 degrees vs LSTM 33.19; robotics roughly tied | Some promising synthetic angular results; wind, protein, and robotics data are generated by `src/angular_data.py` |
| `results/stress_test/stress_test_results.json` | Random-walk RMSE: VM 96.35 vs LSTM 45.44; trend: VM 551.75 vs LSTM 264.51 | Both models fail badly here; VM is worse. Add persistence/trend controls before attributing the failure solely to periodicity |

Do not pool all saved baseline results as one experiment. For example, the
architecture exploration uses a different LSTM configuration (38,796 parameters,
RMSE 5.3882) from the standard benchmark (43,668 parameters, RMSE 6.1786).
The current real-benchmark JSON contains Jena only; cached occupancy/power data
do not imply complete saved benchmark coverage.

## Findings that affect the research claims

### 1. The log VM layer has a direct sine/cosine equivalent

From the implemented equation:

`kappa*cos(x+mu) = (kappa*cos(mu))*cos(x) - (kappa*sin(mu))*sin(x)`.

Thus each log-output layer is a linear map of sine/cosine input features with
polar-parameterized coefficients. A direct numerical reconstruction differed
by at most `2.98e-7` in a float32 check. This does not make the whole stacked or
phase-conditioned architecture linear. It does make a matched sine/cosine
encoder an essential comparator for isolating the parameterization's benefit.

### 2. The frequency bank does not force frequency specialization

`FrequencyBankVonMises.forward()` returns
`kappa_band * sum_i cos(theta_i + mu_ji) + bias_j`, without an exponential.
Its different kappa values scale response amplitudes; they do not change the
cosine frequencies or sharpen the log-response peaks. The following learned
linear phase projection can absorb those fixed scales. Different periods enter
through the timestamp angles. Claims about forced fast/slow decomposition and
sharp bands tracking daily peaks are unverified and overstated in existing prose.

### 3. Periodicity is a property of particular paths

The core VM layer preserves `2*pi` input shifts. The default temporal encoder
also adds a projected raw-sensor residual, which breaks sensor-shift invariance.
With seed 42, a small eval-mode encoder had maximum output difference `4.5175`
after shifting one sensor by `2*pi`, versus `7.15e-7` after shifting one time
angle. This can be intentional for non-angular sensors; the guarantee must say
which inputs it covers. Affine input projections in other variants also require
care before claiming end-to-end periodicity.

### 4. Temporal benchmark success is incompletely enforced

`scripts/benchmark_standard_dataset.py` checks absolute thresholds only for base
VM. Temporal regression is printed as a warning; `all_passed`, `passed`, and the
exit status do not enforce it. R2 regression is defined as an absolute 0.10
decrease, unlike the relative 10% check on error metrics. Direct comparisons also
need explicit finite-value checks, since NaN can evade inequality-based gates.
The saved temporal metrics themselves satisfy the intended checks, but the
automation does not reliably certify this.

### 5. Timestamp and validation edge cases remain

Custom periods use hours since `index[0]`. Because the temporal pipeline calls
the extractor separately on each split, non-special periods can reset their
phase at validation/test boundaries. A 48-hour-period check confirmed this;
default daily/weekly paths do not use that relative-origin formula. Annual
encoding uses day-of-year/365.25 without fractional hours; leap years, year
boundaries, time zones, and DST need explicit semantics and tests.

The current suite covers the specified shapes, gradients, core periodicity and
normalization bypass, but passing it does not resolve these broader claims.
The temporal design's success criteria for improved LSTM-free performance and
interpretable learned time bands have no corresponding saved temporal ablation
or phase-analysis evidence found in this checkout.

## Recommended next work

1. Repair benchmark acceptance/reporting, including finite metrics and temporal
   regression, with targeted failure-case tests.
2. Clarify the mathematical claims and add a parameter/budget-matched sine/cosine
   encoder baseline. Decide which sensor inputs should actually be periodic.
3. Fix custom timestamp phase origins and test calendar boundary behavior.
4. Run temporal ablations: time conditioning, fixed versus learnable scaling,
   conv on/off, and LSTM versus pooling. Compare across multiple seeds.
5. Extend the temporal evaluation to real data and investigate occupancy failure
   with persistence/seasonal baselines and split-distribution diagnostics.
6. Record reproducible environment and run manifests, add a concise README,
   and reconcile older reports with verified behavior and current evidence.

This ordering preserves the current working implementation while establishing
a reliable basis for deciding whether further architecture work is worthwhile.
