# VM methodology implementation audit — 2026-09-08

The reviewed implementation is ready for **disposable pilots**, subject to the
orchestrator's final full-suite check and source freeze. This is an engineering
finding; no stage A/B methodological benefit has been established. Reserved
confirmation functions were not generated or scored during this review.

The reviewer implemented the numerical layers/adapters, but did not implement
the methodology trainer, orchestration runner, or independent analysis package,
and did not execute the engineering training fits. The artifact audit uses
separately written metric equations and gate checks rather than importing the
validator's metric/acceptance helpers. This does not represent an external
review of the research program.

## Engineering artifact replay

Artifacts are under
`results/vm_methodology_v1/implementation_20260908/forecaster_validation/`.
The independent audit recomputed every saved RMSE, MAE, R2 and MAPE exactly using
CPU float32 arithmetic. A separate float64 accumulation changed these reported
metrics by at most **1.27e-6** and retained every acceptance decision.

| Model | Parameters | RMSE (ppm) | MAE (ppm) | R2 | MAPE (%) | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh base VM | 43,148 | 5.357273 | 4.150812 | 0.978959 | 0.691626 | Absolute pass |
| PolarForecasterV2 | 43,148 | 5.357191 | 4.150651 | 0.978959 | 0.691601 | Absolute pass |
| DirectCalendarForecasterV1 | 44,620 | 5.187714 | 4.050339 | 0.980270 | 0.674325 | Absolute and base-regression pass |

All three used seed 42, the unchanged standard trainer, the existing local
calendar condition, and the fixed 50-epoch ceiling. Their saved validation
histories independently select epoch 35 under the historical stopping rule.
Recorded fit times were 14.25, 10.94 and 16.38 seconds respectively. These are
single integration runs, not comparative methodological evidence or a reliable
end-to-end speed benchmark. Direct calendar conversion changes capacity and
optimization geometry.

The saved target arrays agree exactly across all models, each with shape
`(133, 12)`. Every initial and selected state is finite and has the recorded
scalar count. All **77** archived file hashes match the source ZIP. Twelve
directly relevant model, pipeline, validation and configuration files match
their training snapshots. Among **85** files inventoried before implementation,
only `docs/project_state.md` had changed at this audit; the owner notes and
historical sources/protocols were intact.

The numerical profile records **48** passing CPU/CUDA output/gradient parity
cases. This audit checked record consistency, without repeating those profiles
or asserting that parity establishes speed or research benefit.

## Independent implementation review

The A objective, native decay exclusions, three-scalar capacity, full-batch
updates, shared learning rate, diagnostic least-squares endpoint and normalized
trapezoidal curves match the canonical contract. B has the declared 49-scalar
models, shared seeded cyclic batches, signed readouts, explicit readout penalty,
gradient clipping, and validation-only checkpoint selection. The checkpoint is
the absolute validation minimum; a separate significant-improvement anchor
implements patience with `min_delta=1e-6`.

The review found and the responsible owners fixed:

- Interruption handling now returns partial trainer states/history so the
  runner archives them before propagating interruption.
- An original absolute invocation deadline survives resumption and counts
  downtime. Interrupted or failed fits are not retried.
- B initialization metadata, original initialization arrays and the derived
  batch seed are retained. NumPy arrays are converted to tensor artifacts so
  `weights_only=True` checkpoint loading works.
- Ridge selects the smallest lambda within tolerance of the global minimum;
  an intermediate near-tie cannot be skipped by sequential selection.
- The independent analyzer's checkpoint replay matches the separate minimum
  and patience rules. Polar model metadata records its actual version.
- Feasibility now includes measured closed-form p90 costs and invocation
  overhead, with a conservative largest-sample-size multiplier for B ridge.
  Development and lock creation require matching pilot/configuration evidence;
  the lock requires complete audits of both A and B development studies.

Ten additional no-training probes passed: interrupted evidence preservation,
immutable resume deadline, no retries, complete retained timeout matrix,
incomplete-study status, safe initialization serialization, pilot dependency
binding, changed pilot evidence rejection, lock-before-generation ordering,
and validation/ridge selection rule replay (related checks share probe rows).
The independent analyzer also enumerates the complete matrix and refuses
success for missing, duplicate, unplanned, failed or unaudited cells. Pilot
rates need complete finite paired cells; confirmation checks bind source,
configuration, generator, development evidence, rates, comparator and analysis
rules before task generation.

## Remaining limits and evidence locations

Pilot timings still need to establish feasibility. Full-stage estimates now
include measured ridge/reference overhead in addition to gradient fits; these
estimates must pass before larger runs. An abrupt process kill can
prevent the current fit's in-memory state from being saved; its registered
running cell remains incomplete and must not be silently retried or excluded.
Hash checks are procedural integrity controls, not cryptographic secrecy of the
known seed registry. Tests and finite states cannot guarantee all future fits
will finish or remain numerically well conditioned.

Audit JSONs and their reproducible CPU-only scripts are saved in
`results/vm_methodology_v1/implementation_20260908/`:
`independent_engineering_audit.json`, `audit_engineering_artifacts.py`,
`independent_execution_audit.json`, and `audit_execution_contract.py`.
Both audit JSONs report `all_passed=true` and record exact source/artifact
hashes. The engineering audit does not reconstruct model predictions from
checkpoints or independently regenerate the forecasting data; its replay scope
is saved arrays, validation bookkeeping, state finiteness/counts and provenance.

## Supplemental audit — progress storage repair before confirmation lock

The running manifest now stores hash-bound references to completed per-fit
summaries, while terminal manifests retain the full detailed rows. This removes
repeated duplication of large immutable histories during progress updates.
Detailed summary JSONs, checkpoints, training objectives and data are unchanged.

The supplemental independent audit passed **17** no-training probes. A
52-cell disposable stub invocation verified the complete planned matrix before
execution, hash/path/status validation of references, exact summary hydration,
and full-schema finalization when interrupted between fits. Resuming a simulated
abrupt lightweight checkpoint skipped its already completed first fit and ran
each remaining cell once. The completed summary/checkpoint bytes and original
absolute deadline remained unchanged.

Actual function ASTs for `_fit` and `_save_task` match both original A/B
development snapshots. Trainer, task generator, numerical layers, forecaster
adapters, protocol, JSON configuration and canonical plan hashes also match
both snapshots. The pre-lock bridge records both original development manifest
hashes and old/current observer hashes, and becomes a hashed lock dependency.
A fixture verified that any observer source change after locking is still
rejected; the pre-lock bridge introduces no confirmation exemption.

The repair has no remaining blocking findings in this review. Exact hashes and
probe results are preserved in
`results/vm_methodology_v1/implementation_20260908/independent_progress_storage_audit.json`,
with the reproducible `audit_progress_storage.py` beside it. No model fit,
GPU computation or reserved function generation was performed for this audit.
