# Synthetic clock audit, 2026-09-08

The default synthetic test set contains a daylight-saving transition that changes
the relationship between the generated signal and its calendar features. This is
large enough to warrant a clock sensitivity check before interpreting whether
calendar inputs help. It does not invalidate matched VM versus sine/cosine
comparisons that receive the same input representation.

This audit used source inspection and timestamp calculations only. No models
were trained, no sources modified, and no test scores used to choose a clock.
The relevant paths are `generate_synthetic_co2`, `augment_with_time_features`,
and `prepare_temporal_data` in `scripts/benchmark_standard_dataset.py`, and
`extract_angular_time_features` in `src/temporal.py`.

## What changes

The generator advances `t = 0, 1, ...` in elapsed hours. Daily and weekly signal
components use `sin(2*pi*t/24)` and `sin(2*pi*t/168)`. Its index starts at
2021-01-01 00:00 in Europe/Paris. Hand-crafted sine/cosine features use elapsed
hours from that same first timestamp, so they remain aligned throughout.

The temporal features instead use local hour and local Monday-based weekday.
Before DST, daily phase aligns with the generator; weekly phase has a harmless
constant origin offset of 96 hours because January 1 was a Friday. At DST, the
local clock skips from **2021-03-28 01:00+01:00 to 03:00+02:00** while the
generated signal advances only one hour. The calendar-to-generator offset then
changes by **+1 hour**, or **15 degrees daily and 2.142857 degrees weekly**.
This is a change in their learned relationship, not a missing observation:
absolute timestamps remain exactly one hour apart.

| Split | Rows | Local timestamp extent |
| --- | ---: | --- |
| Train | 1728 | Jan 1 00:00+01:00–Mar 13 23:00+01:00 |
| Validation | 216 | Mar 14 00:00+01:00–Mar 22 23:00+01:00 |
| Test | 216 | Mar 23 00:00+01:00–Apr 1 00:00+02:00 |

The transition begins at global row **2066**, test row **122** (zero-based).
Neither training nor validation contains a DST transition.

## Extent of test exposure

With 72 input rows and 12 target rows there are 133 test windows:

- 94 windows (70.68%) contain at least one post-transition target: 83 contain
  only post-transition targets, and 11 straddle the transition.
- 1,062 of 1,596 pooled target entries (66.54%) are post-transition. These entries
  include overlapping targets, not independent observations.
- 82 input windows contain post-transition observations; 11 input windows are
  entirely after the transition.
- The first affected target window has final input timestamp Mar 27 14:00+01:00
  and final target timestamp Mar 28 03:00+02:00. The first window with entirely
  post-transition targets ends its input at Mar 28 01:00+01:00.

Calendar-conditioned models share this difficulty, but their sensitivity can
differ by architecture and optimization. Sensor-only models and the elapsed-time
hand-crafted control do not receive this calendar discontinuity. Thus the
extra-calendar versus sensor-only comparison combines the effect of additional
information with robustness to a clock-to-signal shift. It does not isolate the
value of well-aligned calendar information. Comparisons of VM versus sine/cosine
using identical inputs remain controlled with respect to clock information.
No claim about how much this mismatch changes RMSE follows from timestamps alone.

## Recommended sensitivity protocol

Run a **frozen-checkpoint clock correction first**. Encode calendar inputs in a
fixed UTC+01:00 timezone instead of Europe/Paris, while preserving the original
absolute timestamps, sensor values, split membership and targets. This is
equivalent to UTC calendars with a fixed one-hour phase alignment. Direct
calculation confirms that all training/validation angles and every pre-DST angle
are exactly unchanged; only post-DST test angles differ. Existing checkpoints
and their validation selection can therefore be reused without retraining.

Evaluate all completed seeds for VonMisesTime, SinCosTime and TemporalVM on both
clock representations; retain sensor-only and elapsed hand-crafted controls as
unchanged references. Preserve predictions and report paired score differences
on the full test set plus the prespecified pre-transition, straddling-target and
post-transition window groups. Record counts and avoid significance claims from
the overlapping windows. Keep both outcomes, including regressions. This tests
inference sensitivity to the known clock mismatch and is not independent
generalization evidence.

For a subsequent full **UTC-versus-wall-clock training sensitivity**, change only
the index used for angular extraction to UTC, retaining sensor arrays, targets,
absolute times and the same split/window indices. Refit the same prespecified
models, seeds and budgets with their original validation selection. Raw UTC
calendars introduce constant training phase shifts (-15 degrees daily versus
Paris winter, and the corresponding one-hour weekly offset), so retraining this
arm also measures sensitivity to that coordinate choice. Do not confuse it with
the sharper fixed-offset checkpoint correction.

Do not select the better clock by test score or replace the historical benchmark
silently. The aligned choice is justified by the generator's elapsed-time
equation, while local calendars remain a separate robustness condition. Any new
claims should state which condition they use. The frozen-checkpoint check is
worth running now because most evaluated targets occur after the mismatch and
it requires no optimization budget.

The implementation is `scripts/evaluate_synthetic_clock_sensitivity.py`. After
the source comparison finishes, run:

```powershell
py -3.11 scripts/evaluate_synthetic_clock_sensitivity.py --source-dir results/direction_2026-09-08/controlled_synthetic --output-dir results/direction_2026-09-08/synthetic_clock_sensitivity
```

It checks the source dataset fingerprint and unchanged fitting/target tensors,
requires each original prediction array to reproduce within normalized absolute
and relative tolerances of 1e-5, and then scores the corrected representation.
It records every completed time-aware variant, including temporal ablations;
NoTime must be exactly unchanged. Non-completed source fits are explicitly
excluded. Model and input fingerprints, paired per-seed/group score changes,
and original/corrected gates against the unchanged same-seed base VM are saved.
Execution success does not mean every exploratory variant passes its gates.

## Executed fixed-clock sensitivity

The frozen-checkpoint audit subsequently completed 24 evaluations (eight models × three seeds), with **zero original-prediction reconstruction error** on this GPU. There was no retraining or checkpoint selection. All training/validation inputs and all sensor/target tensors were unchanged. The following RMSE values are mean ± sample SD over the original three trained checkpoints; corrected-minus-original is a paired difference.

| Model | Original RMSE | Fixed-offset RMSE | Paired change (ppm) | Corrected absolute pass seeds |
| --- | ---: | ---: | ---: | ---: |
| VonMisesTime | 5.5782 ± 0.1585 | 5.0462 ± 0.0926 | -0.5320 ± 0.2500 | 3/3 |
| SinCosTime | 5.7089 ± 0.3211 | 4.9673 ± 0.0555 | -0.7416 ± 0.2679 | 3/3 |
| TemporalVM | 5.3551 ± 0.0280 | 4.9888 ± 0.0585 | -0.3663 ± 0.0399 | 3/3 |
| NoTime | 5.3835 ± 0.2076 | 5.3835 ± 0.2076 | +0.0000 ± 0.0000 | 3/3 |
| UnitScales | 5.3522 ± 0.1275 | 4.9750 ± 0.0490 | -0.3772 ± 0.1006 | 3/3 |
| LearnedScales | 5.3499 ± 0.0339 | 4.9905 ± 0.0565 | -0.3594 ± 0.0403 | 3/3 |
| NoConv | 7.8363 ± 0.1500 | 4.8979 ± 0.0935 | -2.9384 ± 0.0750 | 3/3 |
| Pooling | 31.7271 ± 3.9972 | 31.1280 ± 4.1521 | -0.5991 ± 0.2321 | 1/3 |

NoTime predictions and all pre-transition/straddling-target subgroup metrics are exactly unchanged. Other changes occur in the post-transition subgroup. Pooling remains unvalidated in seeds 43 and 44.

This materially changes the architectural interpretation: NoConv now has lower average RMSE than full TemporalVM, rather than the large disadvantage observed with the mismatched clock. The earlier result therefore measures robustness to a clock-to-signal shift as well as the convolution/residual architecture. It does not establish that convolution bridges improve well-aligned forecasting. The full temporal model now improves over NoTime, consistent with useful aligned calendar information in this constructed signal. Simple SinCosTime is also competitive; no distinct phase-conditioning or polar-parameterization advantage is established.

These are paired inference sensitivities on already-inspected test data, motivated by the generator equation. They are not an independent confirmation, and neither condition is discarded. A future protocol should prespecify an aligned elapsed-time calendar and retain wall-clock mismatch as a separate robustness test.

Executed command and source/checkpoint/tensor fingerprints are in `results/direction_2026-09-08/synthetic_clock_sensitivity/manifest.json`; per-seed/subgroup metrics and corrected predictions are stored alongside it.
