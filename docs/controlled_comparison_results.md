# Controlled comparison results, 2026-09-08

The strongest finding is an evaluation issue: correcting a held-out synthetic
clock mismatch reverses the apparent benefit of convolution/residual bridges.
With aligned clocks, simple calendar-augmented SinCos is competitive with
TemporalVM. Occupancy forecasts remain far worse than persistence. These results
support fixing the experimental contract before adding architecture complexity.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run and descriptive validation
- Origin Date: 2026-09-08
- Verification Status: ANALYZED; fresh fits executed, historical standard metrics reproduced
- Version Label: controlled_comparison_v1

Final code verification: **380 tests passed in 7.85 seconds**; diff whitespace
check passed. Benchmark failure cases, mathematical equivalence, gradients,
clock boundaries, contiguous data windows and failure/cancellation manifests
are covered. No dependency or global environment changes were needed.

## Evidence and protocol

The repaired standard benchmark passed. All five previously audited models reproduced their earlier metrics exactly. The new SinCos control passed the same absolute limits with 43,148 parameters: RMSE 5.4556, MAE 4.2725, R2 0.978179, MAPE 0.7110%. This is a single-seed validation, not a superiority result.

The controlled experiments then completed 33 synthetic and 21 occupancy fits: three training seeds (42, 43, 44) per model, fixed datasets and splits, unchanged 50-epoch maximum budgets, and validation-only checkpoint selection. No settings were tuned to the new test metrics. See [the pre-run protocol](controlled_comparison_plan.md).

Unlike the historical standard benchmark, controlled runs reset the training RNG after construction. VM/SinCos start with identical functions, counts and training random streams. Temporal ablations retain common reference weights; unit amplitudes are absorbed into projection columns to match the initial phase function. Thus the controlled seed-42 values intentionally need not equal the standard benchmark. Adam/weight decay operate in different coordinates, and computational cost is not matched.

Environment: Python 3.11.4, PyTorch 2.5.1+cu121, CUDA 12.1, RTX 3060 Laptop GPU, NumPy 1.26.4, pandas 2.3.3. Base revision `90dc9f2` plus the working-tree changes archived with each run. Manifests include package versions, source SHA256s, split boundaries, data hashes and normalization statistics; source ZIPs capture the actual uncommitted code. Predictions, CPU state dictionaries and actual epoch histories are preserved locally.

## Synthetic results

Original 90-day elapsed-time generator, data seed 42; 80/10/10 chronological split; 72-hour input and 12-hour CO2 forecast in ppm; 1,645/133/133 windows. The table reports mean ± sample SD across the three training seeds. SD describes optimization variability on this one signal, not generalization uncertainty.

| Model | Parameters | RMSE | MAE | R2 | MAPE (%) | Absolute pass seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VonMises | 43,148 | 5.4362 ± 0.2728 | 4.2090 ± 0.2477 | 0.9783 ± 0.0022 | 0.6998 ± 0.0415 | 3/3 |
| SinCos | 43,148 | 5.5124 ± 0.1542 | 4.3111 ± 0.1659 | 0.9777 ± 0.0013 | 0.7163 ± 0.0274 | 3/3 |
| VonMisesTime | 43,404 | 5.5782 ± 0.1585 | 4.4248 ± 0.1676 | 0.9772 ± 0.0013 | 0.7368 ± 0.0285 | 3/3 |
| SinCosTime | 43,404 | 5.7089 ± 0.3211 | 4.5854 ± 0.2725 | 0.9761 ± 0.0027 | 0.7640 ± 0.0448 | 3/3 |
| TemporalVM | 52,492 | 5.3551 ± 0.0280 | 4.2673 ± 0.0184 | 0.9790 ± 0.0002 | 0.7112 ± 0.0032 | 3/3 |
| NoTime | 43,980 | 5.3835 ± 0.2076 | 4.2282 ± 0.1854 | 0.9787 ± 0.0016 | 0.7021 ± 0.0311 | 3/3 |
| UnitScales | 52,492 | 5.3522 ± 0.1275 | 4.2274 ± 0.1510 | 0.9790 ± 0.0010 | 0.7041 ± 0.0250 | 3/3 |
| LearnedScales | 52,556 | 5.3499 ± 0.0339 | 4.2608 ± 0.0231 | 0.9790 ± 0.0003 | 0.7101 ± 0.0039 | 3/3 |
| NoConv | 51,660 | 7.8363 ± 0.1500 | 6.4247 ± 0.1274 | 0.9550 ± 0.0017 | 1.0746 ± 0.0209 | 3/3 |
| Pooling | 19,212 | 31.7271 ± 3.9972 | 28.2481 ± 3.6354 | 0.2542 ± 0.1800 | 4.7129 ± 0.6075 | 1/3 |
| HandCrafted | 6,924 | 5.0610 ± 0.0486 | 3.8808 ± 0.0389 | 0.9812 ± 0.0004 | 0.6458 ± 0.0068 | 3/3 |

All core absolute checks and all three TemporalVM-versus-base regression checks passed. Pooling remains **unvalidated**: seed 43 failed MAE (30.8979), R2 (0.1201) and MAPE (5.1532%); seed 44 failed MAE (29.7429) and R2 (0.1837). Both RMSE values pass, demonstrating why all metric gates matter. No numerical failures occurred. The process exited 0 for core validation while explicitly recording `all_absolute_passed=false`.

Reference RMSEs: persistence 52.1991 ppm; daily seasonal persistence 10.9455 ppm. Both use only observations available at each origin.

Selected paired RMSE differences below are candidate minus reference, so negative favors the candidate. They are descriptive, not significance tests.

| Candidate − reference | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD (ppm) |
| --- | ---: | ---: | ---: | ---: |
| SinCos − VonMises | +0.2206 | -0.2489 | +0.2571 | +0.0762 ± 0.2822 |
| TemporalVM − VonMises | +0.2155 | -0.3859 | -0.0728 | -0.0811 ± 0.3008 |
| NoTime − TemporalVM | +0.0328 | +0.2458 | -0.1933 | +0.0284 ± 0.2196 |
| UnitScales − TemporalVM | -0.1401 | -0.0059 | +0.1373 | -0.0029 ± 0.1387 |
| LearnedScales − TemporalVM | +0.0028 | -0.0085 | -0.0100 | -0.0052 ± 0.0070 |
| NoConv − TemporalVM | +2.4910 | +2.3438 | +2.6089 | +2.4813 ± 0.1328 |
| Pooling − TemporalVM | +21.7875 | +29.3156 | +28.0130 | +26.3721 ± 4.0234 |
| HandCrafted − VonMises | -0.1627 | -0.6250 | -0.3378 | -0.3752 ± 0.2334 |

In this original wall-clock condition, the VM/SinCos ordering reverses between
seeds. TemporalVM has slightly lower average RMSE than base VM but higher average
MAE/MAPE. Removing time conditioning has little average RMSE effect here, and
fixed-unit/learned amplitudes remain close to the default. Removing convolution
**and its residual bridge together** degrades performance in this condition.
The clock sensitivity below changes both the time-conditioning and bridge
interpretations. Pooling is substantially worse; an LSTM replacement is not
established. The hand-crafted comparator has the lowest average RMSE in the
original clock condition.

## Calendar sensitivity changes the interpretation

The generator follows elapsed time while local calendar angles cross DST inside
the test split. A fixed UTC+01:00 clock preserves every training and validation
feature while aligning post-DST test features with the generator. We evaluated
24 saved checkpoints without retraining; all original predictions reproduced
exactly. Targets, sensor values, fitting tensors and checkpoints stayed fixed.
See the [clock audit](synthetic_clock_audit.md) for per-seed and subgroup evidence.

| Model | Original RMSE mean | Aligned RMSE mean ± sample SD |
| --- | ---: | ---: |
| VonMisesTime | 5.5782 | 5.0462 ± 0.0926 |
| SinCosTime | 5.7089 | 4.9673 ± 0.0555 |
| TemporalVM | 5.3551 | 4.9888 ± 0.0585 |
| NoTime | 5.3835 | 5.3835 ± 0.2076 |
| UnitScales | 5.3522 | 4.9750 ± 0.0490 |
| LearnedScales | 5.3499 | 4.9905 ± 0.0565 |
| NoConv | 7.8363 | 4.8979 ± 0.0935 |
| Pooling | 31.7271 | 31.1280 ± 4.1521 |

Under aligned clocks, NoConv-minus-TemporalVM RMSE is -0.0909 ± 0.0560 ppm,
reversing its original disadvantage. NoTime-minus-TemporalVM is +0.3947 ± 0.2595,
consistent with useful aligned timestamp information. SinCosTime-minus-TemporalVM
is -0.0215 ± 0.0406, so the special phase pathway has no demonstrated unique
advantage over the simpler matched-information control. Scale changes offer no
clear additional benefit. Pooling still fails in two seeds.

This sensitivity was motivated by source-level clock semantics after the first
run, not by a fresh holdout. It cannot serve as independent confirmation. Both
conditions are retained: aligned calendars test the intended synthetic relation,
and the original condition measures robustness to a clock-to-signal shift.
There is no general convolution-benefit claim from this experiment.

## Occupancy results

Cached UCI occupancy CO2, 346 retained hourly rows, historical 70/15/15 row split (242/52/52 rows), 24-hour input and six-hour forecast. New windows require contiguous hourly inputs and targets. This removes 58 training windows: 155/23/23 valid windows remain. Normalization uses training rows only; no interpolation or smoothing is applied. Hourly bins become available at their right edges. All models and controls use identical held-out origins. This is a changed protocol, not a reproduction of historical neural occupancy metrics.

| Model | Parameters | RMSE | MAE | R2 | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| VonMises | 42,886 | 734.2275 ± 12.1453 | 704.0834 ± 12.1812 | -10.7985 ± 0.3885 | 51.1806 ± 0.8881 |
| SinCos | 42,886 | 734.1205 ± 12.1117 | 704.0268 ± 12.1485 | -10.7951 ± 0.3874 | 51.1796 ± 0.8858 |
| TemporalVM | 52,230 | 735.9881 ± 3.2891 | 704.8905 ± 2.2500 | -10.8532 ± 0.1059 | 51.1974 ± 0.1049 |
| VonMisesTime | 43,014 | 742.3458 ± 7.1402 | 712.0300 ± 7.8877 | -11.0594 ± 0.2317 | 51.7462 ± 0.6471 |
| SinCosTime | 43,014 | 742.2887 ± 7.0166 | 712.0101 ± 7.7574 | -11.0575 ± 0.2277 | 51.7471 ± 0.6368 |
| LSTM | 43,322 | 721.8805 ± 11.6383 | 691.2804 ± 11.3128 | -10.4049 ± 0.3665 | 50.2504 ± 0.8183 |
| HandCrafted | 1,014 | 749.3731 ± 56.6846 | 704.6546 ± 60.9404 | -11.3349 ± 1.8856 | 50.9891 ± 4.6281 |
| Persistence | 0 | 233.4200 | 194.1069 | -0.1922 | 14.7996 |
| SeasonalPersistence24h | 0 | 582.9805 | 526.8829 | -6.4369 | 37.1802 |
| TrainMean | 0 | 740.5043 | 708.9760 | -10.9989 | 51.4291 |

Every trained model performs much worse than persistence on this test segment. All recurrent models select epoch 1 and stop after 16 epochs; the hand-crafted fits select epochs 1–2. Recurrent prediction means lie around 620–660 ppm, near the training-row mean 629.57 ppm, whereas pooled test targets average 1338.55 ppm. The saved predictions support mean-like forecasts and insufficient adaptation in this setup. They do not isolate why this occurs: LSTM and both coefficient parameterizations show the same broad failure. The evaluated target range lies within the overall training range, so a level/distribution shift is more precise than claiming all test targets are outside training support.

The 23 overlapping windows are not independent samples. Three training seeds do not compensate for this short evaluation period. Real results use finite checks, not synthetic CO2 success thresholds; exit 0 certifies completed finite evaluation, not useful forecasts. See [occupancy diagnostics](occupancy_diagnostics.md) for distributions, gaps and exact-hour controls.

The paired SinCos-minus-VM RMSE difference is -0.1070 ± 0.0844 ppm across seeds,
and TemporalVM-minus-VM is +1.7607 ± 13.5379 ppm. These small differences do not
resolve the forecasting failure. The 138 forecast entries cover only 28 unique
future hourly targets. Independent artifact review checked all 21 checkpoints,
aligned predictions and source hashes; CPU metric recomputation differed from
the saved GPU reductions by at most 0.000122 ppm.

## Reproduction and limits

```powershell
py -3.11 scripts/benchmark_standard_dataset.py --output-dir results/new-standard
py -3.11 scripts/controlled_temporal_comparison.py --output-dir results/new-synthetic --seeds 42 43 44 --epochs 50
py -3.11 scripts/controlled_real_comparison.py --output-dir results/new-occupancy --seeds 42 43 44 --epochs 50
```

This run used `results/direction_2026-09-08/{standard,controlled_synthetic,controlled_occupancy}`. Results are ignored by Git; the tables above preserve the evidence. Sum of measured fit/evaluation times: synthetic 298.89 seconds, occupancy 6.87 seconds. These exclude preparation and are not a hardware-controlled speed comparison.

All 11 statistical fallacy categories were considered: Simpson/ecological inference (no pooled cross-dataset inference), selection/Berkson and collider bias (no population or causal claims), base-rate neglect (not a classification study), regression to mean (no selected-extreme improvement claim), survivorship (all seeds and failures retained), look-elsewhere/forking paths (exploratory full reporting; no p-values or test-set tuning), causal attribution and reverse causality (no causal generalization). The important live limitations are small fixed datasets, repeated historical benchmark use, overlapping horizons, and the calendar mismatch. No formal statistical significance or novelty is claimed.

## Direction supported by this stage

Keep the matched SinCos control and simple persistence controls in future
experiments. Freeze clock semantics, then compare SinCosTime, TemporalVM and
NoConv on fresh realizations/origins, with NoTime as an information ablation.
For practical forecasting, prioritize distribution shift and validation behavior
before larger architectures. For circular-input research, specify actual angular
inputs/targets and compare matched parameterizations with circular metrics.
The owner must choose the primary research objective before either becomes a
confirmatory program. Jena and power require the documented
[data-readiness decisions](real_data_readiness.md) before their existing rows can
be treated as fixed-hour forecasts.
