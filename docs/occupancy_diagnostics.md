# Occupancy diagnostic: local cache, 2026-09-08

The cached occupancy test segment has a strong CO2 level shift. A prediction fixed
at the training mean produces R2 **-10.999**, close to the historical VM value
**-10.847** cited in `project_state.md`. Fresh persistence is substantially better,
although its test R2 remains negative. This identifies an inadequate forecasting
setup and missing controls; it does not identify the neural model's failure
mechanism without fresh predictions and checkpoint analysis.

No downloads or neural training were performed. These are deterministic controls
on the three locally cached UCI files, using the existing chronological 70/15/15
split, 24 input rows and six output rows. Command:

```powershell
py -3.11 scripts/diagnose_occupancy.py --output-dir results/direction_2026-09-08/occupancy_diagnostics
```

The ignored output `occupancy_diagnostics.json` contains input/source SHA256s,
Python/NumPy/pandas versions, source revision, split dates, distributions, gap
counts, and full control metrics. The run used Python 3.11.4, NumPy 1.26.4 and
pandas 2.3.3 at base revision `90dc9f2` with the new diagnostic script. No seed,
parameters or optimization budget apply to these fixed controls.

## Split distribution and evaluation targets

The loader returns **346 rows**, not the approximate 432 stated in its docstring.
There are 20,560 raw rows and no duplicate timestamps or NaNs in the cached files.

| Split | Rows | Dates (2015) | CO2 mean | Population SD | CO2 range | Windows |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Train | 242 | Feb 2 14:00–Feb 14 01:00 | 629.57 | 275.60 | 421.80–1974.18 | 213 |
| Validation | 52 | Feb 14 02:00–Feb 16 05:00 | 601.94 | 104.94 | 500.99–862.27 | 23 |
| Test | 52 | Feb 16 06:00–Feb 18 09:00 | 1082.18 | 310.27 | 713.98–1613.12 | 23 |

Units are ppm. These population SDs describe distributions; the existing sensor
normalizer uses the training sample SD. After the split-local lookback, pooled
test targets have mean **1338.55 ppm**, SD **213.77 ppm**, and variance
**45,699.72 ppm²**. Validation targets have mean 673.77 and SD 76.57 ppm.
Each held-out split contributes only 23 overlapping six-step windows, or 138
target entries counting repeated timestamps. These are not 138 independent
observations. The test target range remains inside the broad training range;
the shift concerns the frequency and level of the evaluated observations.

## Fair causal controls

Persistence repeats the last observed CO2 bin. Seasonal persistence reads the
exact timestamp 24 elapsed hours before each target bin, and only when it is
available at the origin within the same split. The train-mean control freezes
the mean at 629.57 ppm. All three use the same 23 valid windows in each held-out
split. Predictions and metrics use original units without normalization.

| Split | Control | RMSE | MAE | R2 | MAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Persistence | 57.16 | 41.92 | 0.443 | 5.89 |
| Validation | Seasonal 24h | 167.62 | 155.25 | -3.793 | 22.33 |
| Validation | Train mean | 88.41 | 63.53 | -0.333 | 8.81 |
| Test | Persistence | 233.42 | 194.11 | -0.192 | 14.80 |
| Test | Seasonal 24h | 582.98 | 526.88 | -6.437 | 37.18 |
| Test | Train mean | 740.50 | 708.98 | -10.999 | 51.43 |

The historical VM result is not a fresh matched neural comparison. Its similarity
to the train-mean control suggests checking mean-like predictions or excessive
shrinkage, but does not prove either explanation. Training-split controls in the
JSON are descriptive in-sample diagnostics: the full training mean is causal
only for held-out evaluation.

## Time gaps and leakage audit

Hourly resampling spans 380 possible bins and drops 34 empty bins. Two jumps
remain: Feb 4 10:00→17:00 (six missing hours) and Feb 10 09:00→Feb 11 14:00
(28 missing hours). Both are inside training. The sequence builder checks NaNs
but does not check timestamp continuity: **58 of 213 training windows cross a
gap**, and a six-row output can extend **34 elapsed hours** from the last input
bin. The 24-row seasonal offset therefore cannot generally stand for 24 hours.
The diagnostic uses timestamp lookup and reports a separate common contiguous
subset: 155 training windows, versus 173 common seasonal-valid row windows.
Validation and test contain no gaps, so their control tables need no filtering.

No smoothing, interpolation, backfill or other imputation occurs in the current
occupancy path. Empty bins are deleted rather than fabricated. Normalization
is fitted on training only. Hourly means have left-edge labels but become known
at the right edge: an input bin labeled 10:00 is available at 11:00. Forecast
origin semantics must respect that edge; otherwise offline labels could be
misread as values already known at 10:00. Partial bins contain 9–61 raw samples
in this cache; the loader does not currently filter incomplete hours.

## Next work

1. Add an explicit contiguous-hour window policy to real-data experiments and
   report excluded-window counts, preserving legacy results as historical.
2. Include persistence and exact-hour seasonal controls in fresh neural runs,
   save predictions and actual checkpoint/epoch information, and inspect
   predicted level versus the training mean.
3. Use several chronological evaluation origins to quantify variability on this
   short record. Keep the current holdout fixed; do not tune model choices
   repeatedly to this 23-window test segment.

Validation: seven focused diagnostic tests cover known periodic/ramp controls,
gaps, unavailable/future seasonal sources, common NaN filtering, constant-target
R2 and refusal to download when the cache is absent. No shared loader or model
behavior was modified by this diagnostic.

## Fresh neural follow-up

The [controlled comparison](controlled_comparison_results.md) subsequently ran
all seven neural models over three training seeds on the 155/23/23 contiguous
windows. Their RMSE means range from 721.88 to 749.37 ppm, versus persistence
233.42 ppm. Recurrent models select epoch 1 and produce test means near the
training mean. This strengthens the mean-like-prediction observation, while
still leaving its causal mechanism unresolved. No test-set tuning followed.
