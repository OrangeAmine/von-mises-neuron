# Cached Jena and power readiness audit, 2026-09-08

Both raw datasets are available locally, but the current loader/window contract
needs explicit continuity and aggregation rules before full temporal-model runs.
This audit read cached files and reproduced loader operations without downloads,
training, or shared-source changes. Counts describe this cache, not a fresh
upstream dataset or model comparison.

The saved output is `results/direction_2026-09-08/real_data_readiness.json`.
The audit is now retained in `scripts/audit_real_data_readiness.py`; reproduce it
from the repository root with a new output directory:

```powershell
py -3.11 scripts/audit_real_data_readiness.py --output-dir results/my-readiness-audit
```

It calls no download function and never modifies raw data. The initial local
script remains with its original output for provenance.

## Available inputs

| Dataset | Extracted file | Bytes | Raw rows | Timestamp extent |
| --- | --- | ---: | ---: | --- |
| Jena | `data/raw/jena/extracted/jena_climate_2009_2016.csv` | 43,164,220 | 420,551 | 2009-01-01 00:10–2017-01-01 00:00 |
| Power | `data/raw/power/extracted/household_power_consumption.txt` | 132,960,755 | 2,075,259 | 2006-12-16 17:24–2010-11-26 21:02 |

Extracted-file SHA256 fingerprints:

- Jena: `9626d3964f7afb3fa94b6f3d3a0e92fe96e4a4477c31ed9875cc64d6717c129f`
- Power: `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`

Cached archives also exist: Jena 13,568,290 bytes and power 20,640,916 bytes.
Timestamps are parsed as timezone-naive. The source metadata was not externally
verified for timezone or DST conventions; calendar experiments must document
their assumption rather than infer a timezone from the dataset's name.

## Missingness and hourly continuity

| Property | Jena | Power |
| --- | ---: | ---: |
| Complete hourly grid bins | 70,129 | 34,589 |
| Retained hourly rows | 70,041 | 34,168 |
| Entirely missing bins deleted | 88 | 421 |
| Remaining timestamp jumps | 2 | 8 |
| Largest jump between retained bins | 74 hours | 120 hours |
| Missing cells after resampling, before fill | 2 | 0 |
| Actual forward-filled cells | 2 | 0 |
| Actual backward-filled cells | 0 | 0 |

Jena's jumps are 2014-09-24 17:00→2014-09-25 09:00 and
2016-10-25 10:00→2016-10-28 12:00. They remove 15 and 73 hourly bins.
Power contains eight missing stretches; all dates are saved in the JSON.
The current `create_sequences` checks NaNs but does not reject time gaps, so
168 input rows and 24 target rows cannot automatically mean 168 and 24 hours.

At the existing 80/10/10 retained-row split and 168/24 windows, continuity
filtering gives these counts. They use the current loader's Jena forward fills:

| Dataset / split | Rows | Candidate windows | Gap-crossing windows | Contiguous finite windows |
| --- | ---: | ---: | ---: | ---: |
| Jena train | 56,032 | 55,841 | 191 | 55,650 |
| Jena validation | 7,004 | 6,813 | 0 | 6,813 |
| Jena test | 7,005 | 6,814 | 191 | 6,623 |
| Power train | 27,334 | 27,143 | 955 | 26,188 |
| Power validation | 3,417 | 3,226 | 191 | 3,035 |
| Power test | 3,417 | 3,226 | 382 | 2,844 |

Jena replaces 18 raw wind-speed `-9999` sentinels with NaN. Hourly aggregation
leaves missing wind speed at **2015-07-13 10:00 and 11:00**, both in validation.
The loader forward-fills both with **6.16 m/s**. Its subsequent `.bfill()` changes
**zero cells** here, so this cache provides no evidence of actual backward-fill
leakage. Nevertheless, the unrestricted code path could use future values for
leading missing observations on another input. Forward-fill is past-dependent,
but should carry an age/missingness record and a maximum-gap rule if retained.

Power has 25,979 missing minute readings in each of its seven channels; its minute
timestamp grid is regular and unique. Resampling uses available readings, drops
entirely empty hours and performs no imputation. Retained hourly power bins have
2–60 valid readings. Jena target bins have 1–12 readings, with some counts above
six due to duplicated raw timestamps. A minimum-coverage rule would change
evaluation windows and must be specified before comparing models.

## Jena aggregation issues

The Jena raw file has 327 duplicated timestamps and some out-of-order blocks.
Duplicates are identical across the five selected sensor columns. Pandas
resampling sorts timestamps but still weights duplicate rows. Removing duplicate
timestamps changes three hourly temperature means, with a maximum difference of
**0.12111 °C**. Deduplication must be explicit and recorded as a protocol change.

`wd (deg)` is currently averaged arithmetically with the other channels. Degrees
wrap at 360: an arithmetic average of 359 and 1 is 180. In this cache, 3,822 hourly
bins differ by more than 90 degrees between arithmetic and an unweighted circular
mean. This flags an angular aggregation issue; it does not establish that an
unweighted circular mean is the correct physical wind estimator. Choose a
documented vector/circular treatment, including calm-wind handling and the
meaning of wind speed, before using this channel for circular-input claims.
Likewise, the current z-scored degree column is not a raw-radian angular input.

Hourly means are left-labeled but become available after the bin ends. Every
forecast origin should mean the right edge of the final input bin. No smoothing
is used by the current real benchmark. Existing train-only normalization is
appropriate; keeping that does not by itself resolve aggregation or gap policy.

## Before expensive evaluation

1. Freeze a protocol for deduplication, partial-hour coverage, Jena wind direction,
   missing sensor values, and timezone-naive calendar semantics. Preserve the
   old protocol as historical rather than silently mixing its results.
2. Require contiguous input and target bins, or explicitly implement a model that
   handles elapsed time and missing observations. Align every neural model and
   persistence/seasonal control to the same valid origins.
3. Remove unconditional backward-fill from the new evaluation path; either reject
   incomplete windows or implement bounded causal filling with provenance.
4. Begin with preparation-only checks and batched inference. Jena's approximately
   56,000 training windows are much larger than occupancy's 155 contiguous
   windows; the occupancy runner's full-tensor validation should not be copied
   to large datasets without reviewing memory use.
5. Save data/source fingerprints, filtering counts, actual epochs, predictions,
   and split-specific target statistics. Use multiple training seeds and label
   their variability separately from uncertainty over evaluation periods.

No neural quality or model readiness is certified by this data audit. The new
occupancy comparison runner was reviewed separately for common windows, causal
24-hour controls, time features, and paired initialization; no blocking issue
was found in those paths for its fixed 24/6 configuration.
