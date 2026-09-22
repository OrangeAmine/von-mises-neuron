# Practical forecasting data contract

Audited 2026-09-08 from the cached power file, the development preparation
manifests and a separate preparation-only calculation of reserved-block
coverage. This record contains no forecast scores. The frozen
[forecasting protocol](practical_forecasting_protocol.md) defines candidate
selection; the [results report](practical_forecasting_results.md) records the
subsequent forecasting evidence.

## Source and hourly observations

The primary source is the locally extracted UCI household power file:
`data/raw/power/extracted/household_power_consumption.txt`.
It contains **2,075,259 rows**, occupies **132,960,755 bytes**, and has SHA256:

```text
4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7
```

The cache was read without downloads, external research or changes to raw data.
Its timestamps span 2006-12-16 17:24 through 2010-11-26 21:02. They are parsed
as naive recorded clock values. No geographic timezone or DST correction is
inferred. Identical repeated sensor rows would be deduplicated and conflicting
values at the same timestamp rejected; this power cache contains **no duplicate
timestamps**.

The seven sensor columns, in input order, are:

| Position | Source column | Treatment |
| --- | --- | --- |
| 0 | `Global_active_power` | Forecast target, hourly mean in kW |
| 1 | `Global_reactive_power` | Sensor; source numerical scale retained before normalization |
| 2 | `Voltage` | Sensor; source numerical scale retained before normalization |
| 3 | `Global_intensity` | Sensor; source numerical scale retained before normalization |
| 4 | `Sub_metering_1` | Sensor; source numerical scale retained before normalization |
| 5 | `Sub_metering_2` | Sensor; source numerical scale retained before normalization |
| 6 | `Sub_metering_3` | Sensor; source numerical scale retained before normalization |

Only the target is reported as kW. This record does not assign unverified units
to the other six columns or treat their arithmetic means as hourly energy sums.
All seven channels are aggregated by arithmetic mean of available readings.

A row labelled **01:00** contains readings in **[00:00, 01:00)** and becomes
available at 01:00. A raw reading timestamped exactly 01:00 belongs to the next
bin. Each sensor needs at least **45 finite minute readings per hour**; fewer
readings make that sensor/hour missing. `?` is missing, nonfinite numerical
values are missing, and no interpolation, smoothing, forward fill or backfill
is applied. Empty and under-covered hours remain on the complete hourly grid.

| Whole-cache coverage | Count |
| --- | ---: |
| Hourly grid bins, 2006-12-16 18:00 through 2010-11-26 22:00 | 34,589 |
| Entirely missing hours before the coverage threshold | 421 |
| Additional hours rejected by the 45-reading threshold | 17 |
| Hours with any missing sensor after the threshold | 438 |
| Hours with all seven sensors accepted | 34,151 |
| Partial hours with 1–59 readings, per sensor | 83 |
| Accepted partial hours with 45–59 readings, per sensor | 66 |
| Missing raw minute values, per sensor | 25,979 |

In this cache, missingness and coverage counts are identical across all seven
channels. Every accepted hourly mean uses 45–60 readings. The implementation
still applies the threshold separately to each sensor, so this coincidence is
not assumed for other inputs.

## Normalization and forecasting inputs

For each fold, fit each sensor's mean and population standard deviation
(`ddof=0`) using only its finite hourly observations in the training interval.
Use `(value - mean) / scale` for every split; a scale at most `1e-12` is replaced
by 1. Statistics are fitted on training rows, including rows that do not survive
the subsequent window filter, rather than repeatedly weighted overlapping
windows. Validation and test observations never change these statistics.

The common model input has shape **(origins, 168, 9)**: seven normalized sensor
columns followed by raw daily and weekly calendar angles in radians. These
angles use the recorded clock hour and Monday-based weekday, respectively, and
are not z-scored. Targets have shape **(origins, 24)** and use the training
target's normalization; saved evaluations also retain targets in kW.

An origin is the availability time of the last input observation. The input
contains that observation and the previous 167 hours; the first target is one
hour after the origin. Every input sensor and every future target must be
finite. Future non-target sensors are not inspected when accepting a window.
A missing input hour therefore prevents forecasts whose 168-hour histories
cross it. Missing timestamp rows supplied to the window helper are restored as
missing hourly rows before filtering.

## Chronological intervals and observed window counts

All boundaries below are midnight **availability timestamps**. Each interval is
half-open: its start is included and its end excluded. In particular, a target
labelled at a split's start represents the raw hour ending at that boundary.

| Evaluation block | Training interval | Validation interval | Evaluation interval |
| --- | --- | --- | --- |
| Development 1 | [2007-07-08, 2008-01-04) | [2008-01-04, 2008-02-01) | [2008-02-01, 2008-02-29) |
| Development 2 | [2008-01-06, 2008-07-04) | [2008-07-04, 2008-08-01) | [2008-08-01, 2008-08-29) |
| Development 3 | [2008-07-08, 2009-01-04) | [2009-01-04, 2009-02-01) | [2009-02-01, 2009-03-01) |
| Reserved final | [2010-04-04, 2010-10-01) | [2010-10-01, 2010-10-29) | [2010-10-29, 2010-11-26) |

Every 24-hour target sequence must lie entirely within its own interval.
Training inputs also stay in training. Validation and evaluation inputs may
use earlier observed context, including earlier splits. Thus a fully observed
28-day block has 649 forecast origins covering 672 unique target hours.

The following counts come from the actual accepted windows. Exclusion columns
are disjoint and applied in order: insufficient history, nonfinite input, then
nonfinite target. Candidate counts already exclude origins whose 24-hour
targets would overrun the split end.

| Block / split | Candidate windows | Insufficient history | Nonfinite input | Nonfinite target | Valid windows | Unique target hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development 1 / train | 4,297 | 168 | 337 | 43 | 3,749 | 3,795 |
| Development 1 / validation | 649 | 0 | 0 | 0 | 649 | 672 |
| Development 1 / evaluation | 649 | 0 | 0 | 0 | 649 | 672 |
| Development 2 / train | 4,297 | 168 | 0 | 0 | 4,129 | 4,152 |
| Development 2 / validation | 649 | 0 | 0 | 0 | 649 | 672 |
| Development 2 / evaluation | 649 | 0 | 0 | 0 | 649 | 672 |
| Development 3 / train | 4,297 | 168 | 336 | 48 | 3,745 | 3,814 |
| Development 3 / validation | 649 | 0 | 0 | 0 | 649 | 672 |
| Development 3 / evaluation | 649 | 0 | 336 | 42 | 271 | 317 |
| Reserved final / train | 4,297 | 168 | 403 | 48 | 3,678 | 3,724 |
| Reserved final / validation | 649 | 0 | 116 | 0 | 533 | 556 |
| Reserved final / evaluation | 649 | 0 | 0 | 0 | 649 | 672 |

Each training grid has 4,320 rows; each validation/evaluation grid has 672.
Finite training observations per sensor are 4,317 / 4,320 / 4,318 / 4,112 for
the four blocks in table order. The third development evaluation has 670 finite
target observations, but only 317 unique target hours survive full-history
filtering. The reserved validation interval has 672 finite targets; gaps in
earlier context still remove 116 origins. Reserved-block counts above were
calculated without fitting a model or computing forecast errors.

Hourly forecasts overlap: a target hour can appear at up to 24 forecast
horizons. The three development blocks contain **1,569 accepted origins** and
**37,656 origin/horizon target pairs**, covering **1,661 unique target hours**.
The reserved final block contains **649 origins**, **15,576 target pairs** and
**672 unique target hours**. These are not independent observations. Earlier
development evaluation periods also enter later folds' historical training
data; these remain chronological backtests on one household.

## Evidence and reuse limits

The saved development preparation manifest and its three `data_manifest.json`
files are under
`results/practical_forecasting_2026-09-08/preparation/`. They retain all sensor
normalization values, target summaries, filtering counts, raw fingerprints,
package versions and implementation hashes. `results/` is ignored by Git; the
counts and primary raw fingerprint above preserve the essential data record
when local artifacts are absent. The implementation is
[`src/operational_data.py`](../src/operational_data.py), with causal and coverage
checks in [`tests/test_operational_data.py`](../tests/test_operational_data.py).

For CSV-based inference, a finite, contiguous hourly table can establish its
column order and timestamp spacing, but cannot prove that its values were
computed from at least 45 valid minute readings. That assurance requires the
raw source or retained per-bin coverage provenance. Incoming rows must already
follow the same availability labels, recorded-clock convention and sensor
scales. This contract supports reproducible offline evaluation; it does not
establish production readiness or forecast usefulness by itself.
