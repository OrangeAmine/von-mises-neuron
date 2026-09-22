# Practical forecasting protocol, 2026-09-08

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: autonomous research engineering under the owner's chosen objective
- Origin Date: 2026-09-08
- Verification Status: UNVERIFIED (protocol frozen before new scores)
- Version Label: practical_forecasting_v1

The owner selected **useful real-world forecasting**. Architecture identity is
secondary to predictive skill against useful simple forecasts. Existing occupancy
results are development evidence, not a fresh test. No new setting will be chosen
to improve its already-inspected 23-window test score.

## Primary data and time contract

Use the locally cached UCI household power data, target `Global_active_power`
in kW. This is a longer, multiyear forecasting evaluation than occupancy.
No download or external upload is needed. Raw timestamps remain naive recorded
clock values; no geographic timezone/DST interpretation is inferred.

Aggregate within hourly bins and label each by its **right-edge availability**.
Require at least 45 valid minute observations per sensor/hour (75% coverage).
Keep the hourly grid, mark insufficient coverage missing, and reject any window
with incomplete sensors or targets. Do not interpolate, smooth or backfill.
Input lookback is 168 hours; predict the next 24 hourly mean power values.

For each fold fit sensor normalization on training rows only. Validation/test
may use earlier observed input context, but every target must lie wholly in its
own half-open interval. No future sensor values enter inputs. Input arrays carry
sensors first and raw 24/168-hour calendar angles last.

## Frozen development folds and reserved final block

Three development evaluation blocks begin 2008-02-01, 2008-08-01 and 2009-02-01,
each lasting 28 days. For each, validation is the immediately preceding 28 days
and training is the preceding 180 days. Dates are fixed before any new model
scores and cover different seasonal positions. They are development backtests,
not three independent datasets.

Reserve the final block **2010-10-29 through 2010-11-26 (exclusive)**, with the
same preceding 28-day validation and 180-day training windows. Do not inspect
its forecast scores until the candidate-selection rule below is applied to the
development folds. Data quality/coverage checks are permitted before scoring.

## Candidates and selection

- Persistence: repeat the latest observed target for all horizons.
- Daily and weekly seasonal persistence: retrieve the observations 24/168 hours
  before each target. All requested sources are inside the 168-hour history.
- Residual ridge: predict changes from the last target using each sensor's
  centered lag history and sine/cosine calendar features at the origin.
  Candidate penalties are fixed at 0.1, 1, 10, 100, 1000. Select penalty using
  validation MSE only; persistence is an explicit validation competitor.
- Residual LSTM, SinCos and TemporalVM: center each sensor on its last observed
  value, retain calendar angles, and predict a correction added to the latest
  observed target. Initialize the output head to zero so the initial function
  is exactly persistence. Use the existing 64-wide encoders (LSTM width68).
  No architecture expansion or hyperparameter search is planned.

Neural seeds are 42,43,44; max50 epochs; Adam lr0.001, weight decay0.0001,
batch64, gradient clipping1, scheduler factor0.5/patience5/minlr1e-5, early
stopping patience15. **Epoch zero competes on validation**, so training cannot
force selection of a checkpoint worse than its persistence initialization on
validation. This does not guarantee test-set safety. Inputs/targets, seeds and
budgets stay comparable; parameter counts and selected epochs are recorded.

Select a deployable candidate family using development aggregate squared error
on the common forecast origins, considering a fixed equal-weight ensemble of the
three neural seeds for each family. Include every simple control in selection;
do not choose the best random seed. Break numerical ties in favor of a simpler
control. Require at least 5% lower development RMSE than the best simple control
and no development fold RMSE more than 10% worse than that fold's best simple
control to recommend a learned candidate. Otherwise retain the best simple
control. These practical gates are declared now and will not be weakened.

After selection, refit the selected family using the reserved block's training
and validation periods and score it once against all simple controls on that
block. The family remains selected even if the final result disappoints; do not
retry or tune it using the reserved scores. A learned family must also meet the
5% RMSE margin on the final block before being described as a useful improvement.

## Verification and reporting

New neural wrappers must first pass the unchanged standard synthetic absolute
thresholds on its 90-day,72/12 setup. Temporal residual also receives the existing
base-VM regression comparison; any failure remains reported, not hidden by a
successful fit. Model tests cover level-shift behavior and finite gradient flow.

Report original-unit RMSE/MAE, R2, per-horizon errors, relative skill against
persistence and seasonal controls, selected epochs, predicted/observed levels,
window exclusions, parameter counts, runtime and environment/source/data hashes.
MAPE is secondary and must state its denominator convention for near-zero power.
Its denominator is clamped at 0.1 kW; also report WAPE without this per-value
clamp (undefined when the total absolute target is zero).
Use all completed seeds, preserve failures/checkpoints/predictions, and distinguish
training variability from variation across chronological folds. No significance
claim is justified by overlapping forecast horizons or a single household.

Local outputs go under `results/practical_forecasting_2026-09-08/`, with fresh
directories and partial-result manifests. GPU runs execute sequentially and are
monitored per fit; a 30-minute per-invocation ceiling bounds development runs.
This phase establishes offline forecasting evidence, not production readiness.
