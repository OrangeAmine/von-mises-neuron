# N07: adaptive trust in calendar forecasts

Material Passport: PLAN; prospective, agent-authored on 2026-09-29 under the
owner's instruction to do next steps 3 and 4. This is a new synthetic study,
not a revision of any N01–N06 result or an external preregistration.

## Question and scope

Can feedback from already received observations reduce reliance on a stale
calendar while retaining its benefit during a stable routine? Test a convex
mixture of the unchanged N04 residual ridge and a training-only seasonal table.
There is no new VM layer and no claim about VM superiority. The unchanged N04
generator supplies paired stable and +3-hour shifted physical routines, with
regular attendance, a two-hour time constant, 15-minute packets delayed by one
step, and identical training, validation and noise streams within each room.
See [the fixed configuration](../configs/calendar_trust_v1.json).

## Information and models

Use chronological 28/14/14-day train/validation/test splits, six hours of sensor
history, hourly origins, and 15-minute to four-hour endpoints. Fit normalization
and the 96-bin UTC seasonal table on received training observations only.
Select the sensor ridge penalty using validation MSE over all horizons; do not
refit on validation. Controls are sensor-only, seasonal-only and a fixed 50:50
mixture. All have the same component fits and available observations.

For each horizon and expert, initialize a loss state with its normalized
validation MSE. These are historical hindcasts from a model selected using the
whole validation set, not prospective validation performance. At the first test
origin all initialization labels and model-selection information must have
arrived. Then update only when an earlier issued forecast's target observation
arrives: L <- (1-a)L + a(error/train_std)^2, a=1-2^(-1/24). The half-life is
24 feedback updates, not elapsed hours when packets are missing. Set weights to
softmax(-eta L); missing labels cause no update. Each horizon has its own state.
Tied arrivals have deterministic target/origin order. No model coefficients are
updated. Receipt times, not offline test truth availability, govern feedback.

Select exactly one eta from 0.5, 2, 8 using the mean one-hour normalized RMSE
over the two development rooms and both regimes; break ties toward lower eta.
Freeze that choice before generating any confirmation room. Confirmation has
12 fresh, disjoint room seeds. No further settings are selected from confirmation.

## Estimands and unchanged success rule

The primary unit is the room, paired across regimes and all methods. Average
the stable and shifted one-hour RMSE after division by the room's training
standard deviation. For each of three controls, report the mean room-level
percentage gain and a paired percentile 95% bootstrap interval, resampling
rooms (10,000 draws, fixed seed). Success requires **all three** mean gains at
least 5% and all three lower endpoints above zero. Also require adaptive RMSE
no more than 10% above seasonal on stable data and no more than 10% above
sensor-only on shifted data, using regime-level mean normalized RMSE. An
intersection gate is used; individual intervals are descriptive, not simultaneous.

Report ppm RMSE/MAE, 15-minute/four-hour diagnostic endpoints, first-48-hour
versus later performance and calendar-weight traces without selecting new
claims from them. Overlapping windows/time steps are not independent samples.
Synthetic room variation and 12 units limit precision and generalization.

## Engineering and execution

Before confirmation, run the full repository suite and a fresh standard
90-day/seed-42 synthetic benchmark. Use its unchanged 72-step history,
12-step horizon, chronological 80/10/10 split and base VM 50-epoch trainer.
Adapt the same residual-ridge/seasonal/mixing algorithm to these hourly data;
all new mixtures must meet RMSE <=35, MAE <=25, R2 >=0.20, MAPE <=5%, and
base-VM regression limits (errors <=10% worse, R2 drop <=0.10). A failed
engineering gate is retained and the algorithm called diagnostic/unvalidated;
confirmation can still describe the fixed hypothesis, not confer validation.
Online mixtures receive observed test labels causally; fixed controls do not
update parameters. This difference is the intervention, not equal adaptation.

Use `py -3.11`, CUDA for fresh base-VM training when available, and the unchanged
CPU closed-form ridge implementation. No dependencies are added. From the
repository root:

```powershell
py -3.11 scripts/run_calendar_trust_v1.py freeze --output results/calendar_trust_v1/run_20260929
py -3.11 scripts/run_calendar_trust_v1.py development --output results/calendar_trust_v1/run_20260929
py -3.11 scripts/validate_calendar_trust_v1.py --output results/calendar_trust_v1/run_20260929
py -3.11 scripts/run_calendar_trust_v1.py confirmation --output results/calendar_trust_v1/run_20260929
py -3.11 scripts/run_calendar_trust_v1.py audit --output results/calendar_trust_v1/run_20260929
```

The validation command runs and saves the full test suite before fitting the
fresh base VM. Its nonzero exit code also discloses a completed acceptance
failure; it must not be interpreted as permission to retry or alter thresholds.

Freeze configuration, protocol, source hashes and a source ZIP before development.
Each stage creates a new directory and refuses overwrite. Record source revision,
environment, fit parameters/counts, selected penalties, label/window counts,
arrival/target/issue arrays, forecasts, feedback counts, weights and timings.
Development/confirmation/engineering caps are 180/600/900 seconds. Monitor
process output and liveness at intervals <=30 seconds; preserve errors and do
not silently restart a formal stage. Recompute confirmation metrics from saved
arrays and verify hashes; this is same-agent artifact replay, not independent
scientific review. Keep compact results tracked because raw results are ignored.

## Interpretation

Online aggregation is an established family of methods; see Wintenberger,
[Optimal learning with Bernstein Online Aggregation](https://arxiv.org/abs/1404.1356).
This simple discounted-error softmax is **not BOA** and inherits no BOA guarantee.
No field validation, calibration guarantee or causal inference is intended.
Review all eleven ARS statistical fallacies in the final evidence record.
