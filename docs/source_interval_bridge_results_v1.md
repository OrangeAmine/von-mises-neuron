# Completed interface for uncertainty and controlled decisions

Material Passport: IMPLEMENTED AND CHECKED, 2026-09-29. This is next step 4,
target/interface alignment, not a combined N05/N06 scientific experiment.

**The bridge now accepts uncertainty about source generation, projects it through
an explicitly named airflow plan, and returns conditional CO2 bounds.** It
rejects N05 concentration intervals as incompatible inputs. The original N05
and N06 results and their failed scientific gates remain unchanged.

The [frozen interface contract](source_interval_contract_v1.md) defines units,
timestamps, delayed source labels, initial-state bounds, queued actuator actions,
calibration provenance and the precise containment claim. The implementation is
[source_interval_bridge_v1.py](../src/source_interval_bridge_v1.py); the runnable
[example](../scripts/demo_source_interval_bridge_v1.py) exports this
[tracked example receipt](source_interval_bridge_example_v1.json).

## What is implemented

| Boundary | Enforced behavior |
| --- | --- |
| Target and units | Accept `source_generation`, `ppm*m3/h`; reject concentration/ppm inputs. |
| Calibration | Use complete, already arrived source paths with the same offsets relative to issuance; future or absent labels are not read. |
| Residuals | Calibrate maximum absolute whole-path source residuals using N05's unchanged empirical order statistic. This reuses arithmetic, not N05's concentration residuals. |
| Insufficient labels | Retain an infinite upper bound and return `fallback required`. No clipping or automatic fallback policy. |
| State timing | Start from a received measurement's event time and integrate the unobserved gap to the decision time. |
| Actions | Require an explicit flow path plus an unchanged prefix of applied and already queued actions. |
| Output | CO2 endpoints after issuance, in ppm, with source-model and action-plan identifiers and conditional scope. |

The exact positive-coefficient plant solution gives deterministic enclosure
**if** the entire source path and initial state are contained, plant parameters
are correct and the specified airflow is applied. This claim does not supply
those premises or prove empirical coverage. A Gaussian measurement noise scale
is not a hard state bound. Truthful model/forecast provenance and correspondence
between the supplied locked prefix and the actuator ledger remain caller duties.

## Checks and example outcome

The full repository suite passed **1,848 tests in 108.97 seconds** after adding
46 tests across steps 3 and 4. Bridge tests cover zero/nearly-zero airflow,
finite and unbounded envelopes, delayed states and labels, whole-path calibration,
incompatible units, changed actuator queues, future-label mutation and the
unchanged N06 seasonal/VM/Fourier source predictors. Numerical trajectories from
SciPy's independent ODE solver stayed inside the projected bounds for sampled
admissible source/state paths. This supports the implementation of the monotone
propagation formula; random numerical samples do not prove its assumptions.

The example uses a fixed illustrative source prediction of 60,000 ppm*m3/h,
20 historical synthetic calibration paths, alpha 0.1, a nine-interval path
including the one-step sensor gap, and a received state interval [590,610] ppm.
It compares two plans with the same three-interval applied/queued prefix:

- Base airflow after the queue: at least one future endpoint's entire CO2 band
  exceeds 900 ppm.
- High airflow after the queue: all future endpoint upper bounds stay at or
  below 900 ppm, conditional on the stated source/state/plant assumptions.

Both example trajectories were contained. The interval classifier sends no
commands, optimizes no cost and claims no policy advantage. The illustrative
truth was deliberately chosen within the input bounds; these two containments
are **not** a measured coverage rate. The original N06 source forecasters retain
their prior diagnostic eligibility; no new learned model was introduced here.

## Ready for the next design decision

The target ambiguity is resolved for a **source-observed synthetic contract**.
A scientifically credible combined trial still needs a source observation or
inference model, a state/parameter uncertainty treatment, calibration pools that
respect the controlled history, a declared fallback and a prospective policy
cost/coverage protocol. Existing N05 CO2 bands cannot fill those gaps.

No real room, safety limit or energy-saving claim has been validated. A future
trial should keep source-path coverage, resulting conditional CO2 containment
and realized decision cost as separate outcomes. The current bridge makes those
outcomes and their assumptions explicit enough to design that trial.

The companion [N07 result](calendar_trust_report_v1.md) concerns adaptive point
forecast weighting only; it does not provide calibrated source intervals for
this bridge. The two follow-ups are not coupled by this completion.
