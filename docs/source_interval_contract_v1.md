# N05/N06 interface: source envelopes and controlled CO2

Material Passport: PLAN / implementation contract, 2026-09-29. This completes
next step 4's interface work. It does not launch a combined scientific campaign
or revise either original study's failed gate.

## Target alignment

| Item | Original N05 | Original N06 | This bridge |
| --- | --- | --- | --- |
| Target | CO2 concentration at a future timestamp | Source generation over a plant interval | Source path first, then CO2 conditional on a named action path |
| Units | ppm | ppm*m3/h | Source ppm*m3/h; resulting CO2 ppm; airflow m3/h |
| Point model | Persistence shared by all arms | Calendar source models | Caller-supplied frozen source forecast, e.g. N06; no new learned forecaster |
| Calibration truth | Delayed noisy CO2 | No uncertainty calibration | Delayed source labels assumed observable, synthetic only |
| Control | Passive trajectory | Each policy has its own controlled history | One explicitly specified action path per envelope |

N05's saved ppm intervals are **incompatible inputs** and are rejected. They
cannot be relabeled as source uncertainty, inverted without assumptions, or
transferred unchanged to a new ventilation policy. Reuse the finite-sample
order-statistic routine only with new residuals in source units.

## Version-1 contract

Implementation: [source_interval_bridge_v1.py](../src/source_interval_bridge_v1.py).
All timestamps are integer UTC nanoseconds. Source entries represent constant
generation on `[start + j*step, start + (j+1)*step)`. Concentration endpoints
are the right edges. Sources are nonnegative. Arrays have one explicit common
path length. Every source envelope carries a model identifier, calibration
cutoff/count, target, units and scope.

1. Receive a state interval at its **measurement event time**, with receipt no
   later than issuance. A Gaussian noise standard deviation is not a hard state
   bound. The caller must supply a defensible state interval and its assumption.
2. Forecast a source path beginning at that event time, covering the unobserved
   state-to-issuance gap and the future horizon. Historical source forecasts in
   that gap must use only information available at issuance.
3. Calibrate on archived predictions from the same frozen source model, path
   length and offsets relative to issuance. Every calibration source label must
   have arrived **after its own prediction**; only fully arrived paths can be
   used at the current issuance. Future/missing truth is not inspected.
4. Use each complete path's maximum absolute source residual as its score. With
   `k=ceil((n+1)*(1-alpha))`, use the kth score, or infinity if unavailable.
   Clip only the lower source bound to physical zero. There is no upper cap.
5. Supply total airflow for every interval and an independent immutable prefix
   from the actuator ledger: already applied intervals plus commands queued
   through the actuator delay. Reject changed prefixes, missing coverage or a
   source cadence different from the plant step.
6. Propagate source and initial-state lower/upper bounds with the unchanged N06
   plant parameters. Return only CO2 endpoints **after issuance**, tagged with
   the plan and source model identifiers. The limit classifier produces a
   conditional signal, not a command or a cost-optimal policy.

Caller-owned facts are explicit: archived predictions genuinely preceded their
labels; model identity and training cutoff are truthful; the supplied locked
prefix agrees with the actual actuator ledger; plant parameters are known;
source and state bounds are defensible. Array validation cannot prove these.

## What propagation guarantees

For fixed total airflow `Q >= 0`, volume `V > 0`, and interval duration `d`,
`C_next = C_out + (C-C_out)*exp(-Q*d/V) + S*(1-exp(-Q*d/V))/Q`.
The source coefficient has limit `d/V` at zero airflow. Both coefficients are
nonnegative, so containment of the initial state and **entire source path**
implies containment of the concentration path under that same flow plan.
The code uses `expm1` for the small-flow limit and propagates infinite upper
bounds explicitly. An unbounded envelope returns `fallback required`; this
interface does not silently substitute a finite band or invent a fallback.

This is deterministic conditional containment. A 90% empirical source band is
not a proved 90% closed-loop CO2 band. Overlapping paths, drift, model selection,
sensor uncertainty, parameter error and feedback can invalidate a naive coverage
interpretation. Marginal per-horizon N05 intervals do not imply simultaneous
source-path coverage. Calibration observations under one action history cannot
be treated as observations of another counterfactual history.

## Reproducible engineering example and remaining science

```powershell
py -3.11 scripts/demo_source_interval_bridge_v1.py --output results/source_interval_bridge_v1/example_20260929.json
py -3.11 -m pytest tests/test_source_interval_bridge_v1.py -q
```

The example uses a fixed illustrative source forecast, bounded synthetic source
residuals and a stated initial-state interval. It projects the same source band
through two action paths with identical pending commands, verifies containment
against exact trajectories and emits serializable receipts. It measures no
forecast skill, empirical coverage, field feasibility, energy saving or policy
advantage. Tests separately exercise the unchanged N06 calendar source adapter.

A later combined study still needs an agreed source-label measurement model
(or separately validated source inference), a state/parameter uncertainty model,
causal calibration pools under control, a declared fallback, comparable policy
budgets and a new prospective coverage/cost protocol. The present interface
is ready for that design discussion; it supplies no missing empirical evidence.
