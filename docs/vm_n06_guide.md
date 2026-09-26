# Forecast timing and a simulated decision — N06 guide

Open [the offline demonstrator](vm_n06_demo/index.html) in a browser, or run this
from the project root and keep the terminal open:

```powershell
py -3.11 scripts/launch_vm_n06.py
```

The server binds only `127.0.0.1:8766`. Ctrl+C stops it. The HTML and its local
`data.js` also work without a server or network. There is no model download,
private data, account, sibling checkout or N05 runtime dependency.

## Start with a prediction

In **Error timing**, decide which hour-long source miss will matter more before
reading the result. Choose either forecast or both. The linked charts show the
stipulated source forecast, the controlled concentration, and applied duty.
The shaded 08:00–10:00 interval is actual attendance. Open **Explain** afterward.
Reset restores the controls; no randomness changes between button presses.

In **Useful lead time**, compare the reactive policy with two policies that know
the future source exactly. One uses only 15 minutes of that information; the
other uses two hours. Change capacity or delay, inspect the actual airflow and
exceedance together, then explain why information did or did not help. Zero
capacity is a negative control: commands change, physical outcomes do not.

In **Change the objective**, the plotted points come from the bounded study,
with each policy's primary development-selected setting held fixed. Change the
effort weight. The points stay put but the lowest-cost non-oracle arm changes
from VM to reactive. This is a retrospective illustration, not a new policy
selection on evaluation data. The oracle remains visible as a separate
information reference. The table below the lesson also reports the secondary
objective's separately development-selected settings; those are distinct from
the fixed-setting illustration. The first two lessons are authored one-day
examples; the third averages three-day outcomes across six rooms and four regimes.

## What actually changes the room?

The plant is a well-mixed 150 m3 room. Concentration C is in ppm, outdoor
concentration is 420 ppm, generation S is in ppm*m3/h, and Q is airflow in m3/h:

`V*dC/dt = S + Q*(Cout-C)`.

An assumed person generates 0.018 m3 CO2/h. Multiplying by 1e6 converts this to
18,000 ppm*m3/h. Twelve people give S=216,000 and S/V=1,440 ppm/h. These are
declared synthetic assumptions, not inferred/calibrated room parameters.

At positive Q, the equilibrium is `Cout+S/Q`, and the response rate is `Q/V`.
Both change with an action. For constant source and flow over dt=0.25 h,

`Cnext = Cout + (C-Cout)*exp(-Q*dt/V) + (S/Q)*(1-exp(-Q*dt/V))`.

The implementation uses `expm1` to avoid cancellation at small Q. If Q=0, the
exact limit is `Cnext=C+S*dt/V`. For example, from 600 ppm with twelve people,
zero flow produces 960 ppm after fifteen minutes. This is independently checked
against numerical ODE integration. N04's fixed-equilibrium response-time sweep
did not represent this controlled mass balance; its frozen code is unchanged.

Base flow is 45 m3/h. A binary actuator usually adds 405 m3/h, with a declared
delay. At maximum normal flow and twelve people, equilibrium is 900 ppm; with
only 60 m3/h extra it is 2,477 ppm. The limited actuator cannot hold that occupied
equilibrium below the study limit. Airflow/duty does not measure electricity,
thermal comfort, health protection, wear or money.

## What can each policy know?

At grid time k, state C[k] exists before interval source S[k] acts. Measurement
C[k]+noise[k] becomes available one step later. The policy sees its own latest
arrived concentration; a known initial value is used until the first arrival.
A command issued at k affects interval k+delay. An action changes the next state
and all later sensor observations. Other policies never borrow that history.

The seasonal, Fourier and VM source estimators use 14 earlier days of ideal
historical source labels and the known future UTC clock. This assumes a past
attendance counter; source is not identifiable from concentration alone in this
setup. No live or future source labels are available to them. They do not adapt
online to a changed routine. All use the same labels. VM uses 16 fixed circular
kernels and 17 fitted ridge coefficients, Fourier eight harmonic pairs and 17
coefficients, and seasonal 96 quarter-hour means. There is no learned geometry.
These are diagnostic models, not validated replacements for the repository's
forecasting architectures.

Reactive requests ventilation from its latest sensor alone. Anticipatory arms
keep that same guard and add the maximum predicted concentration over two hours.
The on trigger is `900-margin`; the off trigger is another 50 ppm lower. Between
the two, retain the previous request. There is no minimum-on or ramp constraint.
Every policy shares delay, capacity, sensing and the three-margin search budget.
The oracle uses the same rule but receives the actual future source. Perfect
information does not make this rule an optimal controller or a bound.

The forecast concentration is computed from the current policy's arrived
observation, propagating through already applied/queued flows. It assumes no
**new** future commands after the queue. An action-conditioned prediction instead
uses an explicitly supplied alternative flow path. Neither is the uncontrolled
trajectory from the start of the experiment. Closed-loop accuracy is scored only
for the action-independent **source** target; concentration outcomes are not
treated as unchanged forecast labels. A calendar-source predictor avoids the
controlled-sensor distribution shift of a frozen sensor-based concentration
model, but remains vulnerable to routine shift and uncertain source labels.

## Read the metrics together

Exceedance duration integrates time when the exact interval curve is strictly
above 900 ppm. Severity integrates `max(C-900,0)` in ppm*h. These distinguish a
brief high peak from a long small excess. Extra-air effort is the integral of
Q minus base Q in m3. On-hours, starts and switches describe actual applied
actuation, not commands waiting in the queue. The 900 ppm limit is a study
threshold; it is not presented as a health standard.

`J = exceedance_hours + lambda * additional_air_m3 / 1000`.

Lambda is explicitly chosen as 0.02 or 2 hours-equivalent per 1,000 m3. Units
make the accounting clear but do not calibrate its value to people or costs.
The trade-off plot shows actual evaluated points. Do not draw a favorable
matched comparison across an untested crossing. The one attained mean-effort
match in the study favored Fourier over VM; all failed matches are retained.

## Reproduce the bounded study

The [contract](vm_n06_protocol_v1.md), [configuration](../configs/vm_n06_v1.json),
[contract lock](vm_n06_contract_lock.json), [source lock](vm_n06_source_lock_v1.json)
and [evaluation lock](vm_n06_evaluation_lock_v1.json) record the original sequence.
The seeds are now consumed. Reusing them is reproduction, not fresh confirmation.
Use a new output namespace; runners refuse to overwrite an existing stage.

```powershell
py -3.11 -m pytest tests/ -q --junitxml=results/vm_n06/reproduction_checks/full_suite.xml
py -3.11 scripts/validate_n06_calendar_v1.py --output-dir results/vm_n06/reproduction_checks/standard
py -3.11 scripts/run_vm_n06_v1.py freeze --output-dir results/vm_n06/reproduction_study
py -3.11 scripts/run_vm_n06_v1.py development --output-dir results/vm_n06/reproduction_study
py -3.11 scripts/prepare_n06_engineering.py --study-dir results/vm_n06/reproduction_study --test-xml results/vm_n06/reproduction_checks/full_suite.xml --standard-dir results/vm_n06/reproduction_checks/standard
py -3.11 scripts/run_vm_n06_v1.py evaluation --output-dir results/vm_n06/reproduction_study
py -3.11 scripts/analyze_vm_n06_v1.py --study-dir results/vm_n06/reproduction_study --output-dir results/vm_n06/reproduction_report
py -3.11 scripts/render_vm_n06_timing.py --source results/vm_n06/reproduction_report/teaching.json --output results/vm_n06/reproduction_report/timing_steps.png
```

The standard adapter command is expected to exit **1**: all three calendar
concentration adapters failed the base-VM regression gates. Its completed JSON
is still evidence; do not edit thresholds or refit for a pass. The unchanged
base-VM comparison uses a verified inherited standard result rather than a new
training run. Full-suite completion and model acceptance remain separate checks.

The study cap is 180 seconds across development and evaluation, versus an
observed 8.14 seconds. The separate standard diagnostic cap is 120 seconds.
Generation, fit, rollout, array and partial-record writes are included in study
timers; final-summary serialization, Python startup, tests, report rendering and
UI inspection are separate. Fitting
uses float64 CUDA when available; scalar integration and plotting use CPU.
No packages or global Python installations were changed.

Full arrays and fitted readouts are under this worktree's ignored
`results/vm_n06/study_v1/`. Compact tracked artifacts include
[all achieved means/SDs](vm_n06_achieved_points_v1.csv),
[selected room-level results](vm_n06_selected_rooms_v1.csv),
[paired effects and every match failure](vm_n06_results_v1.json),
[teaching arrays](vm_n06_teaching_v1.json) and
[acceptance outcomes](vm_n06_standard_acceptance_v1.json).

The optional N05 boundary is deliberately deferred: this delivery takes point
source forecasts only. A later interval extension must declare target, issuance,
availability and action assumptions, verify provenance/calibration eligibility,
and freeze a new policy comparison. Concentration intervals tied to an
uncontrolled target cannot silently become action-conditioned predictions.
