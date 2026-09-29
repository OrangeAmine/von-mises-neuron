# Von Mises inspired forecasting experiments

Research code for circular feature layers and time-series forecasting in PyTorch.
The current models combine trigonometric encoders with an LSTM or pooling. The
temporal variant conditions sensor phases on calendar angles and optionally adds
depthwise convolution/residual bridges.

Start with [the audited project state](docs/project_state.md) and
[the controlled comparison protocol](docs/controlled_comparison_plan.md).
Historical design documents contain hypotheses, not established results.

The six subsequent studies **N01–N06 are complete and integrated**. Start with
the [consolidated research account](docs/research_account_n01_n06.md) or
[interactive findings explorer](learning/vm_playground/research.html).
N01, N04, N05 and N06 failed their overall scientific gates; N02 has mixed
claim-level results, and N03 is teaching implementation. The
[original lessons](learning/vm_playground/index.html) and
[decision demonstrator](docs/vm_n06_demo/index.html) are available offline.
See the [integration record](docs/integration_20260929.md) for current checks.

The completed **VM methodology** objective separates the effects of periodic
encoding, polar optimization, learned circular kernels and angular interactions.
The [approved methodology v1](docs/vm_methodology_results.md) and the
[circular-locality v2 follow-up](docs/vm_locality_v2_results.md) are implemented
and completed through locked confirmation and independent audits. Both overall
continuation gates failed; the reports retain the conditional gains, regressions
and limitations. V2 completed 5,504 records; its historical repository suite passed
1,262 tests. See the [saved literature review](docs/vm_literature_review.md),
[v1 usage guide](docs/vm_methodology_usage.md), and
[v2 protocol](docs/vm_locality_v2_protocol.md) for context and reproduction.
Raw arrays and checkpoints under `results/` remain local and ignored; compact
statistics, configuration and provenance are saved under `docs/`.

The completed **real-world forecasting** direction used the
[practical protocol](docs/practical_forecasting_protocol.md) to compare persistence,
daily/weekly seasonal forecasts, residual ridge, and persistence-initialized
neural models on chronological household-power periods. Architecture identity
does not determine model selection.
The [completed practical evaluation](docs/practical_forecasting_results.md)
selected residual ridge: reserved-period RMSE was 0.783 kW versus 0.934 kW for
daily seasonal persistence. Negative predictions and substantial remaining error
limit operational use; this is evidence from one historical household.

## Run locally

Use Python 3.11 in an isolated environment and install `requirements.txt`.
On the audited Windows machine, `py -3.11` already has the working PyTorch/CUDA
environment; bare `python` selects a different installation without PyTorch.
Scripts select CUDA when available and otherwise use CPU.

```powershell
py -3.11 -m pytest tests/ -q
py -3.11 scripts/benchmark_standard_dataset.py --help
py -3.11 scripts/benchmark_standard_dataset.py --output-dir results/my-benchmark
py -3.11 scripts/controlled_temporal_comparison.py --help
py -3.11 scripts/controlled_temporal_comparison.py --calendar-clock local --output-dir results/my-local-comparison
py -3.11 scripts/controlled_temporal_comparison.py --calendar-clock fixed --output-dir results/my-fixed-comparison
```

Use a fresh output directory for each experiment. The controlled comparison
runner refuses an existing directory, saves individual runs, predictions and a
manifest, and reports all three training seeds. Its completion does not mean
every ablation passed: inspect per-model absolute gates and temporal regression.
The standard benchmark enforces its applicable acceptance gates in the exit
status and JSON. `--skip-baselines` runs base VM and TemporalVM only.

The controlled runner defaults to `--calendar-clock local` for historical
reproduction. `fixed` uses UTC+01:00 throughout, preserving winter training
phases and removing the synthetic test's DST clock mismatch. The choice is
recorded in the manifest; it preserves observations, absolute timestamps, and
elapsed-time hand-crafted features. Keep both protocols explicit; see the
[clock audit](docs/synthetic_clock_audit.md).

## Practical power evaluation

The cache-only runner requires the extracted household-power data under
`data/raw/power/extracted/`. It uses hourly observations available at each bin's
right edge, at least 45 minute readings per sensor/hour, and complete 168-hour
input histories to forecast the next 24 hours. It fits normalization on training
data only and rejects gaps without interpolation.

```powershell
py -3.11 scripts/validate_residual_forecasters.py --output-dir results/my-practical/synthetic
py -3.11 scripts/practical_forecasting.py --prepare-only --output-dir results/my-practical/preparation
py -3.11 scripts/practical_forecasting.py --synthetic-validation results/my-practical/synthetic/validation.json --output-dir results/my-practical/development
py -3.11 scripts/practical_forecasting.py --phase final --selection-dir results/my-practical/development --output-dir results/my-practical/final
```

The synthetic validator exits nonzero if any candidate fails; its explicit
per-candidate eligibility controls which neural families reach development.
Inspect that artifact even after a failed overall exit. The frozen practical
protocol requires seeds 42/43/44 and a maximum 50 epochs. The final command
requires completed development and unchanged implementation, data, protocol,
packages and training settings. It refits only the selected family and compares
it with all simple controls. A completed run is not itself evidence of improved
forecasting: inspect `selection.json` and `final_assessment.json`.

To reproduce the selected final forecast on an hourly sensor CSV:

```powershell
py -3.11 scripts/predict_practical_forecast.py --final-run results/my-practical/final --input-csv hourly-history.csv --output-dir results/my-forecast
```

Supply a `timestamp` column and all seven original sensor columns, in physical
units, with at least 168 consecutive hours. Timestamps must be naive recorded
clock values marking hourly-bin availability. Input means must already satisfy
the [coverage contract](docs/practical_forecasting_data_contract.md); the CSV
interface cannot establish minute-level coverage. The command uses stored
training normalization and the selected model, writes 24 forecast timestamps
plus hashed provenance, and rejects incomplete histories or changed model code.
It applies no clipping. Local audited example outputs are under
`results/practical_forecasting_2026-09-08/inference_example/`.

## Interpretation

For a log-output layer,
`kappa*cos(x+mu) = a*cos(x) + b*sin(x)`, with `a=kappa*cos(mu)` and
`b=-kappa*sin(mu)`. The matched SinCos model tests Cartesian versus polar
parameterization with equal architecture and parameter count. This identity
does not make stacked layers linear in the original inputs.

Frozen kappa bands in the temporal log-output bank scale amplitude; they do
not change frequency or force fast/slow specialization. Timestamp angles supply
the periods. Calendar day/week/year angles use local wall time; custom periods
use elapsed time from a fixed epoch or an explicitly shared origin. Annual
features now use each Gregorian year's actual length.

Core VM layers are periodic in their angular inputs. Composed models need
separate checks: the temporal model's raw-sensor residual retains nonperiodic
sensor information, while its calendar-angle path remains periodic. Real-valued
CO2 and temperature measurements should not be confused with physical angles.

## Repository and evidence

- `src/`: layers, models, data loading, preprocessing and sequence construction.
- `scripts/`: synthetic and real benchmarks, diagnostics and research tools.
- `tests/`: mathematical, model and pipeline checks.
- `docs/`: tracked plans, findings and handoff state.
- `data/raw/` and `results/`: ignored local data and generated artifacts; these
  do not travel with a fresh clone. Reports retain concise evidence in `docs/`.

No general superiority, novelty or LSTM replacement is established by a single
benchmark. For model changes, keep the standard absolute thresholds unchanged
and report failed variants. Presentation scripts additionally need
`python-pptx`, which is not part of the core requirements.
