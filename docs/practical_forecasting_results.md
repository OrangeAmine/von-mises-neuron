# Practical household-power forecasting results — 2026-09-08

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: autonomous research engineering for useful real-world forecasting
- Verification Status: VERIFIED for the recorded offline runs and artifact audits
- Version Label: practical_forecasting_v1; one household, not deployment validation

**Residual ridge passed the frozen development rule and the reserved final
test.** Final RMSE was **0.78274 kW**, a **16.18% reduction** from the best final
simple control, daily seasonal persistence at **0.93386 kW**. This supports a
useful RMSE improvement in this evaluation. It does not establish an acceptable
error level for a real decision or a unique benefit from VM layers.

The [protocol](practical_forecasting_protocol.md) was frozen before new scores.
The [data contract](practical_forecasting_data_contract.md) records raw-byte
provenance, coverage, split dates and exclusions. The final period was scored
once, after development selected ridge; neural models were not evaluated on
that reserved period. No thresholds, dates or model settings were changed in
response to the results.

## Implementation and validation

New persistence-residual wrappers predict a correction to the latest observed
target. Each sensor history is centered on its own last observation; calendar
angles remain unchanged. A zero output head makes the initial forecast exactly
persistence. The batched trainer includes that epoch-zero checkpoint in
validation selection and retains explicit numerical failures.

The cache-only power pipeline labels hourly means by their right-edge
availability, requires at least 45 minute readings per sensor/hour, retains the
hourly grid and rejects incomplete windows. It uses seven observed sensors and
two raw calendar angles, a 168-hour lookback and a 24-hour horizon. No future
sensor values, interpolation or test-fitted normalization enter training.

All new neural candidates first received the unchanged standard synthetic
absolute gates: RMSE <= 35, MAE <= 25, R2 >= 0.20, MAPE <= 5%. Seed 42, maximum 50
epochs, original 90-day/72-hour lookback/12-hour horizon benchmark:

| Model | RMSE (ppm) | Absolute gates | Base-VM regression requirement |
| --- | ---: | --- | --- |
| Fresh base VM reference | 5.35727 | Pass | Reference |
| Residual LSTM | 7.02715 | Pass | Not applicable |
| Residual SinCos | 6.98054 | Pass | Not applicable |
| Residual TemporalVM | 6.06301 | Pass | **Fail; excluded from power trials** |

Temporal residual's RMSE/MAE/MAPE regressions were +13.17%/+16.36%/+16.53%,
exceeding the unchanged 10% allowance. Its R2 drop of 0.00591 passed. The validator
correctly exited 1 and recorded candidate eligibility true/true/false; this was
not treated as a fully passing benchmark. All three residual candidates had
higher RMSE than the fresh base VM in this check. Historical configurations use
different inputs or training settings, so this is not a matched estimate of the
residual wrapper's effect.

The final code verification completed **533 tests in 25.10 seconds**, including
causality, coverage, residual gradients, epoch-zero selection, failure
preservation, selection integrity, inference artifacts and independent audits.
Development and final commands both exited 0 with all fits finite.

## Development: selection includes robustness across periods

Three-seed neural ensembles use seeds 42/43/44 with equal weights. Each neural
fit used at most 50 epochs, Adam with learning rate 0.001, weight decay 0.0001,
batch size 64, gradient clipping at 1 and validation early stopping patience 15.
Ridge considered only the five prespecified penalties and persistence on
validation. All four ridge fits selected alpha 1000.

| Family | Feb 2008 RMSE | Aug 2008 RMSE | Feb 2009 RMSE | Pooled RMSE | Frozen eligibility |
| --- | ---: | ---: | ---: | ---: | --- |
| Persistence | 1.37053 | 0.31619 | 1.17112 | 1.02723 | Simple control |
| Daily seasonal | 1.10547 | 0.35307 | 0.97311 | 0.84889 | Simple control |
| Weekly seasonal | 0.99039 | 0.45743 | 0.90289 | 0.79567 | Best pooled simple |
| **Residual ridge** | **0.81437** | **0.30055** | **0.79118** | **0.64793** | **Selected** |
| Residual LSTM ensemble | 0.78252 | 0.37793 | 0.69692 | 0.62949 | Fails summer regression |
| Residual SinCos ensemble | 0.79331 | 0.43478 | 0.71288 | 0.65291 | Fails summer regression |

All errors above are kW. Pooled RMSE is computed from total squared error and
target count, not the unweighted mean of fold RMSEs. Development origins number
649/649/271; the third development block loses coverage to gaps. There are 37,656
overlapping origin/horizon pairs covering 1,661 unique target hours.

Ridge improves pooled RMSE by 18.57% versus the best pooled simple control and
passes every fold's requirement to stay within 10% of that fold's best simple
RMSE. Although LSTM has the lowest pooled error, its summer RMSE is 19.52% worse
than persistence; SinCos is 37.51% worse. Neither qualifies under the declared
rule. **An RMSE gate does not protect MAE:** summer ridge MAE of 0.18170 kW is
worse than persistence at 0.10956 kW, despite its slightly better RMSE.

Each LSTM has 44,836 parameters; each SinCos model has 44,568. The selected ridge
has 1,180 features and 28,344 fitted coefficients/intercepts for 24 horizons. Three
neural seeds mean three model states at ensemble inference. Development completed
18 neural fits and 3 ridge fits in 326.55 seconds wall time. Selected neural
epochs, ordered by development fold and then seed, were LSTM 40/32/39, 2/6/4,
18/19/24; SinCos 34/35/33, 5/5/9, 14/31/17.
Seed variability and prediction levels are retained in the independent audit;
ensemble performance should not be confused with an average individual model.

## Reserved final period: 2010-10-29 through 2010-11-26 exclusive

Training/validation/evaluation retain 3,678/533/649 windows. Every final forecast
uses only observed context; the 649 origins cover 672 unique future hours, or
15,576 overlapping forecast cells. The selected family was refitted once using
the reserved period's preceding 180-day training and 28-day validation intervals.

| Forecast | RMSE (kW) | MAE (kW) | R2 | WAPE (%) | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | 1.17584 | 0.89847 | -0.95125 | 74.98 | 120.44 |
| Daily seasonal | 0.93386 | 0.62375 | -0.23080 | 52.05 | 73.74 |
| Weekly seasonal | 1.01375 | 0.72367 | -0.45039 | 60.39 | 85.22 |
| **Selected residual ridge** | **0.78274** | **0.58158** | **0.13532** | **48.53** | **75.98** |

Ridge clears the prespecified 5% final RMSE margin and improves final MAE by 6.76%
versus daily seasonal. Its MAPE is worse. MAPE divides by
`max(abs(target), 0.1 kW)`; WAPE divides summed absolute error by summed absolute
targets, without that per-value clamp. Point error remains substantial.
Ridge's horizon 1/12/24 RMSE is 0.61674/0.79922/0.77717 kW. All 24 per-horizon
errors are retained in JSON.

The final ridge forecast averages 1.16000 kW against observed 1.19835 kW, with
mean error -0.03835 kW. **0.976% of its final predictions are negative**, with a
minimum of -0.31273 kW. Development ridge negative fractions were
5.252%/2.106%/1.338%, in fold order. These statistics count origin/horizon pairs,
so repeatedly forecast target hours receive repeated weight.
Forecasts were not clipped or reselected after inspecting outcomes. Physical
constraints need a separately specified and validated policy before operational
use. No prediction intervals, operational cost model or retraining policy have
been established.

The final invocation took 16.38 seconds including loading/provenance; ridge fitting
and scoring took 1.66 seconds. The device was an NVIDIA GeForce RTX 3060 Laptop
GPU for neural training, CPU for ridge. Runtime is local evidence, not a general
performance guarantee.

## Reproduction, artifacts and next decision

Local ignored evidence lives under `results/practical_forecasting_2026-09-08/`:
`synthetic_validation/`, `preparation/`, `development/`, `final/`, prediction
archives, saved states/coefficients, complete training histories, source ZIPs,
data/code hashes, and `development_independent_audit.json` /
`final_independent_audit.json`. See the [README commands](../README.md).

Independent audits recomputed saved forecast metrics, verified common target and
origin arrays, exact equal-seed ensembles, checkpoint-selection histories,
finite checkpoint contents, source/protocol hashes, and raw-byte fingerprints.
They separately reconstructed all fold training normalization statistics from
raw rows. These checks do not claim full forward replay of every neural
checkpoint. The selected final model additionally has an offline CSV inference
command, which applies saved normalization and reconstructs the frozen model.
Replaying the first archived final forecast from its 168-hour CSV history
matched all 24 target timestamps exactly, with maximum prediction difference
**4.44e-16 kW**. This checks that forecast's inference reconstruction; it is not
an additional evaluation period or proof of the CSV's raw-minute coverage.

Environment: Python 3.11.4, torch 2.5.1+cu121, NumPy 1.26.4, pandas 2.3.3,
SciPy 1.17.0, scikit-learn 1.8.0, pytest 9.0.2. Base revision remains 90dc9f2 with
uncommitted implementation captured in each source snapshot. Protocol SHA256:
`10dbb51cbc3e6c7d0989708010e7096969bb4f275160baf5305d08000335008b`.
No dependency installations, global Python changes, raw-data edits or user-note
edits were made; the directly used threadpoolctl dependency was declared.

The forecasting direction is now **robust simple and residual forecasts,
evaluated against operationally relevant controls**. Keep ridge as the current
reference; the evidence does not favor retaining VM-specific architecture for
its own sake. Further useful work needs the owner's actual target, required
horizon and decision/error costs. A single historical household, overlapping
forecasts, missing-data exclusions and one reserved period do not establish
generalization or production readiness. Do not tune another model on this
already-inspected final block and call that a fresh confirmation.
