# Six completed studies: what the project now supports

Integration edition, 29 September 2026. This account synthesizes the completed
VM-N01–N06 deliveries. It introduces no new experiment, pooled significance
test, threshold, or model selection. The original protocols and reports remain
the authoritative records.

## The result in context

Circular structure is useful when it matches the prediction problem. These
deliveries do not establish a general advantage for learned VM geometry.
They instead separate four questions: whether a representation fits well,
whether its parameters identify something, whether predictions remain useful
when routines change, and whether those predictions improve a decision.
Success on one question does not answer the others.

All six deliveries are complete. **N01, N04, N05 and N06 failed their overall
scientific gates. N02 has claim-specific outcomes, not an overall gate. N03 is
a teaching resource, not an efficacy experiment.** Completion, engineering
acceptance and scientific success must remain separate labels.

The earlier [methodology v1](vm_methodology_results.md) and
[locality v2](vm_locality_v2_results.md) findings remain unchanged. In particular,
locality v2's large average gains coexisted with a 47.58% smooth-family regression
against fixed16 and a failed overall gate. N01 follows that question; it does
not replace the earlier result.

## Evidence across the six deliveries

| Delivery and question | Retained result | Decision and boundary |
| --- | --- | --- |
| [N01: smooth Fourier backbone plus local kernels](vm_n01_results.md) | The selected joint-Adam hybrid reduced balanced normalized clean MSE by **19.21%** against Fourier16 (paired 95% interval **8.12–30.59%**), **56.34%** against the fixed hybrid, and **40.40%** against Fourier36. | **Overall gate failed:** smooth error rose **11.97%** against the fixed hybrid, beyond the unchanged 10% guard. Fourier36 was stronger at 256 labels. |
| [N02: phase and width interpretation](vm_n02_interpretation_results.md) | All **36/36** isolated-peak fits met recovery criteria under full support, the correct family and training-derived local initialization. The redundant model had mean clean NRMSE **0.05073**, but only **26/36** fits individually met the accuracy criterion. | **Claim-specific results:** redundant responses failed the across-initialization stability criterion (**0.07209 > 0.05**); centers and widths were unstable. Proxy reliance did not establish a causal mechanism. |
| [N03: learning playground](learning/vm_n03_completion.md) | Three resettable lessons teach polar/Cartesian equivalence, amplitude versus locality, and evidence interpretation. All **720** historical locality-v2 cells remain inspectable. | **Engineering delivery:** no new scientific comparison or learner-outcome study. The integration adds a separate findings explorer without revising the frozen v2 export. |
| [N04: changing routines and physical dynamics](vm_n04_report_v1.md) | Learned calendar geometry improved the primary normalized RMSE by **1.58%** against Fourier (**1.24–1.91%**), but was **2.10% worse** than fixed geometry. | **Overall gate failed:** the Fourier gain missed the 5% requirement; one fast, regular, stable/outage regime regressed **12.86%**, beyond its 10% guard. Calendar benefits depended on routine alignment. |
| [N05: phase-dependent uncertainty](vm_n05_report_v1.md) | VM-scaled intervals improved the primary one-hour interval score by **7.50%** against Fourier, with room-level 95% interval **−4.34 to +17.43%**. Only **4/8** rooms favored VM. | **Overall gate failed:** room-level uncertainty crossed zero and noisy-condition coverage was **83.70% / 83.32%**, below the 85% guards. Ordinary rolling calibration had a better score and narrower intervals overall. |
| [N06: forecasts used for ventilation decisions](vm_n06_report_v1.md) | Under the low-effort-penalty objective, VM improved the objective by **45.74%** against reactive control and **5.34%** against seasonal control. | **Overall gate failed:** the matched Fourier contrast was only **2.73%** (**−0.07 to +6.15%**), below 5% and crossing zero. More air was used than by reactive control; this is not an equal-effort saving. |

Percentages have different denominators and metrics across studies; they should
not be ranked or averaged together. Intervals above retain each study's own
paired analysis. Independent units are generated functions in N01, generated
cases in N02, and rooms in N04–N06; repeated fits, scenarios, timestamps and
rollouts are not additional independent rooms.

## What the findings change

**Local adaptation remains conditional.** N01 provides evidence for combining
smooth and local structure at the tested small data budgets. Its smooth guard
still failed: the actual smooth errors were 0.00204512 versus 0.00182657, a
small absolute difference with a disqualifying relative change under the agreed
rule. Joint Adam did better than variable projection under the recorded time
budgets; this is not a universal optimizer result or an equal-update comparison.
The experiment used six confirmation functions per family, one noise level and
two label budgets. Larger Fourier capacity was competitive or better when more
labels were available.

**Prediction quality does not make components identifiable.** N02's isolated
success depends on its deliberately identifiable setup. Splitting a component
or adding cancelling components can preserve the same response with different
parameters. Even response stability failed in the redundant case. In the proxy
case, aligned NRMSE was 0.14585, which failed its fixed 0.1 accuracy criterion;
shifted NRMSE was 1.34867. The known approximate noise floor explains context,
but does not turn the failure into a pass. Likewise, the isolated doubled-kappa
effect of 0.0999106 narrowly missed its fixed 0.1 effect threshold. Parameter
recovery and deletion sensitivity do not by themselves identify a physical cause.

**A calendar is a source of information whose reliability can change.** In N04,
the learned-calendar model improved over no calendar by about 42.23% in stable
routines, then was about 68.42% worse under the three-hour test shift; other
calendar arms showed the same qualitative failure. The data were synthetic
rooms with sensor delay and outages. These results concern CO₂ prediction,
not measured occupancy recovery or validation in buildings.

**Uncertainty needs the appropriate replication unit.** N05 held the point
forecast fixed at latest-arrived persistence for every arm. Its VM basis had
fixed centers and concentration; only the scale readout was learned. It did
not test adaptive phase/width geometry. The conditional fixed-interval block
bootstrap gave a positive interval, but omitted refitting uncertainty and
cannot replace the room-level interval. At the primary one-hour horizon, mean
VM coverage was 85.85% with width
284.99 ppm, versus rolling calibration's 89.89% and 199.72 ppm. The nominal
90% target is not a coverage guarantee under these dependent, changing data.

**Forecast errors acquire value through the decision and its constraints.** N06
uses an exact controlled mass-balance recurrence. Its fixed-kernel ridge
estimator predicts exogenous source, using ideal historical source labels
available only in the synthetic setup. It is neither the N04 neural forecaster
nor N05's interval model. Under
`J = exceedance hours + 0.02 × extra air / 1000 m³`, VM used 4643.9 m³ of extra
air with 3.217 exceedance hours; reactive control used 2194.4 m³ with 6.056 hours.
Routine shifts removed the benefit, capacity limits defeated even the
perfect-source heuristic, and a larger effort penalty made reactive control
the best non-oracle policy. That oracle is a reference policy, not an optimal
bound. No energy saving, field efficacy or optimal-control claim follows.

## Engineering and evidence status

The N01 forecasting adapter passed its standard absolute acceptance checks:
RMSE 5.15322 ppm, MAE 4.08604 ppm, R² 0.98053 and MAPE 0.67904%. This does not
rescue its separate static scientific gate. N04's four neural adapters passed
absolute thresholds but failed the comparison with base VM; they remain
**unvalidated under the full acceptance contract**. N04 residual ridge passed
both. N06's new calendar concentration adapters likewise failed base-VM
regression checks and remain diagnostic/unvalidated. N05 introduced no new
point forecaster. The original task reports retain their benchmark evidence.

Historical task-level test counts are not additive. Fresh checks of the combined
repository are recorded in the [integration report](integration_20260929.md).
They check compatibility and preservation; they do not reproduce the campaigns.
N02's documented process-monitor amendment and N04's storage-recovery amendment
remain part of the experimental history.

Before integration, all six working source trees and their ignored result
payloads were archived with hashes. Exact task bytes were imported into the
main checkout, duplicate N04 dependencies were verified equal, and original
task checkouts were left intact. The tracked
[preservation receipt](integration/preservation_20260929.json) identifies source
revisions, imported paths and archives. Full arrays remain ignored local
artifacts, now also available under the main checkout's `results/`; they do not
travel with a fresh clone. Compact reports, protocols and metrics do.

## Reading and teaching route

Start with the [playground](../learning/vm_playground/index.html) for the exact
identities. Use the [findings explorer](../learning/vm_playground/research.html)
to compare each retained gain with its limiting evidence. It reads saved
results; changing a view does not train a model or change a scientific gate.
The [offline companion](../learning/vm_playground/guide.html) and
[N06 decision demonstrator](vm_n06_demo/index.html) connect representations,
uncertainty and decision timing.

The next research choices remain open for discussion: testing when to trust
calendar information, and resolving the uncertainty/decision target contract
before combining N05 and N06. Neither follow-up was started by this integration.
