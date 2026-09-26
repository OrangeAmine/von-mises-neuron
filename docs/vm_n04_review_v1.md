# VM-N04 statistical and interpretation review

Date: 2026-09-22. Scope: completed confirmation under scientific protocol v1
and the disclosed storage controller v1.1. The locked scientific gate **failed**.
This review identifies no arithmetic or interpretation defect that changes that
decision. It does not certify deployment, population generality, or engineering
acceptance of the four new neural models.

The reviewer read the frozen protocol, generator, model and runner, the
storage-recovery controller and receipt, the final confirmation summary, and
the tracked [JSON](vm_n04_results_v1.json) and [complete CSV](vm_n04_cells_v1.csv).
The [main report](vm_n04_report_v1.md) owns the full scientific synthesis.
This is an agent review, not external human or blinded peer review. The reviewer
also implemented the frozen summary analyzer; the arithmetic check below was
written separately from that analyzer and is not an independently fitted study.

## Separate numerical check

Starting from the confirmation summary's individual model/room/regime/seed
records, a separate calculation formed each room-by-regime normalized one-hour
RMSE after averaging its two neural initializations. It then averaged the 24
regimes within each room and resampled the six paired rooms 10,000 times with
seed 94004. The two aggregate effects, percentile intervals and all 24 effects
for each comparator matched the tracked analysis within absolute tolerance
1e-14. The CSV's calendar-versus-none condition comparisons were checked
separately. This checks summary arithmetic; prediction-array and checkpoint
replay belong to the separate artifact audit.

| Locked learned-kernel contrast | Error reduction | Paired room 95% interval | Rooms improved | Decision |
|---|---:|---:|---:|---|
| Versus Fourier | 1.5794% | 1.2380% to 1.9099% | 6/6 | Fails the 5% practical threshold and one regime guard |
| Versus fixed kernels | -2.1045% | -2.9240% to -1.2963% | 0/6 | Fails the practical threshold and positive-interval requirement |

The Fourier guard failure is the fastest-response, regular-attendance,
stable-routine, outage regime: normalized RMSE is 0.18062058 for learned versus
0.16003522 for Fourier, a **12.8630% regression**, above the unchanged 10% limit.
Its favorable stable-routine calendar information does not erase this comparison
between basis choices. All 24 fixed-kernel guards pass, but learned is worse in
aggregate. Passing those guards cannot override the other failed requirements.
Room-level learned-versus-Fourier improvements range from 0.9111% to 2.2646%;
against fixed kernels they range from -3.5472% to -0.9052%.

Positive bootstrap intervals alone do not establish the declared practical
benefit. Likewise, the 24 paired regimes, overlapping windows and repeated
initializations do not turn six independent room draws into a larger sample.

## Conditions under which calendar information helps or fails

At the primary one-hour endpoint, all three calendar branches improve over the
matched no-calendar neural arm in **all 12 stable regimes**, and worsen it in
**all 12 shifted regimes**. The following descriptive margins average the
prespecified response times and outage conditions equally; they do not create
new acceptance criteria. Lower normalized RMSE is better.

| Routine / attendance | No calendar | Fourier | Fixed kernels | Learned kernels | Seasonal reference | Learned skill versus no calendar |
|---|---:|---:|---:|---:|---:|---:|
| Stable / regular | 0.670130 | 0.256582 | 0.268939 | 0.260147 | 0.241181 | 61.180% |
| Stable / irregular | 0.680994 | 0.518141 | 0.525680 | 0.520444 | 0.621417 | 23.576% |
| Shifted / regular | 0.676850 | 1.323177 | 1.218431 | 1.267531 | 1.306960 | -87.269% |
| Shifted / irregular | 0.682596 | 1.021481 | 0.993786 | 1.021991 | 1.201017 | -49.721% |

Calendar value and learned-locality value are separate findings. A simple
training-derived seasonal schedule has the lowest mean error among the seven
arms in the stable/regular margin. Under a changed routine, stale calendar
associations cause a large forecasting cost relative to the matched sensor
model. This does not establish that learned centers recovered attendance,
response time, or a physical mechanism.

Across all 24 regimes at one hour, learned kernels have **13.2645% higher**
normalized error than no calendar and **2.9759% higher** error than residual
ridge. They improve over persistence by 16.3208% and over seasonal by 8.9142%.
Thus beating persistence is insufficient to establish added value beyond the
other available controls. At four hours, learned improves over no calendar by
5.0673%, but fixed kernels improve by 6.2017%, and seasonal remains slightly
better than learned. This is a secondary endpoint, not a replacement primary
success. Whole-horizon RMSE uses all 16 target entries before the square root;
it is distinct from the mean of 16 endpoint RMSEs.

## Eleven statistical and methodological checks

| Check | Assessment and claim boundary |
|---|---|
| Simpson's paradox / aggregation | Stable and shifted routines have opposite calendar effects. Equal regime weighting was frozen before confirmation. The balanced mean must accompany these condition-specific results; it cannot support universal calendar benefit. |
| Ecological inference | The six synthetic room/trajectory draws are the independent units. Train/test within each room share its physical parameters. Neither their means nor their windows establish unseen-building transfer, occupant behavior, or building-population performance. |
| Selection bias / Berkson's paradox | Reserved seeds and all 24 regimes were fixed. Every test dataset retains the same 326 hourly origins across methods: six candidates lack split-local history and four lack the full future horizon. No outages or unfavorable predictions are excluded. The parameter distribution and hourly issuance remain a designed sample, not representative operating conditions. |
| Collider bias / outcome-dependent conditioning | No favorable-model outcomes define the primary sample. Boundary/event/outage subgroups are diagnostic generator labels, not predictors or selection criteria. In particular, the actual event boundary is unavailable future information for an ordinary forecaster; a diagnostic score there does not imply it can identify that subgroup in advance. |
| Base-rate and scale neglect | This is regression, not classification, but irregular attendance jointly changes absence probability, timing jitter and amplitude. The response-time sweep also changes signal variation at fixed measurement-noise SD. Reported normalization uses each training trajectory's SD; it does not eliminate these substantive differences. The actual SD range is 21.2894–123.1911 ppm, so the implementation's 1 ppm floor is inactive. |
| Regression to the mean | No room was selected for an extreme baseline error or retrained after confirmation. Fresh confirmation trajectories answer the frozen question within familiar synthetic mechanisms; they do not demonstrate generalization to an unseen domain. |
| Survivorship bias | All 288 confirmation neural fits, 1,152 neural test records, 432 reference records and 144 datasets remain. The original pilot's finalization failure is retained and disclosed; the 48 completed fits were recovered without refitting. Failed standard regression gates remain visible rather than being removed from the comparison. |
| Look-elsewhere effect | One primary endpoint and two conjunctive comparator requirements were fixed, including a 5% practical threshold and every-regime guard. All horizons and regimes remain available. Secondary margins and the four-hour improvement are descriptive and cannot replace the failed primary decision. No subgroup significance search is claimed. |
| Researcher degrees of freedom | One common learning rate was selected from original pilot validation histories; development made no scientific selection. The separately locked v1.1 change repairs NumPy scalar serialization and reconstructs already consumed pilot metadata. It preserves the model, data, training, protocol, metrics and gates. It is a disclosed orchestration deviation, not an uninterrupted successful original pilot. |
| Correlation versus causation | Paired synthetic interventions support contrasts inside this specified simulator. Fixed equilibrium amplitude across response times is a smoothing comparison, not an isolated ventilation intervention. Learned basis parameters and predictive ranking do not establish physical identifiability, energy savings, or field causality. |
| Reverse causality / temporal order | The implementation uses both event and arrival cutoffs, split-local causal filling, and known future UTC coordinates; realized future attendance and sensors are excluded from predictors. Latent and noisy simulator targets retained during outages are offline labels. Tests and artifact review support this stated contract; a valid synthetic pipeline does not validate real sensor revisions or unknown network delays. |

## Timing, boundaries and model-acceptance qualifications

- Each fitting cell has 662 training windows using 10,592 label entries but
  only 2,660 unique target timestamps; validation has 326 windows, 5,216 label
  entries and 1,316 unique timestamps. Labels used for validation are part of
  the information budget. These overlapping uses are not independent samples.
- The outage follows nominal routine time and moves with the shifted routine,
  independently of realized daily attendance and jitter. It is not
  clock-independent dropout; available missingness indicators can carry timing
  structure. `origin_outage` means any outage in the six-hour history, not
  necessarily a missing reading at the exact issuance time.
- The clock-error diagnostic changes only the calendar input in stable,
  outage-free cases. Its one-hour RMSE increases by 21.4351 ppm for Fourier,
  18.3257 ppm for fixed kernels and 20.6867 ppm for learned kernels; the
  no-calendar model is unchanged. Correcting the known offset leaves at most
  1.073e-6 normalized prediction discrepancy, consistent with recorded float32
  arithmetic. This diagnostic is distinct from changing physical routines.
- The 24-row input spans a six-hour acquisition budget, shorter than the
  slowest eight-hour response time. Split-contained windows exclude the first
  six hours following the test boundary. No claim of immediate shift detection,
  online adaptation, or robustness to arbitrary outage length follows.
- Confirmation elapsed time is 248.625 seconds; development is 85.521 seconds.
  The pilot value 107.530 seconds is a conservative reconstructed feasibility
  estimate: 43.547 seconds of wall-clock artifact span, a 60-second margin, and
  3.983 seconds of recovery work. It is not the lost original monotonic runtime.
  Original pilot reference-only timers remain unavailable. Saved confirmation
  runtimes and parameter counts are actual measurements/counts, with training
  time counted once per fitted model rather than four times per test variant.
- The scientific models have 3,920 / 3,953 / 3,953 / 4,017 trainable parameters
  for none / Fourier / fixed / learned. Equal 32-feature calendar branches
  leave 64 extra learned geometry parameters and different function classes.
  Equal epochs do not imply equal runtime or function-space regularization.
- All four new neural standard adapters pass absolute thresholds but fail
  the unchanged base-VM RMSE/MAE/MAPE regression limits. They remain
  **unvalidated** under the receiving project's acceptance rule, irrespective
  of any conditional scientific advantage. Residual ridge passes both gates.
  A finite completed comparison and a validated model are different outcomes.

## Reviewed evidence identities

The compact outputs and raw metric summary consumed by this review are bound by:

| Artifact | SHA256 |
|---|---|
| Tracked analysis JSON | `3ffb3098d4e5e0851577a5d01f057f50196e39f604665e378dd35927dff0d557` |
| Tracked cell CSV | `958bb8cb37b7ac44af94e1465bc53366aa24b4ae30f6c54366ea7710e7e350f0` |
| Confirmation summary | `927aef9db268f3bf6dc10ef2294704a255c23b9f1f43725f4e1b25e8ce3517f4` |
| Successor source lock | `069abf9d9635ef9c6b3345404e300f421ec81147997bca735cdeb5c029bd2d74` |

No frozen source, historical evidence, original checkout, owner note or sibling
project was changed by this final review. Its only new artifact is this document.
The answer remains a completed conditional synthetic finding with a failed
adaptive-locality gate, not a reason to launch an unplanned follow-up search.
