# Reproducing and consuming VM-N05

Use Python 3.11 and the existing requirements, from this repository root. No
sibling checkout, external dataset, network service or N06 code is required.
The N04 generator is imported locally. `docs/vm_n05_dependency_receipt.json`
separates the 156 inherited dependency entries from the new N05 files.

## Reproduce the bounded study

Choose fresh output directories. Every stage refuses to overwrite its outputs.
The engineering command runs the full repository suite once and writes its
JUnit log, exact source snapshot and environment receipt. There is no new point
forecaster and no rerun of the unchanged standard benchmark.

```powershell
py -3.11 scripts/run_vm_n05_v1.py --help
py -3.11 scripts/run_vm_n05_v1.py engineering --output-dir results/vm_n05/reproduction/engineering
py -3.11 scripts/run_vm_n05_v1.py fixture --output-dir results/vm_n05/reproduction/fixture
py -3.11 scripts/run_vm_n05_v1.py freeze --output-dir results/vm_n05/reproduction/study --engineering-dir results/vm_n05/reproduction/engineering --fixture-dir results/vm_n05/reproduction/fixture
py -3.11 scripts/run_vm_n05_v1.py run --output-dir results/vm_n05/reproduction/study
py -3.11 scripts/run_vm_n05_v1.py analyze --study-dir results/vm_n05/reproduction/study --output-dir results/vm_n05/reproduction/analysis
```

The published N05 seeds are consumed after this delivery. These commands
reproduce that evidence; new scientific selection requires a separately frozen
version and new seeds. The CLI validates exact source/config bytes against its
lock before reserved generation or analysis. A failed formal run retains its
progress and cannot be resumed or silently rerun in the same directory.
Never change the frozen protocol to make a failed scientific gate pass.

Files under `results/` are ignored. The study stores one factor JSON per room,
one row-array NPZ and one interval CSV.gz per room/scenario, and hashes for all
payloads. The analysis verifies those hashes before reading intervals. Compact
tracked artifacts preserve the conclusion without requiring the local arrays.
The full local run also retains label budgets, exclusions, generator metadata,
timing and environment. Independent rooms—not rows or interventions—are the
primary replication units.

## Interval artifact contract v1

The compact [sample](vm_n05_interval_contract_sample_v1.csv) contains the first
day of evaluation for the first room in the stable condition, plus four origins
with an outage target from the shifted/noisy/outage condition. It carries all
three horizons and all six arms, with matched point predictions. It is a
documented example, not a separate evaluation set or a deployment feed.

| Field | Meaning |
| --- | --- |
| `room_seed`, `scenario` | Independent room identity and paired intervention; scenario is reporting metadata, never an input |
| `arm` | `static`, `static_matched`, `rolling`, `fourier`, `hour_bin`, or `vm` |
| `issue_time_utc` | ISO 8601 issuance, after processing that instant's arrivals |
| `target_time_utc` | ISO 8601 target event time, issue + horizon |
| `observation_available_time_utc` | Actual arrival of the target measurement; empty if no packet ever arrives |
| `horizon_minutes` | 15, 60, or 240 elapsed minutes |
| `point_model_id` | Frozen `latest_arrived_persistence` definition |
| `point_ppm` | Same point for all calibration arms at the same origin/horizon |
| `lower_ppm`, `upper_ppm` | Symmetric interval bounds, with no physical-support or width clipping |
| `units` | `ppm` for target, point and interval endpoints |
| `calibration_cutoff_utc` | Last permitted information instant; at or before issue |
| `calibration_count` | Number of finite matured residuals actually used at this horizon |
| `source_id` | SHA256 of the complete frozen source-lock JSON |
| `config_id` | SHA256 of the exact canonical config bytes |
| `truth_ppm_offline` | Offline scoring-only simulator observation, including hidden outage truth; **never an issuance input** |

CSV uses UTF-8, a header and decimal physical units. Infinite bounds serialize
as `-inf`/`inf`; a consumer must explicitly support or reject them, never silently
replace them with a finite operational bound. There are no such intervals in
the completed run if the report's `infinite_interval` counter is zero. Empty
arrival fields do not mean that an observation arrived at issuance. Do not use
future availability timestamps as predictor covariates: they are retrospective
audit fields. Check identity and units before joining by
`(room_seed, scenario, issue_time_utc, horizon_minutes, arm)`.

The full CSV.gz adds integer UTC nanoseconds, scale/quantile values, fallback and
clipping flags, origin-known regime fields and the latest target/arrival used by
calibration. It encodes absent arrival as int64 maximum (`9223372036854775807`);
the portable sample maps that sentinel to an empty ISO field. Each pool contains
only earlier issues whose target and actual arrival are no later than its
cutoff. Sorted target time, then issue time, defines the latest 168 residuals.

N06 may ignore intervals and select/deduplicate `point_ppm` by
`(room_seed, scenario, issue_time_utc, horizon_minutes)`. N05 supplies no action
model or evidence of decision benefit. Consuming an interval is optional and
does not require either task to access the other's worktree.

## Method and evidence boundaries

VM here is a fixed peak-normalized circular bank with learned **scale readout**
weights. It does not learn preferred phases or concentrations. All conditional
arms have the same four causal covariates, 24 clock features, fitting labels and
one configuration. Their equal coefficient penalty is not equal functional
regularization. Each scale fit has 29 numerical coefficients including the
intercept; hour-bin's intercept/one-hot representation has linear redundancy,
resolved by the positive ridge penalty.

Primary room-bootstrap uncertainty carries independently fitted calibration
pipelines together. The separately named fixed-interval block bootstrap holds
factors and online quantiles fixed and omits refitting variability. Neither is
a conditional-coverage guarantee, and neither turns this synthetic study into
real-data or controller validation. See the [protocol](vm_n05_protocol_v1.md)
and [result report](vm_n05_report_v1.md).
