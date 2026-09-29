# VM-N01 implementation and reproduction

The scientific contract is [the prospective protocol](vm_n01_protocol.md) and
[its frozen JSON](../configs/vm_n01_v1.json). The inherited snapshot is recorded
separately in [vm_n01_inherited_baseline.json](vm_n01_inherited_baseline.json).
Only new versioned modules implement the experiment; locked v1/v2 files stay
unchanged. See [the result report](vm_n01_results.md) for measured outcomes.

## Public implementation

- `src/smooth_local_v1.py`: integer Fourier + peak-normalized circular features,
  signed linear readout, fixed/adaptive geometry and state replay. A separate
  line Gaussian basis supplies the unrestricted diagnostic.
- `src/smooth_local_training_v1.py`: positive ridge, joint Adam and differentiable
  variable projection, timed checkpoint eligibility and retained failures.
- `src/smooth_local_tasks_v1.py`: fresh synthetic functions, independent nested
  training/validation streams, seed/array identities and reserved-access guard.
- `src/smooth_local_analysis_v1.py`: declared selections, function-level paired
  aggregation, stratified bootstrap intervals and conjunctive benefit gate.
- `src/smooth_local_protocol_v1.py`: frozen config/source/evidence verification.
- `src/smooth_local_forecaster_v1.py`: narrow engineering adapter. It is not a
  static scientific arm or a broader forecasting study.

The basis accepts scalar, vector, or `(...,1)` float32/float64 tensors, returning
`(...,2*M+K)` features in cosine/sine/kernel order. Static fitting is float64;
the forecaster uses float32. Models and inputs must share dtype/device. Fixed
geometry and harmonics are buffers; adaptive centers and raw softplus widths
are parameters. `predict_smooth_local_state` replays a saved state without
refitting. Parameter counts include ridge-solved coefficients even when those
coefficients are buffers rather than optimizer parameters.

## Commands

Use `py -3.11` on the audited machine, and inspect `--help` on another host.
CUDA is selected when available. Tests include CPU and CUDA checks when present.

```powershell
py -3.11 -m pytest tests/ -q
py -3.11 scripts/validate_smooth_local_forecaster_v1.py --output-dir results/vm_n01_v1/engineering_standard
py -3.11 scripts/run_vm_n01.py freeze --checks docs/vm_n01_validation.json --output results/vm_n01_v1/source_lock.json
py -3.11 scripts/run_vm_n01.py pilot --source-lock results/vm_n01_v1/source_lock.json --output-dir results/vm_n01_v1/pilot
py -3.11 scripts/run_vm_n01.py development --source-lock results/vm_n01_v1/source_lock.json --pilot-dir results/vm_n01_v1/pilot --output-dir results/vm_n01_v1/development
py -3.11 scripts/run_vm_n01.py lock --source-lock results/vm_n01_v1/source_lock.json --pilot-dir results/vm_n01_v1/pilot --development-dir results/vm_n01_v1/development --output results/vm_n01_v1/confirmation_lock.json
py -3.11 scripts/run_vm_n01.py confirmation --source-lock results/vm_n01_v1/source_lock.json --pilot-dir results/vm_n01_v1/pilot --confirmation-lock results/vm_n01_v1/confirmation_lock.json --output-dir results/vm_n01_v1/confirmation
py -3.11 scripts/report_vm_n01.py --study-dir results/vm_n01_v1 --lock results/vm_n01_v1/confirmation_lock.json --output-dir docs
```

These are the original execution commands, not instructions to overwrite an
existing study. Every phase requires a new directory; reporting refuses to
overwrite tables. On reproduction use a new contained subdirectory, preserve
the configuration, run the checks and create a fresh check receipt with the
actual log/result hashes. The historical receipt cannot substitute for absent
local results or newly executed checks. `freeze` requires all three engineering
prerequisites and verifies every referenced check file. Its receipt schema is
documented by `docs/vm_n01_validation.json`.

The complete `results/vm_n01_v1/` archive is ignored and not supplied by a fresh
clone. Tracked CSV/JSON/docs preserve compact metrics and provenance. Exact
checkpoint replay requires the original arrays and `.pt` files. Checkpoints are
trusted local experiment files; do not load untrusted pickle-based checkpoints.
An observer can check the archived equations and selections with
`scripts/audit_vm_n01.py --help`; it never regenerates target functions or fits.

Wall-clock budgets can select different update counts on a different run or
machine. Numerical saved-state replay and a new wall-time reproduction are
different checks. Reusing these now-consumed confirmation seeds is reproduction,
not fresh confirmation. New scientific selection requires new seeds and a new
prospective protocol. Do not change this configuration or source lock to obtain
a preferred gate result.
