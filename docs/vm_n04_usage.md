# Reproducing VM-N04

Use Python 3.11 with the existing requirements. No sibling checkout or external
dataset is a runtime dependency. The original research baseline is
`a7431a926c751868b11a3918c6cdd29be6540a32`; the study source locks identify the
additional uncommitted file bytes exactly.

Use the **v1.1 controller** for new runs; v1 is preserved as the exact original
pilot source, including its final-summary NumPy-metadata serialization defect.
The successor changes serialization only. It calls the same frozen generator,
models, trainer, protocol and analysis.

```powershell
py -3.11 -m pytest tests/ -q
py -3.11 scripts/benchmark_standard_dataset.py --output-dir results/<new-id>/standard
py -3.11 scripts/validate_dynamics_forecasters_v1.py --output-dir results/<new-id>/adapters
py -3.11 scripts/run_vm_n04_v1_1.py freeze --output-dir results/<new-id>/study
py -3.11 scripts/run_vm_n04_v1_1.py pilot --output-dir results/<new-id>/study
py -3.11 scripts/run_vm_n04_v1_1.py development --output-dir results/<new-id>/study
```

Before confirmation, save `engineering.json` in that study directory containing
the actual check results and hashes of their evidence. Required fields
`full_suite_completed` and `standard_benchmarks_completed` mean the checks
executed; they do not override failed acceptance. The checked-in engineering
record illustrates the schema but is not proof of a new run. Failed arm gates
must remain reported. Then:

```powershell
py -3.11 scripts/run_vm_n04_v1_1.py confirmation --output-dir results/<new-id>/study
py -3.11 scripts/analyze_vm_n04_v1.py --study-dir results/<new-id>/study --output-dir results/<new-id>/report
```

The original new neural adapters pass absolute standard thresholds but fail
base-VM regression limits; their validation command correctly exits nonzero.
Do not change that result or treat finite scientific output as acceptance.

Directories cannot be overwritten or resumed implicitly. Failed fits remain
failed; source changes invalidate locks. The exceptional `recover-pilot`
command is restricted to an original failed-finalization pilot with every fit
complete and every artifact verified. It is unnecessary for a fresh v1.1 run.

The published seeds are now consumed. Running these commands again reproduces
this experiment; it does not create a fresh confirmation study. The full arrays
and checkpoints under `results/` are ignored and absent from a fresh clone.
Compact tables, source identities and findings remain under `docs/`.
