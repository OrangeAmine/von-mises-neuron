# Integration of the six completed deliveries

29 September 2026. Authorized scope: preserve and integrate N01–N06, reconcile
the handoff, and prepare an internal research account with refreshed learning
resources. The calendar-trust study and N05/N06 target-contract follow-up remain
deferred for discussion.

## Preservation and import

The integration starts from `a7431a926c751868b11a3918c6cdd29be6540a32` on
`master`. Original checkouts remain intact. This is an exact-byte file import,
not a rewrite of the task branches or original experiment commits.

| Delivery | Original worktree | Source state |
| --- | --- | --- |
| N01 | `de12` | Uncommitted over `a7431a9`, archived before import |
| N02 | `c416` | Committed at `368551799c27b9c4ad90d85d89c3b4ab3e817e10` |
| N03 | `63ae` | Uncommitted over `a7431a9`, archived before import |
| N04 | `9c11` | Committed at `843bf96f289d6217b038ecbc2355eef4bf8b5e87` |
| N05 | `7457` | Uncommitted over `a7431a9`, including verified N04 dependency |
| N06 | `a3a8` | Committed at `96c72d05414fa41380e0b6d08d89963bc9a30cc6`; dependency ancestor `c0ee369` |

Worktree paths are under
`C:/Users/Amine/.codex/worktrees/<id>/von-mises-neuron`.
All six working source inventories and all **8,250** ignored result files were
archived and verified by SHA256 before importing **153 unique delivery files**.
N04's duplicate dependency files in N05/N06 matched and were imported once.
The initial main source, including owner notes and dispatch records, also has
a verified archive. Source/result archives total about 120 MB including that
main snapshot. See the tracked [preservation receipt](integration/preservation_20260929.json).

The full manifest and ZIPs are local under `results/integration_20260929/`.
Each task's results were also copied to its original relative `results/` path
in main, with hash verification. No existing result was overwritten with
differing bytes. Ignored files do not travel with a fresh clone; tracked reports
and compact metrics do. No original worktree was archived or removed.
`VM_exploration.md` remains unchanged and untracked.

Original task handoffs remain in their source archives. The main handoff links
all six final reports and the consolidated account; its earlier historical
sections remain intact. Scientific source, configuration, protocol, metric and
report files are unchanged. N03's entry page, guides and README are the intended
teaching edits; its numerical implementation, tests and frozen export remain
unchanged.

## New reading and learning material

- [Research account](research_account_n01_n06.md): cross-study interpretation,
  replication units, conditional gains, failed gates and engineering eligibility.
- [Findings explorer](../learning/vm_playground/research.html): all six studies,
  within-study comparisons, source hashes and JSON pointers, worked answers and
  reset. N05 distinguishes primary from conditional intervals; N06 changes the
  objective while keeping its original scientific decision visible.
- [Offline companion](../learning/vm_playground/guide.html#new-findings) and
  [Markdown guide](learning/vm_playground_guide.md): updated reading routes.

The new exporter resolves metrics directly from eleven pinned tracked sources.
It does not refit models, recompute estimates or revise gates. The original
720-cell locality-v2 export is unchanged. No dependencies were added.

## Integration validation

Fresh integration checks completed in the main checkout on Windows with Python
3.11.4, PyTorch 2.5.1+cu121 (CUDA available) and Node.js 24.14.1:

| Check | Result |
| --- | --- |
| `py -3.11 -m pytest tests/ -q` | **1,802 passed in 135.80 s**; full model/pipeline and teaching suite |
| `node learning/vm_playground/core.test.cjs` | **12 passed**, including original numerical and selection checks |
| Original evidence exporter `--check` | Five pinned hashes and all **720** historical cells agree; export unchanged |
| New research exporter `--check` | Six studies and **eleven** pinned sources agree; metrics resolve to exact saved values |
| JavaScript syntax | New interface and generated export pass `node --check` |
| Preservation | **989** original source-file entries, **8,250** original and copied result files, **13** archive hashes, **149** unchanged imported files, **127** unchanged inherited files, and **126** frozen source checks passed |
| Local links | **121** links in six reading/entry files resolve; HTML fragment targets exist |
| Browser | All six studies and **13** comparison views exercised; gate labels remained unchanged, source disclosure and worked answer opened, reset restored N01/default and closed disclosures |
| Responsive inspection | Actual screenshots inspected at **1280×720** and **390×844**; no page-level horizontal overflow at either size |
| Existing teaching routes | All three original lessons reveal/reset correctly; updated companion and N06 decision demo open successfully |
| Browser console | No captured warnings or errors during these checks |
| Staging integrity | All **168** staged file blobs match their working-file bytes exactly; owner notes and raw results excluded |
| Whitespace | All **19** newly authored or edited integration files pass with CRLF recognized. The frozen N02 generated SVG retains its **2,656** original trailing-space warnings; it was not reformatted because its bytes are part of the evidence record. |

See the [preservation check](integration/preservation_check_20260929.json) and
[validation receipt](integration/validation_20260929.json). Full-suite log and
JUnit XML remain under `results/integration_20260929/`. N02's original runner
hash and its documented operational amendment were checked separately; the
amendment was not rewritten. Reported check counts can overlap and are not a
count of independent experiments.

No scientific campaign or standard training benchmark was rerun. Imported model
implementations are byte-identical to the completed deliveries, so their saved
standard acceptance outcomes remain the applicable evidence. The combined
suite checks compatibility; it does not turn any failed scientific gate or
regression acceptance into a pass. Full prediction-array numerical replays were
not repeated in this integration; the original artifacts and audits are retained.

The explorer uses the same local-server route as the original lessons:

```powershell
py -3.11 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/learning/vm_playground/research.html>. Direct file
opening is supported by the local-script design; the verified route for this
integration is loopback HTTP. Original worktrees and local archives remain for
recovery. Nothing was pushed or published externally.
