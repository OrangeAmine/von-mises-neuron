# VM-N03 completion and worktree handoff

Delivered 2026-09-22 in the isolated assigned worktree:
`C:\Users\Amine\.codex\worktrees\63ae\von-mises-neuron`.

The authorized VM-N03 first delivery is complete: three deterministic, resettable
browser lessons, a standalone HTML/Markdown companion, the historical evidence
export, focused tests, and this handoff. This is teaching implementation and
engineering validation, not a new scientific experiment or learner-outcome study.
No dependency was added and no training was run. VM-N01 is an independent task;
this resource does not depend on its results.

## Open and launch

- [Playground](../../learning/vm_playground/index.html)
- [Offline HTML guide](../../learning/vm_playground/guide.html)
- [Markdown guide](vm_playground_guide.md)
- [Implementation and maintenance commands](../../learning/vm_playground/README.md)

The tested launch command, run from the worktree root, is:

```powershell
py -3.11 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/learning/vm_playground/index.html> or
<http://127.0.0.1:8765/learning/vm_playground/guide.html>.
Use the appropriate Python executable on another host; Ctrl+C stops the server.
The interface uses ordinary local scripts and CSS without fetch, module imports,
CDNs, accounts or model downloads. Direct opening of `index.html` is supported by
its design, but browser-tool URL policy blocked actual `file://` inspection;
loopback HTTP was the verified route. Both routes require retaining the relative
repository layout for evidence links. No internet connection is needed.

## Implemented lessons

1. **Same function, different coordinates:** amplitude/phase controls, polar and
   Cartesian overlay, coefficient and preferred-angle readouts, a probe/wrap
   comparison, exact-zero and near-zero presets, and phase/sign explanations.
   The graph retains a fixed response scale. The guide distinguishes mathematical
   sensor wrapping from physical angles and core guarantees from composed models.
2. **Amplitude versus locality:** linked sum and peak-normalized exponential
   plots, fixed κ=1 references, center/concentration/integer-harmonic controls,
   daily/weekly timestamp mapping and the exact chordal Gaussian overlay. The
   zero case and multiple peaks are explicit; kernels are not called densities.
3. **A gain with a failed gate:** immutable v2 primary-decision cards plus all ten
   model arms in balanced, family and label/noise/compute cell views; fit-time
   and update summaries; exact JSON pointers/CSV lines behind source links.
   The 720 cells, 30 balanced summaries and 120 family means remain inspectable.
   A visually separate authored selection exercise has fixed seed 20260922,
   200 repetitions, nested candidate sets and development-only selection.

Questions precede reveals. Resets restore controls, hide outcomes, close answers
and clear predictions. The guide contains worked equations, four self-checks
with answers, a source-reading map and four evidence labels: exact identity,
engineering validation, empirical finding and open hypothesis. Frozen v1 and
v2 negative findings remain visible.

## Actual focused validation

Environment: Windows, Python 3.11.4, Node.js 24.14.1, Codex in-app browser.

| Check | Result |
| --- | --- |
| `node learning/vm_playground/core.test.cjs` | 12 passed, 0 failed; 598.50 ms in the final recorded run |
| `py -3.11 scripts/learning/build_vm_playground_evidence.py --check` | Passed: five snapshot-pinned source hashes, all 720 CSV/JSON cells, and all exported family/balanced aggregates |
| `py -3.11 -m pytest tests/test_vm_playground_evidence.py -q` | 10 passed in 0.39 s |
| Browser interaction and visual inspection | All three lessons, reset paths, guide and 390×844 narrow layout checked; default viewport 1280×720 |
| Browser console | No captured warnings or errors after interactions |
| Source preservation and local links | Final receipt in [vm_n03_validation.json](vm_n03_validation.json) |
| Whitespace | `git diff --check` passed; new task text also checked for trailing whitespace |

The numerical tests cover polar/Cartesian equality, phase sign, zero and near-zero
limits, boundary and multi-turn wrap, integer harmonic periods, a noninteger
counterexample, an independent Gaussian identity, amplitude versus locality,
the non-density integral at κ=0, reset-default isolation, reproducible selection,
first-index tie breaking, nested minima, adversarial held-out-score intervention,
and recursive freezing of the evidence export. The exporter tests reject changed
source hashes, missing/inconsistent cells and altered summaries.

Browser checks actually exercised the controls through the DOM and inspected
rendered screenshots, including:

- Lesson 1: κ=0 shows no preferred angle; κ=0.001 shows the weak-phase note;
  probing π retains wrap consistency; keyboard phase changes update coefficients;
  reset restores the original readout and hides the lesson workspace.
- Lesson 2: h=3 and T=168 give 56-hour repetition; κ=12 narrows peaks; κ=0 shows
  the constant-kernel exception; reset restores defaults. Paired-plot axis text
  was enlarged after inspection, then visually checked again.
- Lesson 3: balanced and family views, nonperiodic diagnostic labeling, 0.25 s
  versus 2 s, the narrow/512/noise-0.05 Fourier win, and the smooth/64/noise-0.2
  failure cell. Fit-time and update readouts agree with the saved report. View
  changes leave the primary gate cards unchanged.
- Selection exercise: candidate and development-size changes; reset clears and
  hides results, then reproduces the same table (development n=16, 64 candidates:
  development MSE 0.3561; reserved MSE 1.0005). These are toy values, not research
  results. An increased development sample gives a smaller optimism gap.
- Narrow layout: no page-level horizontal overflow at 390 px; the wider history
  table has its own horizontal scroll. Guide navigation and an answer disclosure
  were exercised, and the readable equations were visually inspected.

No full model/pipeline suite or standard training benchmark was run because
research implementations and pipelines are unchanged. This does not revalidate
the historical scientific campaign. Plain isolated `node --test` initially hit
a sandbox child-process `EPERM`; direct Node execution above passed. UI checking
was performed through the available browser tool, not a cross-browser matrix or
a formal assistive-technology audit. JavaScript math may vary at the last bit
across engines; the numerical tolerances and displayed precision reflect this.

## Inherited baseline and preservation

Before any substantive edits, the clean worktree received the immutable launch
snapshot from
`D:\Desktop\Claude\von-mises-neuron\results\task_launches\20260922T112135Z`.
Archive SHA256:
`18191869b3383ee437e33de713b4aaed6a7f103e9ae88c5581ce8ea8a44f31b5`.
All 130 relative paths, sizes and individual SHA256 entries were checked before
and after overlay. Absolute/traversal paths, Git internals and unexpected local
edits were rejected by the bootstrap checks. Only the assigned worktree was
written; the original checkout and context checkout were not modified.

The complete inherited manifest is saved separately as
[vm_n03_inherited_snapshot.json](vm_n03_inherited_snapshot.json). Those 130 files
are the baseline, not VM-N03 authorship. After delivery, all inherited files
except the authorized `docs/project_state.md` handoff retain their exact hashes,
including all frozen source/configs/reports, compact results and
`VM_exploration.md`. The state update inserts a new section after the title and
preserves the inherited title and remaining body bytes exactly.

Initial and delivery HEAD: `a7431a926c751868b11a3918c6cdd29be6540a32`.
The managed worktree has detached HEAD; VM-N03 files remain uncommitted.
No merge, push, scientific source edit or original-checkout edit was performed.

## Read-only research-context receipt

Supplied root: `D:\Desktop\Claude\Hybrid_models_forecasting`.
The supplied `.agents/skills/research-workspace-context/SKILL.md` was followed
with task class **implementation**. Its validator passed using the receiving
project's Python 3.11. Context revision was
`5f48fadda23f3132ecece500370b0c124e408ef4`, checkout state `dirty`, manifest SHA256
`49aed2956b668ffdbf00c531a4f4967515724ebd9747132b1618eb934f0af337`.

Read navigation owners: `AGENTS.md`, `START_HERE.md`,
`docs/research/authority.md`, and the skill's authority-routing reference.
Selected teaching owners: `docs/learning/playgrounds/roadmap.md` and
`docs/learning/labs/model_selection.md`. Their predict/manipulate/observe/explain
pattern and separation of development selection from reserved evaluation informed
this resource. The new toy uses its own declared seed/config and does not copy
the external lesson's numerical results. The explicitly requested lightweight
browser interface takes precedence over the context roadmap's Marimo preference.

Required navigation/teaching resources were available. No raw data, optional
experiment arrays, papers, thesis payloads or new scientific evidence were needed.
Navigation validation does not establish scientific payload identity. No context
thesis, paper, application or study was edited.

## Integration scope

Import the new `learning/vm_playground/`, `scripts/learning/` and VM-N03 documents
and tests, plus the scoped state entry. Preserve the independently recorded
inherited snapshot; do not present its preexisting research files as new work.
A future repository navigation link can point to the playground or companion.
No shared teaching framework, Forecast Lab port, scientific extension, or
additional experiment is a prerequisite or implied follow-up.
