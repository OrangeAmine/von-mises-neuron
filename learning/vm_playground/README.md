# VM-N03 teaching playground

Integration edition, 2026-09-29: [research.html](research.html) adds an offline
explorer for all six completed deliveries. The original three lessons and
historical `evidence.js` remain intact. The HTML/Markdown companions now include
the new findings and their failed gates. See the
[consolidated account](../../docs/research_account_n01_n06.md).

Open `index.html` in a browser, or use the tested local-server route from the
repository root:

```powershell
py -3.11 -m http.server 8765 --bind 127.0.0.1
```

Visit <http://127.0.0.1:8765/learning/vm_playground/index.html>.
[guide.html](guide.html) is the standalone offline reading companion; no script,
external font, CDN or package is needed to read it. Keep the repository layout
for the historical source links. The page has no network requests beyond loading
its local assets and following user-selected links. No private data or model is
required. Browser-tool security prevented direct `file://` inspection during
delivery; the loopback route received actual UI validation.

## Small, separate implementation

- `core.js`: pure teaching equations, immutable reset defaults and fixed-seed
  selection simulation. No research imports or training.
- `app.js`, `index.html`, `style.css`: plain browser interface and SVG plots.
- `evidence.js`: generated immutable copy of the tracked confirmation summaries,
  with exact source pointers and hashes. All 720 cells remain available.
- `core.test.cjs`: numerical, deterministic selection and immutability checks.

The selection exercise uses Mulberry32 streams with Box-Muller normal draws.
The development seed is 20260922; the independent reserved stream uses its XOR
with `0x9e3779b9`. Each of 200 repetitions includes 64 candidates, 128 development
residuals (the first 16 also form the small sample) and 256 reserved residuals
per candidate. First index breaks ties. The selector receives development scores
only. Reset reuses the same authored pool; it does not create untouched data.
These residuals use arbitrary standardized units, so their squared errors are
dimensionless toy scores, not a physical forecasting metric.

## Focused maintenance checks

Run from the repository root with Python 3.11 and Node.js (no npm install):

```powershell
node learning/vm_playground/core.test.cjs
py -3.11 scripts/learning/build_vm_playground_evidence.py --check
py -3.11 -m pytest tests/test_vm_playground_evidence.py -q
py -3.11 scripts/learning/build_vm_research_portfolio.py --check
py -3.11 -m pytest tests/test_vm_research_portfolio.py -q
```

The export command without `--check` regenerates only `evidence.js`. It first
checks five pinned historical source hashes and CSV/JSON consistency. Do not
change those hashes to adopt revised results silently; new evidence needs its
own clearly distinguished lesson edition. The pure numerical test runner avoids
Node test isolation child-process restrictions in this Windows sandbox.

For interface edits, inspect all three reveals, control changes, reset behavior,
historical filters, source links, guide and narrow layout in a browser. The
delivery record describes the actual checks. No standard training benchmark or
full pipeline suite is needed unless research model/pipeline behavior changes.

[Completion and validation](../../docs/learning/vm_n03_completion.md) ·
[Maintained Markdown companion](../../docs/learning/vm_playground_guide.md)

## Maintaining the completed-study explorer

`docs/learning/research_portfolio.json` contains reviewed teaching text and
metric references. Eleven pinned source hashes identify the original reports
and structured results. `build_vm_research_portfolio.py` resolves exact JSON
pointers into `research-evidence.js`, rejects altered scientific decisions and
exports recursively frozen data. It never recomputes estimates or trains models.
Keep a new study edition separate rather than silently changing historical pins.

`research.js` supplies study selection, within-study views, reset and worked
answers. Inspect every study and comparison after UI changes, including N05's
room-versus-conditional interval distinction and N06's objective change. Verify
the original lesson links, companion anchors, narrow layout and source links.
