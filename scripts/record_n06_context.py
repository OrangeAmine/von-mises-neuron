"""One-time read-only provenance receipt for the explicitly supplied contexts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
N04 = Path('C:/Users/Amine/.codex/worktrees/9c11/von-mises-neuron')
CONTEXT = Path('D:/Desktop/Claude/Hybrid_models_forecasting')


def sha(path: Path) -> str:
    """Identify exact source/artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify selected inherited identities without reopening consumed arrays."""
    lock_path = N04 / 'results/vm_n04/study_v1_1/source_lock.json'
    lock = json.loads(lock_path.read_text())
    assert sha(lock_path) == '069abf9d9635ef9c6b3345404e300f421ec81147997bca735cdeb5c029bd2d74'
    verified = {}
    for name, expected in lock['sources'].items():
        assert sha(N04 / name) == sha(ROOT / name) == expected, name
        verified[name] = expected
    engineering = json.loads((ROOT / 'docs/vm_n04_engineering_v1.json').read_text())
    artifact = 'results/vm_n04/standard_unchanged/benchmark_results.json'
    assert sha(N04 / artifact) == engineering['evidence_sha256'][artifact]
    base = json.loads((N04 / artifact).read_text())
    assert base['acceptance']['VonMises']['passed']
    source_files = ['.agents/skills/research-workspace-context/SKILL.md',
                    'docs/research/active_work.md', 'docs/research/manuscript_review_policy.md',
                    'docs/learning/playgrounds/roadmap.md',
                    'docs/learning/labs/decision_consequences.md',
                    'docs/learning/explainers/04a-first-order-systems.md',
                    'docs/learning/explainers/05a-control-on-top-of-physics.md',
                    'docs/learning/chapter7_reading_companion.md']
    validator = json.loads(subprocess.check_output(
        ['py', '-3.11', str(CONTEXT / 'scripts/utility/validate_research_context.py'), str(CONTEXT)], text=True))
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'receiving_project': str(ROOT), 'context_validator': validator,
               'task_class': 'forecast-lab', 'catalog_ids': [],
               'owners': {'PL-03': 'docs/learning/playgrounds/roadmap.md',
                          'chapter7_teaching': 'docs/learning/chapter7_reading_companion.md'},
               'context_files': {p: sha(CONTEXT / p) for p in source_files},
               'unavailable_resources': [], 'n04_source_lock_sha256': sha(lock_path),
               'n04_verified_sources': verified,
               'inherited_standard': {'path': artifact, 'sha256': sha(N04 / artifact),
                                      'config': base['config'], 'base_vm_metrics': base['results']['VonMises']},
               'scope': 'Teaching structure and mass-balance context only; historical prediction arrays/checkpoints not used. No claims about sibling results. No runtime sibling dependency.'}
    path = ROOT / 'docs/vm_n06_context_receipt.json'
    assert not path.exists()
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    prospective = {'created_utc': receipt['created_utc'], 'status': 'before any N06 study generation',
                   'files': {p: sha(ROOT / p) for p in ('configs/vm_n06_v1.json', 'docs/vm_n06_protocol_v1.md')}}
    (ROOT / 'docs/vm_n06_contract_lock.json').write_text(json.dumps(prospective, indent=2) + '\n')
    print('Verified 10 N04 source identities, one standard benchmark artifact, and teaching context.')


if __name__ == '__main__':
    main()
