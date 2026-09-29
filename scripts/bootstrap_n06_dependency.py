"""Verify and safely import the explicitly supplied N04 dependency snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('D:/Desktop/Claude/von-mises-neuron/results/task_launches/20260926T182459Z')
EXPECTED = 'acc928bcc98bd9f7e38fcf496a26da757247344fd77fa3bc312064a90e5e31d7'


def main() -> None:
    """Import missing/clean older files; record and preserve any conflicts."""
    out = ROOT / 'docs/vm_n06_dependency_receipt.json'
    if out.exists():
        raise FileExistsError('Dependency already imported; preserve the receipt and all subsequent edits.')
    archive = SOURCE / 'vm_n04_dependency_snapshot.zip'
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED
    entries = {row['path']: row for row in manifest['files']}
    receipt = {'archive_sha256': EXPECTED, 'source_revision': manifest['source_revision'],
               'receiving_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'files': []}
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(entries)
        for name, item in entries.items():
            target = (ROOT / name).resolve()
            assert target.is_relative_to(ROOT) and '.git' not in Path(name).parts
            data = bundle.read(name)
            assert len(data) == item['bytes']
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            old = target.read_bytes() if target.exists() else None
            action = 'matching'
            if old != data:
                action = 'added'
                if old is not None:
                    tracked = subprocess.run(['git', 'show', f'HEAD:{name}'], cwd=ROOT, capture_output=True)
                    clean = tracked.returncode == 0 and tracked.stdout.replace(b'\r\n', b'\n') == old.replace(b'\r\n', b'\n')
                    known_older_base = receipt['receiving_head'] == 'a7431a926c751868b11a3918c6cdd29be6540a32'
                    action = 'inherited_clean_update' if clean and known_older_base else 'conflict_preserved'
                if action != 'conflict_preserved':
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            receipt['files'].append({**item, 'action': action,
                                     'previous_sha256': hashlib.sha256(old).hexdigest() if old is not None else None})
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: sum(r['action'] == key for r in receipt['files'])
                      for key in ('matching', 'added', 'inherited_clean_update', 'conflict_preserved')}))


if __name__ == '__main__':
    main()
