"""Bind actual completed test and calendar-acceptance records for a fresh run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Read JUnit counts and preserve failed model gates without overriding them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--test-xml', type=Path, required=True)
    parser.add_argument('--standard-dir', type=Path, required=True)
    args = parser.parse_args()
    target = args.study_dir / 'engineering.json'
    if target.exists():
        raise FileExistsError('Existing engineering evidence must be preserved')
    suites = ET.parse(args.test_xml).getroot().findall('.//testsuite')
    count = sum(int(s.attrib['tests']) for s in suites)
    failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in suites)
    acceptance_path = args.standard_dir / 'acceptance.json'
    acceptance = json.loads(acceptance_path.read_text())
    record = {'full_suite_passed': count > 0 and failures == 0,
              'full_suite': {'tests': count, 'failures': failures,
                             'seconds': sum(float(s.attrib['time']) for s in suites)},
              'standard_adapter_completed': set(acceptance['results']) == {'seasonal', 'vm', 'fourier'},
              'standard_adapters_passed': acceptance['all_passed'],
              'evidence': {p.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.test_xml, acceptance_path)}}
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
