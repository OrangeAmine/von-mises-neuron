"""Frozen, bounded N06 development and paired evaluation; all artifacts local."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decision_plant_v1 import Plant, make_room
from src.decision_forecasts_v1 import CalendarSourceV1
from src.decision_policies_v1 import objective, rollout

SOURCES = ['configs/vm_n06_v1.json', 'docs/vm_n06_protocol_v1.md',
           'docs/vm_n06_contract_lock.json', 'docs/vm_n06_context_receipt.json',
           'src/decision_plant_v1.py', 'src/decision_forecasts_v1.py', 'src/decision_policies_v1.py',
           'src/dynamics_models_v1.py', 'src/circular_layers_v2.py',
           'scripts/run_vm_n06_v1.py', 'scripts/validate_n06_calendar_v1.py',
           'scripts/analyze_vm_n06_v1.py',
           'tests/test_decision_n06_v1.py']


def sha(path: Path) -> str:
    """Compute an exact file identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """Write readable JSON with no nonfinite substitutions."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def freeze(output: Path) -> None:
    """Freeze implementation after debug checks, before study generation."""
    contract = json.loads((ROOT / 'docs/vm_n06_contract_lock.json').read_text())
    for name, identity in contract['files'].items():
        if sha(ROOT / name) != identity:
            raise ValueError(f'Prospective contract changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'sources': {p: sha(ROOT / p) for p in SOURCES},
              'config': json.loads((ROOT / 'configs/vm_n06_v1.json').read_text())}
    write(output / 'source_lock.json', record)
    with zipfile.ZipFile(output / 'frozen_sources.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in SOURCES:
            bundle.write(ROOT / name, name)


def verify(output: Path) -> dict:
    """Require the exact frozen implementation before any scientific run."""
    lock = json.loads((output / 'source_lock.json').read_text())
    for name, identity in lock['sources'].items():
        if sha(ROOT / name) != identity:
            raise ValueError(f'Frozen source changed: {name}')
    return lock


def select(rows: list[dict], config: dict) -> dict:
    """Choose margins only from complete development outcomes."""
    choices = {}
    for name, weight in config['objectives'].items():
        choices[name] = {}
        for arm in config['arms']:
            scores = {margin: float(np.mean([objective(row, weight) for row in rows
                                             if row['arm'] == arm and row['margin_ppm'] == margin]))
                      for margin in config['margins_ppm']}
            winner = min(scores, key=lambda m: (scores[m], m))
            choices[name][arm] = {'margin_ppm': winner, 'development_objectives': scores}
    return choices


def run(output: Path, stage: str) -> dict:
    """Generate paired room trajectories, retain every candidate, enforce cap."""
    lock = verify(output)
    config = lock['config']
    torch.set_num_threads(1)
    folder = output / stage
    if stage not in ('development', 'evaluation'):
        raise ValueError('Unknown stage')
    spent = 0.0
    if stage == 'evaluation':
        dev = json.loads((output / 'development/summary.json').read_text())
        if dev['status'] != 'complete':
            raise ValueError('Complete development required')
        if dev['source_lock_sha256'] != sha(output / 'source_lock.json'):
            raise ValueError('Development belongs to a different source lock')
        expected_dev = len(config['development_seeds']) * len(config['regimes']) * len(config['arms']) * len(config['margins_ppm'])
        if dev['actual_rows'] != expected_dev:
            raise ValueError('Development matrix incomplete')
        engineering = json.loads((output / 'engineering.json').read_text())
        if not engineering['full_suite_passed'] or not engineering['standard_adapter_completed']:
            raise ValueError('Required checks must have executed; failed adapter gates stay explicit')
        for name, identity in engineering['evidence'].items():
            if sha(ROOT / name) != identity:
                raise ValueError('Engineering evidence changed')
        spent = dev['runtime_seconds']
        # Bind the complete development record and selected choices BEFORE generation.
        access = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'source_lock_sha256': sha(output / 'source_lock.json'),
                  'development_sha256': sha(output / 'development/summary.json'),
                  'engineering_sha256': sha(output / 'engineering.json'), 'selection': dev['selection']}
        if (output / 'evaluation_lock.json').exists():
            raise FileExistsError('Evaluation access already recorded; no implicit retries')
        write(output / 'evaluation_lock.json', access)
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    rows, fits, scenarios = [], [], []
    status, error = 'complete', None
    try:
        for seed in config[f'{stage}_seeds']:
            room = make_room(seed, config)
            train = room['train_source']
            dt = config['plant']['dt_hours']
            train_hours = np.arange(len(train)) * dt
            n = len(room['source'])
            h = config['forecast']['horizon_steps']
            scoring_hours = (len(train) + np.arange(n + h)) * dt
            predictors = {}
            for arm in ('seasonal', 'fourier', 'vm'):
                if spent + time.perf_counter() - started > config['compute_cap_seconds']:
                    raise TimeoutError('Combined development/evaluation cap reached')
                began = time.perf_counter()
                fitted = CalendarSourceV1(arm, config['forecast']).fit(train_hours, train)
                predictions = fitted.predict(scoring_hours)
                predictors[arm] = predictions
                fits.append({'seed': seed, 'arm': arm, 'training_labels': len(train),
                             'fit_and_predict_seconds': time.perf_counter() - began, **fitted.record()})
            for regime, changes in config['regimes'].items():
                draw = make_room(seed, config, changes['shift_hours'])
                p = replace(Plant(**config['plant']), **{k: v for k, v in changes.items() if k != 'shift_hours'})
                scenario_file = folder / f'{seed}_{regime}_scenario.npz'
                np.savez_compressed(scenario_file, train_source=train, source=draw['source'], noise=draw['noise'])
                scenarios.append({'seed': seed, 'regime': regime, 'metadata': draw['metadata'],
                                  'file': scenario_file.name, 'sha256': sha(scenario_file),
                                  'source_sha256': hashlib.sha256(draw['source'].tobytes()).hexdigest(),
                                  'noise_sha256': hashlib.sha256(draw['noise'].tobytes()).hexdigest()})
                for arm in config['arms']:
                    pred = (None if arm == 'reactive' else np.pad(draw['source'], (0, h))
                            if arm == 'oracle' else predictors[arm])
                    for margin in config['margins_ppm']:
                        if spent + time.perf_counter() - started > config['compute_cap_seconds']:
                            raise TimeoutError('Combined development/evaluation cap reached')
                        result = rollout(draw['source'], draw['noise'], p, margin, pred, h, config['hysteresis_ppm'])
                        artifact = folder / f'{seed}_{regime}_{arm}_{int(margin)}.npz'
                        np.savez_compressed(artifact, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
                                            forecast_source=np.array([]) if pred is None else pred)
                        err = None if pred is None else pred[:n] - draw['source']
                        rows.append({'seed': seed, 'regime': regime, 'arm': arm, 'margin_ppm': margin,
                                     **result['metrics'], 'source_rmse_ppm_m3h': None if err is None else float(np.sqrt(np.mean(err ** 2))),
                                     'source_mae_ppm_m3h': None if err is None else float(np.mean(np.abs(err))),
                                     'artifact': artifact.name, 'sha256': sha(artifact)})
            write(folder / 'partial.json', {'rows': rows, 'fits': fits, 'scenarios': scenarios})
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    runtime = time.perf_counter() - started
    expected = len(config[f'{stage}_seeds']) * len(config['regimes']) * len(config['arms']) * len(config['margins_ppm'])
    if len(rows) != expected or spent + runtime > config['compute_cap_seconds']:
        status, error = 'failed', error or 'Incomplete matrix or runtime cap exceeded'
    summary = {'stage': stage, 'status': status, 'error': error, 'rows': rows, 'fits': fits,
               'scenarios': scenarios, 'expected_rows': expected, 'actual_rows': len(rows),
               'runtime_seconds': runtime, 'combined_runtime_seconds': spent + runtime,
               'source_lock_sha256': sha(output / 'source_lock.json'),
               'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                               'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                               'torch_threads': torch.get_num_threads(), 'simulation_device': 'cpu scalar exact recurrence'},
               'selection': select(rows, config) if stage == 'development' and status == 'complete' else None}
    write(folder / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('stage', 'status', 'actual_rows', 'runtime_seconds', 'error')}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'development', 'evaluation'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output_dir)
    else:
        sys.exit(0 if run(args.output_dir, args.stage)['status'] == 'complete' else 1)
