"""Diagnostic calendar adapters on unchanged standard data and acceptance gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_standard_dataset import (
    generate_synthetic_co2, prepare_benchmark_data, compute_benchmark_metrics,
    check_thresholds, check_regression, THRESHOLDS,
)
from src.split import chronological_split
from src.sequences import create_sequences
from src.decision_forecasts_v1 import CalendarSourceV1


def run(output: Path) -> dict:
    """Record each new calendar estimator's gates without refitting old models."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'configs/vm_n06_v1.json').read_text())
    inherited = json.loads((ROOT / 'docs/vm_n06_context_receipt.json').read_text())['inherited_standard']
    frame = generate_synthetic_co2(n_days=90, seed=42)
    data = prepare_benchmark_data(frame, 72, 12, torch.device('cpu'))
    train, _, test = chronological_split(frame, [0.8, 0.1, 0.1])
    _, raw_truth, indices = create_sequences(test, 72, 12)
    target_hours = test.index.hour.to_numpy()[indices[:, None] + np.arange(1, 13)]
    mean, std = data['norm_stats']['CO2']
    np.testing.assert_allclose(raw_truth, data['y_test'].numpy() * std + mean, atol=1e-4)
    results = {}
    for arm in ('seasonal', 'fourier', 'vm'):
        if time.perf_counter() - started > config['standard_adapter_cap_seconds']:
            raise TimeoutError('Standard adapter cap reached; do not retry')
        fit_started = time.perf_counter()
        model = CalendarSourceV1(arm, config['forecast']).fit(train.index.hour.to_numpy(), train.CO2.to_numpy())
        predicted = model.predict(target_hours)
        fit_seconds = time.perf_counter() - fit_started
        metrics = compute_benchmark_metrics(torch.tensor((predicted - mean) / std),
                                            data['y_test'], data['norm_stats'])
        absolute, failures = check_thresholds(metrics)
        regression, relative_failures = check_regression(metrics, inherited['base_vm_metrics'])
        results[arm] = {'metrics': metrics, 'absolute_passed': absolute,
                        'regression_passed': regression, 'passed': absolute and regression,
                        'failures': failures + relative_failures,
                        'fit_and_predict_seconds': fit_seconds, 'model': model.record()}
        np.savez_compressed(output / f'{arm}.npz', predictions=predicted, targets=raw_truth,
                            target_hours=target_hours)
    record = {'results': results, 'all_passed': all(x['passed'] for x in results.values()),
              'thresholds': THRESHOLDS, 'inherited_base_reference': inherited,
              'data_contract': {'days': 90, 'seed': 42, 'split': [.8, .1, .1], 'lookback': 72,
                                'horizon': 12, 'training_labels': len(train), 'test_windows': len(indices)},
              'fit_budget': 'One closed-form fit per calendar arm, no selection; 50 epochs not applicable to a closed-form estimator.',
              'limitations': 'Calendar-only concentration transfer diagnostic; no sensor branch, weekly term or online correction. Preserves standard local calendar/DST limitations. Passing would not establish closed-loop validity.',
              'runtime_seconds': time.perf_counter() - started,
              'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'src/decision_forecasts_v1.py',
                                          ROOT / 'scripts/benchmark_standard_dataset.py']}}
    (output / 'acceptance.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps({arm: {key: v[key] for key in ('metrics', 'passed')} for arm, v in report['results'].items()}, indent=2))
    sys.exit(0 if report['all_passed'] else 1)
