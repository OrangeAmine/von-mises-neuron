"""Retain N06 achieved points, paired conditional effects and teaching artifacts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decision_plant_v1 import Plant
from src.decision_policies_v1 import rollout, objective

METRICS = ['exceedance_hours', 'severity_ppm_hours', 'additional_air_m3',
           'total_air_m3', 'on_hours', 'activations', 'switches']


def dump(path: Path, value: dict) -> None:
    """Write a portable compact evidence artifact."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path: Path, rows: list[dict]) -> None:
    """Write a rectangular CSV with explicit column units."""
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], config: dict) -> list[dict]:
    """Average independent rooms equally, preserving all achieved menu points."""
    result = []
    for regime in config['regimes']:
        for arm in config['arms']:
            for margin in config['margins_ppm']:
                values = [r for r in rows if r['regime'] == regime and r['arm'] == arm and r['margin_ppm'] == margin]
                result.append({'regime': regime, 'arm': arm, 'margin_ppm': margin, 'rooms': len(values),
                               **{key: float(np.mean([r[key] for r in values])) for key in METRICS},
                               **{key + '_room_sd': float(np.std([r[key] for r in values], ddof=1)) for key in METRICS}})
    return result


def paired_analysis(rows: list[dict], selection: dict, config: dict) -> dict:
    """Compute prespecified ratios and intervals with whole-room resampling."""
    rng = np.random.default_rng(config['bootstrap']['seed'])
    seeds = config['evaluation_seeds']
    draws = rng.integers(0, len(seeds), (config['bootstrap']['draws'], len(seeds)))
    result, selected = {}, []
    for name, weight in config['objectives'].items():
        chosen = [r for r in rows if r['margin_ppm'] == selection[name][r['arm']]['margin_ppm']]
        for r in chosen:
            selected.append({k: v for k, v in {**r, 'objective_name': name, 'objective_hours_equivalent': objective(r, weight)}.items()
                             if k not in ('artifact', 'sha256')})
        by_arm = {arm: np.array([np.mean([objective(r, weight) for r in chosen if r['arm'] == arm and r['seed'] == seed])
                                 for seed in seeds]) for arm in config['arms']}
        contrasts = {}
        for control in config['practical_effect']['comparators']:
            vm, ref = by_arm['vm'], by_arm[control]
            improvements = 1 - vm[draws].mean(axis=1) / ref[draws].mean(axis=1)
            effect = float(1 - vm.mean() / ref.mean())
            interval = np.quantile(improvements, [.025, .975]).tolist()
            regimes = {regime: float(1 - np.mean([objective(r, weight) for r in chosen if r['arm'] == 'vm' and r['regime'] == regime]) /
                                       np.mean([objective(r, weight) for r in chosen if r['arm'] == control and r['regime'] == regime]))
                       for regime in config['regimes']}
            contrasts[control] = {'relative_improvement': effect, 'paired_95_interval': interval,
                                  'paired_room_difference_vm_minus_control': (vm - ref).tolist(),
                                  'regime_improvements': regimes,
                                  'practical_effect_passed': effect >= config['practical_effect']['relative_improvement_min'],
                                  'positive_interval_passed': interval[0] > 0,
                                  'regime_guard_passed': min(regimes.values()) >= -config['practical_effect']['regime_max_regression']}
            contrasts[control]['gate_passed'] = all(contrasts[control][key] for key in
                                                     ('practical_effect_passed', 'positive_interval_passed', 'regime_guard_passed'))
        result[name] = {'weight': weight, 'arms': {arm: {'mean_objective': float(v.mean()),
                                                        'room_sd': float(v.std(ddof=1)), 'room_objectives': v.tolist(),
                                                        'selected_margin_ppm': selection[name][arm]['margin_ppm']}
                                                  for arm, v in by_arm.items()}, 'contrasts': contrasts}
    return {'objectives': result, 'selected_rows': selected,
            'primary_gate_passed': all(x['gate_passed'] for x in result['exceedance_priority']['contrasts'].values())}


def teaching(config: dict) -> dict:
    """Compute authored educational counterfactuals, separate from held-out study."""
    spec = config['teaching']
    p = replace(Plant(**config['plant']), actuator_delay_steps=spec['delay_steps'], noise_ppm=0)
    n, h = 96, config['forecast']['horizon_steps']
    hours = np.arange(n + h) * p.dt_hours
    truth = np.where((hours >= spec['arrival_hour']) & (hours < spec['arrival_hour'] + 2),
                     spec['source_ppm_m3h'], 0.)
    source, noise = truth[:n], np.zeros(n + 1)
    authored = {}
    for name, interval in [('early_miss', spec['early_miss_hours']), ('late_miss', spec['late_miss_hours'])]:
        pred = truth.copy()
        pred[(hours >= interval[0]) & (hours < interval[1])] = 0
        result = rollout(source, noise, p, spec['margin_ppm'], pred, h)
        err = pred[:n] - source
        authored[name] = {'states': result['states'].tolist(), 'actions': result['actions'].tolist(),
                          'source_forecast': pred[:n].tolist(), 'metrics': result['metrics'],
                          'source_mae': float(np.abs(err).mean()), 'source_rmse': float(np.sqrt(np.mean(err ** 2)))}
    constraints = {}
    options = [('normal', 405., 2), ('limited', 60., 2), ('long_delay', 405., 8), ('zero', 0., 2)]
    for name, capacity, delay in options:
        actual = replace(p, extra_flow_m3h=capacity, actuator_delay_steps=delay)
        case = {'extra_flow_m3h': capacity, 'delay_hours': delay * p.dt_hours, 'policies': {}}
        for arm, forecast, lookahead in [('reactive', None, h), ('oracle_short', truth, 1), ('oracle', truth, h)]:
            result = rollout(source, noise, actual, spec['margin_ppm'], forecast, lookahead)
            case['policies'][arm] = {'states': result['states'].tolist(), 'actions': result['actions'].tolist(),
                                     'metrics': result['metrics']}
        constraints[name] = case
    tradeoffs = []
    for arm, forecast in [('reactive', None), ('early_miss', np.array(authored['early_miss']['source_forecast'] + [0.] * h)),
                          ('late_miss', np.array(authored['late_miss']['source_forecast'] + [0.] * h)), ('oracle', truth)]:
        for margin in config['margins_ppm']:
            result = rollout(source, noise, p, margin, forecast, h)
            tradeoffs.append({'arm': arm, 'margin_ppm': margin, **result['metrics']})
    return {'source': source.tolist(), 'hours': hours[:n].tolist(), 'dt_hours': p.dt_hours,
            'limit_ppm': p.limit_ppm, 'plant': vars(p), 'equal_error': authored,
            'constraints': constraints, 'tradeoffs': tradeoffs,
            'information': 'Authored hypothetical forecasts and perfect-source oracle. No training or calibration claim.'}


def analyze(study: Path, output: Path) -> None:
    """Verify and summarize all evidence, without selecting new policies."""
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((study / 'source_lock.json').read_text())
    config = lock['config']
    evaluation = json.loads((study / 'evaluation/summary.json').read_text())
    access = json.loads((study / 'evaluation_lock.json').read_text())
    assert evaluation['status'] == 'complete'
    assert hashlib.sha256((study / 'development/summary.json').read_bytes()).hexdigest() == access['development_sha256']
    for item in evaluation['rows'] + evaluation['scenarios']:
        name = item.get('artifact', item.get('file'))
        assert hashlib.sha256((study / 'evaluation' / name).read_bytes()).hexdigest() == item['sha256']
    rows = evaluation['rows']
    aggregated = aggregate(rows, config)
    analysis = paired_analysis(rows, access['selection'], config)
    table(output / 'achieved_points.csv', aggregated)
    table(output / 'selected_rooms.csv', analysis.pop('selected_rows'))
    matches = []
    for regime in config['regimes']:
        vm = [r for r in aggregated if r['arm'] == 'vm' and r['regime'] == regime]
        for control in config['practical_effect']['comparators']:
            other = [r for r in aggregated if r['arm'] == control and r['regime'] == regime]
            pairs = [{'vm_margin': a['margin_ppm'], 'control_margin': b['margin_ppm'],
                      'effort_m3': a['additional_air_m3'], 'vm_exceedance_hours': a['exceedance_hours'],
                      'control_exceedance_hours': b['exceedance_hours']}
                     for a in vm for b in other if abs(a['additional_air_m3'] - b['additional_air_m3']) <= 1e-9]
            matches.append({'regime': regime, 'control': control, 'status': 'attained_mean_effort_match' if pairs else 'no_match',
                            'pairs': pairs})
    analysis.update({'independent_units': config['evaluation_seeds'], 'selection': access['selection'],
                     'matches': matches, 'runtime_seconds': evaluation['combined_runtime_seconds'],
                     'environment': evaluation['environment'], 'evaluation_rows': len(rows),
                     'source_lock_sha256': hashlib.sha256((study / 'source_lock.json').read_bytes()).hexdigest(),
                     'evaluation_lock_sha256': hashlib.sha256((study / 'evaluation_lock.json').read_bytes()).hexdigest(),
                     'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'eligibility': 'All new calendar concentration adapters failed base regression gates; diagnostic only. N04 gates unchanged.'})
    dump(output / 'results.json', analysis)
    demo = teaching(config)
    dump(output / 'teaching.json', demo)
    dump(output / 'viewer_data.json', {'study': aggregated, 'analysis': analysis, 'teaching': demo})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, regime in zip(axes.flat, config['regimes']):
        for arm in config['arms']:
            values = [r for r in aggregated if r['regime'] == regime and r['arm'] == arm]
            ax.scatter([r['additional_air_m3'] for r in values], [r['exceedance_hours'] for r in values], label=arm, s=45)
        ax.set(title=regime.replace('_', ' '), xlabel='Additional ventilation (m3 / 3 days)', ylabel='Above 900 ppm (hours / 3 days)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('N06: every achieved operating point, means across six independent rooms\nPoints only: no interpolated matches or energy interpretation', fontsize=12)
    fig.savefig(output / 'tradeoffs.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, row in demo['equal_error'].items():
        axes[0].plot(demo['hours'], np.array(row['source_forecast']) / 1000, label=name.replace('_', ' '))
        axes[1].plot(np.arange(97) / 4, row['states'], label=name.replace('_', ' '))
        axes[2].step(demo['hours'], row['actions'], where='post', label=name.replace('_', ' '))
    axes[0].plot(demo['hours'], np.array(demo['source']) / 1000, color='black', linestyle=':', label='Actual source')
    axes[0].set_ylabel('Source (1000 ppm m3/h)')
    axes[1].axhline(900, color='black', linestyle=':')
    axes[1].set_ylabel('Controlled CO2 (ppm)')
    axes[2].set(ylabel='Applied duty (0/1)', xlabel='UTC hour', xlim=(5, 16))
    axes[0].legend()
    fig.suptitle('Equal source errors, separate controlled outcomes (authored teaching case)')
    fig.savefig(output / 'timing.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'primary_gate_passed': analysis['primary_gate_passed'], 'rows': len(rows),
                      'authored_case_metrics': {k: v['metrics'] for k, v in demo['equal_error'].items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.study_dir, args.output_dir)
