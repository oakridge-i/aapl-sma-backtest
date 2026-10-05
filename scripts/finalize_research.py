"""Full frozen-grid research followed by comparable execution sensitivity.

Replay consumes only explicit CSV/JSON inputs, never serialized Python objects.
"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from quant_backtest import experiments
from quant_backtest.closeout import build_policy_comparison, config_sha256, load_verified_snapshot
from quant_backtest.data import frame_sha256
from quant_backtest.evaluation import evaluate_strategy
from quant_backtest.reports import save_research_outputs
from quant_backtest.strategies import SmaParameters


def json_value(value):
    if is_dataclass(value):
        return {'type': type(value).__name__, 'fields': asdict(value)}
    raise TypeError(type(value).__name__)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_comparison(prices, config, schedules, output):
    summary, daily = build_policy_comparison(prices, config, schedules)
    summary.to_csv(output / 'sensitivity.csv', index=False)
    summary[(summary.cost_bps == 10) & (summary.lag == 1)].to_csv(output / 'comparison.csv', index=False)
    daily.to_csv(output / 'daily_audit.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--expected-hash', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay', help='Existing directory with hashed target_manifest.json; skip selection.')
    args = parser.parse_args()
    config_path = Path(args.config)
    config = experiments.load_research_config(config_path)
    prices = load_verified_snapshot(Path(args.snapshot), args.expected_hash)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty: preserved evidence is never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    if args.replay:
        source = Path(args.replay)
        manifest = json.loads((source / 'target_manifest.json').read_text(encoding='utf-8'))
        config_matches = (manifest['config_semantic_sha256'] == config_sha256(config_path)
                          if 'config_semantic_sha256' in manifest
                          else manifest['config_sha256'] == file_hash(config_path))
        if manifest['data_sha256'] != frame_sha256(prices) or not config_matches:
            raise ValueError('Replay data/config hash mismatch.')
        schedules = {}
        for name, receipt in manifest['targets'].items():
            path = source / receipt['file']
            if file_hash(path) != receipt['sha256']:
                raise ValueError(f'Target hash mismatch: {name}')
            schedules[name] = pd.read_csv(path, index_col='Date', parse_dates=True)
        write_comparison(prices, config, schedules, output)
        (output / 'replay_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Verified offline replay complete.', flush=True)
        return

    # Timing wrappers leave arguments, return values, order and selectors unchanged.
    for name in [name for name in vars(experiments) if name.startswith('run_') and name != 'run_research']:
        original = getattr(experiments, name)
        def timed(*positional, _function=original, _name=name, **keywords):
            started = time.monotonic()
            print(f'START {_name}', flush=True)
            result = _function(*positional, **keywords)
            print(f'DONE {_name}: {time.monotonic() - started:.1f}s', flush=True)
            return result
        setattr(experiments, name, timed)
    result = experiments.run_research(config, snapshot_path=Path(args.snapshot))
    save_research_outputs(result, output)
    schedules = {}
    models = {'fixed_sma_20_100': {'params': SmaParameters(20, 100), 'variant': 'long_cash'},
              **result.selected_models}
    for name, model in models.items():
        if model is None:
            continue
        schedules[name] = evaluate_strategy(prices, config.base_ticker, model['params'], model['variant'],
            10, config.initial_capital, name, config.market_regime_short_window,
            config.market_regime_long_window, config.cash_proxy_ticker)['target_weights']
    for name, curve in [('nested_v3', result.nested_oos_curve),
                        ('nested_ensemble', result.nested_ensemble_oos_curve)]:
        columns = [col for col in curve if col.startswith('target_')]
        if not columns:
            raise ValueError(f'Missing nested target audit: {name}')
        schedules[name] = curve[columns].rename(columns=lambda col: col.removeprefix('target_'))
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'data_sha256': frame_sha256(prices), 'config_sha256': file_hash(config_path),
               'config_semantic_sha256': config_sha256(config_path),
               'git_commit': result.run_metadata['git_commit'], 'targets': {},
               'selected_models': models, 'scenario_selection': 'frozen_targets_no_refitting',
               'common_start': config.test_start, 'cash': config.cash_proxy_ticker}
    for name, targets in schedules.items():
        path = output / f'targets_{name}.csv'
        targets.to_csv(path, index_label='Date', lineterminator='\n')
        receipt['targets'][name] = {'file': path.name, 'sha256': file_hash(path)}
    (output / 'target_manifest.json').write_text(json.dumps(receipt, indent=2, default=json_value), encoding='utf-8')
    write_comparison(prices, config, schedules, output)
    print('Full research and frozen sensitivity complete.', flush=True)


if __name__ == '__main__':
    main()
