import numpy as np
import pandas as pd
import pytest
import hashlib
import json
import subprocess
import sys
import yaml
from dataclasses import replace
from pathlib import Path

from quant_backtest.closeout import build_policy_comparison, load_verified_snapshot
from quant_backtest.experiments import ResearchResult
from quant_backtest.research_config import load_research_config


def config_for_test(**kwargs):
    return replace(load_research_config(Path('configs/research_v6.yaml')), **kwargs)


def test_flat_allocations_pay_financed_fees_and_share_calendar():
    dates = pd.bdate_range('2021-01-01', periods=5)
    prices = pd.DataFrame({'AAPL': 100., 'BIL': 100.}, index=dates)
    config = config_for_test(test_start='2021-01-01', test_end='2021-01-08', cash_proxy_ticker='BIL')
    summary, daily = build_policy_comparison(prices, config, {}, costs=(100,), lags=(1,))
    for weight in (.25, .5, .75, 1.):
        name = f'aapl_cash_{int(weight * 100)}'
        rows = daily[daily.policy == name]
        assert list(rows.Date) == list(dates)
        np.testing.assert_allclose(rows.strategy_equity.iloc[1:], config.initial_capital / (1 + .01 * weight))
        assert rows.earning_exposure.iloc[1] == 0
        assert rows.earning_exposure.iloc[2] == weight
    assert len(summary) == 4


def test_cost_and_lag_replay_same_targets_without_mutation():
    dates = pd.bdate_range('2021-01-01', periods=7)
    prices = pd.DataFrame({'AAPL': [100, 100, 110, 99, 100, 101, 103]}, index=dates)
    targets = pd.DataFrame({'AAPL': [1, 0, .5, 1, 0, 0, 1]}, index=dates)
    original = targets.copy()
    config = config_for_test(test_start='2021-01-01', test_end='2021-01-31', cash_proxy_ticker=None)
    summary, daily = build_policy_comparison(prices, config, {'selected_v3': targets})
    for _, rows in daily[daily.policy == 'selected_v3'].groupby(['cost_bps', 'lag']):
        np.testing.assert_array_equal(rows.target_AAPL, targets.AAPL)
    pd.testing.assert_frame_equal(targets, original)
    assert len(summary[summary.policy == 'selected_v3']) == 8
    with pytest.raises(ValueError, match='cover'):
        build_policy_comparison(prices, config, {'broken': targets.iloc[1:]})


def test_verified_snapshot_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_verified_snapshot(tmp_path / 'absent.csv', 'wrong')
    path = tmp_path / 'prices.csv'
    pd.DataFrame({'AAPL': [100., 101.]}, index=pd.bdate_range('2021-01-01', periods=2)).to_csv(path, index_label='Date')
    with pytest.raises(ValueError, match='hash'):
        load_verified_snapshot(path, 'wrong')


def test_research_result_exposes_selected_models_for_frozen_replay():
    assert 'selected_models' in ResearchResult.__dataclass_fields__


def test_period_metrics_include_first_day_pnl_without_account_reset():
    from quant_backtest.closeout import period_results
    daily = pd.DataFrame({'Date': pd.to_datetime(['2021-12-31', '2022-01-03', '2022-01-04']),
                          'policy': 'example', 'cost_bps': 10, 'lag': 1,
                          'strategy_return': [.1, -.1, .2], 'return_reference': 0.,
                          'earning_exposure': .5, 'turnover': [1., 2., 0.]})
    years = period_results(daily)
    row = years[years.year == 2022].iloc[0]
    assert row.total_return == pytest.approx(.08)
    assert row.max_drawdown == pytest.approx(-.1)


def test_cli_offline_replay_verifies_targets_and_matches_direct_engine(tmp_path):
    from quant_backtest.data import frame_sha256
    from quant_backtest.closeout import config_sha256
    dates = pd.bdate_range('2021-01-01', periods=8)
    prices = pd.DataFrame({'AAPL': np.arange(100., 108.), 'BIL': 100.}, index=dates)
    snapshot = tmp_path / 'prices.csv'
    prices.to_csv(snapshot, index_label='Date')
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'targets_fixed.csv'
    targets = pd.DataFrame({'AAPL': [1, 0, 1, .5, 0, 0, 1, 1]}, index=dates)
    targets.to_csv(path, index_label='Date')
    config_path = Path('configs/research_v6.yaml')
    manifest = {'data_sha256': frame_sha256(prices),
                'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'config_semantic_sha256': config_sha256(config_path),
                'targets': {'fixed': {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    (source / 'target_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    portable_config = tmp_path / 'config.yaml'
    portable_config.write_bytes(config_path.read_bytes().replace(b'\r\n', b'\n'))
    command = [sys.executable, '-X', 'utf8', 'scripts/finalize_research.py', '--config', str(portable_config),
               '--snapshot', str(snapshot), '--expected-hash', frame_sha256(prices), '--replay', str(source)]
    output = tmp_path / 'replay'
    completed = subprocess.run(command + ['--output', str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    expected, _ = build_policy_comparison(prices, load_research_config(config_path), {'fixed': targets})
    actual = pd.read_csv(output / 'sensitivity.csv')
    np.testing.assert_allclose(actual.ending_nav, expected.ending_nav, rtol=1e-12)
    path.write_text(path.read_text() + '\n')
    rejected = subprocess.run(command + ['--output', str(tmp_path / 'tampered')], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'Target hash mismatch' in rejected.stderr


def test_config_receipt_is_portable_across_line_endings(tmp_path):
    from quant_backtest.closeout import config_sha256
    lf = tmp_path / 'lf.yaml'
    crlf = tmp_path / 'crlf.yaml'
    lf.write_bytes(b'period:\n  start: 2015-01-01\ncompute:\n  workers: 8\n')
    crlf.write_bytes(lf.read_bytes().replace(b'\n', b'\r\n'))
    assert config_sha256(lf) == config_sha256(crlf)
    crlf.write_bytes(crlf.read_bytes().replace(b'workers: 8', b'workers: 4'))
    assert config_sha256(lf) != config_sha256(crlf)


def test_full_run_with_extra_history_replays_its_effective_snapshot(tmp_path):
    from quant_backtest.data import frame_sha256, load_price_snapshot
    settings = yaml.safe_load(Path('configs/research_v6.yaml').read_text(encoding='utf-8-sig'))
    settings.update(period={'start': '2019-01-01', 'end': '2021-12-31'},
        universe=['AAPL', 'SPY', 'QQQ', 'BIL'], compute={'workers': 0},
        sma_grid={'short': [5], 'long': [20]},
        train_test={'train_start': '2019-01-01', 'train_end': '2020-12-31',
                    'test_start': '2021-01-01', 'test_end': '2021-12-31'},
        walk_forward={'train_years': 1, 'test_years': 1, 'step_years': 1},
        hysteresis={'entry_thresholds': [0.], 'exit_thresholds': [0.],
                    'min_hold_days': [0], 'cooldown_days': [0]},
        capture_model={'enabled': False}, significance={'enabled': False}, pbo={'enabled': False},
        overlays={'enabled': False}, ensemble={'min_members': 2, 'max_members': 2,
                                             'include_trend_baseline': True},
        signal_families={'enabled': True, 'ts_momentum': {'lookbacks': [21, 63]},
            'donchian': {'entry_windows': [], 'exit_windows': []},
            'atr_trend': {'sma_windows': [], 'atr_windows': [], 'scales': []},
            'dual_momentum': {'lookbacks': [], 'market': 'SPY'},
            'high_52w': {'entry_thresholds': [], 'exit_thresholds': []}})
    config_path = tmp_path / 'small.yaml'
    config_path.write_text(yaml.safe_dump(settings), encoding='utf-8')
    dates = pd.bdate_range('2018-01-01', '2022-02-28')
    time = np.arange(len(dates))
    prices = pd.DataFrame({'AAPL': 100 * np.exp(.0005 * time + .02 * np.sin(time / 11)),
                          'SPY': 100 * np.exp(.0003 * time + .01 * np.sin(time / 20)),
                          'QQQ': 100 * np.exp(.0007 * time + .02 * np.sin(time / 15)),
                          'BIL': 100 * np.exp(.0001 * time)}, index=dates)
    snapshot = tmp_path / 'wide.csv'
    prices.to_csv(snapshot, index_label='Date')
    output = tmp_path / 'full'
    command = [sys.executable, '-X', 'utf8', 'scripts/finalize_research.py', '--config', str(config_path)]
    full = subprocess.run(command + ['--snapshot', str(snapshot), '--expected-hash', frame_sha256(prices),
                                    '--output', str(output)], capture_output=True, text=True)
    assert full.returncode == 0, full.stderr
    effective = load_price_snapshot(output / 'data_snapshot.csv')
    manifest = json.loads((output / 'target_manifest.json').read_text(encoding='utf-8'))
    assert manifest['data_sha256'] == frame_sha256(effective)
    assert manifest['input_data_sha256'] == frame_sha256(prices)
    for target in manifest['targets'].values():
        saved = pd.read_csv(output / target['file'], index_col='Date', parse_dates=True)
        assert saved.index.min() >= pd.Timestamp('2019-01-01')
        assert saved.index.max() <= pd.Timestamp('2021-12-31')
    replay = tmp_path / 'replay_effective'
    result = subprocess.run(command + ['--snapshot', str(output / 'data_snapshot.csv'),
        '--expected-hash', frame_sha256(effective), '--replay', str(output), '--output', str(replay)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = pd.read_csv(output / 'sensitivity.csv')
    repeated = pd.read_csv(replay / 'sensitivity.csv')
    np.testing.assert_allclose(original.ending_nav, repeated.ending_nav, rtol=1e-12)
