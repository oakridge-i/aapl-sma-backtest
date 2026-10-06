"""Publish compact evidence from preserved closeout runs; no raw prices copied."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from quant_backtest.closeout import audit_prices, build_policy_comparison, overlap_revisions, period_results
from quant_backtest.data import load_price_snapshot
from quant_backtest.research_config import load_research_config
from quant_backtest.stats import block_bootstrap_summary


def markdown(table):
    def cell(value):
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    return '| ' + ' | '.join(table.columns) + ' |\n| ' + ' | '.join(['---'] * len(table.columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(cell(value) for value in row) + ' |' for row in table.itertuples(index=False, name=None))


def publish_run(source, config_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    config = load_research_config(config_path)
    prices = load_price_snapshot(source / 'data_snapshot.csv')
    audit = audit_prices(prices)
    (destination / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for name in ['comparison.csv', 'sensitivity.csv', 'target_manifest.json', 'run_manifest.json',
                 'nested_walk_forward.csv', 'nested_ensemble_walk_forward.csv',
                 'nested_walk_forward_summary.csv', 'nested_ensemble_summary.csv',
                 'significance_results.csv', 'pbo_results.csv', 'v06_comparison.csv']:
        shutil.copyfile(source / name, destination / name)
    daily = pd.read_csv(source / 'daily_audit.csv', parse_dates=['Date'])
    baseline = daily[(daily.cost_bps == 10) & (daily.lag == 1)].copy()
    period_results(baseline).to_csv(destination / 'period_results.csv', index=False)
    uncertainty = []
    for policy, curve in baseline.groupby('policy', sort=False):
        curve = curve.set_index('Date')
        row = block_bootstrap_summary(curve.strategy_return, n_iterations=1000,
            block_size=21, seed=42, risk_free_rate=curve.return_reference)
        uncertainty.append({'policy': policy, 'scope': 'conditional_frozen_returns_common_calendar', **row})
    pd.DataFrame(uncertainty).to_csv(destination / 'bootstrap_common.csv', index=False)
    nested = {}
    for name, filename in [('nested_v3', 'nested_oos_curve.csv'), ('nested_ensemble', 'nested_ensemble_oos_curve.csv')]:
        curve = pd.read_csv(source / filename, index_col='Date', parse_dates=True)
        columns = [column for column in curve if column.startswith('target_')]
        nested[name] = curve[columns].rename(columns=lambda column: column.removeprefix('target_'))
    start = max(target.index[0] for target in nested.values())
    full_summary, _ = build_policy_comparison(prices, replace(config, test_start=str(start.date())), nested,
                                              costs=(10,), lags=(1,))
    full_summary.to_csv(destination / 'full_nested_comparison.csv', index=False)
    figures = destination / 'figures'
    figures.mkdir(exist_ok=True)
    chosen = ['aapl_cash_100', 'aapl_cash_25', 'aapl_cash_50', 'aapl_cash_75',
              'fixed_sma_20_100', 'selected_v6', 'nested_ensemble']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for policy in chosen:
        curve = baseline[baseline.policy == policy].set_index('Date')
        for axis, series in zip(axes.flat, [curve.strategy_equity / config.initial_capital,
                curve.strategy_drawdown * 100, curve.earning_exposure,
                curve.turnover.cumsum()]):
            axis.plot(series.index, series, label=policy, linewidth=1.1)
    for axis, title in zip(axes.flat, ['Net NAV / initial capital', 'Drawdown (%)',
                                      'Risky weight earning the return', 'Cumulative one-way turnover']):
        axis.set_title(title)
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(f'Common calendar {baseline.Date.min().date()} – {baseline.Date.max().date()} | 10 bps, lag 1')
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(figures / 'common_account.png', dpi=150)
    plt.close(fig)
    sensitivity = pd.read_csv(source / 'sensitivity.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for policy in sensitivity.policy.unique():
        for lag, axis in zip((1, 2), axes):
            curve = sensitivity[(sensitivity.policy == policy) & (sensitivity.lag == lag)]
            axis.plot(curve.cost_bps, curve.cagr * 100, marker='o', label=policy)
            axis.set_title(f'Frozen targets: lag {lag}')
            axis.set_xlabel('One-way fee (bps)')
            axis.set_ylabel('CAGR (%)')
            axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(figures / 'cost_lag_sensitivity.png', dpi=150)
    plt.close(fig)
    return pd.read_csv(source / 'comparison.csv'), audit


def archive_comparison(archive, historical, destination):
    old = pd.read_csv(archive / 'v06_comparison.csv')
    corrected = pd.read_csv(historical / 'v06_comparison.csv')
    rows = []
    for model in ['baseline_sma_20_100', 'selected_v3', 'selected_v6']:
        before, after = old[old.model == model], corrected[corrected.model == model]
        if len(before) != 1 or len(after) != 1:
            continue
        for metric in ['cagr', 'sharpe', 'max_drawdown', 'turnover']:
            rows.append({'scope': 'test_2021_plus', 'model': model, 'metric': metric,
                         'archived': before.iloc[0][metric], 'corrected': after.iloc[0][metric],
                         'delta': after.iloc[0][metric] - before.iloc[0][metric],
                         'archived_label': before.iloc[0]['name'], 'corrected_label': after.iloc[0]['name'],
                         'interpretation': 'combined accounting, warmup, metric and train-selection changes; same prices'})
    for model, filename in [('nested_v3', 'nested_walk_forward_summary.csv'),
                             ('nested_ensemble', 'nested_ensemble_summary.csv')]:
        before = pd.read_csv(archive / filename).iloc[0]
        after = pd.read_csv(historical / filename).iloc[0]
        for metric in ['cagr', 'sharpe', 'max_drawdown']:
            rows.append({'scope': 'full_nested_history', 'model': model, 'metric': metric,
                         'archived': before[metric], 'corrected': after[metric],
                         'delta': after[metric] - before[metric],
                         'archived_label': before['name'], 'corrected_label': after['name'],
                         'interpretation': 'continuous account versus independently reset windows; selection/accounting also changed'})
    pd.DataFrame(rows).to_csv(destination / 'archive_comparison.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--updated')
    parser.add_argument('--destination', default='docs/final')
    args = parser.parse_args()
    destination = Path(args.destination)
    historical, audit = publish_run(Path(args.historical), ROOT / 'configs/closeout_historical.yaml', destination / 'historical')
    archive_comparison(Path(args.archive), Path(args.historical), destination)
    if args.updated:
        publish_run(Path(args.updated), ROOT / 'configs/closeout_updated.yaml', destination / 'updated')
        overlap_revisions(load_price_snapshot(Path(args.historical) / 'data_snapshot.csv'),
                          load_price_snapshot(Path(args.updated) / 'data_snapshot.csv')).to_csv(
                              destination / 'historical_revisions.csv', index=False)
    columns = ['policy', 'cagr', 'sharpe', 'max_drawdown', 'average_earning_exposure', 'turnover', 'fees_paid']
    print(markdown(historical[columns]))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
