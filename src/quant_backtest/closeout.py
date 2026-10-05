"""Comparable, initially-cash replays of frozen instructions; no selection here."""
from pathlib import Path

import pandas as pd

from .costs import BpsCost
from .data import frame_sha256, load_price_snapshot
from .data_quality import price_returns, validate_prices
from .engine import EngineConfig, run_weight_backtest
from .metrics import annualized_turnover, summarize_performance
from .research_config import ResearchConfig
from .research_data import cash_return_series


def load_verified_snapshot(path: Path, expected_hash: str) -> pd.DataFrame:
    prices = load_price_snapshot(path)
    if frame_sha256(prices) != expected_hash:
        raise ValueError('Snapshot content hash does not match the declared immutable input.')
    return prices


def build_policy_comparison(prices: pd.DataFrame, config: ResearchConfig,
                            schedules: dict[str, pd.DataFrame],
                            costs=(0, 10, 20, 50), lags=(1, 2)) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_prices(prices)
    evaluation = prices.loc[config.test_start:config.test_end]
    if evaluation.empty:
        raise ValueError('Empty common evaluation calendar.')
    policies = {f'aapl_cash_{int(weight * 100)}': pd.DataFrame(
        {config.base_ticker: weight}, index=evaluation.index) for weight in (.25, .5, .75, 1.)}
    if policies.keys() & schedules.keys():
        raise ValueError('Reserved baseline policy name.')
    for name, targets in schedules.items():
        if not targets.index.is_unique or not evaluation.index.isin(targets.index).all():
            raise ValueError(f'Targets for {name} must cover every common date once.')
        policies[name] = targets.loc[evaluation.index].copy()
    returns = price_returns(evaluation)
    cash = cash_return_series(evaluation, config.cash_proxy_ticker)
    reference = cash if cash is not None else 0.0
    summaries, audits = [], []
    for name, targets in policies.items():
        for cost in costs:
            for lag in lags:
                result = run_weight_backtest(returns, targets, EngineConfig(
                    config.initial_capital, BpsCost(float(cost)), int(lag)), cash)
                curve = result.curve.copy()
                earning = result.executed_weights.sum(axis=1)
                row = summarize_performance(name, curve.strategy_equity, curve.strategy_return,
                    risk_free_rate=reference, exposure=float((earning > 0).mean()),
                    turnover=annualized_turnover(curve.turnover))
                prior_nav = curve.strategy_equity.shift(1, fill_value=config.initial_capital)
                row.update(policy=name, cost_bps=cost, lag=lag, sessions=len(curve),
                    start=str(curve.index[0].date()), end=str(curve.index[-1].date()),
                    ending_nav=float(curve.strategy_equity.iloc[-1]),
                    average_earning_exposure=float(earning.mean()),
                    fees_paid=float((prior_nav * curve.transaction_cost).sum()))
                summaries.append(row)
                curve['earning_exposure'] = earning
                curve['return_reference'] = reference
                for prefix, weights in [('target', targets), ('earning', result.executed_weights),
                                        ('closing', result.closing_weights)]:
                    for asset in weights:
                        curve[f'{prefix}_{asset}'] = weights[asset]
                curve.index.name = 'Date'
                curve = curve.reset_index()
                curve['policy'], curve['cost_bps'], curve['lag'] = name, cost, lag
                audits.append(curve)
    return pd.DataFrame(summaries), pd.concat(audits, ignore_index=True)


def period_results(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar years extracted from continuous returns; no extra entry or reset."""
    rows = []
    working = daily.copy()
    working['year'] = pd.to_datetime(working.Date).dt.year
    for (policy, cost, lag, year), segment in working.groupby(['policy', 'cost_bps', 'lag', 'year']):
        segment = segment.set_index('Date')
        row = summarize_performance(policy, (1 + segment.strategy_return).cumprod(),
            segment.strategy_return, risk_free_rate=segment.return_reference,
            turnover=annualized_turnover(segment.turnover))
        row.update(policy=policy, cost_bps=cost, lag=lag, year=year,
                   sessions=len(segment), average_earning_exposure=segment.earning_exposure.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def audit_prices(prices: pd.DataFrame) -> dict:
    """Inspect supplied sessions against XNYS; do not repair or fill the data."""
    import exchange_calendars as calendars
    validate_prices(prices)
    expected = calendars.get_calendar('XNYS', start=prices.index[0], end=prices.index[-1]).sessions
    expected = expected.tz_localize(None) if expected.tz is not None else expected
    returns = price_returns(prices)
    split_date = pd.Timestamp('2020-08-31')
    return {'data_sha256': frame_sha256(prices), 'sessions': len(prices),
            'start': str(prices.index[0].date()), 'end': str(prices.index[-1].date()),
            'tickers': list(prices.columns), 'missing_values': int(prices.isna().sum().sum()),
            'duplicate_dates': int(prices.index.duplicated().sum()),
            'calendar': 'XNYS (shared US equity full-day session coverage proxy)',
            'calendar_library_version': calendars.__version__,
            'missing_sessions': [str(date.date()) for date in expected.difference(prices.index)],
            'unexpected_sessions': [str(date.date()) for date in prices.index.difference(expected)],
            'maximum_absolute_daily_return': returns.abs().max().to_dict(),
            'aapl_split_2020_08_31_adjusted_return': float(returns.loc[split_date, 'AAPL'])
                if split_date in returns.index and 'AAPL' in returns else None,
            'adjustments_independently_certified': False,
            'vintage': 'retrospectively adjusted vendor snapshot, not point-in-time data',
            'repair_policy': 'no fill, no deletion, no automatic outlier repair'}


def overlap_revisions(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Separate vendor historical revisions from the additional observations."""
    dates = old.index.intersection(new.index)
    rows = []
    for ticker in old.columns.intersection(new.columns):
        prior, latest = old.loc[dates, ticker], new.loc[dates, ticker]
        ratio = latest / prior
        old_returns = prior.pct_change(fill_method=None)
        new_returns = latest.pct_change(fill_method=None)
        rows.append({'ticker': ticker, 'overlap_sessions': len(dates),
                     'price_ratio_first': ratio.iloc[0], 'price_ratio_last': ratio.iloc[-1],
                     'max_normalized_price_revision': float((ratio / ratio.iloc[0] - 1).abs().max()),
                     'max_daily_return_revision': float((new_returns - old_returns).abs().max()),
                     'new_sessions': len(new.index.difference(old.index)),
                     'missing_old_sessions': len(old.index.difference(new.index))})
    return pd.DataFrame(rows)
