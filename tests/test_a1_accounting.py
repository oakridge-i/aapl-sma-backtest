from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_backtest.costs import BpsCost
from quant_backtest.engine import EngineConfig, run_weight_backtest
from quant_backtest.continuous import continuous_oos
from quant_backtest.data_quality import price_returns, validate_prices
from quant_backtest.evaluation import evaluate_strategy
from quant_backtest.metrics import sharpe_ratio, sortino_ratio
from quant_backtest.significance import _stitched_oos_significance
from quant_backtest.strategies import SmaParameters
from test_m3_overlays import m3_config


def test_next_close_trade_cannot_earn_the_move_into_that_close():
    idx = pd.bdate_range('2024-01-01', periods=4)
    returns = pd.DataFrame({'A': [0, 1.0, 0.1, 0]}, index=idx)
    weights = pd.DataFrame({'A': [1, 1, 1, 1]}, index=idx)
    result = run_weight_backtest(returns, weights, EngineConfig(100, BpsCost(0)))
    assert result.curve.strategy_equity.tolist() == pytest.approx([100, 100, 110, 110])
    assert result.closing_weights.A.tolist() == [0, 1, 1, 1]
    assert result.executed_weights.A.tolist() == [0, 0, 1, 1]
    # Earlier timing gave this post-close signal the move into the next close.
    earlier_timing = run_weight_backtest(returns, weights, EngineConfig(100, BpsCost(0), execution_lag=0))
    assert earlier_timing.curve.strategy_equity.iloc[-1] == pytest.approx(220)


def test_half_weight_rebalances_drift_and_pays_from_nav():
    idx = pd.bdate_range('2024-01-01', periods=3)
    returns = pd.DataFrame({'A': [0, 0, 0.1]}, index=idx)
    target = pd.DataFrame({'A': 0.5}, index=idx)
    result = run_weight_backtest(returns, target, EngineConfig(200, BpsCost(100)))
    # Initial buy: stock=0.5*V, V + 1%*stock = 200.
    after_entry = 200 / 1.005
    assert result.curve.strategy_equity.iloc[1] == pytest.approx(after_entry)
    stock, cash = after_entry * .5 * 1.1, after_entry * .5
    # Sale: V = stock+cash - 1%*(stock - V/2).
    final = (stock + cash - .01 * stock) / .995
    assert result.curve.strategy_equity.iloc[2] == pytest.approx(final)
    assert result.curve.turnover.iloc[2] > 0
    assert result.curve.transaction_cost.iloc[2] * after_entry == pytest.approx((stock - final / 2) * .01)


def test_engine_matches_independent_dollar_ledger():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range('2023-01-01', periods=40)
    r = pd.DataFrame(rng.normal(0, .035, (40, 2)), columns=['A', 'B'], index=idx)
    r.iloc[0] = 0
    w = pd.DataFrame(rng.dirichlet([1, 1, 1], 40)[:, :2], columns=r.columns, index=idx)
    cash_r = pd.Series(.0002, index=idx)
    cash_r.iloc[0] = 0
    result = run_weight_backtest(r, w, EngineConfig(1000, BpsCost(20)), cash_r)
    holdings, cash, expected = np.zeros(2), 1000., []
    for i in range(40):
        holdings *= 1 + r.iloc[i].to_numpy()
        cash *= 1 + cash_r.iloc[i]
        value = holdings.sum() + cash
        target = np.zeros(2) if i == 0 else w.iloc[i - 1].to_numpy()
        lo, hi = 0., value
        for _ in range(60):
            mid = (lo + hi) / 2
            required = mid + .002 * np.abs(target * mid - holdings).sum()
            if required > value:
                hi = mid
            else:
                lo = mid
        nav = (lo + hi) / 2
        holdings = target * nav
        cash = nav - holdings.sum()
        assert cash >= -1e-10
        expected.append(nav)
    np.testing.assert_allclose(result.curve.strategy_equity, expected, rtol=1e-12)


def test_continuous_account_keeps_boundary_move_and_avoids_reentry(tmp_path):
    idx = pd.bdate_range('2024-01-01', periods=8)
    prices = pd.DataFrame({'AAPL': [100, 100, 100, 100, 120, 120, 120, 120]}, index=idx)
    targets = pd.DataFrame({'AAPL': 1.0}, index=idx)
    windows = pd.DataFrame({'test_start': [str(idx[0].date()), str(idx[4].date())],
                            'test_end': [str(idx[3].date()), str(idx[-1].date())]})
    config = replace(m3_config(tmp_path), cash_proxy_ticker=None)
    result = continuous_oos(prices, config, windows, [targets.iloc[:4], targets.iloc[4:]], 'test')
    direct = run_weight_backtest(price_returns(prices), targets, EngineConfig(config.initial_capital, BpsCost(10)))
    pd.testing.assert_series_equal(result['oos_returns'], direct.curve.strategy_return)
    assert result['oos_returns'].loc[idx[4]] == pytest.approx(.2)
    assert result['curve'].turnover.loc[idx[4]] == pytest.approx(0)
    assert result['windows'].iloc[1].cagr > 0


@pytest.mark.parametrize('overlap', [False, True])
def test_continuous_schedule_rejects_ambiguous_coverage(tmp_path, overlap):
    idx = pd.bdate_range('2024-01-01', periods=8)
    p = pd.DataFrame({'AAPL': 100.}, index=idx)
    w = p / 100
    pieces = [w.iloc[:4], w.iloc[3:] if overlap else w.iloc[5:]]
    with pytest.raises(ValueError, match='windows'):
        continuous_oos(p, replace(m3_config(tmp_path), cash_proxy_ticker=None), pd.DataFrame(), pieces, 'test')


def test_daily_reference_is_aligned_and_used_in_denominator():
    idx = pd.bdate_range('2024-01-01', periods=4)
    excess = pd.Series([.01, -.01, .02, -.02], index=idx)
    rf = pd.Series([0, .04, 0, .03], index=idx)
    assert sharpe_ratio(excess + rf, rf) == pytest.approx(sharpe_ratio(excess))
    assert sortino_ratio(excess + rf, rf) == pytest.approx(sortino_ratio(excess))
    with pytest.raises(ValueError, match='cover'):
        sharpe_ratio(excess, rf.iloc[1:])


def test_continuous_summary_matches_significance_with_variable_cash(tmp_path):
    idx = pd.bdate_range('2020-01-01', periods=80)
    p = pd.DataFrame({'AAPL': 100 * np.cumprod(1 + .002 + .01 * np.sin(np.arange(80))),
                      'BIL': 100 * np.cumprod(1 + .0001 + .00005 * np.cos(np.arange(80)))}, index=idx)
    w = pd.DataFrame({'AAPL': .5}, index=idx)
    windows = pd.DataFrame({'test_start': [str(idx[0].date())], 'test_end': [str(idx[-1].date())]})
    config = replace(m3_config(tmp_path), cash_proxy_ticker='BIL', bootstrap_iterations=10)
    result = continuous_oos(p, config, windows, [w], 'test')
    sig = _stitched_oos_significance(result['oos_returns'], p, config)
    for key in ['cagr', 'sharpe', 'max_drawdown']:
        assert result['summary'].iloc[0][key] == pytest.approx(sig['observed_' + key])


@pytest.mark.parametrize('defect', ['nan', 'zero', 'infinity', 'duplicate', 'unsorted'])
def test_invalid_prices_fail_before_signal_generation(defect):
    p = pd.DataFrame({'AAPL': [100., 101., 102.]}, index=pd.bdate_range('2024-01-01', periods=3))
    if defect == 'duplicate':
        p.index = [p.index[0], p.index[0], p.index[2]]
    elif defect == 'unsorted':
        p = p.iloc[::-1]
    else:
        p.iloc[1, 0] = {'nan': np.nan, 'zero': 0., 'infinity': np.inf}[defect]
    with pytest.raises(ValueError):
        evaluate_strategy(p, 'AAPL', SmaParameters(1, 2), 'long_cash', 10, 100, 'test')


def test_missing_adjusted_close_is_not_silently_replaced(monkeypatch):
    import quant_backtest.data as data
    p = pd.DataFrame({'Open': [100.], 'High': [100.], 'Low': [100.], 'Close': [100.], 'Volume': [10]},
                     index=pd.to_datetime(['2024-01-02']))
    monkeypatch.setattr(data.yf, 'download', lambda *a, **kw: p)
    with pytest.raises(ValueError, match='Adjusted Close'):
        data.download_ohlcv('AAPL', '2024-01-01')


def test_missing_cash_proxy_fails():
    p = pd.DataFrame({'AAPL': [100., 101., 102.]}, index=pd.bdate_range('2024-01-01', periods=3))
    with pytest.raises(ValueError, match='cash proxy'):
        evaluate_strategy(p, 'AAPL', SmaParameters(1, 2), 'long_cash', 10, 100, 'test', cash_proxy='BIL')


def test_nested_fallback_cannot_use_global_future_fit(tmp_path, monkeypatch):
    import quant_backtest.ensemble_research as e
    idx = pd.bdate_range('2020-01-01', periods=310)
    p = pd.DataFrame({'AAPL': np.linspace(100, 200, len(idx))}, index=idx)
    config = replace(m3_config(tmp_path), cash_proxy_ticker=None, enable_overlays=False)
    monkeypatch.setattr(e, 'walk_forward_windows', lambda *_: [(1, idx[0], idx[269], idx[270], idx[-1])])
    monkeypatch.setattr(e, 'run_family_sweep', lambda *a, **k: (pd.DataFrame(), []))
    monkeypatch.setattr(e, 'run_ensemble_leaderboard', lambda *a, **k: pd.DataFrame())
    a = e.run_nested_ensemble_walk_forward(p, config, {'params': SmaParameters(1, 2), 'variant': 'long_cash'})
    b = e.run_nested_ensemble_walk_forward(p, config, {'params': SmaParameters(2, 300), 'variant': 'long_cash'})
    pd.testing.assert_series_equal(a['oos_returns'], b['oos_returns'])
    assert a['windows'].iloc[0].selected_label == SmaParameters(20, 100).label()


def test_legacy_and_new_comparisons_share_warm_history_and_metrics(tmp_path):
    from quant_backtest.sweeps import run_v03_comparison, run_v04_comparison
    from quant_backtest.ensemble_research import run_v06_comparison
    from quant_backtest.significance import run_significance_analysis
    idx = pd.bdate_range('2020-01-01', periods=360)
    x = 100 + np.arange(360) * .1 + np.sin(np.arange(360))
    p = pd.DataFrame({'AAPL': x, 'SPY': x, 'QQQ': x, 'BIL': 100 + np.arange(360) * .002}, index=idx)
    config = replace(m3_config(tmp_path), test_start=str(idx[280].date()), test_end=str(idx[-1].date()),
                     cash_proxy_ticker='BIL', bootstrap_iterations=10, permutation_iterations=10)
    model = {'params': SmaParameters(20, 100), 'variant': 'long_cash', 'selection_status': 'test'}
    v3, _ = run_v03_comparison(p, config, model)
    v4, _ = run_v04_comparison(p, config, model, model)
    v6, _ = run_v06_comparison(p, config, model, {}, model)
    sig = run_significance_analysis(p, config, model, model, pd.DataFrame(), pd.DataFrame())
    expected = v3.loc[v3.model.eq('selected_v3')].iloc[0]
    for table, label in [(v4, 'selected_v3'), (v4, 'selected_v4'), (v6, 'selected_v6')]:
        row = table.loc[table.model.eq(label)].iloc[0]
        for metric in ['cagr', 'sharpe', 'max_drawdown']:
            assert row[metric] == pytest.approx(expected[metric])
    for _, row in sig.iterrows():
        assert row.observed_sharpe == pytest.approx(expected.sharpe)
        assert row.permutation_observed_sharpe == pytest.approx(expected.sharpe)


def test_cash_remuneration_and_metric_reference_are_separate():
    idx = pd.bdate_range('2020-01-01', periods=60)
    p = pd.DataFrame({'AAPL': 100 + np.arange(60), 'BIL': 100 + np.arange(60) * .01}, index=idx)
    args = (p, 'AAPL', SmaParameters(2, 10), 'long_cash', 10, 100, 'test')
    default = evaluate_strategy(*args, cash_proxy='BIL')
    zero = evaluate_strategy(*args, cash_proxy='BIL', reference_returns=0.)
    pd.testing.assert_series_equal(default['curve'].strategy_equity, zero['curve'].strategy_equity)
    assert default['row']['sharpe'] != zero['row']['sharpe']
    assert zero['risk_free_rate'] == 0
