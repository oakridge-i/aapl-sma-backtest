"""One self-financing account for an out-of-sample target schedule."""
from __future__ import annotations

import pandas as pd

from .costs import BpsCost
from .data_quality import price_returns
from .engine import EngineConfig, run_weight_backtest
from .metrics import annualized_turnover, summarize_performance
from .research_data import cash_return_series


def continuous_oos(prices, config, windows, target_pieces, label):
    targets = pd.concat(target_pieces).sort_index().fillna(0.0)
    if not targets.index.is_unique:
        raise ValueError("Continuous OOS requires nonoverlapping windows; overlapping selections need an explicit policy.")
    evaluation = prices.loc[targets.index[0]:targets.index[-1]]
    if not targets.index.equals(evaluation.index):
        raise ValueError("Continuous OOS windows must cover every supplied session without gaps.")
    returns = price_returns(evaluation[targets.columns])
    cash = cash_return_series(evaluation, config.cash_proxy_ticker)
    reference = cash if cash is not None else 0.0
    result = run_weight_backtest(returns, targets, EngineConfig(config.initial_capital, BpsCost(10.0)), cash)
    curve = result.curve
    benchmark = price_returns(evaluation[[config.base_ticker]])[config.base_ticker]
    benchmark_equity = config.initial_capital * (1 + benchmark).cumprod()
    table = windows.copy()
    for i, window in table.iterrows():
        segment = curve.loc[window.test_start:window.test_end]
        r = segment.strategy_return
        b = benchmark.reindex(r.index)
        ref = reference.reindex(r.index) if isinstance(reference, pd.Series) else reference
        metrics = summarize_performance(label, (1 + r).cumprod(), r, risk_free_rate=ref)
        bm = summarize_performance("benchmark", (1 + b).cumprod(), b, risk_free_rate=ref)
        for key in ["cagr", "sharpe", "max_drawdown"]:
            if key in table:
                table.loc[i, "independent_window_" + key] = table.loc[i, key]
            table.loc[i, key] = metrics[key]
        table.loc[i, "turnover"] = annualized_turnover(segment.turnover)
        table.loc[i, "exposure"] = (result.executed_weights.loc[r.index].sum(axis=1) > 0).mean()
        table.loc[i, "benchmark_cagr"] = bm["cagr"]
        table.loc[i, "benchmark_sharpe"] = bm["sharpe"]
        table.loc[i, "excess_cagr_vs_benchmark"] = metrics["cagr"] - bm["cagr"]
    table["accounting"] = "continuous_account"
    row = summarize_performance(label, curve.strategy_equity, curve.strategy_return,
                                risk_free_rate=reference, benchmark_equity=benchmark_equity)
    bm = summarize_performance("benchmark_stitched", benchmark_equity, benchmark, risk_free_rate=reference)
    row |= {"windows": len(table), "accounting": "continuous_account",
            "windows_beating_benchmark": int((table.sharpe > table.benchmark_sharpe).sum()),
            "median_window_sharpe": table.sharpe.median(), "median_window_cagr": table.cagr.median()}
    bm |= {"windows": len(table), "accounting": "continuous_account"}
    curve = curve.assign(buy_hold_return=benchmark, buy_hold_equity=benchmark_equity)
    for asset in targets.columns:
        curve[f"target_{asset}"] = targets[asset]
        curve[f"earning_weight_{asset}"] = result.executed_weights[asset]
        curve[f"closing_weight_{asset}"] = result.closing_weights[asset]
    return {"windows": table, "summary": pd.DataFrame([row, bm]),
            "oos_returns": curve.strategy_return, "curve": curve,
            "target_weights": targets, "weights": result.executed_weights,
            "closing_weights": result.closing_weights}
