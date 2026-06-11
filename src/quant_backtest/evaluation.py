"""Single-model evaluation: signals -> weights -> engine -> metric row."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import calculate_closed_trade_returns, calculate_win_rate, count_exposure_episodes
from .costs import BpsCost
from .engine import EngineConfig, run_weight_backtest
from .metrics import (
    annualized_turnover,
    avoided_downside_while_underweight,
    capture_ratio,
    capture_spread,
    holding_periods,
    missed_return_while_underweight,
    summarize_performance,
    trade_frequency_per_year,
)
from .registry import build_strategy, family_for_params
from .research_config import ResearchConfig
from .research_data import cash_return_series
from .strategies import (
    CaptureAwareAllocationParameters,
    SmaParameters,
    TrendAllocationParameters,
    build_capture_aware_weights,
    build_fallback_weights,
    build_hybrid_regime_weights,
    build_regime_fallback_weights,
    build_single_asset_weights,
    build_sma_regime,
)

ParamsLike = SmaParameters | TrendAllocationParameters | CaptureAwareAllocationParameters


def evaluate_strategy(
    prices: pd.DataFrame,
    ticker: str,
    params: ParamsLike,
    variant: str,
    cost_bps: float,
    initial_capital: float,
    label: str,
    market_regime_short_window: int = 50,
    market_regime_long_window: int = 200,
    cash_proxy: str | None = None,
) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in prices.columns:
        raise ValueError(f"Missing ticker in price data: {ticker}")

    family = family_for_params(params)
    needed = [ticker]
    capture_fallback = _capture_fallback_ticker(params) if family.needs_market_context else None
    if variant in {"fallback_spy", "long_spy_regime", "hybrid_spy_regime"} or capture_fallback == "SPY":
        needed.append("SPY")
    if variant in {"fallback_qqq", "long_qqq_regime", "hybrid_qqq_regime"} or capture_fallback == "QQQ":
        needed.append("QQQ")
    needed = list(dict.fromkeys(needed))
    available = [column for column in needed if column in prices.columns]

    price = prices[ticker].dropna()
    strategy = build_strategy(params)
    if family.needs_market_context:
        fallback_ticker = capture_fallback if capture_fallback in prices.columns else None
        market_ticker = fallback_ticker or ("SPY" if "SPY" in prices.columns else "QQQ" if "QQQ" in prices.columns else None)
        market_risk_off = None
        if params.risk.use_market_sma_filter and market_ticker:
            market_regime = build_sma_regime(
                prices[market_ticker],
                market_regime_short_window,
                market_regime_long_window,
            )
            market_risk_off = ~market_regime
        fallback_regime = None
        if fallback_ticker:
            fallback_regime = build_sma_regime(
                prices[fallback_ticker],
                market_regime_short_window,
                market_regime_long_window,
            )
        signals = strategy.generate(
            price,
            market_risk_off=market_risk_off,
            fallback_regime=fallback_regime,
        )
        weights = build_capture_aware_weights(ticker, signals, fallback_ticker)
        available = [column for column in weights.columns if column in prices.columns]
    elif isinstance(params, TrendAllocationParameters):
        signals = strategy.generate(price)
        if variant == "fallback_spy" and "SPY" in prices.columns:
            weights = build_fallback_weights(ticker, "SPY", signals.target_position)
        elif variant == "fallback_qqq" and "QQQ" in prices.columns:
            weights = build_fallback_weights(ticker, "QQQ", signals.target_position)
        elif variant == "long_spy_regime" and "SPY" in prices.columns:
            regime = build_sma_regime(prices["SPY"], market_regime_short_window, market_regime_long_window)
            weights = build_regime_fallback_weights(ticker, "SPY", signals.target_position, regime)
        elif variant == "long_qqq_regime" and "QQQ" in prices.columns:
            regime = build_sma_regime(prices["QQQ"], market_regime_short_window, market_regime_long_window)
            weights = build_regime_fallback_weights(ticker, "QQQ", signals.target_position, regime)
        elif variant == "hybrid_spy_regime" and "SPY" in prices.columns:
            regime = build_sma_regime(prices["SPY"], market_regime_short_window, market_regime_long_window)
            weights = build_hybrid_regime_weights(ticker, "SPY", signals, params, regime)
        elif variant == "hybrid_qqq_regime" and "QQQ" in prices.columns:
            regime = build_sma_regime(prices["QQQ"], market_regime_short_window, market_regime_long_window)
            weights = build_hybrid_regime_weights(ticker, "QQQ", signals, params, regime)
        else:
            weights = build_single_asset_weights(ticker, signals.target_position)
    elif isinstance(params, SmaParameters):
        signals = strategy.generate(price)
        if variant == "fallback_spy" and "SPY" in prices.columns:
            weights = build_fallback_weights(ticker, "SPY", signals.target_position)
        elif variant == "fallback_qqq" and "QQQ" in prices.columns:
            weights = build_fallback_weights(ticker, "QQQ", signals.target_position)
        else:
            weights = build_single_asset_weights(ticker, signals.target_position)
    else:
        # Generic single-asset long/cash family from the registry.
        if family.needs_market_price:
            market_ticker = str(getattr(params, "market_ticker", "SPY")).upper()
            market_price = prices[market_ticker] if market_ticker in prices.columns else None
            signals = strategy.generate(price, market_price=market_price)
        else:
            signals = strategy.generate(price)
        weights = build_single_asset_weights(ticker, signals.target_position)

    cash_returns = cash_return_series(prices, cash_proxy)
    returns = prices[available].pct_change().fillna(0.0)
    engine_result = run_weight_backtest(
        returns=returns,
        target_weights=weights,
        config=EngineConfig(initial_capital=initial_capital, cost_model=BpsCost(cost_bps)),
        cash_returns=cash_returns,
    )
    risk_free_rate = 0.0
    if cash_returns is not None:
        aligned_cash = cash_returns.reindex(returns.index).dropna()
        if not aligned_cash.empty:
            risk_free_rate = float(aligned_cash.mean() * 252)
    curve = _combine_curve(price, signals, engine_result.curve, engine_result.executed_weights, ticker, prices)
    row = summarize_curve(
        curve=curve,
        executed_weights=engine_result.executed_weights,
        ticker=ticker,
        label=label,
        variant=variant,
        params=params,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
    )
    return {
        "row": row,
        "curve": curve,
        "metrics": pd.DataFrame([row]),
        "weights": engine_result.executed_weights,
        "risk_free_rate": risk_free_rate,
    }


def evaluate_equal_weight_signal_portfolio(
    prices: pd.DataFrame,
    universe: list[str],
    params: SmaParameters,
    config: ResearchConfig,
) -> dict[str, Any]:
    weights = []
    valid_tickers = [ticker for ticker in universe if ticker in prices.columns]
    for ticker in valid_tickers:
        signals = build_strategy(params).generate(prices[ticker].dropna())
        weights.append(signals.target_position.rename(ticker))
    target_weights = pd.concat(weights, axis=1).reindex(prices.index).fillna(0.0)
    if valid_tickers:
        target_weights = target_weights / len(valid_tickers)
    returns = prices[valid_tickers].pct_change().fillna(0.0)
    cash_returns = cash_return_series(prices, config.cash_proxy_ticker)
    result = run_weight_backtest(
        returns,
        target_weights,
        EngineConfig(initial_capital=config.initial_capital, cost_model=BpsCost(10.0)),
        cash_returns=cash_returns,
    )

    risk_free_rate = 0.0
    if cash_returns is not None:
        aligned_cash = cash_returns.reindex(returns.index).dropna()
        if not aligned_cash.empty:
            risk_free_rate = float(aligned_cash.mean() * 252)
    basket_return = returns.mean(axis=1)
    benchmark_equity = config.initial_capital * (1.0 + basket_return).cumprod()
    total_exposure = result.executed_weights.abs().sum(axis=1)
    closed = calculate_closed_trade_returns(result.curve["strategy_return"], total_exposure)
    win_rate = calculate_win_rate(closed)
    row = summarize_performance(
        name="equal_weight_signal_portfolio",
        equity=result.curve["strategy_equity"],
        returns=result.curve["strategy_return"],
        trades=count_exposure_episodes(total_exposure),
        win_rate=win_rate,
        risk_free_rate=risk_free_rate,
        exposure=float((total_exposure > 0).mean()),
        turnover=annualized_turnover(result.curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=result.curve["gross_strategy_equity"],
        benchmark_equity=benchmark_equity,
    )
    return row | {
        "ticker": "EQUAL_WEIGHT",
        "label": "multi_asset",
        "variant": "equal_weight_signal_portfolio",
        "short_window": params.short_window,
        "long_window": params.long_window,
        "cost_bps": 10.0,
        "benchmark_cagr": summarize_performance("benchmark", benchmark_equity, basket_return)["cagr"],
        "benchmark_sharpe": summarize_performance("benchmark", benchmark_equity, basket_return)["sharpe"],
        "benchmark_max_drawdown": summarize_performance("benchmark", benchmark_equity, basket_return)["max_drawdown"],
    }


def summarize_curve(
    curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    label: str,
    variant: str,
    params: ParamsLike,
    cost_bps: float,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    total_exposure = executed_weights.abs().sum(axis=1)
    closed = calculate_closed_trade_returns(curve["strategy_return"], total_exposure)
    win_rate = calculate_win_rate(closed)
    benchmark_metrics = summarize_performance(
        "benchmark",
        curve["buy_hold_equity"],
        curve["buy_hold_return"],
        risk_free_rate=risk_free_rate,
    )
    ticker_weight = executed_weights.get(ticker, pd.Series(0.0, index=curve.index)).reindex(curve.index).fillna(0.0)
    fallback_columns = [column for column in executed_weights.columns if column != ticker]
    fallback_exposure = (
        float(executed_weights[fallback_columns].abs().sum(axis=1).mean()) if fallback_columns else 0.0
    )
    holds = holding_periods(ticker_weight)
    # Count exposure episodes, not turnover days: with volatility sizing the
    # weight changes almost daily without opening or closing a trade.
    trade_count = count_exposure_episodes(total_exposure)
    upside = capture_ratio(curve["strategy_return"], curve["buy_hold_return"], "up")
    downside = capture_ratio(curve["strategy_return"], curve["buy_hold_return"], "down")
    risk = getattr(params, "risk", None)
    sizing = getattr(params, "sizing", None)
    row = summarize_performance(
        name=f"{ticker}_{variant}_{params.label()}_{cost_bps:g}bps",
        equity=curve["strategy_equity"],
        returns=curve["strategy_return"],
        trades=trade_count,
        win_rate=win_rate,
        risk_free_rate=risk_free_rate,
        exposure=float((total_exposure > 0).mean()),
        turnover=annualized_turnover(curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=curve["gross_strategy_equity"],
        benchmark_equity=curve["buy_hold_equity"],
    )
    return row | {
        "ticker": ticker,
        "label": label,
        "variant": variant,
        "risk_free_rate": risk_free_rate,
        "short_window": getattr(params, "short_window", np.nan),
        "long_window": getattr(params, "long_window", np.nan),
        "spread_threshold": getattr(params, "spread_threshold", 0.0),
        "momentum_window": np.nan if getattr(params, "momentum_window", None) is None else params.momentum_window,
        "partial_exposure": getattr(params, "partial_exposure", False),
        "entry_threshold": getattr(params, "entry_threshold", 0.0),
        "exit_threshold": getattr(params, "exit_threshold", 0.0),
        "min_hold_days": getattr(params, "min_hold_days", 0),
        "cooldown_days": getattr(params, "cooldown_days", 0),
        "cost_bps": cost_bps,
        "benchmark_total_return": benchmark_metrics["total_return"],
        "benchmark_cagr": benchmark_metrics["cagr"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "upside_capture": upside,
        "downside_capture": downside,
        "capture_spread": capture_spread(upside, downside),
        "missed_return_while_in_cash": missed_return_while_underweight(curve["buy_hold_return"], ticker_weight),
        "avoided_downside_while_out": avoided_downside_while_underweight(curve["buy_hold_return"], ticker_weight),
        "average_holding_days": float(holds.mean()) if not holds.empty else np.nan,
        "median_holding_days": float(holds.median()) if not holds.empty else np.nan,
        "trade_frequency_per_year": trade_frequency_per_year(trade_count, curve.index),
        "fallback_exposure": fallback_exposure,
        "average_target_exposure": float(ticker_weight.abs().mean()),
        "volatility_adjusted_exposure": float(curve["volatility_weight"].mean())
        if "volatility_weight" in curve.columns
        else np.nan,
        "risk_price_sma_window": getattr(risk, "price_sma_window", np.nan),
        "risk_drawdown_threshold": getattr(risk, "rolling_drawdown_threshold", np.nan),
        "risk_volatility_window": getattr(risk, "realized_volatility_window", np.nan),
        "risk_max_realized_volatility": getattr(risk, "max_realized_volatility", np.nan),
        "risk_sharp_loss_threshold": getattr(risk, "sharp_loss_threshold", np.nan),
        "target_volatility": getattr(sizing, "target_volatility", np.nan),
        "sizing_window": getattr(sizing, "realized_volatility_window", np.nan),
        "fallback_asset": getattr(params, "fallback_asset", "cash"),
        "fallback_weight": getattr(params, "fallback_weight", 0.0),
        "fallback_min_hold_days": getattr(params, "fallback_min_hold_days", 0),
        "fallback_cooldown_days": getattr(params, "fallback_cooldown_days", 0),
        "hybrid_fallback": getattr(params, "hybrid_fallback", False),
    }


def _buy_hold_benchmark_row(
    price: pd.Series,
    initial_capital: float,
    ticker: str,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    returns = price.pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + returns).cumprod()
    return summarize_performance(f"{ticker}_buy_hold", equity, returns, risk_free_rate=risk_free_rate) | {
        "ticker": ticker,
        "model": f"{ticker} buy and hold",
        "variant": "buy_hold",
    }


def _capture_fallback_ticker(params: CaptureAwareAllocationParameters) -> str | None:
    fallback = params.fallback_asset.upper()
    if fallback in {"", "CASH"} or params.fallback_weight <= 0:
        return None
    return fallback


def _combine_curve(
    price: pd.Series,
    signals,
    engine_curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    curve = engine_curve.copy()
    curve["price"] = price.reindex(curve.index)
    curve["short_sma"] = signals.short_sma.reindex(curve.index)
    curve["long_sma"] = signals.long_sma.reindex(curve.index)
    curve["signal"] = signals.target_position.reindex(curve.index).fillna(0.0)
    curve["position"] = executed_weights.get(ticker, pd.Series(0.0, index=curve.index))
    fallback_columns = [column for column in executed_weights.columns if column != ticker]
    curve["fallback_position"] = (
        executed_weights[fallback_columns].abs().sum(axis=1) if fallback_columns else 0.0
    )
    for column in [
        "trend_position",
        "risk_off",
        "volatility_weight",
        "realized_volatility",
        "price_sma",
        "rolling_drawdown",
        "sharp_loss",
        "fallback_target",
    ]:
        if hasattr(signals, column):
            curve[column] = getattr(signals, column).reindex(curve.index)
    curve["buy_hold_return"] = prices[ticker].pct_change().reindex(curve.index).fillna(0.0)
    curve["buy_hold_equity"] = engine_curve["strategy_equity"].iloc[0] * (1.0 + curve["buy_hold_return"]).cumprod()
    curve["buy_hold_drawdown"] = curve["buy_hold_equity"] / curve["buy_hold_equity"].cummax() - 1.0
    return curve


def build_trade_log(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or "position" not in curve.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    in_trade = False
    entry_date = None
    previous_date = None
    for date, row in curve.iterrows():
        exposed = float(row.get("position", 0.0) or 0.0) > 0.0
        if exposed and not in_trade:
            entry_date = date
            in_trade = True
        elif in_trade and not exposed and entry_date is not None:
            rows.append(_trade_log_row(curve, entry_date, date, "closed"))
            in_trade = False
            entry_date = None
        previous_date = date
    if in_trade and entry_date is not None and previous_date is not None:
        rows.append(_trade_log_row(curve, entry_date, previous_date, "open"))
    return pd.DataFrame(rows)


def _trade_log_row(curve: pd.DataFrame, entry_date, exit_date, status: str) -> dict[str, Any]:
    trade = curve.loc[entry_date:exit_date]
    entry_price = float(trade["price"].iloc[0])
    exit_price = float(trade["price"].iloc[-1])
    price_path = trade["price"] / entry_price - 1.0
    return {
        "entry_date": pd.Timestamp(entry_date).date().isoformat(),
        "exit_date": pd.Timestamp(exit_date).date().isoformat(),
        "status": status,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "holding_days": int(len(trade)),
        "trade_return": float((1.0 + trade["strategy_return"]).prod() - 1.0),
        "max_favorable_excursion": float(price_path.max()),
        "max_adverse_excursion": float(price_path.min()),
        "regime_at_entry": str(trade["regime"].iloc[0]) if "regime" in trade else "",
        "volatility_at_entry": float(trade["realized_volatility"].iloc[0])
        if "realized_volatility" in trade and pd.notna(trade["realized_volatility"].iloc[0])
        else np.nan,
        "trend_spread_at_entry": float(trade["spread"].iloc[0])
        if "spread" in trade and pd.notna(trade["spread"].iloc[0])
        else np.nan,
    }


def run_regime_results(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or "regime" not in curve.columns:
        return pd.DataFrame()
    rows = []
    for regime, group in curve.groupby("regime"):
        if group.empty:
            continue
        upside = capture_ratio(group["strategy_return"], group["buy_hold_return"], "up")
        downside = capture_ratio(group["strategy_return"], group["buy_hold_return"], "down")
        rows.append(
            {
                "regime": regime,
                "days": int(len(group)),
                "strategy_return": float((1.0 + group["strategy_return"]).prod() - 1.0),
                "benchmark_return": float((1.0 + group["buy_hold_return"]).prod() - 1.0),
                "upside_capture": upside,
                "downside_capture": downside,
                "capture_spread": capture_spread(upside, downside),
                "average_exposure": float(group["position"].abs().mean()) if "position" in group else np.nan,
                "fallback_exposure": float(group["fallback_position"].abs().mean())
                if "fallback_position" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
