from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    annualized_turnover,
    drawdown_series,
    exposure_percentage,
    metrics_table,
    summarize_performance,
)


@dataclass(frozen=True)
class BacktestConfig:
    ticker: str = "AAPL"
    short_window: int = 20
    long_window: int = 100
    cost_bps: float = 10.0
    initial_capital: float = 10_000.0
    price_column: str = "Adj Close"
    risk_free_rate: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: pd.DataFrame
    closed_trade_returns: pd.Series


def run_sma_backtest(prices: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    _validate_config(config)
    price = _select_price_series(prices, config.price_column)

    short_sma = price.rolling(config.short_window, min_periods=config.short_window).mean()
    long_sma = price.rolling(config.long_window, min_periods=config.long_window).mean()
    signal = build_sma_signal(short_sma, long_sma)
    position = signal.shift(1).fillna(0.0)

    returns_frame = apply_position_and_costs(
        price=price,
        position=position,
        cost_bps=config.cost_bps,
        initial_capital=config.initial_capital,
    )

    equity_curve = pd.DataFrame(
        {
            "price": price,
            "short_sma": short_sma,
            "long_sma": long_sma,
            "signal": signal,
            "position": position,
            **returns_frame,
        }
    )
    equity_curve["strategy_drawdown"] = drawdown_series(equity_curve["strategy_equity"])
    equity_curve["buy_hold_drawdown"] = drawdown_series(equity_curve["buy_hold_equity"])

    closed_trade_returns = calculate_closed_trade_returns(
        returns=equity_curve["strategy_return"],
        position=equity_curve["position"],
    )
    win_rate = calculate_win_rate(closed_trade_returns)
    trade_count = int(equity_curve["trade"].sum())

    metrics = metrics_table(
        [
            summarize_performance(
                "strategy",
                equity_curve["strategy_equity"],
                equity_curve["strategy_return"],
                trades=trade_count,
                win_rate=win_rate,
                risk_free_rate=config.risk_free_rate,
                exposure=exposure_percentage(equity_curve["position"]),
                turnover=annualized_turnover(equity_curve["trade"]),
                closed_trade_returns=closed_trade_returns,
                gross_equity=config.initial_capital
                * (1.0 + equity_curve["position"] * equity_curve["asset_return"]).cumprod(),
                benchmark_equity=equity_curve["buy_hold_equity"],
            ),
            summarize_performance(
                "buy_hold",
                equity_curve["buy_hold_equity"],
                equity_curve["buy_hold_return"],
                trades=1,
                win_rate=None,
                risk_free_rate=config.risk_free_rate,
                exposure=1.0,
                turnover=0.0,
            ),
        ]
    )

    return BacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        closed_trade_returns=closed_trade_returns,
    )


def build_sma_signal(short_sma: pd.Series, long_sma: pd.Series) -> pd.Series:
    signal = (short_sma > long_sma).astype(float)
    signal = signal.where(long_sma.notna(), 0.0)
    signal.name = "signal"
    return signal


def apply_position_and_costs(
    price: pd.Series,
    position: pd.Series,
    cost_bps: float,
    initial_capital: float,
) -> dict[str, pd.Series]:
    aligned_position = position.reindex(price.index).fillna(0.0).astype(float)
    daily_return = price.pct_change().fillna(0.0)
    trade = aligned_position.diff().abs().fillna(aligned_position.abs())
    cost_rate = cost_bps / 10_000.0
    transaction_cost = trade * cost_rate

    strategy_return = aligned_position * daily_return - transaction_cost
    buy_hold_return = daily_return
    strategy_equity = initial_capital * (1.0 + strategy_return).cumprod()
    buy_hold_equity = initial_capital * (1.0 + buy_hold_return).cumprod()

    return {
        "asset_return": daily_return,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "trade": trade,
        "transaction_cost": transaction_cost,
        "strategy_equity": strategy_equity,
        "buy_hold_equity": buy_hold_equity,
    }


def calculate_closed_trade_returns(
    returns: pd.Series,
    position: pd.Series,
    epsilon: float = 1e-9,
) -> pd.Series:
    """Compound strategy returns over exposure episodes.

    A trade is one contiguous period where the absolute position exceeds
    ``epsilon``. Daily weight changes inside an episode (e.g. volatility
    sizing) do not open or close trades; only crossing to or from flat does.
    The first flat day is included so exit costs land inside the trade.
    """
    exposed = position.astype(float).fillna(0.0).abs() > epsilon
    previous = exposed.shift(1, fill_value=False)
    entry_dates = list(position.index[exposed & ~previous])
    exit_dates = list(position.index[~exposed & previous])

    closed_returns: list[float] = []
    for entry_date, exit_date in zip(entry_dates, exit_dates):
        trade_returns = returns.loc[entry_date:exit_date]
        closed_returns.append(float((1.0 + trade_returns).prod() - 1.0))

    return pd.Series(closed_returns, name="closed_trade_return", dtype=float)


def count_exposure_episodes(position: pd.Series, epsilon: float = 1e-9) -> int:
    """Number of exposure episodes (entries), including a still-open one."""
    exposed = position.astype(float).fillna(0.0).abs() > epsilon
    previous = exposed.shift(1, fill_value=False)
    return int((exposed & ~previous).sum())


def calculate_win_rate(closed_trade_returns: pd.Series) -> float:
    if closed_trade_returns.empty:
        return np.nan
    return float((closed_trade_returns > 0).mean())


def _select_price_series(prices: pd.DataFrame, preferred_column: str) -> pd.Series:
    if preferred_column in prices.columns:
        price = prices[preferred_column]
    elif "Close" in prices.columns:
        price = prices["Close"]
    else:
        raise ValueError(f"DataFrame must contain {preferred_column!r} or 'Close'.")

    clean = price.dropna().astype(float)
    if clean.empty:
        raise ValueError("Price series is empty after dropping missing values.")
    clean.name = "price"
    return clean


def _validate_config(config: BacktestConfig) -> None:
    if config.short_window <= 0 or config.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if config.short_window >= config.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if config.cost_bps < 0:
        raise ValueError("cost_bps must be non-negative.")
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
