from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_backtest.backtest import (
    BacktestConfig,
    apply_position_and_costs,
    build_sma_signal,
    run_sma_backtest,
)
from quant_backtest.metrics import cagr, max_drawdown


def test_sma_signal_and_position_are_shifted_one_day() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"Adj Close": [10, 10, 10, 12, 14, 16]}, index=dates)
    config = BacktestConfig(short_window=2, long_window=3)

    result = run_sma_backtest(prices, config)
    curve = result.equity_curve

    expected_signal = build_sma_signal(curve["short_sma"], curve["long_sma"])
    expected_position = expected_signal.shift(1).fillna(0.0)

    pd.testing.assert_series_equal(curve["signal"], expected_signal, check_names=False)
    pd.testing.assert_series_equal(curve["position"], expected_position, check_names=False)
    assert curve.loc[dates[3], "signal"] == 1.0
    assert curve.loc[dates[3], "position"] == 0.0
    assert curve.loc[dates[4], "position"] == 1.0


def test_sma_summary_counts_entries_and_exposure_earning_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"Adj Close": [10, 10, 10, 12, 14, 16]}, index=dates)
    result = run_sma_backtest(prices, BacktestConfig(short_window=2, long_window=3, cost_bps=100))
    assert result.metrics.loc["strategy", "trades"] == 1
    assert result.metrics.loc["strategy", "exposure"] == pytest.approx(1 / 6)
    assert result.equity_curve["earning_position"].tolist() == [0, 0, 0, 0, 0, 1]


def test_transaction_costs_apply_on_entry_and_exit() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    price = pd.Series([100.0, 110.0, 121.0], index=dates)
    position = pd.Series([0.0, 1.0, 0.0], index=dates)

    frame = apply_position_and_costs(
        price=price,
        position=position,
        cost_bps=100.0,
        initial_capital=100.0,
    )

    expected_returns = pd.Series([0.0, -0.01 / 1.01, 0.089], index=dates)
    pd.testing.assert_series_equal(frame["strategy_return"], expected_returns, check_names=False)
    assert frame["trade"].sum() == pytest.approx(1 / 1.01 + 1.1)
    assert frame["transaction_cost"].sum() == pytest.approx(0.01 / 1.01 + 0.011)
    assert frame["strategy_equity"].iloc[-1] == pytest.approx(100 / 1.01 * 1.1 * 0.99)


def test_metrics_on_synthetic_equity() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2025-01-01"])
    equity = pd.Series([100.0, 120.0, 90.0, 150.0], index=dates)

    assert max_drawdown(equity) == pytest.approx(-0.25)
    assert cagr(equity) == pytest.approx(0.5, rel=0.01)


def test_backtest_smoke_without_network() -> None:
    dates = pd.date_range("2024-01-01", periods=140, freq="B")
    values = [100 + index * 0.3 + math.sin(index / 5) for index in range(len(dates))]
    prices = pd.DataFrame({"Adj Close": values}, index=dates)

    result = run_sma_backtest(
        prices,
        BacktestConfig(short_window=5, long_window=20, cost_bps=10.0, initial_capital=10_000.0),
    )

    assert not result.equity_curve.empty
    assert {"strategy", "buy_hold"} == set(result.metrics.index)
    assert result.equity_curve["strategy_equity"].iloc[-1] > 0
    assert result.equity_curve["buy_hold_equity"].iloc[-1] > 0
