from __future__ import annotations

import pandas as pd
import pytest

from quant_backtest.costs import BpsCost
from quant_backtest.engine import EngineConfig, run_weight_backtest


def test_bps_cost_scenarios() -> None:
    turnover = pd.Series([0.0, 1.0, 2.0])

    assert BpsCost(0).calculate(turnover).sum() == 0.0
    assert BpsCost(5).calculate(turnover).sum() == pytest.approx(0.0015)
    assert BpsCost(10).calculate(turnover).sum() == pytest.approx(0.003)
    assert BpsCost(20).calculate(turnover).sum() == pytest.approx(0.006)
    assert BpsCost(50).calculate(turnover).sum() == pytest.approx(0.015)


def test_engine_applies_one_day_execution_lag() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"AAPL": [0.0, 0.10, 0.10, 0.10]}, index=dates)
    target_weights = pd.DataFrame({"AAPL": [0.0, 1.0, 1.0, 1.0]}, index=dates)

    result = run_weight_backtest(
        returns=returns,
        target_weights=target_weights,
        config=EngineConfig(initial_capital=100.0, cost_model=BpsCost(0.0)),
    )

    assert result.executed_weights.loc[dates[1], "AAPL"] == 0.0
    assert result.closing_weights.loc[dates[2], "AAPL"] == 1.0
    assert result.executed_weights.loc[dates[2], "AAPL"] == 0.0
    assert result.executed_weights.loc[dates[3], "AAPL"] == 1.0
    assert result.curve.loc[dates[1], "strategy_return"] == 0.0
    assert result.curve.loc[dates[2], "strategy_return"] == 0.0
    assert result.curve.loc[dates[3], "strategy_return"] == pytest.approx(0.10)


def test_engine_turnover_and_cost_drag() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, 0.05, 0.05]}, index=dates)
    target_weights = pd.DataFrame({"AAPL": [1.0, 1.0, 0.0, 0.0]}, index=dates)

    result = run_weight_backtest(
        returns=returns,
        target_weights=target_weights,
        config=EngineConfig(initial_capital=100.0, cost_model=BpsCost(100.0)),
    )

    assert result.curve["turnover"].sum() == pytest.approx(1 / 1.01 + 1.05)
    assert result.curve["transaction_cost"].sum() == pytest.approx(0.01 / 1.01 + 0.0105)
    assert result.curve["gross_strategy_equity"].iloc[-1] > result.curve["strategy_equity"].iloc[-1]
