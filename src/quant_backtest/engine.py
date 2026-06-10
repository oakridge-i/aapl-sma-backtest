from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .costs import BpsCost, CostModel
from .metrics import drawdown_series


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 10_000.0
    cost_model: CostModel = BpsCost(10.0)
    execution_lag: int = 1


@dataclass(frozen=True)
class EngineResult:
    curve: pd.DataFrame
    executed_weights: pd.DataFrame


def run_weight_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: EngineConfig,
    cash_returns: pd.Series | None = None,
) -> EngineResult:
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if config.execution_lag < 0:
        raise ValueError("execution_lag must be non-negative.")

    aligned_returns, aligned_targets = returns.align(target_weights, join="left", axis=0)
    aligned_targets = aligned_targets.reindex(columns=aligned_returns.columns).fillna(0.0).astype(float)
    aligned_returns = aligned_returns.fillna(0.0).astype(float)

    executed_weights = aligned_targets.shift(config.execution_lag).fillna(0.0)
    turnover = executed_weights.diff().abs().sum(axis=1).fillna(executed_weights.abs().sum(axis=1))
    transaction_cost = config.cost_model.calculate(turnover)
    gross_return = (executed_weights * aligned_returns).sum(axis=1)
    cash_weight = (1.0 - executed_weights.sum(axis=1)).clip(lower=0.0)
    if cash_returns is not None:
        aligned_cash = cash_returns.reindex(aligned_returns.index).fillna(0.0).astype(float)
        gross_return = gross_return + cash_weight * aligned_cash
    net_return = gross_return - transaction_cost
    equity = config.initial_capital * (1.0 + net_return).cumprod()
    gross_equity = config.initial_capital * (1.0 + gross_return).cumprod()

    curve = pd.DataFrame(
        {
            "gross_return": gross_return,
            "strategy_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "cash_weight": cash_weight,
            "strategy_equity": equity,
            "gross_strategy_equity": gross_equity,
            "strategy_drawdown": drawdown_series(equity),
        },
        index=aligned_returns.index,
    )
    return EngineResult(curve=curve, executed_weights=executed_weights)
