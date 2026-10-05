from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import BpsCost, CostModel


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 10_000.0
    cost_model: CostModel = BpsCost(10.0)
    # Signal after close t -> trade at close t+lag -> exposure next bar.
    execution_lag: int = 1


@dataclass(frozen=True)
class EngineResult:
    curve: pd.DataFrame
    executed_weights: pd.DataFrame  # weights earning this bar's return
    closing_weights: pd.DataFrame  # after this close's trades


def run_weight_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: EngineConfig,
    cash_returns: pd.Series | None = None,
) -> EngineResult:
    """Self-financing close rebalancing, initially in cash.

    Costs are one-way fractions of traded risky notional. Cash is a synthetic
    remunerated account, NOT a traded ETF; its supplied yield is a proxy.
    Mark old holdings, charge trades against drifted holdings, then allocate
    post-cost NAV. No terminal liquidation is assumed.
    """
    if not np.isfinite(config.initial_capital) or config.initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")
    if not isinstance(config.execution_lag, int) or config.execution_lag < 0:
        raise ValueError("execution_lag must be a non-negative integer.")
    if returns.empty or not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Returns require a nonempty, unique, increasing index.")
    if not returns.columns.is_unique or not target_weights.columns.is_unique:
        raise ValueError("Asset columns must be unique.")
    if not target_weights.index.equals(returns.index):
        raise ValueError("Targets and returns must have identical dates.")
    if set(target_weights.columns) - set(returns.columns):
        raise ValueError("Target asset has no return data.")
    values = returns.to_numpy(dtype=float)
    targets = target_weights.reindex(columns=returns.columns, fill_value=0.0).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Returns must be finite and greater than -1; missing data cannot be zero-filled.")
    if not np.isfinite(targets).all() or (targets < -1e-12).any() or (targets.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Targets must be finite, nonnegative, and unlevered.")
    targets = np.maximum(targets, 0.0)
    cash = np.zeros(len(returns)) if cash_returns is None else cash_returns.reindex(returns.index).to_numpy(dtype=float)
    if not np.isfinite(cash).all() or (cash <= -1).any():
        raise ValueError("Cash returns must cover every date and be finite and greater than -1.")
    fees = config.cost_model.calculate(pd.Series([0.0, 1.0, 2.0])).to_numpy()
    rate = float(fees[1])
    if not np.isfinite(fees).all() or not 0 <= rate < 0.1 or not np.allclose(fees, [0, rate, 2 * rate]):
        raise ValueError("Engine requires proportional one-way costs below 1000 bps.")
    lag = config.execution_lag
    desired = np.zeros_like(targets)
    if lag == 0:
        desired[:] = targets
    elif lag < len(targets):
        desired[lag:] = targets[:-lag]

    # Daily rebalancing fixes closing proportions; mark them before trading.
    held = np.vstack([np.zeros((1, targets.shape[1])), desired[:-1]])
    held_cash = np.maximum(1.0 - held.sum(axis=1), 0.0)
    marked = held * (1.0 + values)
    growth = marked.sum(axis=1) + held_cash * (1.0 + cash)
    drifted = marked / growth[:, None]
    post_cost_fraction = np.ones(len(returns))
    if rate:
        # Contraction solves cost financing, including full-investment entries.
        for _ in range(16):
            updated = 1.0 - rate * np.abs(desired * post_cost_fraction[:, None] - drifted).sum(axis=1)
            if np.max(np.abs(updated - post_cost_fraction)) < 1e-14:
                post_cost_fraction = updated
                break
            post_cost_fraction = updated
    traded = np.abs(desired * post_cost_fraction[:, None] - drifted).sum(axis=1)
    cost = growth * rate * traded  # fraction of previous closing NAV
    gross = growth - 1.0
    net = growth * post_cost_fraction - 1.0
    equity = config.initial_capital * np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.maximum(equity, config.initial_capital))
    curve = pd.DataFrame({
        "gross_return": gross,
        "strategy_return": net,
        "turnover": growth * traded,
        "transaction_cost": cost,
        "cash_weight": held_cash,
        "closing_cash_weight": np.maximum(1.0 - desired.sum(axis=1), 0.0),
        "strategy_equity": equity,
        "gross_strategy_equity": config.initial_capital * np.cumprod(growth),
        "strategy_drawdown": equity / peak - 1.0,
    }, index=returns.index)
    return EngineResult(curve, pd.DataFrame(held, index=returns.index, columns=returns.columns),
                        pd.DataFrame(desired, index=returns.index, columns=returns.columns))
