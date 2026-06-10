"""Significance diagnostics for already-selected models.

These are reporting steps: they quantify how much of an observed result could
be luck. Nothing here feeds back into model selection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evaluation import evaluate_strategy
from .research_config import ResearchConfig
from .research_data import cash_return_series
from .stats import (
    block_bootstrap_summary,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    timing_permutation_pvalue,
)


def run_significance_analysis(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v3_model: dict[str, Any],
    selected_v4_model: dict[str, Any] | None,
    allocation_leaderboard: pd.DataFrame,
    capture_leaderboard: pd.DataFrame,
    nested_oos_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Bootstrap, Deflated Sharpe, and permutation diagnostics on the test period."""
    test_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    models: list[tuple[str, Any, str, pd.DataFrame]] = [
        ("selected_v3", selected_v3_model["params"], selected_v3_model["variant"], allocation_leaderboard),
    ]
    if selected_v4_model is not None:
        models.append(
            ("selected_v4", selected_v4_model["params"], selected_v4_model["variant"], capture_leaderboard)
        )

    rows: list[dict[str, Any]] = []
    for model_label, params, variant, trials_table in models:
        result = evaluate_strategy(
            prices=test_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="significance_test",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        curve = result["curve"]
        returns = curve["strategy_return"]
        risk_free_rate = result["risk_free_rate"]

        row: dict[str, Any] = {
            "model": model_label,
            "variant": variant,
            "period_start": str(curve.index.min().date()),
            "period_end": str(curve.index.max().date()),
            "n_obs": int(returns.dropna().shape[0]),
            "observed_cagr": result["row"]["cagr"],
            "observed_sharpe": result["row"]["sharpe"],
            "observed_max_drawdown": result["row"]["max_drawdown"],
        }
        row |= block_bootstrap_summary(
            returns,
            n_iterations=config.bootstrap_iterations,
            block_size=config.bootstrap_block_size,
            seed=config.significance_seed,
            risk_free_rate=risk_free_rate,
        )
        trial_sharpes = (
            trials_table["sharpe"] if not trials_table.empty and "sharpe" in trials_table else pd.Series(dtype=float)
        )
        row |= deflated_sharpe_ratio(returns, trial_sharpes, risk_free_rate=risk_free_rate)
        cash_returns = cash_return_series(test_prices, config.cash_proxy_ticker)
        weight_returns = test_prices[result["weights"].columns].pct_change().fillna(0.0)
        row |= timing_permutation_pvalue(
            executed_weights=result["weights"],
            asset_returns=weight_returns,
            cost_bps=10.0,
            cash_returns=cash_returns,
            n_permutations=config.permutation_iterations,
            seed=config.significance_seed,
        )
        rows.append(row)

    if nested_oos_returns is not None and not nested_oos_returns.empty:
        rows.append(_stitched_oos_significance(nested_oos_returns, prices, config))
    return pd.DataFrame(rows)


def _stitched_oos_significance(
    oos_returns: pd.Series,
    prices: pd.DataFrame,
    config: ResearchConfig,
) -> dict[str, Any]:
    """Bootstrap diagnostics for the stitched nested walk-forward OOS series.

    Deflated Sharpe and the permutation test are intentionally omitted here:
    the candidate set differs per window and the stitched series has no single
    weight path, so those tests would not be well defined.
    """
    cash_returns = cash_return_series(prices, config.cash_proxy_ticker)
    risk_free_rate = 0.0
    if cash_returns is not None:
        aligned = cash_returns.reindex(oos_returns.index).dropna()
        if not aligned.empty:
            risk_free_rate = float(aligned.mean() * 252)
    clean = oos_returns.dropna()
    equity = (1.0 + clean).cumprod()
    n_obs = int(clean.shape[0])
    observed_sharpe = float("nan")
    std = float(clean.std(ddof=0))
    if std > 0:
        observed_sharpe = float((clean.mean() * 252 - risk_free_rate) / (std * (252**0.5)))
    row: dict[str, Any] = {
        "model": "nested_oos_stitched",
        "variant": "per_window_selection",
        "period_start": str(clean.index.min().date()),
        "period_end": str(clean.index.max().date()),
        "n_obs": n_obs,
        "observed_cagr": float(equity.iloc[-1] ** (252.0 / n_obs) - 1.0) if n_obs else float("nan"),
        "observed_sharpe": observed_sharpe,
        "observed_max_drawdown": float((equity / equity.cummax() - 1.0).min()) if n_obs else float("nan"),
    }
    row |= block_bootstrap_summary(
        clean,
        n_iterations=config.bootstrap_iterations,
        block_size=config.bootstrap_block_size,
        seed=config.significance_seed,
        risk_free_rate=risk_free_rate,
    )
    return row


def run_pbo_analysis(
    prices: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """CSCV PBO over the full-period hysteresis grid.

    This is a diagnostic of the selection *process*: it measures how often the
    in-sample winner of the configured grid underperforms out of sample under
    combinatorial splits. It uses the full sample (including test data), which
    is standard for CSCV and acceptable because nothing here feeds selection.
    """
    from .sweeps import run_hysteresis_sweep

    table, candidate_returns = run_hysteresis_sweep(
        prices, config, period_name="pbo_full", collect_returns=True
    )
    if candidate_returns is None or candidate_returns.empty:
        return pd.DataFrame()
    summary = probability_of_backtest_overfitting(
        candidate_returns,
        n_blocks=config.pbo_blocks,
        max_candidates=config.pbo_max_candidates,
    )
    if not summary:
        return pd.DataFrame()
    summary |= {
        "grid": "trend_hysteresis",
        "grid_size": int(len(table)),
        "period_start": str(candidate_returns.index.min().date()),
        "period_end": str(candidate_returns.index.max().date()),
    }
    return pd.DataFrame([summary])
