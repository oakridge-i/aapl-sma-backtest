"""Significance diagnostics for already-selected models.

These are reporting steps: they quantify how much of an observed result could
be luck. Nothing here feeds back into model selection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evaluation import evaluate_strategy
from .data_quality import price_returns
from .metrics import summarize_performance
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
    extra_models: list[tuple[str, Any, str, pd.DataFrame]] | None = None,
    extra_stitched: list[tuple[str, pd.Series]] | None = None,
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
    if extra_models:
        models.extend(extra_models)

    rows: list[dict[str, Any]] = []
    for model_label, params, variant, trials_table in models:
        # Match the warm-history evaluation used by the v6 comparison.
        model_prices = prices.loc[: config.test_end or prices.index.max()]
        result = evaluate_strategy(
            prices=model_prices,
            evaluation_start=config.test_start,
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
        risk_free_rate = result["reference_returns"]

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
        weight_returns = price_returns(test_prices[result["target_weights"].columns])
        row |= timing_permutation_pvalue(
            executed_weights=result["target_weights"],
            weights_are_targets=True,
            risk_free_rate=risk_free_rate,
            asset_returns=weight_returns,
            cost_bps=10.0,
            cash_returns=cash_returns,
            n_permutations=config.permutation_iterations,
            seed=config.significance_seed,
        )
        row["dsr_scope"] = "supplied_candidates_only_independence_approximation"
        row["search_history_complete"] = False
        row["bootstrap_scope"] = "fixed_returns_not_selection_pipeline"
        row["probability_interpretation"] = "diagnostics_not_probability_of_alpha_or_future_profit"
        rows.append(row)

    if nested_oos_returns is not None and not nested_oos_returns.empty:
        rows.append(_stitched_oos_significance(nested_oos_returns, prices, config))
    for label, stitched_returns in extra_stitched or []:
        if stitched_returns is not None and not stitched_returns.empty:
            rows.append(_stitched_oos_significance(stitched_returns, prices, config, model_label=label))
    return pd.DataFrame(rows)


def _stitched_oos_significance(
    oos_returns: pd.Series,
    prices: pd.DataFrame,
    config: ResearchConfig,
    model_label: str = "nested_oos_stitched",
) -> dict[str, Any]:
    """Bootstrap diagnostics for the stitched nested walk-forward OOS series.

    Deflated Sharpe and the permutation test are intentionally omitted here:
    the candidate set differs per window. A pipeline-level test would need
    to repeat selection; resampling the final path alone does not do this.
    """
    cash_returns = cash_return_series(prices, config.cash_proxy_ticker)
    clean = oos_returns.dropna()
    reference = cash_returns.reindex(clean.index).copy() if cash_returns is not None else 0.0
    if isinstance(reference, pd.Series):
        reference.iloc[0] = 0.0  # same initial valuation as the continuous engine
    risk_free_rate = reference
    equity = (1.0 + clean).cumprod()
    metrics = summarize_performance(model_label, equity, clean, risk_free_rate=reference)
    row = {
        "model": model_label,
        "variant": "continuous_per_window_selection",
        "period_start": str(clean.index.min().date()),
        "period_end": str(clean.index.max().date()),
        "n_obs": len(clean),
        "observed_cagr": metrics["cagr"],
        "observed_sharpe": metrics["sharpe"],
        "observed_max_drawdown": metrics["max_drawdown"],
        "bootstrap_scope": "fixed_returns_not_selection_pipeline",
        "search_history_complete": False,
        "probability_interpretation": "resample_fractions_not_probability_of_alpha",
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
        "scope": "hysteresis_grid_only_not_v6_pipeline",
        "candidate_subset": "evenly_spaced_when_capped",
        "selection_metric": "raw_return_sharpe",
        "grid_size": int(len(table)),
        "period_start": str(candidate_returns.index.min().date()),
        "period_end": str(candidate_returns.index.max().date()),
    }
    return pd.DataFrame([summary])
