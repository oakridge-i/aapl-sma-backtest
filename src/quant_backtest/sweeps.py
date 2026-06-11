"""Research sweeps, leaderboards, comparisons, and walk-forward runs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evaluation import (
    _buy_hold_benchmark_row,
    build_trade_log,
    evaluate_equal_weight_signal_portfolio,
    evaluate_strategy,
    run_regime_results,
)
from .metrics import summarize_performance
from .parallel import EvaluationContext, EvaluationJob, evaluate_grid
from .research_config import ResearchConfig
from .research_data import cash_return_series
from .selection import (
    add_capture_selection_score,
    add_parameter_stability,
    add_selection_score,
    allocation_selection_mask,
    capture_parameter_grid,
    capture_variant,
    parameter_grid,
    select_allocation_model,
    select_best_parameters,
    select_capture_model,
    select_top_trend_candidates,
    trend_parameter_grid,
)
from .strategies import (
    CaptureAwareAllocationParameters,
    RiskFilterParameters,
    SmaParameters,
    TrendAllocationParameters,
    VolatilitySizingParameters,
    classify_market_regime,
)


def _grid_context(prices: pd.DataFrame, config: ResearchConfig) -> EvaluationContext:
    return EvaluationContext(
        prices=prices,
        ticker=config.base_ticker,
        initial_capital=config.initial_capital,
        market_regime_short_window=config.market_regime_short_window,
        market_regime_long_window=config.market_regime_long_window,
        cash_proxy=config.cash_proxy_ticker,
    )


def run_parameter_sweep(prices: pd.DataFrame, config: ResearchConfig, period_name: str = "full") -> pd.DataFrame:
    jobs = [
        EvaluationJob(SmaParameters(short_window=short, long_window=long), "long_cash", 10.0, period_name)
        for short, long in parameter_grid(config)
    ]
    rows, _ = evaluate_grid(jobs, _grid_context(prices, config), config.parallel_workers)
    table = pd.DataFrame(rows)
    table = add_parameter_stability(table)
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run_cost_sensitivity(prices: pd.DataFrame, config: ResearchConfig, selected_params: SmaParameters) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenarios = [
        ("base_20_100", SmaParameters(20, 100)),
        ("selected", selected_params),
    ]
    for scenario_name, params in scenarios:
        for cost_bps in config.cost_bps:
            result = evaluate_strategy(
                prices=prices,
                ticker=config.base_ticker,
                params=params,
                variant="long_cash",
                cost_bps=cost_bps,
                initial_capital=config.initial_capital,
                label=scenario_name,
                cash_proxy=config.cash_proxy_ticker,
            )
            rows.append(result["row"])
    table = pd.DataFrame(rows)
    return table.sort_values(["label", "cost_bps"]).reset_index(drop=True)


def run_train_test(prices: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    train_prices = prices.loc[config.train_start : config.train_end]
    test_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    train_sweep = run_parameter_sweep(train_prices, config, period_name="train")
    selected = select_best_parameters(train_sweep, config)
    rows = []
    for period_name, period_prices in [("train", train_prices), ("test", test_prices)]:
        result = evaluate_strategy(
            prices=period_prices,
            ticker=config.base_ticker,
            params=selected,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=period_name,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"] | {"selected_on": "train"})
    return pd.DataFrame(rows)


def walk_forward_windows(
    prices: pd.DataFrame, config: ResearchConfig
) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows = []
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end or prices.index.max())
    train_offset = pd.DateOffset(years=config.walk_forward_train_years)
    test_offset = pd.DateOffset(years=config.walk_forward_test_years)
    step_offset = pd.DateOffset(years=config.walk_forward_step_years)

    train_start = start
    window_id = 1
    while True:
        train_end = train_start + train_offset - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + test_offset - pd.DateOffset(days=1)
        if test_start > end:
            break
        if test_end > end:
            test_end = end
        windows.append((window_id, train_start, train_end, test_start, test_end))
        train_start = train_start + step_offset
        window_id += 1
    return windows


def run_walk_forward(prices: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_id, train_start, train_end, test_start, test_end in walk_forward_windows(prices, config):
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        if len(train_prices) > 250 and len(test_prices) > 20:
            selected = select_best_parameters(run_parameter_sweep(train_prices, config, period_name="wf_train"), config)
            result = evaluate_strategy(
                prices=test_prices,
                ticker=config.base_ticker,
                params=selected,
                variant="long_cash",
                cost_bps=10.0,
                initial_capital=config.initial_capital,
                label="walk_forward_test",
                cash_proxy=config.cash_proxy_ticker,
            )
            rows.append(
                result["row"]
                | {
                    "window_id": window_id,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def run_final_model_walk_forward(
    prices: pd.DataFrame,
    config: ResearchConfig,
    models: list[tuple[str, Any, str]],
) -> pd.DataFrame:
    """Evaluate the final (fixed) models across every walk-forward test window.

    Parameters stay frozen, so this shows whether the selected models hold up
    across sub-periods rather than relying on one favorable train/test split.
    """
    rows: list[dict[str, Any]] = []
    for window_id, _, _, test_start, test_end in walk_forward_windows(prices, config):
        test_prices = prices.loc[test_start:test_end]
        if len(test_prices) <= 20:
            continue
        for model_label, params, variant in models:
            result = evaluate_strategy(
                prices=test_prices,
                ticker=config.base_ticker,
                params=params,
                variant=variant,
                cost_bps=10.0,
                initial_capital=config.initial_capital,
                label="final_walk_forward",
                market_regime_short_window=config.market_regime_short_window,
                market_regime_long_window=config.market_regime_long_window,
                cash_proxy=config.cash_proxy_ticker,
            )
            row = result["row"]
            rows.append(
                {
                    "model": model_label,
                    "variant": variant,
                    "window_id": window_id,
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "cagr": row["cagr"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                    "turnover": row["turnover"],
                    "exposure": row["exposure"],
                    "benchmark_cagr": row["benchmark_cagr"],
                    "benchmark_sharpe": row["benchmark_sharpe"],
                    "excess_cagr_vs_benchmark": row["excess_cagr_vs_benchmark"],
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["model", "window_id"]).reset_index(drop=True)


def run_nested_walk_forward(prices: pd.DataFrame, config: ResearchConfig) -> dict[str, Any]:
    """Walk-forward *model selection*: re-run the v0.3 selection per window.

    For every walk-forward window the full selection pipeline (hysteresis sweep
    on the window's train slice -> top candidates -> allocation leaderboard on
    the same train slice -> selection rule) runs from scratch, and the selected
    model is evaluated once on the window's test slice. The stitched
    out-of-sample return series is selection-clean by construction and is the
    primary scoreboard for model upgrades.

    Returns a dict with ``windows`` (per-window table), ``summary`` (aggregate
    rows), and ``oos_returns`` (stitched daily OOS returns).
    """
    window_rows: list[dict[str, Any]] = []
    oos_pieces: list[pd.Series] = []
    benchmark_pieces: list[pd.Series] = []

    for window_id, train_start, train_end, test_start, test_end in walk_forward_windows(prices, config):
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        if len(train_prices) <= 250 or len(test_prices) <= 20:
            continue

        window_config = config
        train_hysteresis = run_hysteresis_sweep(
            train_prices, window_config, period_name=f"nested_train_{window_id}"
        )
        candidates = select_top_trend_candidates(train_hysteresis, window_config)
        leaderboard = run_allocation_leaderboard(
            train_prices,
            window_config,
            candidates,
            train_slice_override=train_prices,
        )
        selected = select_allocation_model(leaderboard, window_config)

        result = evaluate_strategy(
            prices=test_prices,
            ticker=config.base_ticker,
            params=selected["params"],
            variant=selected["variant"],
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="nested_oos",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        row = result["row"]
        window_rows.append(
            {
                "window_id": window_id,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_variant": selected["variant"],
                "selected_label": selected["params"].label(),
                "selection_status": selected["selection_status"],
                "candidates_evaluated": int(len(train_hysteresis) + len(leaderboard)),
                "cagr": row["cagr"],
                "sharpe": row["sharpe"],
                "max_drawdown": row["max_drawdown"],
                "turnover": row["turnover"],
                "exposure": row["exposure"],
                "benchmark_cagr": row["benchmark_cagr"],
                "benchmark_sharpe": row["benchmark_sharpe"],
                "excess_cagr_vs_benchmark": row["excess_cagr_vs_benchmark"],
            }
        )
        oos_pieces.append(result["curve"]["strategy_return"])
        benchmark_pieces.append(result["curve"]["buy_hold_return"])

    windows_table = pd.DataFrame(window_rows)
    if windows_table.empty:
        return {"windows": windows_table, "summary": pd.DataFrame(), "oos_returns": pd.Series(dtype=float)}

    oos_returns = pd.concat(oos_pieces).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    benchmark_returns = pd.concat(benchmark_pieces).sort_index()
    benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep="first")]

    oos_equity = config.initial_capital * (1.0 + oos_returns).cumprod()
    benchmark_equity = config.initial_capital * (1.0 + benchmark_returns).cumprod()
    summary_rows = [
        summarize_performance("nested_oos_stitched", oos_equity, oos_returns, benchmark_equity=benchmark_equity)
        | {
            "windows": int(len(windows_table)),
            "windows_beating_benchmark": int(
                (windows_table["sharpe"] > windows_table["benchmark_sharpe"]).sum()
            ),
            "median_window_sharpe": float(windows_table["sharpe"].median()),
            "median_window_cagr": float(windows_table["cagr"].median()),
        },
        summarize_performance("benchmark_stitched", benchmark_equity, benchmark_returns)
        | {"windows": int(len(windows_table))},
    ]
    summary = pd.DataFrame(summary_rows)
    return {"windows": windows_table, "summary": summary, "oos_returns": oos_returns}


def run_multi_asset(prices: pd.DataFrame, config: ResearchConfig, params: SmaParameters) -> pd.DataFrame:
    rows = []
    for ticker in config.universe:
        result = evaluate_strategy(
            prices=prices,
            ticker=ticker,
            params=params,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="multi_asset",
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"])
    portfolio = evaluate_equal_weight_signal_portfolio(prices, config.universe, params, config)
    rows.append(portfolio)
    table = pd.DataFrame(rows)
    table["beats_benchmark_cagr"] = table["cagr"] > table["benchmark_cagr"]
    table["beats_benchmark_sharpe"] = table["sharpe"] > table["benchmark_sharpe"]
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run_model_leaderboard(prices: pd.DataFrame, config: ResearchConfig, selected_params: SmaParameters) -> pd.DataFrame:
    """Rank long-only variants on the train period only (no test leakage)."""
    rows = []
    evaluation_prices = prices.loc[config.train_start : config.train_end]
    variants = [
        ("long_cash", selected_params),
        ("fallback_spy", selected_params),
        ("fallback_qqq", selected_params),
        ("partial_exposure", SmaParameters(selected_params.short_window, selected_params.long_window, partial_exposure=True)),
        (
            "spread_threshold",
            SmaParameters(
                selected_params.short_window,
                selected_params.long_window,
                spread_threshold=config.spread_threshold,
            ),
        ),
        ("momentum_3m", SmaParameters(selected_params.short_window, selected_params.long_window, momentum_window=63)),
        ("momentum_6m", SmaParameters(selected_params.short_window, selected_params.long_window, momentum_window=126)),
    ]
    for variant, params in variants:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="leaderboard_train",
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"])
    table = pd.DataFrame(rows)
    table["passes_selection"] = (
        (table["cagr"] > 0)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & (table["turnover"] < 8.0)
    )
    return table.sort_values(["passes_selection", "sharpe"], ascending=[False, False]).reset_index(drop=True)


def run_hysteresis_sweep(
    prices: pd.DataFrame,
    config: ResearchConfig,
    period_name: str = "full_hysteresis",
    collect_returns: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame | None]:
    jobs = [
        EvaluationJob(params, "long_cash_hysteresis", 10.0, period_name)
        for params in trend_parameter_grid(config)
    ]
    rows, returns_matrix = evaluate_grid(
        jobs, _grid_context(prices, config), config.parallel_workers, collect_returns=collect_returns
    )
    table = pd.DataFrame(rows)
    table = add_parameter_stability(table)
    table = add_selection_score(table)
    sorted_table = table.sort_values("selection_score", ascending=False)
    order = list(sorted_table.index)
    sorted_table = sorted_table.reset_index(drop=True)
    if collect_returns:
        ordered_returns = None
        if returns_matrix is not None:
            # Keep column i aligned with table row i after sorting.
            ordered_returns = returns_matrix.iloc[:, order]
            ordered_returns.columns = [f"candidate_{idx}" for idx in range(len(order))]
        return sorted_table, ordered_returns
    return sorted_table


def run_allocation_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[TrendAllocationParameters],
    train_slice_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank allocation variants on the train period only (no test leakage)."""
    evaluation_prices = (
        train_slice_override
        if train_slice_override is not None
        else prices.loc[config.train_start : config.train_end]
    )
    variants = [
        "long_cash_hysteresis",
        "long_spy_regime",
        "long_qqq_regime",
        "hybrid_spy_regime",
        "hybrid_qqq_regime",
    ]
    jobs: list[EvaluationJob] = []
    for params in candidates:
        for variant in variants:
            jobs.append(EvaluationJob(params, variant, 10.0, "allocation_train"))
            jobs.append(EvaluationJob(params, variant, 20.0, "allocation_train_20bps"))
    outputs, _ = evaluate_grid(jobs, _grid_context(evaluation_prices, config), config.parallel_workers)

    rows: list[dict[str, Any]] = []
    for base_row, stress_row in zip(outputs[0::2], outputs[1::2]):
        rows.append(
            base_row
            | {
                "cagr_20bps": stress_row["cagr"],
                "sharpe_20bps": stress_row["sharpe"],
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = add_selection_score(table)
    table["robust_20bps"] = (table["cagr_20bps"] > 0) & (table["sharpe_20bps"] > 0)
    table["passes_selection"] = allocation_selection_mask(table, config)
    return table.sort_values(["passes_selection", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def run_capture_analysis(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_model: dict[str, Any],
) -> pd.DataFrame:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    scenarios: list[tuple[str, Any, str]] = [
        ("v2_sma_5_50", SmaParameters(5, 50), "long_cash"),
        ("low_turnover_sma_10_200", SmaParameters(10, 200), "long_cash"),
        (
            "selected_v3",
            selected_model["params"],
            selected_model["variant"],
        ),
    ]
    rows = []
    for model_label, params, variant in scenarios:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=model_label,
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        row = result["row"]
        rows.append(
            {
                "model": model_label,
                "variant": variant,
                "cagr": row["cagr"],
                "sharpe": row["sharpe"],
                "turnover": row["turnover"],
                "upside_capture": row["upside_capture"],
                "downside_capture": row["downside_capture"],
                "missed_return_while_in_cash": row["missed_return_while_in_cash"],
                "fallback_exposure": row["fallback_exposure"],
                "selection_status": selected_model["selection_status"] if model_label == "selected_v3" else "comparison",
            }
        )
    return pd.DataFrame(rows)


def run_turnover_analysis(allocation_leaderboard: pd.DataFrame, model_leaderboard: pd.DataFrame) -> pd.DataFrame:
    parts = []
    if not model_leaderboard.empty:
        parts.append(
            model_leaderboard.assign(source="v2_leaderboard")[
                ["source", "variant", "cagr", "sharpe", "max_drawdown", "turnover", "trades", "cost_drag"]
            ]
        )
    if not allocation_leaderboard.empty:
        parts.append(
            allocation_leaderboard.assign(source="v3_allocation")[
                ["source", "variant", "cagr", "sharpe", "max_drawdown", "turnover", "trades", "cost_drag"]
            ].head(50)
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_v03_comparison(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    scenarios: list[tuple[str, Any, str]] = [
        ("baseline_sma_20_100", SmaParameters(20, 100), "long_cash"),
        ("v2_sma_5_50", SmaParameters(5, 50), "long_cash"),
        ("v2_fallback_spy", SmaParameters(5, 50), "fallback_spy"),
        ("v2_fallback_qqq", SmaParameters(5, 50), "fallback_qqq"),
        ("selected_v3", selected_model["params"], selected_model["variant"]),
    ]
    rows = []
    selected_curve = pd.DataFrame()
    for model_label, params, variant in scenarios:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=model_label,
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        row = result["row"] | {
            "model": model_label,
            "selection_status": selected_model["selection_status"] if model_label == "selected_v3" else "comparison",
        }
        rows.append(row)
        if model_label == "selected_v3":
            selected_curve = result["curve"]
    return pd.DataFrame(rows), selected_curve


def run_v03_cost_sensitivity(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_model: dict[str, Any],
) -> pd.DataFrame:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    rows = []
    for cost_bps in config.cost_bps:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=selected_model["params"],
            variant=selected_model["variant"],
            cost_bps=cost_bps,
            initial_capital=config.initial_capital,
            label="selected_v3",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"] | {"selection_status": selected_model["selection_status"]})
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def run_v04_research(
    prices: pd.DataFrame,
    config: ResearchConfig,
    trend_candidates: list[TrendAllocationParameters],
    selected_v3_model: dict[str, Any],
) -> dict[str, Any]:
    train_prices = prices.loc[config.train_start : config.train_end]
    test_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    risk_filter_sweep = run_risk_filter_sweep(train_prices, config, trend_candidates)
    capture_params = capture_parameter_grid(config, risk_filter_sweep)

    # The v0.3 hurdle and every v0.4 candidate are evaluated on the train
    # period; the selected model sees the test period only in the comparison
    # tables below.
    v3_train = evaluate_strategy(
        prices=train_prices,
        ticker=config.base_ticker,
        params=selected_v3_model["params"],
        variant=selected_v3_model["variant"],
        cost_bps=10.0,
        initial_capital=config.initial_capital,
        label="selected_v3_train",
        market_regime_short_window=config.market_regime_short_window,
        market_regime_long_window=config.market_regime_long_window,
        cash_proxy=config.cash_proxy_ticker,
    )
    capture_leaderboard = run_capture_leaderboard(train_prices, config, capture_params, v3_train["row"])
    selected_v4_model = select_capture_model(capture_leaderboard, config, v3_train["row"], selected_v3_model)
    v04_comparison, v04_curve = run_v04_comparison(prices, config, selected_v3_model, selected_v4_model)
    v04_cost_sensitivity = run_v04_cost_sensitivity(prices, config, selected_v4_model)
    benchmark_comparison = run_benchmark_comparison(prices, config, selected_v3_model, selected_v4_model)

    if not v04_curve.empty:
        market_ticker = "SPY" if "SPY" in test_prices.columns else config.base_ticker
        regimes = classify_market_regime(
            test_prices[config.base_ticker],
            test_prices[market_ticker],
            config.market_regime_short_window,
            config.market_regime_long_window,
        )
        v04_curve = v04_curve.copy()
        v04_curve["regime"] = regimes.reindex(v04_curve.index).fillna("sideways")

    return {
        "capture_leaderboard": capture_leaderboard,
        "risk_filter_sweep": risk_filter_sweep,
        "regime_results": run_regime_results(v04_curve),
        "trade_log": build_trade_log(v04_curve),
        "benchmark_comparison": benchmark_comparison,
        "v04_comparison": v04_comparison,
        "v04_cost_sensitivity": v04_cost_sensitivity,
        "v04_curve": v04_curve,
        "selected_v4_model": selected_v4_model,
    }


def run_risk_filter_sweep(
    prices: pd.DataFrame,
    config: ResearchConfig,
    trend_candidates: list[TrendAllocationParameters],
) -> pd.DataFrame:
    candidates = trend_candidates[: config.capture_trend_candidates] or [TrendAllocationParameters(5, 200)]
    jobs: list[EvaluationJob] = []
    for trend in candidates:
        for vol_window in config.volatility_windows or [20, 40, 60]:
            for target_vol in config.target_volatilities or [0.15, 0.20, 0.25]:
                for max_vol in config.max_realized_volatilities or [0.35, 0.45]:
                    for drawdown_threshold in config.rolling_drawdown_thresholds or [0.08, 0.12]:
                        for sharp_loss in config.sharp_loss_thresholds or [-0.06, -0.08]:
                            params = CaptureAwareAllocationParameters(
                                trend=trend,
                                risk=RiskFilterParameters(
                                    rolling_drawdown_threshold=drawdown_threshold,
                                    realized_volatility_window=vol_window,
                                    max_realized_volatility=max_vol,
                                    sharp_loss_threshold=sharp_loss,
                                ),
                                sizing=VolatilitySizingParameters(
                                    realized_volatility_window=vol_window,
                                    target_volatility=target_vol,
                                ),
                                fallback_asset="cash",
                                fallback_weight=0.0,
                            )
                            jobs.append(EvaluationJob(params, "capture_cash", 10.0, "capture_train"))
    rows, _ = evaluate_grid(jobs, _grid_context(prices, config), config.parallel_workers)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = add_capture_selection_score(table)
    return table.sort_values("selection_score", ascending=False).reset_index(drop=True)


def run_capture_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[CaptureAwareAllocationParameters],
    v3_row: dict[str, Any],
) -> pd.DataFrame:
    from .selection import capture_selection_mask

    jobs: list[EvaluationJob] = []
    for params in candidates:
        variant = capture_variant(params)
        jobs.append(EvaluationJob(params, variant, 10.0, "capture_train"))
        jobs.append(EvaluationJob(params, variant, 20.0, "capture_train_20bps"))
    outputs, _ = evaluate_grid(jobs, _grid_context(prices, config), config.parallel_workers)

    rows: list[dict[str, Any]] = []
    for base_row, stress_row in zip(outputs[0::2], outputs[1::2]):
        rows.append(
            base_row
            | {
                "cagr_20bps": stress_row["cagr"],
                "sharpe_20bps": stress_row["sharpe"],
                "capture_spread_20bps": stress_row["capture_spread"],
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = add_capture_selection_score(table)
    table["robust_20bps"] = (
        (table["cagr_20bps"] > 0)
        & (table["sharpe_20bps"] > 0)
        & (table["capture_spread_20bps"] > 0)
    )
    table["passes_selection"] = capture_selection_mask(table, config, v3_row)
    return table.sort_values(["passes_selection", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def run_v04_comparison(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v3_model: dict[str, Any],
    selected_v4_model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    scenarios: list[tuple[str, Any, str, str]] = [
        ("baseline_sma_20_100", SmaParameters(20, 100), "long_cash", "comparison"),
        ("v2_sma_5_50", SmaParameters(5, 50), "long_cash", "comparison"),
        ("selected_v3", selected_v3_model["params"], selected_v3_model["variant"], selected_v3_model["selection_status"]),
        ("selected_v4", selected_v4_model["params"], selected_v4_model["variant"], selected_v4_model["selection_status"]),
    ]
    rows = []
    selected_curve = pd.DataFrame()
    for model_label, params, variant, status in scenarios:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=model_label,
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"] | {"model": model_label, "selection_status": status})
        if model_label == "selected_v4":
            selected_curve = result["curve"]
    return pd.DataFrame(rows), selected_curve


def run_v04_cost_sensitivity(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v4_model: dict[str, Any],
) -> pd.DataFrame:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    rows = []
    for cost_bps in config.cost_bps:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=selected_v4_model["params"],
            variant=selected_v4_model["variant"],
            cost_bps=cost_bps,
            initial_capital=config.initial_capital,
            label="selected_v4",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"] | {"selection_status": selected_v4_model["selection_status"]})
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def run_benchmark_comparison(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v3_model: dict[str, Any],
    selected_v4_model: dict[str, Any],
) -> pd.DataFrame:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    cash_returns = cash_return_series(evaluation_prices, config.cash_proxy_ticker)
    risk_free_rate = float(cash_returns.mean() * 252) if cash_returns is not None and not cash_returns.empty else 0.0
    rows = []
    for ticker in [config.base_ticker, "SPY", "QQQ"]:
        if ticker in evaluation_prices.columns:
            rows.append(
                _buy_hold_benchmark_row(
                    evaluation_prices[ticker], config.initial_capital, ticker, risk_free_rate=risk_free_rate
                )
            )
    if {config.base_ticker, "QQQ"}.issubset(evaluation_prices.columns):
        blend_return = evaluation_prices[[config.base_ticker, "QQQ"]].pct_change().fillna(0.0).mean(axis=1)
        blend_equity = config.initial_capital * (1.0 + blend_return).cumprod()
        rows.append(
            summarize_performance("50_50_aapl_qqq", blend_equity, blend_return, risk_free_rate=risk_free_rate)
            | {"ticker": "AAPL_QQQ", "model": "50% AAPL / 50% QQQ", "variant": "static_blend"}
        )
    rows.append(
        evaluate_strategy(
            evaluation_prices,
            config.base_ticker,
            SmaParameters(1, 200),
            "long_cash",
            10.0,
            config.initial_capital,
            "aapl_sma200_filter",
            cash_proxy=config.cash_proxy_ticker,
        )["row"]
        | {"model": "AAPL SMA200 filter"}
    )
    for model, selected in [("selected_v3", selected_v3_model), ("selected_v4", selected_v4_model)]:
        rows.append(
            evaluate_strategy(
                evaluation_prices,
                config.base_ticker,
                selected["params"],
                selected["variant"],
                10.0,
                config.initial_capital,
                model,
                config.market_regime_short_window,
                config.market_regime_long_window,
                cash_proxy=config.cash_proxy_ticker,
            )["row"]
            | {"model": model, "selection_status": selected["selection_status"]}
        )
    return pd.DataFrame(rows)
