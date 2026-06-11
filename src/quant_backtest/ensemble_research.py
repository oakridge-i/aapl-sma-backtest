"""Signal-family research and ensemble selection (the 0.6 model search).

The search space is deliberately small: each family has a handful of
parameterizations, one champion per family is picked on the train period,
and the ensemble search is over *which families participate*, not over a
dense parameter grid. All ranking happens on train data; the test period is
touched once per final model in the comparison tables.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from .evaluation import evaluate_strategy
from .parallel import EvaluationContext, EvaluationJob, evaluate_grid
from .registry import family_for_params
from .research_config import ResearchConfig
from .selection import add_selection_score
from .signal_families import (
    AtrTrendParameters,
    DonchianParameters,
    DualMomentumParameters,
    EnsembleParameters,
    High52WeekParameters,
    TimeSeriesMomentumParameters,
)
from .strategies import SmaParameters, TrendAllocationParameters
from .sweeps import walk_forward_windows


# The fixed trend member available to ensembles. This is the published v0.3
# parameterization, frozen a priori; it is never re-fitted, so including it
# adds no selection degrees of freedom.
CANONICAL_TREND_MEMBER = TrendAllocationParameters(
    short_window=5,
    long_window=200,
    entry_threshold=0.01,
    exit_threshold=-0.01,
    min_hold_days=10,
    cooldown_days=5,
)


def family_parameter_grids(config: ResearchConfig) -> dict[str, list[Any]]:
    raw = config.signal_family_grids or {}

    ts = raw.get("ts_momentum", {})
    donchian = raw.get("donchian", {})
    atr = raw.get("atr_trend", {})
    dual = raw.get("dual_momentum", {})
    high = raw.get("high_52w", {})

    grids: dict[str, list[Any]] = {
        "ts_momentum": [
            TimeSeriesMomentumParameters(lookback_days=int(lookback))
            for lookback in ts.get("lookbacks", [63, 126, 252])
        ],
        "donchian_breakout": [
            DonchianParameters(entry_window=int(entry), exit_window=int(exit_window))
            for entry in donchian.get("entry_windows", [55, 100, 252])
            for exit_window in donchian.get("exit_windows", [20, 50])
            if int(exit_window) < int(entry)
        ],
        "atr_trend": [
            AtrTrendParameters(sma_window=int(sma), atr_window=int(atr_window), scale=float(scale))
            for sma in atr.get("sma_windows", [100, 200])
            for atr_window in atr.get("atr_windows", [20])
            for scale in atr.get("scales", [2.0, 3.0])
        ],
        "dual_momentum": [
            DualMomentumParameters(lookback_days=int(lookback), market_ticker=str(dual.get("market", "SPY")).upper())
            for lookback in dual.get("lookbacks", [126, 252])
        ],
        "high_52w": [
            High52WeekParameters(
                window=int(high.get("window", 252)),
                entry_threshold=float(entry),
                exit_threshold=float(exit_threshold),
            )
            for entry in high.get("entry_thresholds", [0.90, 0.95])
            for exit_threshold in high.get("exit_thresholds", [0.80, 0.85])
            if float(exit_threshold) < float(entry)
        ],
    }
    return {family: params for family, params in grids.items() if params}


def run_family_sweep(
    prices: pd.DataFrame,
    config: ResearchConfig,
    period_name: str = "family_train",
) -> tuple[pd.DataFrame, list[Any]]:
    """Evaluate every family parameterization; returns (table, aligned params).

    ``table.iloc[i]`` corresponds to ``params_list[i]`` via the
    ``param_index`` column, so champions can be reconstructed exactly.
    """
    grids = family_parameter_grids(config)
    params_list: list[Any] = []
    families: list[str] = []
    for family_name, family_params in grids.items():
        for params in family_params:
            params_list.append(params)
            families.append(family_name)

    jobs = [
        EvaluationJob(params, family_name, 10.0, period_name)
        for params, family_name in zip(params_list, families)
    ]
    context = EvaluationContext(
        prices=prices,
        ticker=config.base_ticker,
        initial_capital=config.initial_capital,
        market_regime_short_window=config.market_regime_short_window,
        market_regime_long_window=config.market_regime_long_window,
        cash_proxy=config.cash_proxy_ticker,
    )
    rows, _ = evaluate_grid(jobs, context, config.parallel_workers)
    table = pd.DataFrame(rows)
    table["family"] = families
    table["param_index"] = range(len(params_list))
    table = add_selection_score(table)
    return table.sort_values("selection_score", ascending=False).reset_index(drop=True), params_list


def select_family_champions(
    table: pd.DataFrame,
    params_list: list[Any],
    config: ResearchConfig,
) -> dict[str, Any]:
    """Pick the best train-period parameterization per family."""
    champions: dict[str, Any] = {}
    if table.empty:
        return champions
    for family_name, group in table.groupby("family"):
        eligible = group[(group["cagr"] > 0) & (group["turnover"] <= config.ensemble_turnover_limit)]
        source = eligible if not eligible.empty else group
        best = source.sort_values("selection_score", ascending=False).iloc[0]
        champions[str(family_name)] = params_list[int(best["param_index"])]
    return champions


def build_ensemble_candidates(
    champions: dict[str, Any],
    config: ResearchConfig,
) -> list[EnsembleParameters]:
    """Subsets of family champions (plus the fixed trend member) by size."""
    pool: list[Any] = list(champions.values())
    if config.ensemble_include_trend_baseline:
        pool.append(CANONICAL_TREND_MEMBER)
    if not pool:
        return []
    min_members = max(1, config.ensemble_min_members)
    max_members = min(len(pool), config.ensemble_max_members)
    candidates = [
        EnsembleParameters(members=tuple(subset))
        for size in range(min_members, max_members + 1)
        for subset in itertools.combinations(pool, size)
    ]
    return candidates


def run_ensemble_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[EnsembleParameters],
) -> pd.DataFrame:
    """Rank ensemble compositions on the train period with a 20 bps stress."""
    if not candidates:
        return pd.DataFrame()
    jobs: list[EvaluationJob] = []
    for params in candidates:
        jobs.append(EvaluationJob(params, "ensemble_vote", 10.0, "ensemble_train"))
        jobs.append(EvaluationJob(params, "ensemble_vote", 20.0, "ensemble_train_20bps"))
    context = EvaluationContext(
        prices=prices,
        ticker=config.base_ticker,
        initial_capital=config.initial_capital,
        market_regime_short_window=config.market_regime_short_window,
        market_regime_long_window=config.market_regime_long_window,
        cash_proxy=config.cash_proxy_ticker,
    )
    outputs, _ = evaluate_grid(jobs, context, config.parallel_workers)

    rows: list[dict[str, Any]] = []
    for candidate_idx, (base_row, stress_row) in enumerate(zip(outputs[0::2], outputs[1::2])):
        rows.append(
            base_row
            | {
                "candidate_index": candidate_idx,
                "members": len(candidates[candidate_idx].members),
                "member_labels": "+".join(member.label() for member in candidates[candidate_idx].members),
                "cagr_20bps": stress_row["cagr"],
                "sharpe_20bps": stress_row["sharpe"],
            }
        )
    table = pd.DataFrame(rows)
    table = add_selection_score(table)
    table["robust_20bps"] = (table["cagr_20bps"] > 0) & (table["sharpe_20bps"] > 0)
    table["passes_selection"] = (
        (table["cagr"] > 0)
        & (table["turnover"] <= config.ensemble_turnover_limit)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & table["robust_20bps"]
    )
    return table.sort_values(["passes_selection", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def select_ensemble_model(
    leaderboard: pd.DataFrame,
    candidates: list[EnsembleParameters],
    fallback_model: dict[str, Any],
) -> dict[str, Any]:
    if not leaderboard.empty:
        passing = leaderboard[leaderboard["passes_selection"]]
        if not passing.empty:
            best = passing.sort_values("selection_score", ascending=False).iloc[0]
            return {
                "params": candidates[int(best["candidate_index"])],
                "variant": "ensemble_vote",
                "selection_status": "selected_v6",
            }
    return {
        "params": fallback_model["params"],
        "variant": fallback_model["variant"],
        "selection_status": "no_robust_upgrade_baseline_retained",
    }


def run_v06_comparison(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v3_model: dict[str, Any],
    family_champions: dict[str, Any],
    selected_v6_model: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One test-period evaluation per final model (the single test touch)."""
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    scenarios: list[tuple[str, Any, str, str]] = [
        ("baseline_sma_20_100", SmaParameters(20, 100), "long_cash", "comparison"),
        ("selected_v3", selected_v3_model["params"], selected_v3_model["variant"], selected_v3_model["selection_status"]),
    ]
    for family_name, params in sorted(family_champions.items()):
        scenarios.append((f"champion_{family_name}", params, family_name, "family_champion"))
    scenarios.append(
        ("selected_v6", selected_v6_model["params"], selected_v6_model["variant"], selected_v6_model["selection_status"])
    )

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
        if model_label == "selected_v6":
            selected_curve = result["curve"]
    return pd.DataFrame(rows), selected_curve


def run_v06_cost_sensitivity(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v6_model: dict[str, Any],
) -> pd.DataFrame:
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    rows = []
    for cost_bps in config.cost_bps:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=selected_v6_model["params"],
            variant=selected_v6_model["variant"],
            cost_bps=cost_bps,
            initial_capital=config.initial_capital,
            label="selected_v6",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"] | {"selection_status": selected_v6_model["selection_status"]})
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def run_nested_ensemble_walk_forward(
    prices: pd.DataFrame,
    config: ResearchConfig,
    fallback_model: dict[str, Any],
) -> dict[str, Any]:
    """Walk-forward ensemble selection: champions and composition re-picked per window.

    The stitched OOS series is the primary scoreboard for the ensemble model:
    every window's selection sees only that window's train slice.
    """
    window_rows: list[dict[str, Any]] = []
    oos_pieces: list[pd.Series] = []
    benchmark_pieces: list[pd.Series] = []

    for window_id, train_start, train_end, test_start, test_end in walk_forward_windows(prices, config):
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        if len(train_prices) <= 250 or len(test_prices) <= 20:
            continue

        family_table, params_list = run_family_sweep(
            train_prices, config, period_name=f"nested_family_{window_id}"
        )
        champions = select_family_champions(family_table, params_list, config)
        candidates = build_ensemble_candidates(champions, config)
        leaderboard = run_ensemble_leaderboard(train_prices, config, candidates)
        selected = select_ensemble_model(leaderboard, candidates, fallback_model)

        if config.enable_overlays:
            from .overlay_research import overlay_parameter_grid, run_overlay_leaderboard, select_overlay_model

            overlay_candidates = overlay_parameter_grid(config, selected["params"])
            overlay_leaderboard = run_overlay_leaderboard(train_prices, config, overlay_candidates)
            selected = select_overlay_model(overlay_leaderboard, overlay_candidates, selected)

        result = evaluate_strategy(
            prices=test_prices,
            ticker=config.base_ticker,
            params=selected["params"],
            variant=selected["variant"],
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="nested_ensemble_oos",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        row = result["row"]
        selected_label = (
            selected["params"].label() if hasattr(selected["params"], "label") else str(selected["params"])
        )
        window_rows.append(
            {
                "window_id": window_id,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_variant": selected["variant"],
                "selected_label": selected_label,
                "selection_status": selected["selection_status"],
                "candidates_evaluated": int(len(family_table) + len(leaderboard)),
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

    from .metrics import summarize_performance

    oos_equity = config.initial_capital * (1.0 + oos_returns).cumprod()
    benchmark_equity = config.initial_capital * (1.0 + benchmark_returns).cumprod()
    summary_rows = [
        summarize_performance(
            "nested_ensemble_oos_stitched", oos_equity, oos_returns, benchmark_equity=benchmark_equity
        )
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
