"""Research orchestration.

This module wires the research pipeline together and re-exports the public
API of the split modules (``research_config``, ``research_data``,
``evaluation``, ``selection``, ``sweeps``, ``significance``) so that
``from quant_backtest.experiments import ...`` keeps working.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import frame_sha256, load_price_snapshot
from .evaluation import (  # noqa: F401  (re-exported)
    build_trade_log,
    evaluate_equal_weight_signal_portfolio,
    evaluate_strategy,
    run_regime_results,
    summarize_curve,
)
from .research_config import (  # noqa: F401  (re-exported)
    AUTO_WORKERS,
    DEFAULT_VARIANTS,
    ResearchConfig,
    load_research_config,
)
from .research_data import (  # noqa: F401  (re-exported)
    cash_return_series,
    create_fixture_prices,
    download_research_prices,
    research_tickers,
)
from .selection import (  # noqa: F401  (re-exported)
    add_capture_selection_score,
    add_parameter_stability,
    add_selection_score,
    allocation_selection_mask,
    capture_parameter_grid,
    capture_params_from_row,
    capture_selection_mask,
    capture_variant,
    parameter_grid,
    select_allocation_model,
    select_best_parameters,
    select_capture_model,
    select_top_trend_candidates,
    trend_parameter_grid,
    trend_params_from_row,
    trend_selection_mask,
)
from .ensemble_research import (  # noqa: F401  (re-exported)
    CANONICAL_TREND_MEMBER,
    build_ensemble_candidates,
    family_parameter_grids,
    run_ensemble_leaderboard,
    run_family_sweep,
    run_nested_ensemble_walk_forward,
    run_v06_comparison,
    run_v06_cost_sensitivity,
    select_ensemble_model,
    select_family_champions,
)
from .significance import run_pbo_analysis, run_significance_analysis  # noqa: F401  (re-exported)
from .strategies import SmaParameters
from .sweeps import (  # noqa: F401  (re-exported)
    run_allocation_leaderboard,
    run_benchmark_comparison,
    run_capture_analysis,
    run_capture_leaderboard,
    run_cost_sensitivity,
    run_final_model_walk_forward,
    run_hysteresis_sweep,
    run_model_leaderboard,
    run_multi_asset,
    run_nested_walk_forward,
    run_parameter_sweep,
    run_risk_filter_sweep,
    run_train_test,
    run_turnover_analysis,
    run_v03_comparison,
    run_v03_cost_sensitivity,
    run_v04_comparison,
    run_v04_cost_sensitivity,
    run_v04_research,
    run_walk_forward,
    walk_forward_windows,
)

# Backward-compatible aliases for pre-0.6 private names.
_download_prices = download_research_prices
_research_tickers = research_tickers
_cash_return_series = cash_return_series
_walk_forward_windows = walk_forward_windows


@dataclass(frozen=True)
class ResearchResult:
    prices: pd.DataFrame
    baseline_curve: pd.DataFrame
    base_backtest: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    parameter_sweep: pd.DataFrame
    train_test_results: pd.DataFrame
    walk_forward_results: pd.DataFrame
    multi_asset_results: pd.DataFrame
    model_leaderboard: pd.DataFrame
    hysteresis_sweep: pd.DataFrame
    allocation_leaderboard: pd.DataFrame
    capture_analysis: pd.DataFrame
    turnover_analysis: pd.DataFrame
    v03_comparison: pd.DataFrame
    v03_cost_sensitivity: pd.DataFrame
    v03_curve: pd.DataFrame
    capture_leaderboard: pd.DataFrame
    risk_filter_sweep: pd.DataFrame
    regime_results: pd.DataFrame
    trade_log: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    v04_comparison: pd.DataFrame
    v04_cost_sensitivity: pd.DataFrame
    v04_curve: pd.DataFrame
    final_walk_forward: pd.DataFrame = field(default_factory=pd.DataFrame)
    significance_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    nested_walk_forward: pd.DataFrame = field(default_factory=pd.DataFrame)
    nested_walk_forward_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    pbo_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    family_leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    ensemble_leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    v06_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    v06_cost_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    v06_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    nested_ensemble_walk_forward: pd.DataFrame = field(default_factory=pd.DataFrame)
    nested_ensemble_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_metadata: dict[str, Any] = field(default_factory=dict)


def run_research(
    config: ResearchConfig,
    fixture_data: bool = False,
    snapshot_path: Path | None = None,
) -> ResearchResult:
    if snapshot_path is not None:
        prices = load_price_snapshot(Path(snapshot_path))
        data_source = f"snapshot:{snapshot_path}"
    elif fixture_data:
        prices = create_fixture_prices(config)
        data_source = "fixture"
    else:
        prices = download_research_prices(config)
        data_source = "yfinance"
    prices = prices.loc[config.start : config.end or prices.index.max()]

    base_params = SmaParameters(short_window=20, long_window=100)
    baseline = evaluate_strategy(
        prices=prices,
        ticker=config.base_ticker,
        params=base_params,
        variant="long_cash",
        cost_bps=10.0,
        initial_capital=config.initial_capital,
        label="baseline",
        cash_proxy=config.cash_proxy_ticker,
    )
    parameter_sweep = run_parameter_sweep(prices, config)
    train_prices = prices.loc[config.train_start : config.train_end]
    selected_params = select_best_parameters(run_parameter_sweep(train_prices, config, period_name="train"), config)

    cost_sensitivity = run_cost_sensitivity(prices, config, selected_params)
    train_test_results = run_train_test(prices, config)
    walk_forward_results = run_walk_forward(prices, config)
    multi_asset_results = run_multi_asset(prices, config, selected_params)
    # All candidate ranking below happens on the train period only. The test
    # period is touched exactly once per final model, in the comparison and
    # significance steps, so reported out-of-sample numbers are not the result
    # of picking the best test outcome.
    leaderboard = run_model_leaderboard(prices, config, selected_params)
    train_hysteresis = run_hysteresis_sweep(train_prices, config, period_name="train_hysteresis")
    top_trend_candidates = select_top_trend_candidates(train_hysteresis, config)
    hysteresis_sweep = run_hysteresis_sweep(prices, config)
    allocation_leaderboard = run_allocation_leaderboard(prices, config, top_trend_candidates)
    selected_model = select_allocation_model(allocation_leaderboard, config)
    capture_analysis = run_capture_analysis(prices, config, selected_model)
    turnover_analysis = run_turnover_analysis(allocation_leaderboard, leaderboard)
    v03_comparison, v03_curve = run_v03_comparison(prices, config, selected_model)
    v03_cost_sensitivity = run_v03_cost_sensitivity(prices, config, selected_model)
    if config.enable_capture_model:
        v04 = run_v04_research(prices, config, top_trend_candidates, selected_model)
    else:
        v04 = {
            "capture_leaderboard": pd.DataFrame(),
            "risk_filter_sweep": pd.DataFrame(),
            "regime_results": pd.DataFrame(),
            "trade_log": pd.DataFrame(),
            "benchmark_comparison": pd.DataFrame(),
            "v04_comparison": pd.DataFrame(),
            "v04_cost_sensitivity": pd.DataFrame(),
            "v04_curve": pd.DataFrame(),
            "selected_v4_model": None,
        }

    # M2: signal families and ensemble selection (train-only ranking).
    family_leaderboard = pd.DataFrame()
    ensemble_leaderboard = pd.DataFrame()
    v06_comparison = pd.DataFrame()
    v06_cost_sensitivity = pd.DataFrame()
    v06_curve = pd.DataFrame()
    selected_v6_model: dict[str, Any] | None = None
    family_champions: dict[str, Any] = {}
    if config.enable_signal_families:
        family_leaderboard, family_params = run_family_sweep(train_prices, config)
        family_champions = select_family_champions(family_leaderboard, family_params, config)
        ensemble_candidates = build_ensemble_candidates(family_champions, config)
        ensemble_leaderboard = run_ensemble_leaderboard(train_prices, config, ensemble_candidates)
        selected_v6_model = select_ensemble_model(ensemble_leaderboard, ensemble_candidates, selected_model)
        v06_comparison, v06_curve = run_v06_comparison(
            prices, config, selected_model, family_champions, selected_v6_model
        )
        v06_cost_sensitivity = run_v06_cost_sensitivity(prices, config, selected_v6_model)

    final_models: list[tuple[str, Any, str]] = [
        ("baseline_sma_20_100", SmaParameters(20, 100), "long_cash"),
        ("selected_v2", selected_params, "long_cash"),
        ("selected_v3", selected_model["params"], selected_model["variant"]),
    ]
    if v04.get("selected_v4_model"):
        v4_model = v04["selected_v4_model"]
        final_models.append(("selected_v4", v4_model["params"], v4_model["variant"]))
    if selected_v6_model is not None:
        final_models.append(("selected_v6", selected_v6_model["params"], selected_v6_model["variant"]))
    final_walk_forward = run_final_model_walk_forward(prices, config, final_models)

    nested = {"windows": pd.DataFrame(), "summary": pd.DataFrame(), "oos_returns": pd.Series(dtype=float)}
    nested_ensemble = {"windows": pd.DataFrame(), "summary": pd.DataFrame(), "oos_returns": pd.Series(dtype=float)}
    if config.enable_nested_walk_forward:
        nested = run_nested_walk_forward(prices, config)
        if config.enable_signal_families:
            nested_ensemble = run_nested_ensemble_walk_forward(prices, config, selected_model)

    significance_results = pd.DataFrame()
    if config.enable_significance:
        extra_models = []
        extra_stitched = []
        if selected_v6_model is not None:
            extra_models.append(
                ("selected_v6", selected_v6_model["params"], selected_v6_model["variant"], ensemble_leaderboard)
            )
        if not nested_ensemble["oos_returns"].empty:
            extra_stitched.append(("nested_ensemble_oos_stitched", nested_ensemble["oos_returns"]))
        significance_results = run_significance_analysis(
            prices=prices,
            config=config,
            selected_v3_model=selected_model,
            selected_v4_model=v04.get("selected_v4_model"),
            allocation_leaderboard=allocation_leaderboard,
            capture_leaderboard=v04["capture_leaderboard"],
            nested_oos_returns=nested["oos_returns"] if config.enable_nested_walk_forward else None,
            extra_models=extra_models or None,
            extra_stitched=extra_stitched or None,
        )

    pbo_results = pd.DataFrame()
    if config.enable_pbo:
        pbo_results = run_pbo_analysis(prices, config)

    run_metadata = _build_run_metadata(config, prices, data_source)

    return ResearchResult(
        prices=prices,
        baseline_curve=baseline["curve"],
        base_backtest=baseline["metrics"],
        cost_sensitivity=cost_sensitivity,
        parameter_sweep=parameter_sweep,
        train_test_results=train_test_results,
        walk_forward_results=walk_forward_results,
        multi_asset_results=multi_asset_results,
        model_leaderboard=leaderboard,
        hysteresis_sweep=hysteresis_sweep,
        allocation_leaderboard=allocation_leaderboard,
        capture_analysis=capture_analysis,
        turnover_analysis=turnover_analysis,
        v03_comparison=v03_comparison,
        v03_cost_sensitivity=v03_cost_sensitivity,
        v03_curve=v03_curve,
        capture_leaderboard=v04["capture_leaderboard"],
        risk_filter_sweep=v04["risk_filter_sweep"],
        regime_results=v04["regime_results"],
        trade_log=v04["trade_log"],
        benchmark_comparison=v04["benchmark_comparison"],
        v04_comparison=v04["v04_comparison"],
        v04_cost_sensitivity=v04["v04_cost_sensitivity"],
        v04_curve=v04["v04_curve"],
        final_walk_forward=final_walk_forward,
        significance_results=significance_results,
        nested_walk_forward=nested["windows"],
        nested_walk_forward_summary=nested["summary"],
        pbo_results=pbo_results,
        family_leaderboard=family_leaderboard,
        ensemble_leaderboard=ensemble_leaderboard,
        v06_comparison=v06_comparison,
        v06_cost_sensitivity=v06_cost_sensitivity,
        v06_curve=v06_curve,
        nested_ensemble_walk_forward=nested_ensemble["windows"],
        nested_ensemble_summary=nested_ensemble["summary"],
        run_metadata=run_metadata,
    )


def _build_run_metadata(config: ResearchConfig, prices: pd.DataFrame, data_source: str) -> dict[str, Any]:
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    try:
        import yfinance

        versions["yfinance"] = yfinance.__version__
    except Exception:
        pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
        "data_sha256": frame_sha256(prices),
        "data_start": str(prices.index.min().date()),
        "data_end": str(prices.index.max().date()),
        "tickers": list(prices.columns),
        "selection_period": "train",
        "git_commit": _git_commit(),
        "versions": versions,
        "config": dict(config.__dict__),
    }


def _git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    commit = output.stdout.strip()
    return commit or None
