from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .backtest import calculate_closed_trade_returns, calculate_win_rate, count_exposure_episodes
from .costs import BpsCost
from .data import default_end_date, download_adjusted_close, frame_sha256, load_price_snapshot
from .engine import EngineConfig, run_weight_backtest
from .stats import block_bootstrap_summary, deflated_sharpe_ratio, timing_permutation_pvalue
from .metrics import (
    annualized_turnover,
    avoided_downside_while_underweight,
    capture_ratio,
    capture_spread,
    holding_periods,
    missed_return_while_underweight,
    summarize_performance,
    trade_frequency_per_year,
)
from .strategies import (
    CaptureAwareAllocationParameters,
    CaptureAwareTrendStrategy,
    RiskFilterParameters,
    SmaCrossoverStrategy,
    SmaParameters,
    TrendAllocationParameters,
    TrendAllocationStrategy,
    VolatilitySizingParameters,
    build_capture_aware_weights,
    build_fallback_weights,
    build_hybrid_regime_weights,
    build_regime_fallback_weights,
    build_single_asset_weights,
    build_sma_regime,
    classify_market_regime,
)


DEFAULT_VARIANTS = (
    "long_cash",
    "fallback_spy",
    "fallback_qqq",
    "partial_exposure",
    "spread_threshold",
    "momentum_3m",
    "momentum_6m",
)


@dataclass(frozen=True)
class ResearchConfig:
    start: str
    end: str | None
    initial_capital: float
    base_ticker: str
    universe: list[str]
    cost_bps: list[float]
    short_windows: list[int]
    long_windows: list[int]
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None
    walk_forward_train_years: int
    walk_forward_test_years: int
    walk_forward_step_years: int
    spread_threshold: float = 0.01
    output_dir: str = "outputs"
    entry_thresholds: list[float] | None = None
    exit_thresholds: list[float] | None = None
    min_hold_days: list[int] | None = None
    cooldown_days: list[int] | None = None
    top_candidates: int = 20
    final_turnover_limit: float = 6.0
    market_regime_short_window: int = 50
    market_regime_long_window: int = 200
    enable_capture_model: bool = False
    capture_trend_candidates: int = 8
    capture_risk_candidates: int = 20
    capture_turnover_limit: float = 2.5
    min_capture_spread: float = 0.10
    min_upside_capture: float = 0.60
    max_downside_capture: float = 0.50
    max_drawdown_slippage: float = 0.03
    target_volatilities: list[float] | None = None
    volatility_windows: list[int] | None = None
    max_realized_volatilities: list[float] | None = None
    rolling_drawdown_thresholds: list[float] | None = None
    sharp_loss_thresholds: list[float] | None = None
    fallback_assets: list[str] | None = None
    fallback_weights: list[float] | None = None
    fallback_min_hold_days: list[int] | None = None
    fallback_cooldown_days: list[int] | None = None
    cash_proxy_ticker: str | None = None
    enable_significance: bool = True
    bootstrap_iterations: int = 1000
    bootstrap_block_size: int = 21
    permutation_iterations: int = 500
    significance_seed: int = 42


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
    run_metadata: dict[str, Any] = field(default_factory=dict)


def load_research_config(path: Path) -> ResearchConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    period = raw["period"]
    train_test = raw["train_test"]
    walk_forward = raw["walk_forward"]
    grid = raw["sma_grid"]
    hysteresis = raw.get("hysteresis", {})
    selection = raw.get("selection", {})
    market_regime = raw.get("market_regime", {})
    capture_model = raw.get("capture_model", {})
    risk_filters = raw.get("risk_filters", {})
    volatility_sizing = raw.get("volatility_sizing", {})
    fallback = raw.get("fallback", {})
    cash = raw.get("cash", {})
    significance = raw.get("significance", {})
    cash_proxy = cash.get("proxy")
    if cash_proxy is not None and str(cash_proxy).lower() in ("", "none", "cash", "null"):
        cash_proxy = None
    return ResearchConfig(
        start=str(period["start"]),
        end=None if period.get("end") in (None, "latest") else str(period["end"]),
        initial_capital=float(raw.get("initial_capital", 10_000.0)),
        base_ticker=str(raw["base_ticker"]).upper(),
        universe=[str(ticker).upper() for ticker in raw["universe"]],
        cost_bps=[float(value) for value in raw["cost_bps"]],
        short_windows=[int(value) for value in grid["short"]],
        long_windows=[int(value) for value in grid["long"]],
        train_start=str(train_test["train_start"]),
        train_end=str(train_test["train_end"]),
        test_start=str(train_test["test_start"]),
        test_end=None if train_test.get("test_end") in (None, "latest") else str(train_test["test_end"]),
        walk_forward_train_years=int(walk_forward["train_years"]),
        walk_forward_test_years=int(walk_forward["test_years"]),
        walk_forward_step_years=int(walk_forward["step_years"]),
        spread_threshold=float(raw.get("spread_threshold", 0.01)),
        output_dir=str(raw.get("output_dir", "outputs")),
        entry_thresholds=[float(value) for value in hysteresis.get("entry_thresholds", [0.0, 0.005, 0.01, 0.02])],
        exit_thresholds=[float(value) for value in hysteresis.get("exit_thresholds", [0.0, -0.005, -0.01])],
        min_hold_days=[int(value) for value in hysteresis.get("min_hold_days", [0, 10, 20])],
        cooldown_days=[int(value) for value in hysteresis.get("cooldown_days", [0, 5, 10])],
        top_candidates=int(selection.get("top_candidates", 20)),
        final_turnover_limit=float(selection.get("final_turnover_limit", 6.0)),
        market_regime_short_window=int(market_regime.get("short_window", 50)),
        market_regime_long_window=int(market_regime.get("long_window", 200)),
        enable_capture_model=bool(capture_model.get("enabled", False)),
        capture_trend_candidates=int(capture_model.get("trend_candidates", 8)),
        capture_risk_candidates=int(capture_model.get("risk_candidates", 20)),
        capture_turnover_limit=float(selection.get("capture_turnover_limit", 2.5)),
        min_capture_spread=float(selection.get("min_capture_spread", 0.10)),
        min_upside_capture=float(selection.get("min_upside_capture", 0.60)),
        max_downside_capture=float(selection.get("max_downside_capture", 0.50)),
        max_drawdown_slippage=float(selection.get("max_drawdown_slippage", 0.03)),
        target_volatilities=[float(value) for value in volatility_sizing.get("target_volatilities", [0.15, 0.20, 0.25])],
        volatility_windows=[int(value) for value in volatility_sizing.get("windows", [20, 40, 60])],
        max_realized_volatilities=[
            float(value) for value in risk_filters.get("max_realized_volatilities", [0.35, 0.45])
        ],
        rolling_drawdown_thresholds=[
            float(value) for value in risk_filters.get("rolling_drawdown_thresholds", [0.08, 0.12])
        ],
        sharp_loss_thresholds=[float(value) for value in risk_filters.get("sharp_loss_thresholds", [-0.06, -0.08])],
        fallback_assets=[str(value).upper() for value in fallback.get("assets", ["cash", "SPY", "QQQ", "hybrid_QQQ"])],
        fallback_weights=[float(value) for value in fallback.get("weights", [0.5, 0.75, 1.0])],
        fallback_min_hold_days=[int(value) for value in fallback.get("min_hold_days", [0, 10])],
        fallback_cooldown_days=[int(value) for value in fallback.get("cooldown_days", [0, 5])],
        cash_proxy_ticker=None if cash_proxy is None else str(cash_proxy).upper(),
        enable_significance=bool(significance.get("enabled", True)),
        bootstrap_iterations=int(significance.get("bootstrap_iterations", 1000)),
        bootstrap_block_size=int(significance.get("block_size", 21)),
        permutation_iterations=int(significance.get("permutation_iterations", 500)),
        significance_seed=int(significance.get("seed", 42)),
    )


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
        prices = _download_prices(config)
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

    final_models: list[tuple[str, Any, str]] = [
        ("baseline_sma_20_100", SmaParameters(20, 100), "long_cash"),
        ("selected_v2", selected_params, "long_cash"),
        ("selected_v3", selected_model["params"], selected_model["variant"]),
    ]
    if v04.get("selected_v4_model"):
        v4_model = v04["selected_v4_model"]
        final_models.append(("selected_v4", v4_model["params"], v4_model["variant"]))
    final_walk_forward = run_final_model_walk_forward(prices, config, final_models)

    significance_results = pd.DataFrame()
    if config.enable_significance:
        significance_results = run_significance_analysis(
            prices=prices,
            config=config,
            selected_v3_model=selected_model,
            selected_v4_model=v04.get("selected_v4_model"),
            allocation_leaderboard=allocation_leaderboard,
            capture_leaderboard=v04["capture_leaderboard"],
        )

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


def run_parameter_sweep(prices: pd.DataFrame, config: ResearchConfig, period_name: str = "full") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for short_window, long_window in parameter_grid(config):
        params = SmaParameters(short_window=short_window, long_window=long_window)
        result = evaluate_strategy(
            prices=prices,
            ticker=config.base_ticker,
            params=params,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=period_name,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"])
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


def _walk_forward_windows(
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
    for window_id, train_start, train_end, test_start, test_end in _walk_forward_windows(prices, config):
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
    for window_id, _, _, test_start, test_end in _walk_forward_windows(prices, config):
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


def run_significance_analysis(
    prices: pd.DataFrame,
    config: ResearchConfig,
    selected_v3_model: dict[str, Any],
    selected_v4_model: dict[str, Any] | None,
    allocation_leaderboard: pd.DataFrame,
    capture_leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap, Deflated Sharpe, and permutation diagnostics on the test period.

    These are reporting steps for already-selected models: they quantify how
    much of the single observed test-period result could be luck.
    """
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
        cash_returns = _cash_return_series(test_prices, config.cash_proxy_ticker)
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
    return pd.DataFrame(rows)


def _cash_return_series(prices: pd.DataFrame, cash_proxy: str | None) -> pd.Series | None:
    if not cash_proxy or cash_proxy not in prices.columns:
        return None
    return prices[cash_proxy].dropna().pct_change().fillna(0.0)


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


def run_hysteresis_sweep(prices: pd.DataFrame, config: ResearchConfig, period_name: str = "full_hysteresis") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for params in trend_parameter_grid(config):
        result = evaluate_strategy(
            prices=prices,
            ticker=config.base_ticker,
            params=params,
            variant="long_cash_hysteresis",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=period_name,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(result["row"])
    table = pd.DataFrame(rows)
    table = add_parameter_stability(table)
    table = add_selection_score(table)
    return table.sort_values("selection_score", ascending=False).reset_index(drop=True)


def run_allocation_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[TrendAllocationParameters],
) -> pd.DataFrame:
    """Rank allocation variants on the train period only (no test leakage)."""
    evaluation_prices = prices.loc[config.train_start : config.train_end]
    variants = [
        "long_cash_hysteresis",
        "long_spy_regime",
        "long_qqq_regime",
        "hybrid_spy_regime",
        "hybrid_qqq_regime",
    ]
    rows: list[dict[str, Any]] = []
    for params in candidates:
        for variant in variants:
            result = evaluate_strategy(
                prices=evaluation_prices,
                ticker=config.base_ticker,
                params=params,
                variant=variant,
                cost_bps=10.0,
                initial_capital=config.initial_capital,
                label="allocation_train",
                market_regime_short_window=config.market_regime_short_window,
                market_regime_long_window=config.market_regime_long_window,
                cash_proxy=config.cash_proxy_ticker,
            )
            stress = evaluate_strategy(
                prices=evaluation_prices,
                ticker=config.base_ticker,
                params=params,
                variant=variant,
                cost_bps=20.0,
                initial_capital=config.initial_capital,
                label="allocation_train_20bps",
                market_regime_short_window=config.market_regime_short_window,
                market_regime_long_window=config.market_regime_long_window,
                cash_proxy=config.cash_proxy_ticker,
            )
            row = result["row"] | {
                "cagr_20bps": stress["row"]["cagr"],
                "sharpe_20bps": stress["row"]["sharpe"],
            }
            rows.append(row)

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
) -> dict[str, pd.DataFrame]:
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
    rows: list[dict[str, Any]] = []
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
                            result = evaluate_strategy(
                                prices=prices,
                                ticker=config.base_ticker,
                                params=params,
                                variant="capture_cash",
                                cost_bps=10.0,
                                initial_capital=config.initial_capital,
                                label="capture_train",
                                market_regime_short_window=config.market_regime_short_window,
                                market_regime_long_window=config.market_regime_long_window,
                                cash_proxy=config.cash_proxy_ticker,
                            )
                            rows.append(result["row"])
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = add_capture_selection_score(table)
    return table.sort_values("selection_score", ascending=False).reset_index(drop=True)


def capture_parameter_grid(
    config: ResearchConfig,
    risk_filter_sweep: pd.DataFrame,
) -> list[CaptureAwareAllocationParameters]:
    if risk_filter_sweep.empty:
        base_params = [
            CaptureAwareAllocationParameters(
                trend=TrendAllocationParameters(5, 200),
                risk=RiskFilterParameters(),
                sizing=VolatilitySizingParameters(),
            )
        ]
    else:
        base_params = [
            capture_params_from_row(row)
            for _, row in risk_filter_sweep.head(config.capture_risk_candidates).iterrows()
        ]

    params: list[CaptureAwareAllocationParameters] = []
    for base in base_params:
        for raw_asset in config.fallback_assets or ["cash", "SPY", "QQQ", "hybrid_QQQ"]:
            asset = str(raw_asset).upper()
            hybrid = asset.startswith("HYBRID_")
            fallback_asset = asset.replace("HYBRID_", "")
            if fallback_asset == "CASH":
                params.append(
                    CaptureAwareAllocationParameters(
                        trend=base.trend,
                        risk=base.risk,
                        sizing=base.sizing,
                        fallback_asset="cash",
                        fallback_weight=0.0,
                    )
                )
                continue
            for weight in config.fallback_weights or [0.5, 0.75, 1.0]:
                for min_hold in config.fallback_min_hold_days or [0, 10]:
                    for cooldown in config.fallback_cooldown_days or [0, 5]:
                        params.append(
                            CaptureAwareAllocationParameters(
                                trend=base.trend,
                                risk=base.risk,
                                sizing=base.sizing,
                                fallback_asset=fallback_asset,
                                fallback_weight=weight,
                                fallback_min_hold_days=min_hold,
                                fallback_cooldown_days=cooldown,
                                hybrid_fallback=hybrid,
                            )
                        )
    return list({param.label(): param for param in params}.values())


def run_capture_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[CaptureAwareAllocationParameters],
    v3_row: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for params in candidates:
        variant = capture_variant(params)
        result = evaluate_strategy(
            prices=prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="capture_train",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        stress = evaluate_strategy(
            prices=prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=20.0,
            initial_capital=config.initial_capital,
            label="capture_train_20bps",
            market_regime_short_window=config.market_regime_short_window,
            market_regime_long_window=config.market_regime_long_window,
            cash_proxy=config.cash_proxy_ticker,
        )
        rows.append(
            result["row"]
            | {
                "cagr_20bps": stress["row"]["cagr"],
                "sharpe_20bps": stress["row"]["sharpe"],
                "capture_spread_20bps": stress["row"]["capture_spread"],
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
    cash_returns = _cash_return_series(evaluation_prices, config.cash_proxy_ticker)
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


def run_regime_results(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or "regime" not in curve.columns:
        return pd.DataFrame()
    rows = []
    for regime, group in curve.groupby("regime"):
        if group.empty:
            continue
        upside = capture_ratio(group["strategy_return"], group["buy_hold_return"], "up")
        downside = capture_ratio(group["strategy_return"], group["buy_hold_return"], "down")
        rows.append(
            {
                "regime": regime,
                "days": int(len(group)),
                "strategy_return": float((1.0 + group["strategy_return"]).prod() - 1.0),
                "benchmark_return": float((1.0 + group["buy_hold_return"]).prod() - 1.0),
                "upside_capture": upside,
                "downside_capture": downside,
                "capture_spread": capture_spread(upside, downside),
                "average_exposure": float(group["position"].abs().mean()) if "position" in group else np.nan,
                "fallback_exposure": float(group["fallback_position"].abs().mean())
                if "fallback_position" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def build_trade_log(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or "position" not in curve.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    in_trade = False
    entry_date = None
    previous_date = None
    for date, row in curve.iterrows():
        exposed = float(row.get("position", 0.0) or 0.0) > 0.0
        if exposed and not in_trade:
            entry_date = date
            in_trade = True
        elif in_trade and not exposed and entry_date is not None:
            rows.append(_trade_log_row(curve, entry_date, date, "closed"))
            in_trade = False
            entry_date = None
        previous_date = date
    if in_trade and entry_date is not None and previous_date is not None:
        rows.append(_trade_log_row(curve, entry_date, previous_date, "open"))
    return pd.DataFrame(rows)


def select_capture_model(
    table: pd.DataFrame,
    config: ResearchConfig,
    v3_row: dict[str, Any],
    selected_v3_model: dict[str, Any],
) -> dict[str, Any]:
    if not table.empty:
        passing = table[table["passes_selection"]]
        if not passing.empty:
            best = passing.sort_values("selection_score", ascending=False).iloc[0]
            return {
                "params": capture_params_from_row(best),
                "variant": str(best["variant"]),
                "selection_status": "selected_v4",
            }
    return {
        "params": selected_v3_model["params"],
        "variant": selected_v3_model["variant"],
        "selection_status": "no_robust_upgrade_baseline_retained",
    }


def add_capture_selection_score(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    enriched = table.copy()
    if "capture_spread" not in enriched.columns:
        enriched["capture_spread"] = enriched.apply(
            lambda row: capture_spread(float(row["upside_capture"]), float(row["downside_capture"])),
            axis=1,
        )
    hot_pixel = enriched.get("hot_pixel_risk", pd.Series(0.0, index=enriched.index)).fillna(0.0).clip(lower=0.0)
    enriched["selection_score"] = (
        enriched["sharpe"].fillna(0.0)
        + enriched["cagr"].fillna(0.0)
        + 1.50 * enriched["capture_spread"].fillna(0.0)
        - 0.75 * enriched["downside_capture"].fillna(0.0).clip(lower=0.0)
        - 0.04 * enriched["turnover"].fillna(0.0)
        + 0.50 * enriched["drawdown_improvement_vs_benchmark"].fillna(0.0)
        - 0.25 * hot_pixel
    )
    return enriched


def capture_selection_mask(table: pd.DataFrame, config: ResearchConfig, v3_row: dict[str, Any]) -> pd.Series:
    v3_cagr = float(v3_row.get("cagr", 0.0) or 0.0)
    v3_sharpe = float(v3_row.get("sharpe", 0.0) or 0.0)
    v3_drawdown = float(v3_row.get("max_drawdown", -1.0) or -1.0)
    return (
        (table["cagr"] > v3_cagr)
        & (table["sharpe"] > v3_sharpe)
        & (table["turnover"] <= config.capture_turnover_limit)
        & (table["max_drawdown"] >= v3_drawdown - config.max_drawdown_slippage)
        & (table["upside_capture"] >= config.min_upside_capture)
        & (table["downside_capture"] <= config.max_downside_capture)
        & (table["capture_spread"] >= config.min_capture_spread)
        & table["robust_20bps"]
    )


def capture_params_from_row(row: pd.Series) -> CaptureAwareAllocationParameters:
    fallback_asset = _row_value(row, "fallback_asset", "cash")
    return CaptureAwareAllocationParameters(
        trend=trend_params_from_row(row),
        risk=RiskFilterParameters(
            price_sma_window=int(_row_value(row, "risk_price_sma_window", 200)),
            rolling_drawdown_threshold=float(_row_value(row, "risk_drawdown_threshold", 0.10)),
            realized_volatility_window=int(_row_value(row, "risk_volatility_window", 20)),
            max_realized_volatility=float(_row_value(row, "risk_max_realized_volatility", 0.45)),
            sharp_loss_threshold=float(_row_value(row, "risk_sharp_loss_threshold", -0.08)),
        ),
        sizing=VolatilitySizingParameters(
            realized_volatility_window=int(_row_value(row, "sizing_window", 20)),
            target_volatility=float(_row_value(row, "target_volatility", 0.20)),
        ),
        fallback_asset=str(fallback_asset),
        fallback_weight=float(_row_value(row, "fallback_weight", 0.0)),
        fallback_min_hold_days=int(_row_value(row, "fallback_min_hold_days", 0)),
        fallback_cooldown_days=int(_row_value(row, "fallback_cooldown_days", 0)),
        hybrid_fallback=bool(_row_value(row, "hybrid_fallback", False)),
    )


def capture_variant(params: CaptureAwareAllocationParameters) -> str:
    fallback = params.fallback_asset.lower()
    if fallback == "cash" or params.fallback_weight <= 0:
        return "capture_cash"
    prefix = "capture_hybrid" if params.hybrid_fallback else "capture_fallback"
    return f"{prefix}_{fallback}_regime"


def _buy_hold_benchmark_row(
    price: pd.Series,
    initial_capital: float,
    ticker: str,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    returns = price.pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + returns).cumprod()
    return summarize_performance(f"{ticker}_buy_hold", equity, returns, risk_free_rate=risk_free_rate) | {
        "ticker": ticker,
        "model": f"{ticker} buy and hold",
        "variant": "buy_hold",
    }


def _trade_log_row(curve: pd.DataFrame, entry_date, exit_date, status: str) -> dict[str, Any]:
    trade = curve.loc[entry_date:exit_date]
    entry_price = float(trade["price"].iloc[0])
    exit_price = float(trade["price"].iloc[-1])
    price_path = trade["price"] / entry_price - 1.0
    return {
        "entry_date": pd.Timestamp(entry_date).date().isoformat(),
        "exit_date": pd.Timestamp(exit_date).date().isoformat(),
        "status": status,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "holding_days": int(len(trade)),
        "trade_return": float((1.0 + trade["strategy_return"]).prod() - 1.0),
        "max_favorable_excursion": float(price_path.max()),
        "max_adverse_excursion": float(price_path.min()),
        "regime_at_entry": str(trade["regime"].iloc[0]) if "regime" in trade else "",
        "volatility_at_entry": float(trade["realized_volatility"].iloc[0])
        if "realized_volatility" in trade and pd.notna(trade["realized_volatility"].iloc[0])
        else np.nan,
        "trend_spread_at_entry": float(trade["spread"].iloc[0])
        if "spread" in trade and pd.notna(trade["spread"].iloc[0])
        else np.nan,
    }


def _capture_fallback_ticker(params: CaptureAwareAllocationParameters) -> str | None:
    fallback = params.fallback_asset.upper()
    if fallback in {"", "CASH"} or params.fallback_weight <= 0:
        return None
    return fallback


def _row_value(row: pd.Series, key: str, default):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def select_top_trend_candidates(table: pd.DataFrame, config: ResearchConfig) -> list[TrendAllocationParameters]:
    if table.empty:
        return [TrendAllocationParameters(5, 50)]
    candidates = table[trend_selection_mask(table, config)]
    source = candidates if not candidates.empty else table
    return [
        trend_params_from_row(row)
        for _, row in source.sort_values("selection_score", ascending=False).head(config.top_candidates).iterrows()
    ]


def select_allocation_model(table: pd.DataFrame, config: ResearchConfig) -> dict[str, Any]:
    if not table.empty:
        passing = table[table["passes_selection"]]
        if not passing.empty:
            best = passing.sort_values("selection_score", ascending=False).iloc[0]
            return {
                "params": trend_params_from_row(best),
                "variant": str(best["variant"]),
                "selection_status": "selected_v3",
            }
    return {
        "params": SmaParameters(5, 50),
        "variant": "long_cash",
        "selection_status": "no_robust_upgrade_baseline_retained",
    }


def add_selection_score(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    enriched = table.copy()
    hot_pixel = enriched.get("hot_pixel_risk", pd.Series(0.0, index=enriched.index)).fillna(0.0).clip(lower=0.0)
    enriched["selection_score"] = (
        enriched["sharpe"].fillna(0.0)
        + enriched["cagr"].fillna(0.0)
        - 0.04 * enriched["turnover"].fillna(0.0)
        + 0.50 * enriched["drawdown_improvement_vs_benchmark"].fillna(0.0)
        - 0.25 * hot_pixel
    )
    return enriched


def trend_selection_mask(table: pd.DataFrame, config: ResearchConfig) -> pd.Series:
    return (
        (table["cagr"] > 0)
        & (table["turnover"] <= config.final_turnover_limit)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & (table.get("neighbor_sharpe", table["sharpe"]).fillna(table["sharpe"]) > 0)
    )


def allocation_selection_mask(table: pd.DataFrame, config: ResearchConfig) -> pd.Series:
    return (
        (table["cagr"] > 0)
        & (table["turnover"] <= config.final_turnover_limit)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & table["robust_20bps"]
    )


def trend_parameter_grid(config: ResearchConfig) -> list[TrendAllocationParameters]:
    entry_thresholds = config.entry_thresholds or [0.0, 0.005, 0.01, 0.02]
    exit_thresholds = config.exit_thresholds or [0.0, -0.005, -0.01]
    min_hold_days = config.min_hold_days or [0, 10, 20]
    cooldown_days = config.cooldown_days or [0, 5, 10]
    params = []
    for short_window, long_window in parameter_grid(config):
        for entry_threshold in entry_thresholds:
            for exit_threshold in exit_thresholds:
                for min_hold in min_hold_days:
                    for cooldown in cooldown_days:
                        if entry_threshold >= exit_threshold:
                            params.append(
                                TrendAllocationParameters(
                                    short_window=short_window,
                                    long_window=long_window,
                                    entry_threshold=entry_threshold,
                                    exit_threshold=exit_threshold,
                                    min_hold_days=min_hold,
                                    cooldown_days=cooldown,
                                )
                            )
    return params


def trend_params_from_row(row: pd.Series) -> TrendAllocationParameters:
    return TrendAllocationParameters(
        short_window=int(row["short_window"]),
        long_window=int(row["long_window"]),
        entry_threshold=float(row.get("entry_threshold", 0.0) or 0.0),
        exit_threshold=float(row.get("exit_threshold", 0.0) or 0.0),
        min_hold_days=int(row.get("min_hold_days", 0) or 0),
        cooldown_days=int(row.get("cooldown_days", 0) or 0),
    )


def evaluate_strategy(
    prices: pd.DataFrame,
    ticker: str,
    params: SmaParameters | TrendAllocationParameters | CaptureAwareAllocationParameters,
    variant: str,
    cost_bps: float,
    initial_capital: float,
    label: str,
    market_regime_short_window: int = 50,
    market_regime_long_window: int = 200,
    cash_proxy: str | None = None,
) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in prices.columns:
        raise ValueError(f"Missing ticker in price data: {ticker}")

    needed = [ticker]
    capture_fallback = _capture_fallback_ticker(params) if isinstance(params, CaptureAwareAllocationParameters) else None
    if variant in {"fallback_spy", "long_spy_regime", "hybrid_spy_regime"} or capture_fallback == "SPY":
        needed.append("SPY")
    if variant in {"fallback_qqq", "long_qqq_regime", "hybrid_qqq_regime"} or capture_fallback == "QQQ":
        needed.append("QQQ")
    needed = list(dict.fromkeys(needed))
    available = [column for column in needed if column in prices.columns]

    price = prices[ticker].dropna()
    if isinstance(params, CaptureAwareAllocationParameters):
        fallback_ticker = capture_fallback if capture_fallback in prices.columns else None
        market_ticker = fallback_ticker or ("SPY" if "SPY" in prices.columns else "QQQ" if "QQQ" in prices.columns else None)
        market_risk_off = None
        if params.risk.use_market_sma_filter and market_ticker:
            market_regime = build_sma_regime(
                prices[market_ticker],
                market_regime_short_window,
                market_regime_long_window,
            )
            market_risk_off = ~market_regime
        fallback_regime = None
        if fallback_ticker:
            fallback_regime = build_sma_regime(
                prices[fallback_ticker],
                market_regime_short_window,
                market_regime_long_window,
            )
        signals = CaptureAwareTrendStrategy(params).generate(
            price,
            market_risk_off=market_risk_off,
            fallback_regime=fallback_regime,
        )
        weights = build_capture_aware_weights(ticker, signals, fallback_ticker)
        available = [column for column in weights.columns if column in prices.columns]
    elif isinstance(params, TrendAllocationParameters):
        signals = TrendAllocationStrategy(params).generate(price)
        if variant == "fallback_spy" and "SPY" in prices.columns:
            weights = build_fallback_weights(ticker, "SPY", signals.target_position)
        elif variant == "fallback_qqq" and "QQQ" in prices.columns:
            weights = build_fallback_weights(ticker, "QQQ", signals.target_position)
        elif variant == "long_spy_regime" and "SPY" in prices.columns:
            regime = build_sma_regime(prices["SPY"], market_regime_short_window, market_regime_long_window)
            weights = build_regime_fallback_weights(ticker, "SPY", signals.target_position, regime)
        elif variant == "long_qqq_regime" and "QQQ" in prices.columns:
            regime = build_sma_regime(prices["QQQ"], market_regime_short_window, market_regime_long_window)
            weights = build_regime_fallback_weights(ticker, "QQQ", signals.target_position, regime)
        elif variant == "hybrid_spy_regime" and "SPY" in prices.columns:
            regime = build_sma_regime(prices["SPY"], market_regime_short_window, market_regime_long_window)
            weights = build_hybrid_regime_weights(ticker, "SPY", signals, params, regime)
        elif variant == "hybrid_qqq_regime" and "QQQ" in prices.columns:
            regime = build_sma_regime(prices["QQQ"], market_regime_short_window, market_regime_long_window)
            weights = build_hybrid_regime_weights(ticker, "QQQ", signals, params, regime)
        else:
            weights = build_single_asset_weights(ticker, signals.target_position)
    else:
        signals = SmaCrossoverStrategy(params).generate(price)
        if variant == "fallback_spy" and "SPY" in prices.columns:
            weights = build_fallback_weights(ticker, "SPY", signals.target_position)
        elif variant == "fallback_qqq" and "QQQ" in prices.columns:
            weights = build_fallback_weights(ticker, "QQQ", signals.target_position)
        else:
            weights = build_single_asset_weights(ticker, signals.target_position)

    cash_returns = _cash_return_series(prices, cash_proxy)
    returns = prices[available].pct_change().fillna(0.0)
    engine_result = run_weight_backtest(
        returns=returns,
        target_weights=weights,
        config=EngineConfig(initial_capital=initial_capital, cost_model=BpsCost(cost_bps)),
        cash_returns=cash_returns,
    )
    risk_free_rate = 0.0
    if cash_returns is not None:
        aligned_cash = cash_returns.reindex(returns.index).dropna()
        if not aligned_cash.empty:
            risk_free_rate = float(aligned_cash.mean() * 252)
    curve = _combine_curve(price, signals, engine_result.curve, engine_result.executed_weights, ticker, prices)
    row = summarize_curve(
        curve=curve,
        executed_weights=engine_result.executed_weights,
        ticker=ticker,
        label=label,
        variant=variant,
        params=params,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
    )
    return {
        "row": row,
        "curve": curve,
        "metrics": pd.DataFrame([row]),
        "weights": engine_result.executed_weights,
        "risk_free_rate": risk_free_rate,
    }


def evaluate_equal_weight_signal_portfolio(
    prices: pd.DataFrame,
    universe: list[str],
    params: SmaParameters,
    config: ResearchConfig,
) -> dict[str, Any]:
    weights = []
    valid_tickers = [ticker for ticker in universe if ticker in prices.columns]
    for ticker in valid_tickers:
        signals = SmaCrossoverStrategy(params).generate(prices[ticker].dropna())
        weights.append(signals.target_position.rename(ticker))
    target_weights = pd.concat(weights, axis=1).reindex(prices.index).fillna(0.0)
    if valid_tickers:
        target_weights = target_weights / len(valid_tickers)
    returns = prices[valid_tickers].pct_change().fillna(0.0)
    cash_returns = _cash_return_series(prices, config.cash_proxy_ticker)
    result = run_weight_backtest(
        returns,
        target_weights,
        EngineConfig(initial_capital=config.initial_capital, cost_model=BpsCost(10.0)),
        cash_returns=cash_returns,
    )

    risk_free_rate = 0.0
    if cash_returns is not None:
        aligned_cash = cash_returns.reindex(returns.index).dropna()
        if not aligned_cash.empty:
            risk_free_rate = float(aligned_cash.mean() * 252)
    basket_return = returns.mean(axis=1)
    benchmark_equity = config.initial_capital * (1.0 + basket_return).cumprod()
    total_exposure = result.executed_weights.abs().sum(axis=1)
    closed = calculate_closed_trade_returns(result.curve["strategy_return"], total_exposure)
    win_rate = calculate_win_rate(closed)
    row = summarize_performance(
        name="equal_weight_signal_portfolio",
        equity=result.curve["strategy_equity"],
        returns=result.curve["strategy_return"],
        trades=count_exposure_episodes(total_exposure),
        win_rate=win_rate,
        risk_free_rate=risk_free_rate,
        exposure=float((total_exposure > 0).mean()),
        turnover=annualized_turnover(result.curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=result.curve["gross_strategy_equity"],
        benchmark_equity=benchmark_equity,
    )
    return row | {
        "ticker": "EQUAL_WEIGHT",
        "label": "multi_asset",
        "variant": "equal_weight_signal_portfolio",
        "short_window": params.short_window,
        "long_window": params.long_window,
        "cost_bps": 10.0,
        "benchmark_cagr": summarize_performance("benchmark", benchmark_equity, basket_return)["cagr"],
        "benchmark_sharpe": summarize_performance("benchmark", benchmark_equity, basket_return)["sharpe"],
        "benchmark_max_drawdown": summarize_performance("benchmark", benchmark_equity, basket_return)["max_drawdown"],
    }


def summarize_curve(
    curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    label: str,
    variant: str,
    params: SmaParameters | TrendAllocationParameters | CaptureAwareAllocationParameters,
    cost_bps: float,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    total_exposure = executed_weights.abs().sum(axis=1)
    closed = calculate_closed_trade_returns(curve["strategy_return"], total_exposure)
    win_rate = calculate_win_rate(closed)
    benchmark_metrics = summarize_performance(
        "benchmark",
        curve["buy_hold_equity"],
        curve["buy_hold_return"],
        risk_free_rate=risk_free_rate,
    )
    ticker_weight = executed_weights.get(ticker, pd.Series(0.0, index=curve.index)).reindex(curve.index).fillna(0.0)
    fallback_columns = [column for column in executed_weights.columns if column != ticker]
    fallback_exposure = (
        float(executed_weights[fallback_columns].abs().sum(axis=1).mean()) if fallback_columns else 0.0
    )
    holds = holding_periods(ticker_weight)
    # Count exposure episodes, not turnover days: with volatility sizing the
    # weight changes almost daily without opening or closing a trade.
    trade_count = count_exposure_episodes(total_exposure)
    upside = capture_ratio(curve["strategy_return"], curve["buy_hold_return"], "up")
    downside = capture_ratio(curve["strategy_return"], curve["buy_hold_return"], "down")
    risk = getattr(params, "risk", None)
    sizing = getattr(params, "sizing", None)
    row = summarize_performance(
        name=f"{ticker}_{variant}_{params.label()}_{cost_bps:g}bps",
        equity=curve["strategy_equity"],
        returns=curve["strategy_return"],
        trades=trade_count,
        win_rate=win_rate,
        risk_free_rate=risk_free_rate,
        exposure=float((total_exposure > 0).mean()),
        turnover=annualized_turnover(curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=curve["gross_strategy_equity"],
        benchmark_equity=curve["buy_hold_equity"],
    )
    return row | {
        "ticker": ticker,
        "label": label,
        "variant": variant,
        "risk_free_rate": risk_free_rate,
        "short_window": params.short_window,
        "long_window": params.long_window,
        "spread_threshold": getattr(params, "spread_threshold", 0.0),
        "momentum_window": np.nan if getattr(params, "momentum_window", None) is None else params.momentum_window,
        "partial_exposure": getattr(params, "partial_exposure", False),
        "entry_threshold": getattr(params, "entry_threshold", 0.0),
        "exit_threshold": getattr(params, "exit_threshold", 0.0),
        "min_hold_days": getattr(params, "min_hold_days", 0),
        "cooldown_days": getattr(params, "cooldown_days", 0),
        "cost_bps": cost_bps,
        "benchmark_total_return": benchmark_metrics["total_return"],
        "benchmark_cagr": benchmark_metrics["cagr"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "upside_capture": upside,
        "downside_capture": downside,
        "capture_spread": capture_spread(upside, downside),
        "missed_return_while_in_cash": missed_return_while_underweight(curve["buy_hold_return"], ticker_weight),
        "avoided_downside_while_out": avoided_downside_while_underweight(curve["buy_hold_return"], ticker_weight),
        "average_holding_days": float(holds.mean()) if not holds.empty else np.nan,
        "median_holding_days": float(holds.median()) if not holds.empty else np.nan,
        "trade_frequency_per_year": trade_frequency_per_year(trade_count, curve.index),
        "fallback_exposure": fallback_exposure,
        "average_target_exposure": float(ticker_weight.abs().mean()),
        "volatility_adjusted_exposure": float(curve["volatility_weight"].mean())
        if "volatility_weight" in curve.columns
        else np.nan,
        "risk_price_sma_window": getattr(risk, "price_sma_window", np.nan),
        "risk_drawdown_threshold": getattr(risk, "rolling_drawdown_threshold", np.nan),
        "risk_volatility_window": getattr(risk, "realized_volatility_window", np.nan),
        "risk_max_realized_volatility": getattr(risk, "max_realized_volatility", np.nan),
        "risk_sharp_loss_threshold": getattr(risk, "sharp_loss_threshold", np.nan),
        "target_volatility": getattr(sizing, "target_volatility", np.nan),
        "sizing_window": getattr(sizing, "realized_volatility_window", np.nan),
        "fallback_asset": getattr(params, "fallback_asset", "cash"),
        "fallback_weight": getattr(params, "fallback_weight", 0.0),
        "fallback_min_hold_days": getattr(params, "fallback_min_hold_days", 0),
        "fallback_cooldown_days": getattr(params, "fallback_cooldown_days", 0),
        "hybrid_fallback": getattr(params, "hybrid_fallback", False),
    }


def select_best_parameters(table: pd.DataFrame, config: ResearchConfig) -> SmaParameters:
    if table.empty:
        return SmaParameters(20, 100)
    candidates = table[
        (table["cagr"] > 0)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & (table["turnover"] < 8.0)
        & (table["neighbor_sharpe"].fillna(table["sharpe"]) > 0)
    ]
    source = candidates if not candidates.empty else table
    best = source.sort_values("sharpe", ascending=False).iloc[0]
    return SmaParameters(short_window=int(best["short_window"]), long_window=int(best["long_window"]))


def add_parameter_stability(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    enriched = table.copy()
    neighbor_sharpes = []
    stable_counts = []
    for _, row in enriched.iterrows():
        neighbors = enriched[
            (enriched["short_window"].sub(row["short_window"]).abs() <= 25)
            & (enriched["long_window"].sub(row["long_window"]).abs() <= 100)
            & ~(
                (enriched["short_window"] == row["short_window"])
                & (enriched["long_window"] == row["long_window"])
            )
        ]
        neighbor_sharpes.append(float(neighbors["sharpe"].mean()) if not neighbors.empty else np.nan)
        stable_counts.append(int((neighbors["sharpe"] > 0).sum()) if not neighbors.empty else 0)
    enriched["neighbor_sharpe"] = neighbor_sharpes
    enriched["stable_neighbor_count"] = stable_counts
    enriched["hot_pixel_risk"] = enriched["sharpe"] - enriched["neighbor_sharpe"]
    return enriched


def parameter_grid(config: ResearchConfig) -> list[tuple[int, int]]:
    return [
        (short_window, long_window)
        for short_window in config.short_windows
        for long_window in config.long_windows
        if short_window < long_window
    ]


def create_fixture_prices(config: ResearchConfig) -> pd.DataFrame:
    dates = pd.date_range(config.start, config.end or "2026-05-15", freq="B")
    base = np.arange(len(dates))
    prices = {}
    for idx, ticker in enumerate(config.universe):
        drift = 0.00035 + idx * 0.000025
        seasonal = 0.015 * np.sin(base / (18 + idx))
        shock = 0.01 * np.sin(base / (7 + idx))
        returns = drift + seasonal / 252 + shock / 252
        prices[ticker] = 100 * (1.0 + pd.Series(returns, index=dates)).cumprod()
    cash_ticker = config.cash_proxy_ticker
    if cash_ticker and cash_ticker not in prices:
        # Deterministic ~3% annual yield for the cash proxy.
        daily_yield = 0.03 / 252
        prices[cash_ticker] = 100 * (1.0 + pd.Series(daily_yield, index=dates)).cumprod()
    return pd.DataFrame(prices, index=dates)


def _research_tickers(config: ResearchConfig) -> list[str]:
    tickers = list(config.universe)
    if config.cash_proxy_ticker and config.cash_proxy_ticker not in tickers:
        tickers.append(config.cash_proxy_ticker)
    return tickers


def _download_prices(config: ResearchConfig) -> pd.DataFrame:
    return download_adjusted_close(_research_tickers(config), start=config.start, end=config.end or default_end_date())


def _combine_curve(
    price: pd.Series,
    signals,
    engine_curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    curve = engine_curve.copy()
    curve["price"] = price.reindex(curve.index)
    curve["short_sma"] = signals.short_sma.reindex(curve.index)
    curve["long_sma"] = signals.long_sma.reindex(curve.index)
    curve["signal"] = signals.target_position.reindex(curve.index).fillna(0.0)
    curve["position"] = executed_weights.get(ticker, pd.Series(0.0, index=curve.index))
    fallback_columns = [column for column in executed_weights.columns if column != ticker]
    curve["fallback_position"] = (
        executed_weights[fallback_columns].abs().sum(axis=1) if fallback_columns else 0.0
    )
    for column in [
        "trend_position",
        "risk_off",
        "volatility_weight",
        "realized_volatility",
        "price_sma",
        "rolling_drawdown",
        "sharp_loss",
        "fallback_target",
    ]:
        if hasattr(signals, column):
            curve[column] = getattr(signals, column).reindex(curve.index)
    curve["buy_hold_return"] = prices[ticker].pct_change().reindex(curve.index).fillna(0.0)
    curve["buy_hold_equity"] = engine_curve["strategy_equity"].iloc[0] * (1.0 + curve["buy_hold_return"]).cumprod()
    curve["buy_hold_drawdown"] = curve["buy_hold_equity"] / curve["buy_hold_equity"].cummax() - 1.0
    return curve
