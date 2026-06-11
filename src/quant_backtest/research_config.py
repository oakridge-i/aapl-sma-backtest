"""Research configuration: the dataclass and its YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_VARIANTS = (
    "long_cash",
    "fallback_spy",
    "fallback_qqq",
    "partial_exposure",
    "spread_threshold",
    "momentum_3m",
    "momentum_6m",
)

AUTO_WORKERS = -1


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
    # 0 = serial, AUTO_WORKERS (-1) = one worker per CPU, N > 0 = exactly N.
    parallel_workers: int = 0
    enable_nested_walk_forward: bool = False
    enable_pbo: bool = False
    pbo_max_candidates: int = 200
    pbo_blocks: int = 12
    enable_signal_families: bool = False
    signal_family_grids: dict | None = None
    ensemble_min_members: int = 3
    ensemble_max_members: int = 6
    ensemble_include_trend_baseline: bool = True
    ensemble_turnover_limit: float = 6.0


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
    compute = raw.get("compute", {})
    nested = raw.get("nested_walk_forward", {})
    pbo = raw.get("pbo", {})
    signal_families = raw.get("signal_families", {})
    ensemble = raw.get("ensemble", {})
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
        parallel_workers=_parse_workers(compute.get("workers", 0)),
        enable_nested_walk_forward=bool(nested.get("enabled", False)),
        enable_pbo=bool(pbo.get("enabled", False)),
        pbo_max_candidates=int(pbo.get("max_candidates", 200)),
        pbo_blocks=int(pbo.get("blocks", 12)),
        enable_signal_families=bool(signal_families.get("enabled", False)),
        signal_family_grids={
            key: value for key, value in signal_families.items() if key != "enabled" and isinstance(value, dict)
        }
        or None,
        ensemble_min_members=int(ensemble.get("min_members", 3)),
        ensemble_max_members=int(ensemble.get("max_members", 6)),
        ensemble_include_trend_baseline=bool(ensemble.get("include_trend_baseline", True)),
        ensemble_turnover_limit=float(ensemble.get("turnover_limit", 6.0)),
    )


def _parse_workers(value: object) -> int:
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return AUTO_WORKERS
        return int(value)
    if value is None:
        return 0
    return int(value)
