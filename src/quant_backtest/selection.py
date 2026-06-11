"""Candidate grids, scoring, masks, and model selection rules.

Everything in this module operates on tables of train-period metrics; the
selection rules never see test-period data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import capture_spread
from .research_config import ResearchConfig
from .strategies import (
    CaptureAwareAllocationParameters,
    RiskFilterParameters,
    SmaParameters,
    TrendAllocationParameters,
    VolatilitySizingParameters,
)


def parameter_grid(config: ResearchConfig) -> list[tuple[int, int]]:
    return [
        (short_window, long_window)
        for short_window in config.short_windows
        for long_window in config.long_windows
        if short_window < long_window
    ]


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


def _row_value(row: pd.Series, key: str, default):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value
