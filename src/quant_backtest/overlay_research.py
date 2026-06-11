"""Overlay search around an already-selected base model (the M3 step).

The overlay grid is tiny by design (a handful of trailing-stop multiples,
regime scaling on/off, volatility targeting on/off) and always contains the
identity combination, so the base model competes on equal terms. Ranking is
train-only with a 20 bps cost stress, exactly like every other leaderboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .overlays import (
    OverlayParameters,
    RegimeScalingParameters,
    TrailingStopParameters,
    VolTargetParameters,
)
from .parallel import EvaluationContext, EvaluationJob, evaluate_grid
from .research_config import ResearchConfig
from .selection import add_selection_score


def overlay_parameter_grid(config: ResearchConfig, base_params: Any) -> list[OverlayParameters]:
    raw = config.overlay_grids or {}
    trailing_raw = raw.get("trailing_stop", {})
    regime_raw = raw.get("regime_scaling", {})
    vol_raw = raw.get("vol_target", {})

    trailing_options: list[TrailingStopParameters | None] = [None] + [
        TrailingStopParameters(atr_window=int(window), multiple=float(multiple))
        for window in trailing_raw.get("atr_windows", [20])
        for multiple in trailing_raw.get("multiples", [3.0, 4.0, 5.0])
    ]
    regime_options: list[RegimeScalingParameters | None] = [None]
    if regime_raw.get("enabled", True):
        regime_options.append(
            RegimeScalingParameters(
                sma_window=int(regime_raw.get("sma_window", 200)),
                bear_cut=float(regime_raw.get("bear_cut", 0.5)),
                bull_boost=float(regime_raw.get("bull_boost", 1.25)),
            )
        )
    vol_options: list[VolTargetParameters | None] = [None] + [
        VolTargetParameters(window=int(vol_raw.get("window", 20)), target=float(target))
        for target in vol_raw.get("targets", [0.20])
    ]

    if isinstance(base_params, OverlayParameters):
        base_params = base_params.base
    return [
        OverlayParameters(
            base=base_params,
            trailing_stop=trailing_stop,
            regime_scaling=regime_scaling,
            vol_target=vol_target,
        )
        for trailing_stop in trailing_options
        for regime_scaling in regime_options
        for vol_target in vol_options
    ]


def run_overlay_leaderboard(
    prices: pd.DataFrame,
    config: ResearchConfig,
    candidates: list[OverlayParameters],
) -> pd.DataFrame:
    """Rank overlay combinations on the train period with a 20 bps stress."""
    if not candidates:
        return pd.DataFrame()
    jobs: list[EvaluationJob] = []
    for params in candidates:
        jobs.append(EvaluationJob(params, "overlay", 10.0, "overlay_train"))
        jobs.append(EvaluationJob(params, "overlay", 20.0, "overlay_train_20bps"))
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
        candidate = candidates[candidate_idx]
        rows.append(
            base_row
            | {
                "candidate_index": candidate_idx,
                "is_identity": candidate.is_identity(),
                "trailing_stop": "" if candidate.trailing_stop is None else candidate.trailing_stop.label(),
                "regime_scaling": "" if candidate.regime_scaling is None else candidate.regime_scaling.label(),
                "vol_target": "" if candidate.vol_target is None else candidate.vol_target.label(),
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


def select_overlay_model(
    leaderboard: pd.DataFrame,
    candidates: list[OverlayParameters],
    base_model: dict[str, Any],
) -> dict[str, Any]:
    """Pick the best overlay combination; the identity row keeps the base model.

    To displace the base model an overlay must not just win the score: it must
    pass the hard filters while improving on the identity row's score. If the
    identity combination wins (or nothing passes), the base model is returned
    unchanged.
    """
    if leaderboard.empty:
        return base_model
    passing = leaderboard[leaderboard["passes_selection"]]
    if passing.empty:
        return base_model
    best = passing.sort_values("selection_score", ascending=False).iloc[0]
    candidate = candidates[int(best["candidate_index"])]
    if candidate.is_identity():
        return base_model
    return {
        "params": candidate,
        "variant": "overlay",
        "selection_status": "selected_v6_overlay",
    }
