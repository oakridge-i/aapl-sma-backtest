"""Long-only signal families beyond the SMA crossover, plus an ensemble.

Every family produces an :class:`~quant_backtest.strategies.SmaSignalFrame`
(the shared signal container) with a ``target_position`` in ``[0, 1]``.
Families register themselves with the strategy registry at import time, so
the evaluation and sweep machinery picks them up without modification.

All signals use information available at the close; execution lag is applied
by the engine, exactly as for the SMA families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .registry import StrategyFamily, family_for_params, register_family
from .strategies import SmaSignalFrame, _apply_binary_hold_rules


def _nan_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index)


# --- time-series momentum -----------------------------------------------------


@dataclass(frozen=True)
class TimeSeriesMomentumParameters:
    lookback_days: int = 126
    threshold: float = 0.0
    min_hold_days: int = 0
    cooldown_days: int = 0

    def label(self) -> str:
        parts = [f"tsmom_{self.lookback_days}"]
        if self.threshold:
            parts.append(f"thr_{self.threshold:.3f}")
        if self.min_hold_days:
            parts.append(f"hold_{self.min_hold_days}")
        if self.cooldown_days:
            parts.append(f"cool_{self.cooldown_days}")
        return "_".join(parts)


class TimeSeriesMomentumStrategy:
    name = "ts_momentum"

    def __init__(self, params: TimeSeriesMomentumParameters) -> None:
        if params.lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        if params.min_hold_days < 0 or params.cooldown_days < 0:
            raise ValueError("min_hold_days and cooldown_days must be non-negative.")
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        momentum = clean.pct_change(self.params.lookback_days)
        desired = (momentum > self.params.threshold).astype(float).where(momentum.notna(), 0.0)
        target = _apply_binary_hold_rules(desired, self.params.min_hold_days, self.params.cooldown_days)
        return SmaSignalFrame(
            target_position=target.rename("target_position"),
            short_sma=_nan_series(clean.index),
            long_sma=_nan_series(clean.index),
            spread=momentum.rename("spread"),
            momentum=momentum,
        )


# --- Donchian channel breakout -------------------------------------------------


@dataclass(frozen=True)
class DonchianParameters:
    entry_window: int = 55
    exit_window: int = 20

    def label(self) -> str:
        return f"donchian_{self.entry_window}_{self.exit_window}"


class DonchianBreakoutStrategy:
    name = "donchian_breakout"

    def __init__(self, params: DonchianParameters) -> None:
        if params.entry_window <= 0 or params.exit_window <= 0:
            raise ValueError("Donchian windows must be positive.")
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        # Yesterday's channel: entering on a new high must not see today's bar.
        upper = clean.rolling(self.params.entry_window, min_periods=self.params.entry_window).max().shift(1)
        lower = clean.rolling(self.params.exit_window, min_periods=self.params.exit_window).min().shift(1)

        position = 0.0
        values: list[float] = []
        for date in clean.index:
            top = upper.loc[date]
            bottom = lower.loc[date]
            level = clean.loc[date]
            if np.isnan(top):
                position = 0.0
            elif position == 0.0 and level > top:
                position = 1.0
            elif position == 1.0 and not np.isnan(bottom) and level < bottom:
                position = 0.0
            values.append(position)

        target = pd.Series(values, index=clean.index, name="target_position")
        spread = (clean / upper - 1.0).rename("spread")
        return SmaSignalFrame(
            target_position=target,
            short_sma=lower.rename("short_sma"),
            long_sma=upper.rename("long_sma"),
            spread=spread,
            momentum=None,
        )


# --- ATR-scaled trend strength --------------------------------------------------


@dataclass(frozen=True)
class AtrTrendParameters:
    sma_window: int = 200
    atr_window: int = 20
    scale: float = 3.0

    def label(self) -> str:
        return f"atr_trend_{self.sma_window}_{self.atr_window}_{self.scale:g}"


class AtrTrendStrategy:
    """Continuous exposure proportional to trend strength in ATR units.

    The ATR proxy is the rolling mean of absolute close-to-close moves (the
    dataset has no intraday highs/lows). Exposure ramps linearly from 0 at
    the SMA to 1 at ``scale`` ATRs above it.
    """

    name = "atr_trend"

    def __init__(self, params: AtrTrendParameters) -> None:
        if params.sma_window <= 0 or params.atr_window <= 0:
            raise ValueError("ATR trend windows must be positive.")
        if params.scale <= 0:
            raise ValueError("scale must be positive.")
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        sma = clean.rolling(self.params.sma_window, min_periods=self.params.sma_window).mean()
        atr = clean.diff().abs().rolling(self.params.atr_window, min_periods=self.params.atr_window).mean()
        strength = (clean - sma) / (self.params.scale * atr)
        target = strength.clip(lower=0.0, upper=1.0).where(sma.notna() & atr.notna(), 0.0)
        return SmaSignalFrame(
            target_position=target.rename("target_position"),
            short_sma=_nan_series(clean.index),
            long_sma=sma.rename("long_sma"),
            spread=strength.rename("spread"),
            momentum=None,
        )


# --- dual momentum ---------------------------------------------------------------


@dataclass(frozen=True)
class DualMomentumParameters:
    lookback_days: int = 252
    market_ticker: str = "SPY"

    def label(self) -> str:
        return f"dual_mom_{self.lookback_days}_{self.market_ticker.lower()}"


class DualMomentumStrategy:
    """Long only when the asset beats the market and its own zero hurdle.

    Without a market series the relative leg degrades to pass-through and the
    strategy becomes absolute momentum.
    """

    name = "dual_momentum"

    def __init__(self, params: DualMomentumParameters) -> None:
        if params.lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        self.params = params

    def generate(self, price: pd.Series, market_price: pd.Series | None = None) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        asset_momentum = clean.pct_change(self.params.lookback_days)
        absolute_ok = asset_momentum > 0
        if market_price is not None:
            market = market_price.dropna().astype(float).reindex(clean.index).ffill()
            market_momentum = market.pct_change(self.params.lookback_days)
            relative_ok = asset_momentum > market_momentum
            relative_ok = relative_ok & market_momentum.notna()
        else:
            relative_ok = pd.Series(True, index=clean.index)
        target = (absolute_ok & relative_ok & asset_momentum.notna()).astype(float)
        return SmaSignalFrame(
            target_position=target.rename("target_position"),
            short_sma=_nan_series(clean.index),
            long_sma=_nan_series(clean.index),
            spread=asset_momentum.rename("spread"),
            momentum=asset_momentum,
        )


# --- 52-week-high proximity --------------------------------------------------------


@dataclass(frozen=True)
class High52WeekParameters:
    window: int = 252
    entry_threshold: float = 0.95
    exit_threshold: float = 0.85

    def label(self) -> str:
        return f"high52_{self.window}_{self.entry_threshold:.2f}_{self.exit_threshold:.2f}"


class High52WeekStrategy:
    """Hysteresis on proximity to the rolling high.

    Enter when price recovers to ``entry_threshold`` of the rolling maximum,
    exit only when it sinks below ``exit_threshold`` — the gap avoids churn
    around a single level.
    """

    name = "high_52w"

    def __init__(self, params: High52WeekParameters) -> None:
        if params.window <= 0:
            raise ValueError("window must be positive.")
        if not 0 < params.exit_threshold <= params.entry_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 < exit <= entry <= 1.")
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        rolling_high = clean.rolling(self.params.window, min_periods=self.params.window).max()
        proximity = clean / rolling_high

        position = 0.0
        values: list[float] = []
        for value in proximity:
            if np.isnan(value):
                position = 0.0
            elif position == 0.0 and value >= self.params.entry_threshold:
                position = 1.0
            elif position == 1.0 and value < self.params.exit_threshold:
                position = 0.0
            values.append(position)

        target = pd.Series(values, index=clean.index, name="target_position")
        return SmaSignalFrame(
            target_position=target,
            short_sma=_nan_series(clean.index),
            long_sma=rolling_high.rename("long_sma"),
            spread=(proximity - 1.0).rename("spread"),
            momentum=None,
        )


# --- ensemble -----------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleParameters:
    members: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("Ensemble needs at least one member.")

    def label(self) -> str:
        return "ensemble[" + "+".join(member.label() for member in self.members) + "]"


class EnsembleVoteStrategy:
    """Equal-vote ensemble: exposure is the mean of member target positions.

    Selection happens at the composition level (which families participate),
    which leaves far fewer degrees of freedom than picking the best pixel of
    a parameter grid.
    """

    name = "ensemble_vote"

    def __init__(self, params: EnsembleParameters) -> None:
        self.params = params

    def generate(self, price: pd.Series, market_price: pd.Series | None = None) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        positions = []
        for member in self.params.members:
            family = family_for_params(member)
            strategy = family.strategy_type(member)
            if family.needs_market_price:
                signals = strategy.generate(clean, market_price=market_price)
            else:
                signals = strategy.generate(clean)
            positions.append(signals.target_position.reindex(clean.index).fillna(0.0))
        votes = pd.concat(positions, axis=1)
        target = votes.mean(axis=1).clip(lower=0.0, upper=1.0)
        return SmaSignalFrame(
            target_position=target.rename("target_position"),
            short_sma=_nan_series(clean.index),
            long_sma=_nan_series(clean.index),
            spread=target.rename("spread"),
            momentum=None,
        )


register_family(
    StrategyFamily(
        name="ts_momentum",
        params_type=TimeSeriesMomentumParameters,
        strategy_type=TimeSeriesMomentumStrategy,
    )
)
register_family(
    StrategyFamily(
        name="donchian_breakout",
        params_type=DonchianParameters,
        strategy_type=DonchianBreakoutStrategy,
    )
)
register_family(
    StrategyFamily(
        name="atr_trend",
        params_type=AtrTrendParameters,
        strategy_type=AtrTrendStrategy,
    )
)
register_family(
    StrategyFamily(
        name="dual_momentum",
        params_type=DualMomentumParameters,
        strategy_type=DualMomentumStrategy,
        needs_market_price=True,
    )
)
register_family(
    StrategyFamily(
        name="high_52w",
        params_type=High52WeekParameters,
        strategy_type=High52WeekStrategy,
    )
)
register_family(
    StrategyFamily(
        name="ensemble_vote",
        params_type=EnsembleParameters,
        strategy_type=EnsembleVoteStrategy,
        needs_market_price=True,
    )
)
