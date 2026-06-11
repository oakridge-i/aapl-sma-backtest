"""Exit, sizing, and regime overlays for any base strategy.

An overlay post-processes the base strategy's target position before the
engine applies execution lag and costs. Overlays compose in a fixed order:

1. regime scaling - cut exposure when the market trades below its long SMA,
   modestly boost it (capped at 1) when above;
2. volatility targeting - scale exposure down when realized volatility runs
   above target;
3. ATR trailing stop - the final risk control: force the position flat after
   price falls a multiple of ATR from the post-entry peak, and re-arm only on
   a new high or after the base signal itself resets.

The order is deliberate: the trailing stop must see the final sized exposure,
otherwise a later scaling step could re-open a stopped-out position.

All overlay state uses information available at the close; the engine's
one-day execution lag still applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .registry import StrategyFamily, family_for_params, register_family
from .strategies import SmaSignalFrame


EXPOSURE_EPSILON = 1e-9


@dataclass(frozen=True)
class TrailingStopParameters:
    atr_window: int = 20
    multiple: float = 4.0

    def label(self) -> str:
        return f"ts_{self.atr_window}_{self.multiple:g}"


@dataclass(frozen=True)
class RegimeScalingParameters:
    sma_window: int = 200
    bear_cut: float = 0.5
    bull_boost: float = 1.25

    def label(self) -> str:
        return f"rs_{self.sma_window}_{self.bear_cut:g}_{self.bull_boost:g}"


@dataclass(frozen=True)
class VolTargetParameters:
    window: int = 20
    target: float = 0.20

    def label(self) -> str:
        return f"vt_{self.window}_{self.target:g}"


@dataclass(frozen=True)
class OverlayParameters:
    base: Any
    trailing_stop: TrailingStopParameters | None = None
    regime_scaling: RegimeScalingParameters | None = None
    vol_target: VolTargetParameters | None = None

    def is_identity(self) -> bool:
        return self.trailing_stop is None and self.regime_scaling is None and self.vol_target is None

    def label(self) -> str:
        parts = [self.base.label()]
        if self.regime_scaling is not None:
            parts.append(self.regime_scaling.label())
        if self.vol_target is not None:
            parts.append(self.vol_target.label())
        if self.trailing_stop is not None:
            parts.append(self.trailing_stop.label())
        return "|".join(parts)


class OverlayStrategy:
    name = "overlay"

    def __init__(self, params: OverlayParameters) -> None:
        _validate_overlay_params(params)
        self.params = params

    def generate(self, price: pd.Series, market_price: pd.Series | None = None) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        base_family = family_for_params(self.params.base)
        base_strategy = base_family.strategy_type(self.params.base)
        if base_family.needs_market_price:
            base_signals = base_strategy.generate(clean, market_price=market_price)
        else:
            base_signals = base_strategy.generate(clean)

        target = base_signals.target_position.reindex(clean.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if self.params.regime_scaling is not None:
            target = _apply_regime_scaling(target, market_price, self.params.regime_scaling)
        if self.params.vol_target is not None:
            target = _apply_vol_target(target, clean, self.params.vol_target)
        if self.params.trailing_stop is not None:
            target = _apply_trailing_stop(target, clean, self.params.trailing_stop)

        return SmaSignalFrame(
            target_position=target.rename("target_position"),
            short_sma=base_signals.short_sma.reindex(clean.index),
            long_sma=base_signals.long_sma.reindex(clean.index),
            spread=base_signals.spread.reindex(clean.index),
            momentum=base_signals.momentum,
        )


def _apply_regime_scaling(
    target: pd.Series,
    market_price: pd.Series | None,
    params: RegimeScalingParameters,
) -> pd.Series:
    if market_price is None:
        return target
    market = market_price.dropna().astype(float).reindex(target.index).ffill()
    market_sma = market.rolling(params.sma_window, min_periods=params.sma_window).mean()
    bull = market > market_sma
    factor = pd.Series(params.bear_cut, index=target.index).where(~bull, params.bull_boost)
    # Before the market SMA exists there is no regime call: leave exposure as is.
    factor = factor.where(market_sma.notna(), 1.0)
    return (target * factor).clip(lower=0.0, upper=1.0)


def _apply_vol_target(target: pd.Series, price: pd.Series, params: VolTargetParameters) -> pd.Series:
    realized = price.pct_change().rolling(params.window, min_periods=params.window).std(ddof=0) * (252.0**0.5)
    weight = (params.target / realized).clip(upper=1.0)
    weight = weight.where(realized.notna() & (realized > 0), 1.0)
    return (target * weight).clip(lower=0.0, upper=1.0)


def _apply_trailing_stop(target: pd.Series, price: pd.Series, params: TrailingStopParameters) -> pd.Series:
    atr = price.diff().abs().rolling(params.atr_window, min_periods=params.atr_window).mean()

    values: list[float] = []
    peak = np.nan
    stopped = False
    was_exposed = False
    for date in target.index:
        desired = float(target.loc[date])
        level = float(price.loc[date])
        band = atr.loc[date]
        exposed = desired > EXPOSURE_EPSILON

        if not exposed:
            # The base signal reset the episode; the stop re-arms.
            peak = np.nan
            stopped = False
            was_exposed = False
            values.append(0.0)
            continue

        if not was_exposed:
            peak = level
            stopped = False
        else:
            peak = level if np.isnan(peak) else max(peak, level)

        if stopped:
            if level >= peak:
                # New high after the stop-out: re-enter at the base exposure.
                stopped = False
                values.append(desired)
            else:
                values.append(0.0)
        elif not np.isnan(band) and level < peak - params.multiple * float(band):
            stopped = True
            values.append(0.0)
        else:
            values.append(desired)
        was_exposed = True

    return pd.Series(values, index=target.index)


def _validate_overlay_params(params: OverlayParameters) -> None:
    if params.base is None:
        raise ValueError("Overlay needs a base strategy params object.")
    if isinstance(params.base, OverlayParameters):
        raise ValueError("Overlays must not be nested.")
    if params.trailing_stop is not None:
        if params.trailing_stop.atr_window <= 0:
            raise ValueError("atr_window must be positive.")
        if params.trailing_stop.multiple <= 0:
            raise ValueError("trailing stop multiple must be positive.")
    if params.regime_scaling is not None:
        if params.regime_scaling.sma_window <= 0:
            raise ValueError("regime sma_window must be positive.")
        if not 0 <= params.regime_scaling.bear_cut <= 1:
            raise ValueError("bear_cut must be between 0 and 1.")
        if params.regime_scaling.bull_boost < 1:
            raise ValueError("bull_boost must be at least 1.")
    if params.vol_target is not None:
        if params.vol_target.window <= 0:
            raise ValueError("vol target window must be positive.")
        if params.vol_target.target <= 0:
            raise ValueError("vol target must be positive.")


register_family(
    StrategyFamily(
        name="overlay",
        params_type=OverlayParameters,
        strategy_type=OverlayStrategy,
        needs_market_price=True,
    )
)
