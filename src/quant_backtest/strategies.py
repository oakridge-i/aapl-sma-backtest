from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SmaParameters:
    short_window: int
    long_window: int
    spread_threshold: float = 0.0
    momentum_window: int | None = None
    partial_exposure: bool = False

    def label(self) -> str:
        parts = [f"sma_{self.short_window}_{self.long_window}"]
        if self.spread_threshold:
            parts.append(f"thr_{self.spread_threshold:.3f}")
        if self.momentum_window:
            parts.append(f"mom_{self.momentum_window}")
        if self.partial_exposure:
            parts.append("partial")
        return "_".join(parts)


@dataclass(frozen=True)
class SmaSignalFrame:
    target_position: pd.Series
    short_sma: pd.Series
    long_sma: pd.Series
    spread: pd.Series
    momentum: pd.Series | None


@dataclass(frozen=True)
class TrendAllocationParameters:
    short_window: int
    long_window: int
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    min_hold_days: int = 0
    cooldown_days: int = 0

    def label(self) -> str:
        return (
            f"trend_{self.short_window}_{self.long_window}"
            f"_entry_{self.entry_threshold:.3f}"
            f"_exit_{self.exit_threshold:.3f}"
            f"_hold_{self.min_hold_days}"
            f"_cool_{self.cooldown_days}"
        )


@dataclass(frozen=True)
class RiskFilterParameters:
    price_sma_window: int = 200
    use_price_sma_filter: bool = True
    use_market_sma_filter: bool = True
    rolling_drawdown_window: int = 63
    rolling_drawdown_threshold: float = 0.10
    realized_volatility_window: int = 20
    max_realized_volatility: float = 0.45
    sharp_loss_window: int = 20
    sharp_loss_threshold: float = -0.08

    def label(self) -> str:
        return (
            f"risk_sma_{self.price_sma_window}"
            f"_dd_{self.rolling_drawdown_threshold:.2f}"
            f"_vol_{self.max_realized_volatility:.2f}"
            f"_loss_{abs(self.sharp_loss_threshold):.2f}"
        )


@dataclass(frozen=True)
class VolatilitySizingParameters:
    enabled: bool = True
    realized_volatility_window: int = 20
    target_volatility: float = 0.20
    max_weight: float = 1.0

    def label(self) -> str:
        if not self.enabled:
            return "fixed_weight"
        return f"vol_{self.realized_volatility_window}_{self.target_volatility:.2f}"


@dataclass(frozen=True)
class CaptureAwareAllocationParameters:
    trend: TrendAllocationParameters
    risk: RiskFilterParameters = field(default_factory=RiskFilterParameters)
    sizing: VolatilitySizingParameters = field(default_factory=VolatilitySizingParameters)
    fallback_asset: str = "cash"
    fallback_weight: float = 0.0
    fallback_min_hold_days: int = 0
    fallback_cooldown_days: int = 0
    hybrid_fallback: bool = False

    @property
    def short_window(self) -> int:
        return self.trend.short_window

    @property
    def long_window(self) -> int:
        return self.trend.long_window

    @property
    def entry_threshold(self) -> float:
        return self.trend.entry_threshold

    @property
    def exit_threshold(self) -> float:
        return self.trend.exit_threshold

    @property
    def min_hold_days(self) -> int:
        return self.trend.min_hold_days

    @property
    def cooldown_days(self) -> int:
        return self.trend.cooldown_days

    def label(self) -> str:
        fallback = self.fallback_asset.lower()
        if fallback != "cash":
            fallback = f"{fallback}_{self.fallback_weight:.2f}"
        return "_".join(
            [
                "capture",
                self.trend.label(),
                self.risk.label(),
                self.sizing.label(),
                fallback,
                "hybrid" if self.hybrid_fallback else "fallback",
            ]
        )


@dataclass(frozen=True)
class CaptureAwareSignalFrame:
    target_position: pd.Series
    short_sma: pd.Series
    long_sma: pd.Series
    spread: pd.Series
    momentum: pd.Series | None
    trend_position: pd.Series
    risk_off: pd.Series
    volatility_weight: pd.Series
    realized_volatility: pd.Series
    price_sma: pd.Series
    rolling_drawdown: pd.Series
    sharp_loss: pd.Series
    fallback_target: pd.Series


class SmaCrossoverStrategy:
    name = "sma_crossover"

    def __init__(self, params: SmaParameters) -> None:
        _validate_params(params)
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        short_sma = clean.rolling(self.params.short_window, min_periods=self.params.short_window).mean()
        long_sma = clean.rolling(self.params.long_window, min_periods=self.params.long_window).mean()
        spread = short_sma / long_sma - 1.0

        strong_trend = spread > self.params.spread_threshold
        valid = long_sma.notna()
        if self.params.momentum_window:
            momentum = clean.pct_change(self.params.momentum_window)
            strong_trend = strong_trend & (momentum > 0)
            valid = valid & momentum.notna()
        else:
            momentum = None

        if self.params.partial_exposure:
            weak_trend = clean > long_sma
            target_position = pd.Series(0.0, index=clean.index, name="target_position")
            target_position = target_position.where(~(weak_trend & valid), 0.5)
            target_position = target_position.where(~(strong_trend & valid), 1.0)
        else:
            target_position = (strong_trend & valid).astype(float)
            target_position.name = "target_position"

        target_position = target_position.where(valid, 0.0)
        return SmaSignalFrame(
            target_position=target_position,
            short_sma=short_sma,
            long_sma=long_sma,
            spread=spread,
            momentum=momentum,
        )


class TrendAllocationStrategy:
    name = "trend_allocation"

    def __init__(self, params: TrendAllocationParameters) -> None:
        _validate_trend_params(params)
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        short_sma = clean.rolling(self.params.short_window, min_periods=self.params.short_window).mean()
        long_sma = clean.rolling(self.params.long_window, min_periods=self.params.long_window).mean()
        spread = short_sma / long_sma - 1.0
        valid = long_sma.notna()

        position = 0.0
        hold_days = 0
        cooldown_days = 0
        values: list[float] = []

        for date in clean.index:
            if not bool(valid.loc[date]):
                position = 0.0
                hold_days = 0
                cooldown_days = 0
                values.append(0.0)
                continue

            current_spread = float(spread.loc[date])
            if position == 0.0:
                if cooldown_days > 0:
                    cooldown_days -= 1
                elif current_spread > self.params.entry_threshold:
                    position = 1.0
                    hold_days = 1
            else:
                if hold_days < self.params.min_hold_days:
                    hold_days += 1
                elif current_spread < self.params.exit_threshold:
                    position = 0.0
                    hold_days = 0
                    cooldown_days = self.params.cooldown_days
                else:
                    hold_days += 1

            values.append(position)

        target_position = pd.Series(values, index=clean.index, name="target_position")
        return SmaSignalFrame(
            target_position=target_position,
            short_sma=short_sma,
            long_sma=long_sma,
            spread=spread,
            momentum=None,
        )


class CaptureAwareTrendStrategy:
    name = "capture_aware_trend"

    def __init__(self, params: CaptureAwareAllocationParameters) -> None:
        _validate_capture_params(params)
        self.params = params

    def generate(
        self,
        price: pd.Series,
        market_risk_off: pd.Series | None = None,
        fallback_regime: pd.Series | None = None,
    ) -> CaptureAwareSignalFrame:
        trend_signals = TrendAllocationStrategy(self.params.trend).generate(price)
        clean = price.dropna().astype(float)
        risk = self.params.risk
        sizing = self.params.sizing

        price_sma = clean.rolling(risk.price_sma_window, min_periods=risk.price_sma_window).mean()
        rolling_high = clean.rolling(risk.rolling_drawdown_window, min_periods=1).max()
        rolling_drawdown = clean / rolling_high - 1.0
        sharp_loss = clean.pct_change(risk.sharp_loss_window)
        realized_volatility = clean.pct_change().rolling(
            risk.realized_volatility_window,
            min_periods=risk.realized_volatility_window,
        ).std(ddof=0) * (252**0.5)

        risk_off = pd.Series(False, index=clean.index, name="risk_off")
        if risk.use_price_sma_filter:
            risk_off = risk_off | (clean < price_sma)
        risk_off = risk_off | (rolling_drawdown <= -abs(risk.rolling_drawdown_threshold))
        risk_off = risk_off | (realized_volatility > risk.max_realized_volatility)
        risk_off = risk_off | (sharp_loss <= risk.sharp_loss_threshold)
        if risk.use_market_sma_filter and market_risk_off is not None:
            risk_off = risk_off | market_risk_off.reindex(clean.index).fillna(False).astype(bool)

        if sizing.enabled:
            vol_weight = (sizing.target_volatility / realized_volatility).clip(lower=0.0, upper=sizing.max_weight)
            vol_weight = vol_weight.where(realized_volatility.notna(), sizing.max_weight)
        else:
            vol_weight = pd.Series(sizing.max_weight, index=clean.index, name="volatility_weight")
        vol_weight = vol_weight.fillna(0.0).rename("volatility_weight")

        trend_position = trend_signals.target_position.reindex(clean.index).fillna(0.0)
        target_position = (trend_position * vol_weight).where(~risk_off, 0.0).rename("target_position")

        if fallback_regime is None or self.params.fallback_asset.lower() == "cash" or self.params.fallback_weight <= 0:
            fallback_target = pd.Series(0.0, index=clean.index, name="fallback_target")
        else:
            regime = fallback_regime.reindex(clean.index).fillna(False).astype(bool)
            if self.params.hybrid_fallback:
                desired = ((1.0 - target_position).clip(lower=0.0) * self.params.fallback_weight).where(regime, 0.0)
            else:
                desired = pd.Series(self.params.fallback_weight, index=clean.index).where(
                    (target_position <= 0.0) & regime,
                    0.0,
                )
            fallback_target = _apply_binary_hold_rules(
                desired.clip(lower=0.0, upper=1.0),
                self.params.fallback_min_hold_days,
                self.params.fallback_cooldown_days,
            ).rename("fallback_target")

        return CaptureAwareSignalFrame(
            target_position=target_position,
            short_sma=trend_signals.short_sma,
            long_sma=trend_signals.long_sma,
            spread=trend_signals.spread,
            momentum=None,
            trend_position=trend_position.rename("trend_position"),
            risk_off=risk_off,
            volatility_weight=vol_weight,
            realized_volatility=realized_volatility.rename("realized_volatility"),
            price_sma=price_sma.rename("price_sma"),
            rolling_drawdown=rolling_drawdown.rename("rolling_drawdown"),
            sharp_loss=sharp_loss.rename("sharp_loss"),
            fallback_target=fallback_target,
        )


def build_single_asset_weights(ticker: str, target_position: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({ticker: target_position.astype(float)})


def build_fallback_weights(
    target_ticker: str,
    fallback_ticker: str,
    target_position: pd.Series,
) -> pd.DataFrame:
    position = target_position.astype(float).clip(lower=0.0, upper=1.0)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: position})
    return pd.DataFrame(
        {
            target_ticker: position,
            fallback_ticker: 1.0 - position,
        }
    )


def build_regime_fallback_weights(
    target_ticker: str,
    fallback_ticker: str,
    target_position: pd.Series,
    fallback_regime: pd.Series,
) -> pd.DataFrame:
    target = target_position.astype(float).clip(lower=0.0, upper=1.0)
    regime = fallback_regime.reindex(target.index).fillna(False).astype(bool)
    fallback = (1.0 - target).where(regime, 0.0)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: target})
    return pd.DataFrame({target_ticker: target, fallback_ticker: fallback})


def build_hybrid_regime_weights(
    target_ticker: str,
    fallback_ticker: str,
    signals: SmaSignalFrame,
    params: TrendAllocationParameters,
    fallback_regime: pd.Series,
) -> pd.DataFrame:
    target = signals.target_position.astype(float).clip(lower=0.0, upper=1.0)
    regime = fallback_regime.reindex(target.index).fillna(False).astype(bool)
    weak_trend = (
        (target == 0.0)
        & signals.long_sma.reindex(target.index).notna()
        & (signals.spread.reindex(target.index) > params.exit_threshold)
        & (signals.spread.reindex(target.index) <= params.entry_threshold)
        & regime
    )
    target_weight = target.where(~weak_trend, 0.5)
    fallback_weight = pd.Series(0.0, index=target.index, name=fallback_ticker)
    fallback_weight = fallback_weight.where(~((target == 0.0) & regime), 1.0)
    fallback_weight = fallback_weight.where(~weak_trend, 0.5)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: target_weight})
    return pd.DataFrame({target_ticker: target_weight, fallback_ticker: fallback_weight})


def build_sma_regime(price: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    clean = price.dropna().astype(float)
    short_sma = clean.rolling(short_window, min_periods=short_window).mean()
    long_sma = clean.rolling(long_window, min_periods=long_window).mean()
    return (short_sma > long_sma).rename("regime")


def build_capture_aware_weights(
    target_ticker: str,
    signals: CaptureAwareSignalFrame,
    fallback_ticker: str | None = None,
) -> pd.DataFrame:
    target = signals.target_position.astype(float).clip(lower=0.0, upper=1.0)
    if fallback_ticker is None:
        return pd.DataFrame({target_ticker: target})

    fallback = signals.fallback_target.reindex(target.index).fillna(0.0).astype(float)
    fallback = fallback.clip(lower=0.0, upper=(1.0 - target).clip(lower=0.0))
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: target})
    return pd.DataFrame({target_ticker: target, fallback_ticker: fallback})


def classify_market_regime(
    target_price: pd.Series,
    market_price: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    target = target_price.dropna().astype(float)
    market = market_price.dropna().astype(float).reindex(target.index).ffill()
    target_sma_short = target.rolling(short_window, min_periods=short_window).mean()
    target_sma_long = target.rolling(long_window, min_periods=long_window).mean()
    market_sma_long = market.rolling(long_window, min_periods=long_window).mean()
    target_drawdown = target / target.rolling(63, min_periods=1).max() - 1.0
    target_return_63d = target.pct_change(63)
    spread = target_sma_short / target_sma_long - 1.0

    labels = pd.Series("sideways", index=target.index, name="regime")
    market_positive = market > market_sma_long
    target_positive = target > target_sma_long
    labels = labels.mask(~market_positive.fillna(False), "bear")
    labels = labels.mask((market_positive & target_positive & (target_return_63d > 0)).fillna(False), "bull")
    labels = labels.mask((market_positive & (target_drawdown <= -0.10)).fillna(False), "correction")
    labels = labels.mask((market_positive & ~target_positive & (target > target_sma_short)).fillna(False), "recovery")
    labels = labels.mask((market_positive & spread.abs().lt(0.02)).fillna(False), "sideways")
    return labels


def _apply_binary_hold_rules(target: pd.Series, min_hold_days: int, cooldown_days: int) -> pd.Series:
    state = 0.0
    hold_days = 0
    cooldown = 0
    values: list[float] = []
    for desired in target.fillna(0.0).astype(float):
        wants_on = desired > 0.0
        if state == 0.0:
            if cooldown > 0:
                cooldown -= 1
            elif wants_on:
                state = desired
                hold_days = 1
        else:
            if hold_days < min_hold_days:
                hold_days += 1
                if wants_on:
                    state = desired
            elif wants_on:
                state = desired
                hold_days += 1
            else:
                state = 0.0
                hold_days = 0
                cooldown = cooldown_days
        values.append(state)
    return pd.Series(values, index=target.index)


def _validate_params(params: SmaParameters) -> None:
    if params.short_window <= 0 or params.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if params.short_window >= params.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if params.spread_threshold < 0:
        raise ValueError("spread_threshold must be non-negative.")
    if params.momentum_window is not None and params.momentum_window <= 0:
        raise ValueError("momentum_window must be positive when provided.")


def _validate_trend_params(params: TrendAllocationParameters) -> None:
    if params.short_window <= 0 or params.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if params.short_window >= params.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if params.entry_threshold < params.exit_threshold:
        raise ValueError("entry_threshold must be greater than or equal to exit_threshold.")
    if params.min_hold_days < 0 or params.cooldown_days < 0:
        raise ValueError("min_hold_days and cooldown_days must be non-negative.")


def _validate_capture_params(params: CaptureAwareAllocationParameters) -> None:
    _validate_trend_params(params.trend)
    if params.risk.price_sma_window <= 0:
        raise ValueError("price_sma_window must be positive.")
    if params.risk.rolling_drawdown_window <= 0:
        raise ValueError("rolling_drawdown_window must be positive.")
    if params.risk.realized_volatility_window <= 0:
        raise ValueError("realized_volatility_window must be positive.")
    if params.risk.sharp_loss_window <= 0:
        raise ValueError("sharp_loss_window must be positive.")
    if params.sizing.realized_volatility_window <= 0:
        raise ValueError("realized_volatility_window must be positive.")
    if params.sizing.target_volatility <= 0:
        raise ValueError("target_volatility must be positive.")
    if params.sizing.max_weight <= 0 or params.sizing.max_weight > 1:
        raise ValueError("max_weight must be in (0, 1].")
    if params.fallback_weight < 0 or params.fallback_weight > 1:
        raise ValueError("fallback_weight must be between 0 and 1.")
    if params.fallback_min_hold_days < 0 or params.fallback_cooldown_days < 0:
        raise ValueError("fallback hold and cooldown values must be non-negative.")
