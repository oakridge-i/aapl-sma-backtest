"""Strategy family registry.

Maps a family name to its parameter dataclass and strategy class, so that
evaluation code dispatches on registered families instead of hard-coded
isinstance chains, and new signal families (momentum, breakout, ...) can be
added without touching the sweep machinery.

A family's strategy class must expose ``generate(price, **context)`` returning
a signal frame with a ``target_position`` series. Families whose ``generate``
needs market context (market regime, fallback regime) set
``needs_market_context=True`` and receive ``market_risk_off`` and
``fallback_regime`` keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .strategies import (
    CaptureAwareAllocationParameters,
    CaptureAwareTrendStrategy,
    SmaCrossoverStrategy,
    SmaParameters,
    TrendAllocationParameters,
    TrendAllocationStrategy,
)


@dataclass(frozen=True)
class StrategyFamily:
    name: str
    params_type: type
    strategy_type: type
    needs_market_context: bool = False
    params_from_config: Callable[[dict[str, Any]], Any] | None = None


_FAMILIES: dict[str, StrategyFamily] = {}
_FAMILIES_BY_PARAMS: dict[type, StrategyFamily] = {}


def register_family(family: StrategyFamily) -> StrategyFamily:
    if family.name in _FAMILIES:
        raise ValueError(f"Strategy family already registered: {family.name}")
    if family.params_type in _FAMILIES_BY_PARAMS:
        raise ValueError(f"Params type already registered: {family.params_type.__name__}")
    _FAMILIES[family.name] = family
    _FAMILIES_BY_PARAMS[family.params_type] = family
    return family


def family_by_name(name: str) -> StrategyFamily:
    try:
        return _FAMILIES[name]
    except KeyError:
        raise KeyError(f"Unknown strategy family: {name!r}. Registered: {sorted(_FAMILIES)}") from None


def family_for_params(params: Any) -> StrategyFamily:
    family = _FAMILIES_BY_PARAMS.get(type(params))
    if family is None:
        raise KeyError(
            f"No strategy family registered for params type {type(params).__name__}. "
            f"Registered: {sorted(_FAMILIES)}"
        )
    return family


def build_strategy(params: Any) -> Any:
    return family_for_params(params).strategy_type(params)


def params_from_config(name: str, raw: dict[str, Any]) -> Any:
    family = family_by_name(name)
    if family.params_from_config is None:
        return family.params_type(**raw)
    return family.params_from_config(raw)


def iter_families() -> Iterator[StrategyFamily]:
    return iter(_FAMILIES.values())


register_family(
    StrategyFamily(
        name="sma_crossover",
        params_type=SmaParameters,
        strategy_type=SmaCrossoverStrategy,
    )
)
register_family(
    StrategyFamily(
        name="trend_allocation",
        params_type=TrendAllocationParameters,
        strategy_type=TrendAllocationStrategy,
    )
)
register_family(
    StrategyFamily(
        name="capture_aware_trend",
        params_type=CaptureAwareAllocationParameters,
        strategy_type=CaptureAwareTrendStrategy,
        needs_market_context=True,
    )
)
