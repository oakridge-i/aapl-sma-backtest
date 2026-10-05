"""Fail-closed price validation; no implicit forward-fill or zero-return gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices require a nonempty DatetimeIndex.")
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique, nonmissing, and increasing.")
    if not prices.columns.is_unique:
        raise ValueError("Price columns must be unique.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive on every supplied date; resolve gaps explicitly.")


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    validate_prices(prices)
    result = prices.pct_change(fill_method=None)
    result.iloc[0] = 0.0  # first close is the starting valuation, not a missing quote
    return result
