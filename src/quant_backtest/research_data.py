"""Price acquisition for the research workflow: download, fixtures, cash."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import default_end_date, download_adjusted_close
from .research_config import ResearchConfig


def research_tickers(config: ResearchConfig) -> list[str]:
    tickers = list(config.universe)
    if config.cash_proxy_ticker and config.cash_proxy_ticker not in tickers:
        tickers.append(config.cash_proxy_ticker)
    return tickers


def download_research_prices(config: ResearchConfig) -> pd.DataFrame:
    return download_adjusted_close(research_tickers(config), start=config.start, end=config.end or default_end_date())


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


def cash_return_series(prices: pd.DataFrame, cash_proxy: str | None) -> pd.Series | None:
    if not cash_proxy or cash_proxy not in prices.columns:
        return None
    return prices[cash_proxy].dropna().pct_change().fillna(0.0)
