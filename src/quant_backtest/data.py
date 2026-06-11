from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".cache" / "yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    retries: int = 3,
    retry_wait_seconds: float = 15.0,
) -> pd.DataFrame:
    """Download daily OHLCV data and return a clean single-ticker DataFrame.

    Yahoo Finance intermittently returns empty frames under rate limiting, so
    empty responses are retried with a pause before giving up.
    """
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker must not be empty.")

    data = pd.DataFrame()
    for attempt in range(max(1, retries)):
        if attempt:
            time.sleep(retry_wait_seconds)
        data = yf.download(
            normalized_ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if not data.empty:
            break
    if data.empty:
        raise ValueError(f"No price data returned for {normalized_ticker}.")

    data = _flatten_yfinance_columns(data, normalized_ticker)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns from data source: {missing}")

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    data = data.sort_index()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data


def download_adjusted_close(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        data = download_ohlcv(ticker=ticker, start=start, end=end)
        column = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices[ticker.strip().upper()] = data[column].rename(ticker.strip().upper())
    frame = pd.DataFrame(prices).sort_index()
    frame.index.name = "Date"
    return frame.dropna(how="all")


def frame_sha256(prices: pd.DataFrame) -> str:
    """Stable content hash of a price frame, for run manifests."""
    canonical = prices.sort_index().round(8).to_csv(index_label="Date", lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_price_snapshot(prices: pd.DataFrame, path: Path) -> str:
    """Write the price frame to CSV and return its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.sort_index().to_csv(path, index_label="Date", lineterminator="\n")
    return frame_sha256(prices)


def load_price_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="Date", parse_dates=True)
    frame.index.name = "Date"
    return frame.sort_index().astype(float)


def default_end_date() -> str:
    """Use an exclusive end date so yfinance includes the latest completed day."""
    return date.today().isoformat()


def _flatten_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    columns = data.columns
    ticker_upper = ticker.upper()

    for level in range(columns.nlevels):
        level_values = [str(value).upper() for value in columns.get_level_values(level)]
        if ticker_upper in level_values:
            return data.xs(ticker, axis=1, level=level, drop_level=True).copy()

    first_level = set(str(value) for value in columns.get_level_values(0))
    if {"Open", "High", "Low", "Close"}.issubset(first_level):
        return data.droplevel(1, axis=1).copy()

    raise ValueError("Could not normalize yfinance MultiIndex columns.")
