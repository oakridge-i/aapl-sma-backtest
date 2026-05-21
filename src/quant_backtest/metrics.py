from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


def total_return(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return math.nan
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    clean = equity.dropna()
    if len(clean) < 2:
        return math.nan

    years = (clean.index[-1] - clean.index[0]).days / CALENDAR_DAYS_PER_YEAR
    if years <= 0:
        return math.nan

    ending_ratio = clean.iloc[-1] / clean.iloc[0]
    if ending_ratio <= 0:
        return math.nan
    return float(ending_ratio ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan
    return float(clean.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan

    volatility = annualized_volatility(clean)
    if volatility == 0 or math.isnan(volatility):
        return math.nan

    annualized_return = clean.mean() * TRADING_DAYS_PER_YEAR
    return float((annualized_return - risk_free_rate) / volatility)


def downside_deviation(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan
    downside = clean[clean < 0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan
    downside = downside_deviation(clean)
    if downside == 0 or math.isnan(downside):
        return math.nan
    annualized_return = clean.mean() * TRADING_DAYS_PER_YEAR
    return float((annualized_return - risk_free_rate) / downside)


def drawdown_series(equity: pd.Series) -> pd.Series:
    clean = equity.astype(float)
    running_max = clean.cummax()
    return clean / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    drawdowns = drawdown_series(equity).dropna()
    if drawdowns.empty:
        return math.nan
    return float(drawdowns.min())


def calmar_ratio(equity: pd.Series) -> float:
    annual_return = cagr(equity)
    drawdown = abs(max_drawdown(equity))
    if drawdown == 0 or math.isnan(drawdown):
        return math.nan
    return float(annual_return / drawdown)


def exposure_percentage(position_or_weights: pd.Series | pd.DataFrame) -> float:
    if isinstance(position_or_weights, pd.DataFrame):
        exposure = position_or_weights.abs().sum(axis=1)
    else:
        exposure = position_or_weights.abs()
    clean = exposure.dropna()
    if clean.empty:
        return math.nan
    return float((clean > 0).mean())


def annualized_turnover(turnover: pd.Series) -> float:
    clean = turnover.dropna()
    if clean.empty:
        return math.nan
    return float(clean.mean() * TRADING_DAYS_PER_YEAR)


def years_between(index: pd.Index) -> float:
    if len(index) < 2:
        return math.nan
    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])
    years = (end - start).days / CALENDAR_DAYS_PER_YEAR
    return float(years) if years > 0 else math.nan


def trade_frequency_per_year(trades: int, index: pd.Index) -> float:
    years = years_between(index)
    if math.isnan(years) or years == 0:
        return math.nan
    return float(trades / years)


def capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, direction: str) -> float:
    aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join="inner")
    mask = aligned_benchmark > 0 if direction == "up" else aligned_benchmark < 0
    if not mask.any():
        return math.nan
    benchmark_sum = aligned_benchmark[mask].sum()
    if benchmark_sum == 0:
        return math.nan
    return float(aligned_strategy[mask].sum() / benchmark_sum)


def capture_spread(upside_capture: float, downside_capture: float) -> float:
    if math.isnan(upside_capture) or math.isnan(downside_capture):
        return math.nan
    return float(upside_capture - downside_capture)


def missed_return_while_underweight(benchmark_returns: pd.Series, target_exposure: pd.Series) -> float:
    aligned_returns, aligned_exposure = benchmark_returns.align(target_exposure, join="inner")
    underweight = 1.0 - aligned_exposure.astype(float).clip(lower=0.0, upper=1.0)
    return float((aligned_returns * underweight).sum())


def avoided_downside_while_underweight(benchmark_returns: pd.Series, target_exposure: pd.Series) -> float:
    aligned_returns, aligned_exposure = benchmark_returns.align(target_exposure, join="inner")
    underweight = 1.0 - aligned_exposure.astype(float).clip(lower=0.0, upper=1.0)
    downside = aligned_returns.where(aligned_returns < 0.0, 0.0)
    return float(-(downside * underweight).sum())


def holding_periods(position: pd.Series) -> pd.Series:
    exposure = position.astype(float).fillna(0.0) > 0
    periods: list[int] = []
    current = 0
    for is_exposed in exposure:
        if is_exposed:
            current += 1
        elif current:
            periods.append(current)
            current = 0
    if current:
        periods.append(current)
    return pd.Series(periods, dtype=float, name="holding_period_days")


def trade_distribution(closed_trade_returns: pd.Series) -> dict[str, float]:
    clean = closed_trade_returns.dropna()
    if clean.empty:
        return {
            "average_trade_return": math.nan,
            "average_win": math.nan,
            "average_loss": math.nan,
            "profit_factor": math.nan,
        }

    wins = clean[clean > 0]
    losses = clean[clean < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "average_trade_return": float(clean.mean()),
        "average_win": math.nan if wins.empty else float(wins.mean()),
        "average_loss": math.nan if losses.empty else float(losses.mean()),
        "profit_factor": math.nan if gross_loss == 0 else float(gross_profit / gross_loss),
    }


def summarize_performance(
    name: str,
    equity: pd.Series,
    returns: pd.Series,
    trades: int | None = None,
    win_rate: float | None = None,
    risk_free_rate: float = 0.0,
    exposure: float | None = None,
    turnover: float | None = None,
    closed_trade_returns: pd.Series | None = None,
    gross_equity: pd.Series | None = None,
    benchmark_equity: pd.Series | None = None,
) -> dict[str, float | int | str]:
    trade_stats = trade_distribution(
        pd.Series(dtype=float) if closed_trade_returns is None else closed_trade_returns
    )
    row = {
        "name": name,
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "trades": math.nan if trades is None else int(trades),
        "win_rate": math.nan if win_rate is None else float(win_rate),
        "exposure": math.nan if exposure is None else float(exposure),
        "turnover": math.nan if turnover is None else float(turnover),
        **trade_stats,
    }
    if gross_equity is not None:
        row["cost_drag"] = total_return(gross_equity) - total_return(equity)
    else:
        row["cost_drag"] = math.nan
    if benchmark_equity is not None:
        row["excess_cagr_vs_benchmark"] = cagr(equity) - cagr(benchmark_equity)
        row["drawdown_improvement_vs_benchmark"] = max_drawdown(equity) - max_drawdown(benchmark_equity)
    else:
        row["excess_cagr_vs_benchmark"] = math.nan
        row["drawdown_improvement_vs_benchmark"] = math.nan
    return row


def metrics_table(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    return table.set_index("name")
