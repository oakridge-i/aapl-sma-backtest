"""Statistical significance tools for backtest results.

The functions here answer one question: how likely is it that an observed
backtest result is luck rather than skill? Three complementary tools are
provided:

- circular block bootstrap confidence intervals for CAGR, Sharpe, and max
  drawdown;
- the Deflated Sharpe Ratio of Bailey and Lopez de Prado, which corrects the
  observed Sharpe for the number of candidate models that were tried;
- a circular-shift permutation test that asks whether the strategy's timing
  adds value over random alignment of the same exposure profile with the same
  cost structure.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS_PER_YEAR


EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def block_bootstrap_summary(
    returns: pd.Series,
    n_iterations: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Circular block bootstrap of daily returns.

    Resamples the return series in contiguous blocks (preserving short-range
    autocorrelation), recomputes CAGR, Sharpe, and max drawdown for each
    sample, and returns 5th/50th/95th percentiles plus the probability of a
    negative outcome.
    """
    clean = returns.dropna().astype(float).to_numpy()
    n_obs = len(clean)
    if n_obs < block_size or n_iterations <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n_obs / block_size))
    # Start positions for every block of every iteration, sampled at once.
    starts = rng.integers(0, n_obs, size=(n_iterations, n_blocks))
    offsets = np.arange(block_size)
    # Circular indexing: each block wraps around the end of the sample.
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    samples = clean[indices.reshape(n_iterations, -1)[:, :n_obs]]

    growth = np.cumprod(1.0 + samples, axis=1)
    ending = growth[:, -1]
    years = n_obs / TRADING_DAYS_PER_YEAR
    valid = ending > 0
    cagr_samples = np.full(n_iterations, np.nan)
    cagr_samples[valid] = ending[valid] ** (1.0 / years) - 1.0

    std = samples.std(axis=1, ddof=0)
    mean = samples.mean(axis=1)
    sharpe_samples = np.where(
        std > 0,
        (mean * TRADING_DAYS_PER_YEAR - risk_free_rate) / (std * math.sqrt(TRADING_DAYS_PER_YEAR)),
        np.nan,
    )

    running_max = np.maximum.accumulate(growth, axis=1)
    drawdown_samples = (growth / running_max - 1.0).min(axis=1)

    def percentiles(values: np.ndarray, name: str) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {f"{name}_p05": math.nan, f"{name}_p50": math.nan, f"{name}_p95": math.nan}
        p05, p50, p95 = np.percentile(finite, [5, 50, 95])
        return {f"{name}_p05": float(p05), f"{name}_p50": float(p50), f"{name}_p95": float(p95)}

    summary: dict[str, float] = {"bootstrap_iterations": float(n_iterations)}
    summary |= percentiles(cagr_samples, "cagr")
    summary |= percentiles(sharpe_samples, "sharpe")
    summary |= percentiles(drawdown_samples, "max_drawdown")
    finite_sharpe = sharpe_samples[np.isfinite(sharpe_samples)]
    finite_cagr = cagr_samples[np.isfinite(cagr_samples)]
    summary["prob_negative_sharpe"] = float((finite_sharpe < 0).mean()) if finite_sharpe.size else math.nan
    summary["prob_negative_cagr"] = float((finite_cagr < 0).mean()) if finite_cagr.size else math.nan
    return summary


def expected_max_sharpe(trial_sharpes: pd.Series) -> float:
    """Expected maximum Sharpe among N independent trials with zero true skill.

    Uses the extreme-value approximation from Bailey & Lopez de Prado (2014).
    Inputs and output are in the same (per-period) units as ``trial_sharpes``.
    """
    clean = trial_sharpes.dropna().astype(float)
    n_trials = len(clean)
    if n_trials < 2:
        return math.nan
    trial_std = float(clean.std(ddof=1))
    if not math.isfinite(trial_std) or trial_std == 0:
        return 0.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return trial_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes_annual: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Deflated Sharpe Ratio: P(true Sharpe > 0) after multiple testing.

    ``returns`` are the daily returns of the *selected* model on the
    evaluation period. ``trial_sharpes_annual`` are the annualized Sharpe
    ratios of every candidate that was evaluated during selection; their
    dispersion and count set the hurdle (expected max Sharpe under no skill).
    """
    clean = returns.dropna().astype(float)
    n_obs = len(clean)
    if n_obs < 20:
        return {}

    std = float(clean.std(ddof=0))
    if std == 0:
        return {}
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    observed_sr = (float(clean.mean()) - daily_rf) / std

    trials_daily = trial_sharpes_annual.dropna().astype(float) / math.sqrt(TRADING_DAYS_PER_YEAR)
    hurdle = expected_max_sharpe(trials_daily)
    if math.isnan(hurdle):
        return {}

    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0  # pandas returns excess kurtosis
    denominator = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr**2
    if denominator <= 0:
        return {}
    statistic = (observed_sr - hurdle) * math.sqrt(n_obs - 1) / math.sqrt(denominator)
    return {
        "n_trials": float(len(trials_daily)),
        "observed_sharpe_daily": observed_sr,
        "expected_max_sharpe_annual": hurdle * math.sqrt(TRADING_DAYS_PER_YEAR),
        "deflated_sharpe_prob": _NORMAL.cdf(statistic),
    }


def probability_of_backtest_overfitting(
    candidate_returns: pd.DataFrame,
    n_blocks: int = 12,
    max_candidates: int = 200,
) -> dict[str, float]:
    """CSCV Probability of Backtest Overfitting (Bailey et al., 2017).

    ``candidate_returns`` holds one column of daily net returns per candidate
    configuration over a common period. The sample is cut into ``n_blocks``
    contiguous blocks; for every combination of half the blocks (in-sample)
    the best candidate by IS Sharpe is picked and its *relative rank* by OOS
    Sharpe on the complementary blocks is recorded. PBO is the share of
    combinations where the IS winner lands in the worse half OOS — for pure
    noise it sits near 0.5, for a real edge near 0, and above 0.5 means the
    selection actively picks OOS losers.

    Candidate count is capped with an evenly spaced subset (columns are
    expected to be ordered by selection score, so the subset spans the
    quality spectrum).
    """
    import itertools

    clean = candidate_returns.dropna(axis=0, how="any")
    n_obs, n_candidates = clean.shape
    if n_candidates < 2 or n_blocks < 4 or n_blocks % 2 != 0 or n_obs < n_blocks * 21:
        return {}

    if n_candidates > max_candidates:
        keep = np.unique(np.linspace(0, n_candidates - 1, max_candidates).astype(int))
        clean = clean.iloc[:, keep]
        n_candidates = clean.shape[1]

    block_length = n_obs // n_blocks
    usable = block_length * n_blocks
    values = clean.to_numpy()[:usable]
    blocks = values.reshape(n_blocks, block_length, n_candidates)
    block_sum = blocks.sum(axis=1).T  # [candidates, blocks]
    block_sumsq = (blocks**2).sum(axis=1).T

    half = n_blocks // 2
    combos = list(itertools.combinations(range(n_blocks), half))
    mask = np.zeros((len(combos), n_blocks))
    for combo_idx, combo in enumerate(combos):
        mask[combo_idx, list(combo)] = 1.0

    def sharpe_matrix(sums: np.ndarray, sumsqs: np.ndarray, n: float) -> np.ndarray:
        mean = sums / n
        variance = np.maximum(sumsqs / n - mean**2, 0.0)
        std = np.sqrt(variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(std > 0, mean / std * math.sqrt(TRADING_DAYS_PER_YEAR), -np.inf)
        return sharpe

    n_is = float(half * block_length)
    is_sum = block_sum @ mask.T  # [candidates, combos]
    is_sumsq = block_sumsq @ mask.T
    oos_sum = block_sum.sum(axis=1, keepdims=True) - is_sum
    oos_sumsq = block_sumsq.sum(axis=1, keepdims=True) - is_sumsq
    is_sharpe = sharpe_matrix(is_sum, is_sumsq, n_is)
    oos_sharpe = sharpe_matrix(oos_sum, oos_sumsq, n_is)

    best = is_sharpe.argmax(axis=0)  # [combos]
    best_oos = oos_sharpe[best, np.arange(len(combos))]
    # Relative OOS rank of the IS winner among all candidates.
    below = (oos_sharpe < best_oos[None, :]).sum(axis=0)
    omega = (below + 1.0) / (n_candidates + 1.0)
    logits = np.log(omega / (1.0 - omega))

    return {
        "pbo": float((logits < 0).mean()),
        "n_candidates": float(n_candidates),
        "n_blocks": float(n_blocks),
        "n_combinations": float(len(combos)),
        "mean_logit": float(logits.mean()),
        "median_oos_relative_rank": float(np.median(omega)),
    }


def timing_permutation_pvalue(
    executed_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float,
    cash_returns: pd.Series | None = None,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test for timing skill.

    Circularly shifts the whole executed weight matrix relative to the asset
    returns. Each shifted version keeps the exposure profile, trade frequency,
    and transaction costs of the original strategy but destroys the alignment
    between signals and subsequent returns. The p-value is the share of
    shifted versions with a Sharpe at least as high as the observed one.
    """
    weights = executed_weights.fillna(0.0).astype(float)
    returns = asset_returns.reindex(weights.index).reindex(columns=weights.columns).fillna(0.0).astype(float)
    n_obs = len(weights)
    if n_obs < 30 or n_permutations <= 0:
        return {}

    weight_values = weights.to_numpy()
    return_values = returns.to_numpy()
    cost_rate = cost_bps / 10_000.0
    cash_values = (
        cash_returns.reindex(weights.index).fillna(0.0).astype(float).to_numpy()
        if cash_returns is not None
        else None
    )

    def strategy_sharpe(shifted_weights: np.ndarray) -> float:
        turnover = np.abs(np.diff(shifted_weights, axis=0, prepend=np.zeros((1, shifted_weights.shape[1])))).sum(axis=1)
        net = (shifted_weights * return_values).sum(axis=1) - turnover * cost_rate
        if cash_values is not None:
            cash_weight = np.clip(1.0 - shifted_weights.sum(axis=1), 0.0, None)
            net = net + cash_weight * cash_values
        std = net.std()
        if std == 0:
            return math.nan
        return float(net.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))

    observed = strategy_sharpe(weight_values)
    if math.isnan(observed):
        return {}

    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, n_obs, size=n_permutations)
    permuted = np.array([strategy_sharpe(np.roll(weight_values, int(offset), axis=0)) for offset in offsets])
    finite = permuted[np.isfinite(permuted)]
    if finite.size == 0:
        return {}
    p_value = (1.0 + float((finite >= observed).sum())) / (finite.size + 1.0)
    return {
        "permutation_iterations": float(finite.size),
        "permutation_sharpe_mean": float(finite.mean()),
        "permutation_p_value": p_value,
    }
