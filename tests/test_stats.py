from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_backtest.stats import (
    block_bootstrap_summary,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    timing_permutation_pvalue,
)


def _daily_returns(seed: int = 7, n: int = 500, mean: float = 0.0005, std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, size=n), index=dates)


def test_block_bootstrap_is_deterministic_and_ordered() -> None:
    returns = _daily_returns()

    first = block_bootstrap_summary(returns, n_iterations=200, seed=11)
    second = block_bootstrap_summary(returns, n_iterations=200, seed=11)

    assert first == second
    assert first["cagr_p05"] <= first["cagr_p50"] <= first["cagr_p95"]
    assert first["sharpe_p05"] <= first["sharpe_p50"] <= first["sharpe_p95"]
    assert first["max_drawdown_p50"] <= 0.0
    assert 0.0 <= first["prob_negative_sharpe"] <= 1.0


def test_block_bootstrap_flags_zero_mean_strategy() -> None:
    raw = _daily_returns(mean=0.0)
    returns = raw - raw.mean()  # force the sample Sharpe to exactly zero

    summary = block_bootstrap_summary(returns, n_iterations=500, seed=3)

    # A zero-edge strategy should have a wide Sharpe interval straddling zero.
    assert summary["sharpe_p05"] < 0 < summary["sharpe_p95"]
    assert 0.1 < summary["prob_negative_sharpe"] < 0.9


def test_expected_max_sharpe_grows_with_trial_count() -> None:
    rng = np.random.default_rng(5)
    few = pd.Series(rng.normal(0.0, 0.02, size=10))
    many = pd.Series(rng.normal(0.0, 0.02, size=1000))

    assert expected_max_sharpe(many) > expected_max_sharpe(few) > 0.0


def test_deflated_sharpe_penalizes_wide_searches() -> None:
    returns = _daily_returns(mean=0.0008)
    narrow_search = pd.Series(np.linspace(0.4, 0.6, 5))
    wide_search = pd.Series(np.linspace(-2.0, 2.0, 500))

    narrow = deflated_sharpe_ratio(returns, narrow_search)
    wide = deflated_sharpe_ratio(returns, wide_search)

    assert 0.0 <= wide["deflated_sharpe_prob"] <= 1.0
    assert wide["deflated_sharpe_prob"] < narrow["deflated_sharpe_prob"]
    assert wide["expected_max_sharpe_annual"] > narrow["expected_max_sharpe_annual"]


def test_permutation_pvalue_is_one_for_constant_exposure() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({"AAPL": rng.normal(0.001, 0.02, size=300)}, index=dates)
    weights = pd.DataFrame({"AAPL": 1.0}, index=dates)

    result = timing_permutation_pvalue(weights, returns, cost_bps=10.0, n_permutations=100, seed=1)

    # Constant weights are invariant under circular shifts, so every permuted
    # Sharpe equals the observed one.
    assert result["permutation_p_value"] == pytest.approx(1.0)


def test_permutation_detects_perfect_foresight() -> None:
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(9)
    asset = rng.normal(0.0, 0.02, size=400)
    returns = pd.DataFrame({"AAPL": asset}, index=dates)
    weights = pd.DataFrame({"AAPL": (asset > 0).astype(float)}, index=dates)

    result = timing_permutation_pvalue(weights, returns, cost_bps=0.0, n_permutations=200, seed=4)

    assert result["permutation_p_value"] < 0.05
