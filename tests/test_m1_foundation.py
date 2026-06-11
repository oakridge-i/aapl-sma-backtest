from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_backtest.experiments import ResearchConfig, run_research
from quant_backtest.registry import (
    build_strategy,
    family_by_name,
    family_for_params,
    iter_families,
)
from quant_backtest.research_config import AUTO_WORKERS, load_research_config
from quant_backtest.research_data import create_fixture_prices
from quant_backtest.significance import run_pbo_analysis
from quant_backtest.stats import probability_of_backtest_overfitting
from quant_backtest.strategies import (
    CaptureAwareAllocationParameters,
    SmaParameters,
    TrendAllocationParameters,
)
from quant_backtest.sweeps import run_hysteresis_sweep, run_nested_walk_forward


def m1_config(tmp_path: Path, **overrides) -> ResearchConfig:
    base = ResearchConfig(
        start="2019-01-01",
        end="2021-12-31",
        initial_capital=10_000.0,
        base_ticker="AAPL",
        universe=["AAPL", "MSFT", "SPY", "QQQ"],
        cost_bps=[0.0, 10.0],
        short_windows=[5],
        long_windows=[20],
        train_start="2019-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-12-31",
        walk_forward_train_years=1,
        walk_forward_test_years=1,
        walk_forward_step_years=1,
        entry_thresholds=[0.0, 0.01],
        exit_thresholds=[0.0],
        min_hold_days=[0],
        cooldown_days=[0],
        top_candidates=3,
        cash_proxy_ticker="BIL",
        bootstrap_iterations=200,
        permutation_iterations=50,
        output_dir=str(tmp_path),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- registry ---------------------------------------------------------------


def test_registry_contains_builtin_families() -> None:
    names = {family.name for family in iter_families()}

    assert {"sma_crossover", "trend_allocation", "capture_aware_trend"}.issubset(names)


def test_registry_dispatches_by_params_type() -> None:
    assert family_for_params(SmaParameters(5, 20)).name == "sma_crossover"
    assert family_for_params(TrendAllocationParameters(5, 20)).name == "trend_allocation"
    capture = CaptureAwareAllocationParameters(trend=TrendAllocationParameters(5, 20))
    family = family_for_params(capture)
    assert family.name == "capture_aware_trend"
    assert family.needs_market_context


def test_registry_builds_working_strategy() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    price = pd.Series(np.linspace(100, 130, 60), index=dates)

    strategy = build_strategy(SmaParameters(5, 20))
    signals = strategy.generate(price)

    assert signals.target_position.iloc[-1] == 1.0


def test_registry_rejects_unknown_names_and_params() -> None:
    with pytest.raises(KeyError, match="Unknown strategy family"):
        family_by_name("does_not_exist")
    with pytest.raises(KeyError, match="No strategy family registered"):
        family_for_params(object())


# --- parallel execution ------------------------------------------------------


def test_parallel_sweep_matches_serial(tmp_path: Path) -> None:
    # 16 pairs x 3 entry thresholds = 48 jobs, above the 32-job threshold, so
    # the process pool genuinely engages for the parallel run.
    config = m1_config(
        tmp_path,
        short_windows=[3, 5, 8, 10],
        long_windows=[20, 30, 40, 60],
        entry_thresholds=[0.0, 0.005, 0.01],
    )
    prices = create_fixture_prices(config)

    serial = run_hysteresis_sweep(prices, config)
    parallel_config = dataclasses.replace(config, parallel_workers=2)
    parallel = run_hysteresis_sweep(prices, parallel_config)

    pd.testing.assert_frame_equal(serial, parallel)


def test_workers_auto_parses_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
period: {start: 2020-01-01, end: 2021-12-31}
base_ticker: AAPL
universe: [AAPL, SPY]
cost_bps: [10]
sma_grid: {short: [5], long: [20]}
train_test: {train_start: 2020-01-01, train_end: 2020-12-31, test_start: 2021-01-01, test_end: 2021-12-31}
walk_forward: {train_years: 1, test_years: 1, step_years: 1}
compute: {workers: auto}
nested_walk_forward: {enabled: true}
pbo: {enabled: true, max_candidates: 50, blocks: 8}
""",
        encoding="utf-8",
    )

    config = load_research_config(config_path)

    assert config.parallel_workers == AUTO_WORKERS
    assert config.enable_nested_walk_forward
    assert config.enable_pbo
    assert config.pbo_max_candidates == 50
    assert config.pbo_blocks == 8


# --- nested walk-forward ------------------------------------------------------


def test_nested_walk_forward_selects_per_window(tmp_path: Path) -> None:
    config = m1_config(tmp_path)
    prices = create_fixture_prices(config)

    nested = run_nested_walk_forward(prices, config)

    windows = nested["windows"]
    assert not windows.empty
    assert (windows["candidates_evaluated"] > 0).all()
    # OOS slices must start strictly after each window's train end.
    assert (pd.to_datetime(windows["test_start"]) > pd.to_datetime(windows["train_end"])).all()
    assert not nested["oos_returns"].empty
    assert not nested["summary"].empty
    assert "nested_oos_stitched" in set(nested["summary"]["name"])


def test_nested_oos_returns_only_cover_test_windows(tmp_path: Path) -> None:
    config = m1_config(tmp_path)
    prices = create_fixture_prices(config)

    nested = run_nested_walk_forward(prices, config)

    oos = nested["oos_returns"]
    first_test_start = pd.to_datetime(nested["windows"]["test_start"]).min()
    assert oos.index.min() >= first_test_start
    assert not oos.index.duplicated().any()


# --- PBO ----------------------------------------------------------------------


def _candidate_matrix(edge: float, n_candidates: int = 40, n_obs: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_obs, freq="B")
    noise = rng.normal(0.0, 0.01, size=(n_obs, n_candidates))
    if edge:
        noise[:, 0] += edge
    return pd.DataFrame(noise, index=dates, columns=[f"candidate_{idx}" for idx in range(n_candidates)])


def test_pbo_near_half_for_pure_noise() -> None:
    # A single noise dataset has a high-variance PBO estimate (the CSCV splits
    # overlap), so check the average across several independent datasets.
    values = [
        probability_of_backtest_overfitting(_candidate_matrix(edge=0.0, seed=seed), n_blocks=8)["pbo"]
        for seed in range(7)
    ]

    assert 0.3 <= float(np.mean(values)) <= 0.7


def test_pbo_low_when_one_candidate_has_real_edge() -> None:
    summary = probability_of_backtest_overfitting(_candidate_matrix(edge=0.002), n_blocks=8)

    assert summary["pbo"] < 0.15


def test_pbo_caps_candidates() -> None:
    summary = probability_of_backtest_overfitting(_candidate_matrix(edge=0.0), n_blocks=8, max_candidates=10)

    assert summary["n_candidates"] == 10.0


def test_pbo_analysis_runs_on_fixture(tmp_path: Path) -> None:
    config = m1_config(tmp_path, pbo_blocks=8)
    prices = create_fixture_prices(config)

    table = run_pbo_analysis(prices, config)

    assert not table.empty
    assert 0.0 <= table.loc[0, "pbo"] <= 1.0
    assert table.loc[0, "grid"] == "trend_hysteresis"


# --- end-to-end with M1 features enabled --------------------------------------


def test_run_research_with_nested_and_pbo(tmp_path: Path) -> None:
    config = m1_config(tmp_path, enable_nested_walk_forward=True, enable_pbo=True, pbo_blocks=8)

    result = run_research(config, fixture_data=True)

    assert not result.nested_walk_forward.empty
    assert not result.nested_walk_forward_summary.empty
    assert not result.pbo_results.empty
    assert "nested_oos_stitched" in set(result.significance_results["model"])
