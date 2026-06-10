from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_backtest.backtest import calculate_closed_trade_returns, count_exposure_episodes
from quant_backtest.costs import BpsCost
from quant_backtest.data import frame_sha256, load_price_snapshot, save_price_snapshot
from quant_backtest.engine import EngineConfig, run_weight_backtest
from quant_backtest.experiments import (
    ResearchConfig,
    create_fixture_prices,
    evaluate_strategy,
    run_research,
)
from quant_backtest.reports import save_research_outputs
from quant_backtest.strategies import SmaParameters


def v05_config(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(
        start="2020-01-01",
        end="2021-12-31",
        initial_capital=10_000.0,
        base_ticker="AAPL",
        universe=["AAPL", "MSFT", "SPY", "QQQ"],
        cost_bps=[0.0, 10.0],
        short_windows=[5],
        long_windows=[20],
        train_start="2020-01-01",
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


def test_engine_cash_weight_earns_cash_return() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame({"AAPL": [0.0] * 5}, index=dates)
    target_weights = pd.DataFrame({"AAPL": [0.0] * 5}, index=dates)
    cash_returns = pd.Series(0.01, index=dates)

    result = run_weight_backtest(
        returns=returns,
        target_weights=target_weights,
        config=EngineConfig(initial_capital=100.0, cost_model=BpsCost(0.0)),
        cash_returns=cash_returns,
    )

    assert result.curve["cash_weight"].eq(1.0).all()
    assert result.curve["strategy_equity"].iloc[-1] == pytest.approx(100.0 * 1.01**5)


def test_cash_leg_scales_with_invested_weight() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"AAPL": [0.0] * 4}, index=dates)
    target_weights = pd.DataFrame({"AAPL": [0.5] * 4}, index=dates)
    cash_returns = pd.Series(0.02, index=dates)

    result = run_weight_backtest(
        returns=returns,
        target_weights=target_weights,
        config=EngineConfig(initial_capital=100.0, cost_model=BpsCost(0.0)),
        cash_returns=cash_returns,
    )

    # Day 0 has no executed position (lag), later days hold 50% cash.
    assert result.curve["strategy_return"].iloc[0] == pytest.approx(0.02)
    assert result.curve["strategy_return"].iloc[-1] == pytest.approx(0.01)


def test_closed_trades_ignore_fractional_resizing() -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    position = pd.Series([0.0, 0.5, 1.0, 0.6, 0.8, 0.0, 0.0], index=dates)
    returns = pd.Series([0.0, 0.01, 0.01, 0.01, 0.01, -0.002, 0.0], index=dates)

    closed = calculate_closed_trade_returns(returns, position)

    assert len(closed) == 1
    assert closed.iloc[0] == pytest.approx((1.01**4) * 0.998 - 1.0)
    assert count_exposure_episodes(position) == 1


def test_open_episode_is_counted_but_not_closed() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    position = pd.Series([0.0, 1.0, 1.0, 1.0], index=dates)
    returns = pd.Series(0.01, index=dates)

    assert len(calculate_closed_trade_returns(returns, position)) == 0
    assert count_exposure_episodes(position) == 1


def test_fixture_prices_include_cash_proxy(tmp_path: Path) -> None:
    config = v05_config(tmp_path)
    prices = create_fixture_prices(config)

    assert "BIL" in prices.columns
    annual_yield = prices["BIL"].iloc[-1] / prices["BIL"].iloc[0]
    assert annual_yield > 1.0


def test_evaluate_strategy_uses_cash_yield_as_risk_free(tmp_path: Path) -> None:
    config = v05_config(tmp_path)
    prices = create_fixture_prices(config)

    with_cash = evaluate_strategy(
        prices=prices,
        ticker="AAPL",
        params=SmaParameters(5, 20),
        variant="long_cash",
        cost_bps=10.0,
        initial_capital=10_000.0,
        label="test",
        cash_proxy="BIL",
    )
    without_cash = evaluate_strategy(
        prices=prices,
        ticker="AAPL",
        params=SmaParameters(5, 20),
        variant="long_cash",
        cost_bps=10.0,
        initial_capital=10_000.0,
        label="test",
    )

    assert with_cash["risk_free_rate"] > 0.0
    assert without_cash["risk_free_rate"] == 0.0
    # Cash yield can only help a long/cash strategy's equity.
    assert (
        with_cash["curve"]["strategy_equity"].iloc[-1]
        >= without_cash["curve"]["strategy_equity"].iloc[-1]
    )


def test_research_selection_happens_on_train_only(tmp_path: Path) -> None:
    config = v05_config(tmp_path)

    result = run_research(config, fixture_data=True)

    train_end = pd.Timestamp(config.train_end)
    assert (result.allocation_leaderboard["label"] == "allocation_train").all()
    assert (result.model_leaderboard["label"] == "leaderboard_train").all()
    assert result.run_metadata["selection_period"] == "train"
    # The v0.3 comparison (the single test-period touch) starts after train.
    assert result.v03_curve.index.min() > train_end


def test_research_produces_significance_and_final_walk_forward(tmp_path: Path) -> None:
    config = v05_config(tmp_path)

    result = run_research(config, fixture_data=True)

    assert not result.final_walk_forward.empty
    assert {"baseline_sma_20_100", "selected_v2", "selected_v3"}.issubset(set(result.final_walk_forward["model"]))
    assert not result.significance_results.empty
    row = result.significance_results.iloc[0]
    assert 0.0 <= row["permutation_p_value"] <= 1.0
    assert row["sharpe_p05"] <= row["sharpe_p95"]
    assert row["n_trials"] > 1


def test_outputs_include_snapshot_and_manifest(tmp_path: Path) -> None:
    config = v05_config(tmp_path)

    result = run_research(config, fixture_data=True)
    save_research_outputs(result, tmp_path)

    assert (tmp_path / "significance_results.csv").exists()
    assert (tmp_path / "final_model_walk_forward.csv").exists()
    snapshot_path = tmp_path / "data_snapshot.csv"
    assert snapshot_path.exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_sha256"] == frame_sha256(result.prices)
    assert manifest["selection_period"] == "train"
    assert manifest["config"]["base_ticker"] == "AAPL"

    # Rerunning from the snapshot reproduces the same data hash.
    reloaded = load_price_snapshot(snapshot_path)
    assert frame_sha256(reloaded) == manifest["data_sha256"]


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    config = v05_config(tmp_path)
    prices = create_fixture_prices(config)
    path = tmp_path / "snapshot.csv"

    digest = save_price_snapshot(prices, path)
    reloaded = load_price_snapshot(path)

    assert digest == frame_sha256(reloaded)
    pd.testing.assert_index_equal(reloaded.index, prices.index, check_names=False)
    assert list(reloaded.columns) == list(prices.columns)


def test_config_replace_keeps_frozen_dataclass_usable(tmp_path: Path) -> None:
    config = v05_config(tmp_path)

    replaced = dataclasses.replace(config, output_dir="elsewhere")

    assert replaced.output_dir == "elsewhere"
    assert replaced.cash_proxy_ticker == "BIL"
