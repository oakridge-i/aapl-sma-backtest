from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_backtest.ensemble_research import (
    CANONICAL_TREND_MEMBER,
    build_ensemble_candidates,
    family_parameter_grids,
    run_ensemble_leaderboard,
    run_family_sweep,
    run_nested_ensemble_walk_forward,
    select_ensemble_model,
    select_family_champions,
)
from quant_backtest.evaluation import evaluate_strategy
from quant_backtest.experiments import ResearchConfig, run_research
from quant_backtest.registry import family_for_params
from quant_backtest.research_config import load_research_config
from quant_backtest.research_data import create_fixture_prices
from quant_backtest.signal_families import (
    AtrTrendParameters,
    AtrTrendStrategy,
    DonchianBreakoutStrategy,
    DonchianParameters,
    DualMomentumParameters,
    DualMomentumStrategy,
    EnsembleParameters,
    EnsembleVoteStrategy,
    High52WeekParameters,
    High52WeekStrategy,
    TimeSeriesMomentumParameters,
    TimeSeriesMomentumStrategy,
)


def m2_config(tmp_path: Path, **overrides) -> ResearchConfig:
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
        entry_thresholds=[0.0],
        exit_thresholds=[0.0],
        min_hold_days=[0],
        cooldown_days=[0],
        top_candidates=2,
        cash_proxy_ticker="BIL",
        bootstrap_iterations=200,
        permutation_iterations=50,
        enable_signal_families=True,
        signal_family_grids={
            "ts_momentum": {"lookbacks": [63, 126]},
            "donchian": {"entry_windows": [55], "exit_windows": [20]},
            "atr_trend": {"sma_windows": [100], "atr_windows": [20], "scales": [3.0]},
            "dual_momentum": {"lookbacks": [126], "market": "SPY"},
            "high_52w": {"window": 126, "entry_thresholds": [0.95], "exit_thresholds": [0.85]},
        },
        ensemble_min_members=2,
        ensemble_max_members=4,
        output_dir=str(tmp_path),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _uptrend(n: int = 300) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(100.0, 200.0, n), index=dates)


def _crash_after_rally(n_up: int = 300, n_down: int = 100) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n_up + n_down, freq="B")
    up = np.linspace(100.0, 200.0, n_up)
    down = np.linspace(200.0, 100.0, n_down)
    return pd.Series(np.concatenate([up, down]), index=dates)


# --- individual families -------------------------------------------------------


def test_ts_momentum_long_in_uptrend_flat_after_crash() -> None:
    strategy = TimeSeriesMomentumStrategy(TimeSeriesMomentumParameters(lookback_days=63))
    price = _crash_after_rally()

    signals = strategy.generate(price)

    assert signals.target_position.iloc[250] == 1.0
    assert signals.target_position.iloc[-1] == 0.0


def test_donchian_enters_on_breakout_and_exits_on_breakdown() -> None:
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    flat = [100.0] * 60
    rally = list(np.linspace(101.0, 120.0, 30))
    crash = list(np.linspace(119.0, 80.0, 30))
    price = pd.Series(flat + rally + crash, index=dates)
    strategy = DonchianBreakoutStrategy(DonchianParameters(entry_window=50, exit_window=10))

    signals = strategy.generate(price)

    assert signals.target_position.iloc[59] == 0.0  # before the breakout
    assert signals.target_position.iloc[65] == 1.0  # after a fresh 50-day high
    assert signals.target_position.iloc[-1] == 0.0  # after the breakdown


def test_atr_trend_exposure_is_bounded_and_scales_with_strength() -> None:
    strategy = AtrTrendStrategy(AtrTrendParameters(sma_window=50, atr_window=10, scale=3.0))
    price = _uptrend()

    signals = strategy.generate(price)

    target = signals.target_position
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0
    assert target.iloc[-1] > 0.5  # steady uptrend: well above the SMA


def test_dual_momentum_requires_beating_the_market() -> None:
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    asset = pd.Series(np.linspace(100.0, 130.0, 300), index=dates)
    strong_market = pd.Series(np.linspace(100.0, 200.0, 300), index=dates)
    weak_market = pd.Series(np.linspace(100.0, 105.0, 300), index=dates)
    strategy = DualMomentumStrategy(DualMomentumParameters(lookback_days=126))

    losing = strategy.generate(asset, market_price=strong_market)
    winning = strategy.generate(asset, market_price=weak_market)

    assert losing.target_position.iloc[-1] == 0.0
    assert winning.target_position.iloc[-1] == 1.0


def test_high_52w_hysteresis_enters_near_high_exits_below() -> None:
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    up = list(np.linspace(100.0, 150.0, 300))
    dip = list(np.linspace(150.0, 120.0, 100))  # 20% off the high
    price = pd.Series(up + dip, index=dates)
    strategy = High52WeekStrategy(High52WeekParameters(window=252, entry_threshold=0.95, exit_threshold=0.85))

    signals = strategy.generate(price)

    assert signals.target_position.iloc[290] == 1.0  # at the highs
    assert signals.target_position.iloc[-1] == 0.0  # 0.8 of the high < exit


def test_family_params_validation() -> None:
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(TimeSeriesMomentumParameters(lookback_days=0))
    with pytest.raises(ValueError):
        High52WeekStrategy(High52WeekParameters(entry_threshold=0.8, exit_threshold=0.9))
    with pytest.raises(ValueError):
        AtrTrendStrategy(AtrTrendParameters(scale=0.0))


# --- ensemble -------------------------------------------------------------------


def test_ensemble_exposure_is_mean_of_member_votes() -> None:
    price = _uptrend()
    always_long = TimeSeriesMomentumParameters(lookback_days=21)  # uptrend: long
    rarely_long = High52WeekParameters(window=63, entry_threshold=0.999, exit_threshold=0.99)
    ensemble = EnsembleVoteStrategy(EnsembleParameters(members=(always_long, rarely_long)))

    signals = ensemble.generate(price)
    member_a = TimeSeriesMomentumStrategy(always_long).generate(price).target_position
    member_b = High52WeekStrategy(rarely_long).generate(price).target_position
    expected = (member_a + member_b) / 2.0

    pd.testing.assert_series_equal(signals.target_position, expected, check_names=False)


def test_ensemble_registry_dispatch_and_evaluation(tmp_path: Path) -> None:
    config = m2_config(tmp_path)
    prices = create_fixture_prices(config)
    ensemble = EnsembleParameters(
        members=(TimeSeriesMomentumParameters(lookback_days=63), CANONICAL_TREND_MEMBER)
    )

    family = family_for_params(ensemble)
    assert family.name == "ensemble_vote"
    assert family.needs_market_price

    result = evaluate_strategy(
        prices=prices,
        ticker="AAPL",
        params=ensemble,
        variant="ensemble_vote",
        cost_bps=10.0,
        initial_capital=10_000.0,
        label="test",
        cash_proxy="BIL",
    )
    assert result["row"]["exposure"] > 0
    assert result["weights"].columns.tolist() == ["AAPL"]
    assert result["weights"]["AAPL"].max() <= 1.0


def test_ensemble_requires_members() -> None:
    with pytest.raises(ValueError):
        EnsembleParameters(members=())


# --- selection pipeline ----------------------------------------------------------


def test_family_grids_respect_config(tmp_path: Path) -> None:
    config = m2_config(tmp_path)

    grids = family_parameter_grids(config)

    assert set(grids) == {"ts_momentum", "donchian_breakout", "atr_trend", "dual_momentum", "high_52w"}
    assert len(grids["ts_momentum"]) == 2
    assert len(grids["donchian_breakout"]) == 1


def test_family_sweep_and_champions_are_train_only(tmp_path: Path) -> None:
    config = m2_config(tmp_path)
    prices = create_fixture_prices(config)
    train_prices = prices.loc[config.train_start : config.train_end]

    table, params_list = run_family_sweep(train_prices, config)
    champions = select_family_champions(table, params_list, config)

    assert (table["label"] == "family_train").all()
    assert len(table) == len(params_list)
    assert set(champions) <= set(family_parameter_grids(config))
    assert len(champions) == 5


def test_ensemble_candidates_and_leaderboard(tmp_path: Path) -> None:
    config = m2_config(tmp_path)
    prices = create_fixture_prices(config)
    train_prices = prices.loc[config.train_start : config.train_end]
    table, params_list = run_family_sweep(train_prices, config)
    champions = select_family_champions(table, params_list, config)

    candidates = build_ensemble_candidates(champions, config)
    leaderboard = run_ensemble_leaderboard(train_prices, config, candidates)

    # 6 members (5 champions + trend baseline), subset sizes 2..4.
    assert len(candidates) == 15 + 20 + 15
    assert len(leaderboard) == len(candidates)
    assert (leaderboard["label"] == "ensemble_train").all()
    assert {"robust_20bps", "passes_selection", "candidate_index", "member_labels"}.issubset(leaderboard.columns)


def test_select_ensemble_model_falls_back_to_baseline(tmp_path: Path) -> None:
    fallback = {"params": CANONICAL_TREND_MEMBER, "variant": "long_cash_hysteresis", "selection_status": "selected_v3"}

    selected = select_ensemble_model(pd.DataFrame(), [], fallback)

    assert selected["selection_status"] == "no_robust_upgrade_baseline_retained"
    assert selected["params"] is CANONICAL_TREND_MEMBER


def test_nested_ensemble_walk_forward_runs(tmp_path: Path) -> None:
    config = m2_config(tmp_path)
    prices = create_fixture_prices(config)
    fallback = {"params": CANONICAL_TREND_MEMBER, "variant": "long_cash_hysteresis", "selection_status": "selected_v3"}

    nested = run_nested_ensemble_walk_forward(prices, config, fallback)

    assert not nested["windows"].empty
    assert (pd.to_datetime(nested["windows"]["test_start"]) > pd.to_datetime(nested["windows"]["train_end"])).all()
    assert not nested["oos_returns"].empty
    assert "nested_ensemble_oos_stitched" in set(nested["summary"]["name"])


# --- end-to-end -------------------------------------------------------------------


def test_run_research_with_signal_families(tmp_path: Path) -> None:
    config = m2_config(tmp_path, enable_nested_walk_forward=True)

    result = run_research(config, fixture_data=True)

    assert not result.family_leaderboard.empty
    assert not result.ensemble_leaderboard.empty
    assert not result.v06_comparison.empty
    assert "selected_v6" in set(result.v06_comparison["model"])
    assert not result.v06_cost_sensitivity.empty
    assert "selected_v6" in set(result.final_walk_forward["model"])
    assert "selected_v6" in set(result.significance_results["model"])
    assert "nested_ensemble_oos_stitched" in set(result.significance_results["model"])
    assert not result.nested_ensemble_walk_forward.empty


def test_v6_yaml_parses_signal_families() -> None:
    config = load_research_config(Path("configs/research_v6.yaml"))

    assert config.enable_signal_families
    assert "ts_momentum" in (config.signal_family_grids or {})
    assert config.ensemble_min_members == 3
    assert config.ensemble_max_members == 6
    assert config.ensemble_include_trend_baseline
