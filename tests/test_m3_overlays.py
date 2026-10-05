from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_backtest.ensemble_research import run_nested_ensemble_walk_forward
from quant_backtest.evaluation import evaluate_strategy
from quant_backtest.experiments import run_research
from quant_backtest.overlay_research import (
    overlay_parameter_grid,
    run_overlay_leaderboard,
    select_overlay_model,
)
from quant_backtest.overlays import (
    OverlayParameters,
    OverlayStrategy,
    RegimeScalingParameters,
    TrailingStopParameters,
    VolTargetParameters,
)
from quant_backtest.registry import family_for_params
from quant_backtest.research_config import load_research_config
from quant_backtest.research_data import create_fixture_prices
from quant_backtest.signal_families import TimeSeriesMomentumParameters
from quant_backtest.strategies import TrendAllocationParameters

from test_m2_signal_families import m2_config


def m3_config(tmp_path: Path, **overrides):
    base = m2_config(
        tmp_path,
        enable_overlays=True,
        overlay_grids={
            "trailing_stop": {"atr_windows": [20], "multiples": [4.0]},
            "regime_scaling": {"enabled": True, "sma_window": 100},
            "vol_target": {"window": 20, "targets": [0.20]},
        },
    )
    return dataclasses.replace(base, **overrides) if overrides else base


ALWAYS_LONG_BASE = TimeSeriesMomentumParameters(lookback_days=21)


def _rally_then_crash(n_up: int = 250, n_down: int = 60) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n_up + n_down, freq="B")
    up = np.linspace(100.0, 200.0, n_up)
    down = np.linspace(200.0, 120.0, n_down)
    return pd.Series(np.concatenate([up, down]), index=dates)


# --- trailing stop ---------------------------------------------------------------


def test_trailing_stop_exits_before_the_base_signal() -> None:
    price = _rally_then_crash()
    base = OverlayStrategy(OverlayParameters(base=ALWAYS_LONG_BASE)).generate(price)
    stopped = OverlayStrategy(
        OverlayParameters(base=ALWAYS_LONG_BASE, trailing_stop=TrailingStopParameters(atr_window=20, multiple=4.0))
    ).generate(price)

    crash_start = 250
    base_flat = base.target_position.iloc[crash_start:].eq(0.0).idxmax()
    stop_flat = stopped.target_position.iloc[crash_start:].eq(0.0).idxmax()

    assert stopped.target_position.iloc[200] == 1.0  # long during the rally
    assert stopped.target_position.iloc[-1] == 0.0  # flat after the crash
    # The stop reacts to the drawdown well before the slow momentum signal.
    assert stop_flat < base_flat


def test_trailing_stop_reenters_on_new_high() -> None:
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rally = np.linspace(100.0, 150.0, 150)
    dip = np.linspace(150.0, 120.0, 50)
    recovery = np.linspace(120.0, 170.0, 100)
    price = pd.Series(np.concatenate([rally, dip, recovery]), index=dates)
    signals = OverlayStrategy(
        OverlayParameters(base=ALWAYS_LONG_BASE, trailing_stop=TrailingStopParameters(atr_window=20, multiple=3.0))
    ).generate(price)

    assert signals.target_position.iloc[195] == 0.0  # stopped out in the dip
    assert signals.target_position.iloc[-1] == 1.0  # back in after a new high


# --- regime scaling ----------------------------------------------------------------


def test_regime_scaling_cuts_in_bear_and_caps_boost() -> None:
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    price = pd.Series(np.linspace(100.0, 160.0, 400), index=dates)
    bull_market = pd.Series(np.linspace(100.0, 200.0, 400), index=dates)
    bear_market = pd.Series(np.linspace(200.0, 100.0, 400), index=dates)
    params = OverlayParameters(
        base=ALWAYS_LONG_BASE,
        regime_scaling=RegimeScalingParameters(sma_window=100, bear_cut=0.5, bull_boost=1.25),
    )

    bull = OverlayStrategy(params).generate(price, market_price=bull_market)
    bear = OverlayStrategy(params).generate(price, market_price=bear_market)

    assert bull.target_position.iloc[-1] == 1.0  # boost capped at 1
    assert bear.target_position.iloc[-1] == 0.5  # bear cut applied


def test_regime_scaling_without_market_is_identity() -> None:
    price = _rally_then_crash()
    params = OverlayParameters(base=ALWAYS_LONG_BASE, regime_scaling=RegimeScalingParameters())

    with_market_missing = OverlayStrategy(params).generate(price, market_price=None)
    identity = OverlayStrategy(OverlayParameters(base=ALWAYS_LONG_BASE)).generate(price)

    pd.testing.assert_series_equal(with_market_missing.target_position, identity.target_position)


# --- vol target ----------------------------------------------------------------------


def test_vol_target_scales_down_in_high_volatility() -> None:
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    calm = rng.normal(0.0008, 0.005, size=150)
    wild = rng.normal(0.0008, 0.04, size=150)
    price = pd.Series(100 * np.cumprod(1 + np.concatenate([calm, wild])), index=dates)
    params = OverlayParameters(base=ALWAYS_LONG_BASE, vol_target=VolTargetParameters(window=20, target=0.20))

    signals = OverlayStrategy(params).generate(price)

    calm_exposure = signals.target_position.iloc[100:140].mean()
    wild_exposure = signals.target_position.iloc[200:290].mean()
    assert wild_exposure < calm_exposure
    assert signals.target_position.max() <= 1.0


# --- composition and validation ---------------------------------------------------------


def test_identity_overlay_matches_base() -> None:
    price = _rally_then_crash()
    from quant_backtest.signal_families import TimeSeriesMomentumStrategy

    base = TimeSeriesMomentumStrategy(ALWAYS_LONG_BASE).generate(price)
    identity = OverlayStrategy(OverlayParameters(base=ALWAYS_LONG_BASE)).generate(price)

    pd.testing.assert_series_equal(identity.target_position, base.target_position, check_names=False)


def test_overlay_registry_and_evaluation(tmp_path: Path) -> None:
    config = m3_config(tmp_path)
    prices = create_fixture_prices(config)
    params = OverlayParameters(
        base=TrendAllocationParameters(5, 20),
        trailing_stop=TrailingStopParameters(),
        regime_scaling=RegimeScalingParameters(sma_window=100),
    )

    family = family_for_params(params)
    assert family.name == "overlay"
    assert family.needs_market_price

    result = evaluate_strategy(
        prices=prices,
        ticker="AAPL",
        params=params,
        variant="overlay",
        cost_bps=10.0,
        initial_capital=10_000.0,
        label="test",
        cash_proxy="BIL",
    )
    assert result["weights"]["AAPL"].max() <= 1.0
    assert result["row"]["exposure"] > 0


def test_overlay_validation_rejects_nesting_and_bad_values() -> None:
    with pytest.raises(ValueError, match="nested"):
        OverlayStrategy(OverlayParameters(base=OverlayParameters(base=ALWAYS_LONG_BASE)))
    with pytest.raises(ValueError):
        OverlayStrategy(OverlayParameters(base=ALWAYS_LONG_BASE, trailing_stop=TrailingStopParameters(multiple=0.0)))
    with pytest.raises(ValueError):
        OverlayStrategy(
            OverlayParameters(base=ALWAYS_LONG_BASE, regime_scaling=RegimeScalingParameters(bull_boost=0.5))
        )


# --- selection pipeline ----------------------------------------------------------------


def test_overlay_grid_contains_identity(tmp_path: Path) -> None:
    config = m3_config(tmp_path)

    grid = overlay_parameter_grid(config, ALWAYS_LONG_BASE)

    assert any(candidate.is_identity() for candidate in grid)
    # (1 + 1 trailing) x (1 + regime) x (1 + vol) = 8 combinations.
    assert len(grid) == 8
    assert all(candidate.base is ALWAYS_LONG_BASE for candidate in grid)


def test_overlay_grid_unwraps_overlay_base(tmp_path: Path) -> None:
    config = m3_config(tmp_path)
    wrapped = OverlayParameters(base=ALWAYS_LONG_BASE, trailing_stop=TrailingStopParameters())

    grid = overlay_parameter_grid(config, wrapped)

    assert all(not isinstance(candidate.base, OverlayParameters) for candidate in grid)


def test_overlay_leaderboard_and_selection(tmp_path: Path) -> None:
    config = m3_config(tmp_path)
    prices = create_fixture_prices(config)
    train_prices = prices.loc[config.train_start : config.train_end]
    base_model = {"params": ALWAYS_LONG_BASE, "variant": "ts_momentum", "selection_status": "selected_v6"}

    candidates = overlay_parameter_grid(config, ALWAYS_LONG_BASE)
    leaderboard = run_overlay_leaderboard(train_prices, config, candidates)
    selected = select_overlay_model(leaderboard, candidates, base_model)

    assert len(leaderboard) == len(candidates)
    assert (leaderboard["label"] == "overlay_train").all()
    assert {"is_identity", "robust_20bps", "passes_selection"}.issubset(leaderboard.columns)
    assert selected["selection_status"] in {"selected_v6", "selected_v6_overlay"}
    if selected["selection_status"] == "selected_v6_overlay":
        assert isinstance(selected["params"], OverlayParameters)
        assert not selected["params"].is_identity()
    else:
        assert selected["params"] is ALWAYS_LONG_BASE


def test_select_overlay_model_keeps_base_when_nothing_passes() -> None:
    base_model = {"params": ALWAYS_LONG_BASE, "variant": "ts_momentum", "selection_status": "selected_v6"}

    assert select_overlay_model(pd.DataFrame(), [], base_model) is base_model


# --- end-to-end ---------------------------------------------------------------------------


def test_nested_ensemble_with_overlays(tmp_path: Path) -> None:
    config = m3_config(tmp_path)
    prices = create_fixture_prices(config)
    fallback = {"params": ALWAYS_LONG_BASE, "variant": "ts_momentum", "selection_status": "selected_v3"}

    nested = run_nested_ensemble_walk_forward(prices, config, fallback)

    assert not nested["windows"].empty
    assert not nested["oos_returns"].empty


def test_run_research_with_overlays(tmp_path: Path) -> None:
    config = m3_config(tmp_path)

    result = run_research(config, fixture_data=True)

    assert not result.overlay_leaderboard.empty
    assert "selected_v6" in set(result.v06_comparison["model"])
    v6_row = result.v06_comparison[result.v06_comparison["model"] == "selected_v6"].iloc[0]
    assert v6_row["selection_status"] in {"selected_v6", "selected_v6_overlay", "no_robust_upgrade_baseline_retained"}
    assert "selected_v6" in set(result.significance_results["model"])


def test_v6_yaml_parses_overlays() -> None:
    config = load_research_config(Path("configs/research_v6.yaml"))

    assert config.enable_overlays
    assert "trailing_stop" in (config.overlay_grids or {})
    assert config.overlay_grids["trailing_stop"]["multiples"] == [3.0, 4.0, 5.0]
