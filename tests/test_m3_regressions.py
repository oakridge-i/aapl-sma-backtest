from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quant_backtest.evaluation import evaluate_strategy
from quant_backtest.overlay_research import overlay_parameter_grid, select_overlay_model
from quant_backtest.overlays import OverlayParameters, RegimeScalingParameters, TrailingStopParameters
from quant_backtest.signal_families import DualMomentumParameters, TimeSeriesMomentumParameters
from quant_backtest.strategies import SmaParameters, TrendAllocationParameters

from test_m3_overlays import m3_config


def evaluate(prices, params, variant="long_cash", **kwargs):
    return evaluate_strategy(
        prices, "AAPL", params, variant, 10.0, 10_000.0, "regression",
        market_regime_short_window=5, market_regime_long_window=20, **kwargs,
    )


@pytest.fixture
def prices():
    dates = pd.bdate_range("2020-01-01", periods=330)
    return pd.DataFrame(
        {
            "AAPL": np.concatenate([np.linspace(100, 160, 150), np.linspace(160, 100, 80),
                                     np.linspace(100, 180, 100)]),
            "SPY": np.linspace(100, 160, 330),
            "QQQ": np.linspace(100, 110, 330),
        }, index=dates,
    )


@pytest.mark.parametrize("variant", [
    "fallback_spy", "fallback_qqq", "long_spy_regime", "long_qqq_regime",
    "hybrid_spy_regime", "hybrid_qqq_regime",
])
def test_identity_preserves_full_base_portfolio(prices, variant):
    base = TrendAllocationParameters(5, 20, entry_threshold=0.02, exit_threshold=-0.01)
    plain = evaluate(prices, base, variant)
    wrapped = evaluate(prices, OverlayParameters(base, base_variant=variant), "overlay")
    pd.testing.assert_frame_equal(plain["weights"], wrapped["weights"])
    pd.testing.assert_frame_equal(plain["curve"], wrapped["curve"])
    assert plain["weights"].drop(columns="AAPL").to_numpy().sum() > 0


@pytest.mark.parametrize("variant", ["fallback_spy", "fallback_qqq"])
def test_identity_preserves_sma_fallback(prices, variant):
    base = SmaParameters(5, 20)
    plain = evaluate(prices, base, variant)
    wrapped = evaluate(prices, OverlayParameters(base, base_variant=variant), "overlay")
    pd.testing.assert_frame_equal(plain["curve"], wrapped["curve"])


def test_identity_and_active_overlay_use_base_market_ticker():
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({
        "AAPL": np.linspace(100, 120, 100), "SPY": np.linspace(100, 140, 100),
        "QQQ": np.linspace(100, 105, 100),
    }, index=dates)
    base = DualMomentumParameters(21, "QQQ")
    plain = evaluate(prices, base, "dual_momentum")
    identity = evaluate(prices, OverlayParameters(base), "overlay")
    # A rising price never triggers this stop: the active wrapper must also
    # preserve QQQ as the base signal's comparison market.
    active = evaluate(prices, OverlayParameters(base, trailing_stop=TrailingStopParameters()), "overlay")
    assert plain["weights"]["AAPL"].iloc[-1] == 1.0
    pd.testing.assert_frame_equal(identity["weights"], plain["weights"])
    pd.testing.assert_frame_equal(active["weights"], plain["weights"])


def test_overlays_keep_fallback_holdings_and_release_to_cash(prices):
    base = TrendAllocationParameters(5, 20)
    plain = evaluate(prices, base, "fallback_spy")
    wrapped = evaluate(prices, OverlayParameters(
        base, base_variant="fallback_spy", trailing_stop=TrailingStopParameters(5, 1.0),
    ), "overlay")
    pd.testing.assert_series_equal(wrapped["weights"]["SPY"], plain["weights"]["SPY"])
    assert (wrapped["weights"]["AAPL"] < plain["weights"]["AAPL"]).any()
    assert wrapped["weights"].sum(axis=1).max() <= 1.0


def test_regime_boost_cannot_leverage_hybrid_portfolio(prices):
    base = TrendAllocationParameters(5, 20, entry_threshold=0.5, exit_threshold=-0.1)
    plain = evaluate(prices, base, "hybrid_spy_regime")
    assert (plain["weights"]["AAPL"] == 0.5).any()
    wrapped = evaluate(prices, OverlayParameters(
        base, base_variant="hybrid_spy_regime",
        regime_scaling=RegimeScalingParameters(20, 0.5, 1.25),
    ), "overlay")
    pd.testing.assert_series_equal(wrapped["weights"]["SPY"], plain["weights"]["SPY"])
    assert wrapped["weights"].sum(axis=1).max() <= 1.0


def test_grid_preserves_base_variant_on_wrap_and_unwrap(tmp_path):
    config = m3_config(tmp_path)
    base = TrendAllocationParameters(5, 20)
    grid = overlay_parameter_grid(config, base, "fallback_qqq")
    assert all(p.base_variant == "fallback_qqq" for p in grid)
    unwrapped = overlay_parameter_grid(config, grid[-1])
    assert all(p.base is base and p.base_variant == "fallback_qqq" for p in unwrapped)
    assert grid[0].label() != OverlayParameters(base).label()


@pytest.mark.parametrize("identity_score,identity_passes,overlay_score,expected_overlay", [
    (2.0, False, 1.0, False), (2.0, True, 1.0, False),
    (1.0, True, 1.0, False), (1.0, False, 1.0, False),
    (1.0, False, 2.0, True), (1.0, True, 2.0, True),
    (np.nan, False, 2.0, False),
])
def test_selection_requires_strict_identity_improvement(
    identity_score, identity_passes, overlay_score, expected_overlay,
):
    base = TimeSeriesMomentumParameters(21)
    model = {"params": base, "variant": "ts_momentum", "selection_status": "selected_v6"}
    candidates = [OverlayParameters(base), OverlayParameters(base, regime_scaling=RegimeScalingParameters())]
    board = pd.DataFrame([
        {"candidate_index": 1, "is_identity": False, "passes_selection": True,
         "selection_score": overlay_score},
        {"candidate_index": 0, "is_identity": True, "passes_selection": identity_passes,
         "selection_score": identity_score},
    ])
    selected = select_overlay_model(board, candidates, model)
    assert (selected is not model) == expected_overlay


def warm_prices():
    dates = pd.bdate_range("2020-01-01", periods=330)
    return pd.DataFrame({"AAPL": np.linspace(100, 180, 330), "SPY": np.linspace(200, 100, 330)}, index=dates)


def warm_model():
    return OverlayParameters(TimeSeriesMomentumParameters(21), regime_scaling=RegimeScalingParameters(200))


def test_history_warms_short_test_window_without_training_pnl():
    prices = warm_prices()
    start = prices.index[250]
    result = evaluate(prices, warm_model(), "overlay", evaluation_start=start)
    curve = result["curve"]
    pd.testing.assert_index_equal(curve.index, prices.index[250:])
    assert curve["signal"].eq(0.5).all()
    # The first test close seeds execution; all training P&L is excluded.
    assert curve["strategy_equity"].iloc[0] == 10_000.0
    assert curve["position"].iloc[0] == 0.0
    assert curve["position"].iloc[1] == 0.5
    assert curve["transaction_cost"].iloc[1] == pytest.approx(0.0005)
    expected = 10_000 * (1 + curve["strategy_return"]).cumprod()
    np.testing.assert_allclose(curve["strategy_equity"], expected)
    assert curve["buy_hold_equity"].iloc[0] == 10_000.0


def test_warmed_signals_do_not_use_future_prices():
    prices = warm_prices()
    cutoff = prices.index[290]
    original = evaluate(prices, warm_model(), "overlay", evaluation_start=prices.index[250])
    changed = prices.copy()
    changed.loc[cutoff:, "SPY"] *= 10
    changed.loc[cutoff:, "AAPL"] *= 0.5
    perturbed = evaluate(changed, warm_model(), "overlay", evaluation_start=prices.index[250])
    pd.testing.assert_frame_equal(original["curve"].loc[:prices.index[289]], perturbed["curve"].loc[:prices.index[289]])


def test_trailing_stop_state_survives_test_boundary():
    dates = pd.bdate_range("2020-01-01", periods=330)
    asset = np.concatenate([np.linspace(100, 200, 250), np.linspace(190, 180, 80)])
    prices = pd.DataFrame({"AAPL": asset, "SPY": np.linspace(100, 120, 330)}, index=dates)
    # Momentum is still positive at the split, but the close-based stop has
    # already fired in the history and must not reset to the new test price.
    params = OverlayParameters(TimeSeriesMomentumParameters(200), trailing_stop=TrailingStopParameters(20, 2))
    result = evaluate(prices, params, "overlay", evaluation_start=dates[251])
    assert result["curve"]["signal"].eq(0.0).all()


def test_nested_overlay_selection_uses_train_only_and_evaluation_uses_history(tmp_path, monkeypatch):
    import quant_backtest.ensemble_research as ensemble
    import quant_backtest.overlay_research as overlays

    prices = warm_prices()
    start, end = prices.index[265], prices.index[-1]
    config = dataclasses.replace(m3_config(tmp_path), cash_proxy_ticker=None)
    window = (1, prices.index[0], prices.index[264], start, end)
    monkeypatch.setattr(ensemble, "walk_forward_windows", lambda *_: [window])
    monkeypatch.setattr(ensemble, "run_family_sweep", lambda *a, **k: (pd.DataFrame(), []))
    monkeypatch.setattr(ensemble, "run_ensemble_leaderboard", lambda *a: pd.DataFrame())
    model = warm_model()
    monkeypatch.setattr(overlays, "overlay_parameter_grid", lambda *a: [OverlayParameters(model.base), model])

    def leaderboard(train_prices, *_):
        assert train_prices.index.max() == prices.index[264]
        return pd.DataFrame([
            {"candidate_index": 0, "is_identity": True, "passes_selection": True, "selection_score": 1.0},
            {"candidate_index": 1, "is_identity": False, "passes_selection": True, "selection_score": 2.0},
        ])

    monkeypatch.setattr(overlays, "run_overlay_leaderboard", leaderboard)
    result = ensemble.run_nested_ensemble_walk_forward(
        prices, config, {"params": model.base, "variant": "ts_momentum"},
    )
    expected = evaluate(prices, model, "overlay", evaluation_start=start)
    pd.testing.assert_series_equal(result["oos_returns"], expected["curve"]["strategy_return"])
    assert result["windows"]["candidates_evaluated"].iloc[0] == 2


def test_v6_reports_share_the_same_warmed_evaluation(tmp_path, monkeypatch):
    from quant_backtest.ensemble_research import run_v06_comparison, run_v06_cost_sensitivity
    from quant_backtest.significance import run_significance_analysis
    from quant_backtest.sweeps import run_final_model_walk_forward
    import quant_backtest.sweeps as sweeps

    prices = warm_prices()
    start, end = prices.index[250], prices.index[-1]
    config = dataclasses.replace(m3_config(tmp_path), test_start=str(start.date()), test_end=str(end.date()),
                                 cash_proxy_ticker=None, cost_bps=[10.0])
    params = warm_model()
    model = {"params": params, "variant": "overlay", "selection_status": "selected_v6_overlay"}
    base = {"params": params.base, "variant": "ts_momentum", "selection_status": "selected_v3"}
    comparison, curve = run_v06_comparison(prices, config, base, {}, model)
    costs = run_v06_cost_sensitivity(prices, config, model)
    sig = run_significance_analysis(prices, config, base, None, pd.DataFrame(), pd.DataFrame(),
                                   extra_models=[("selected_v6", params, "overlay", pd.DataFrame())])
    monkeypatch.setattr(sweeps, "walk_forward_windows", lambda *_: [(1, prices.index[0], prices.index[249], start, end)])
    wf = run_final_model_walk_forward(prices, config, [("selected_v6", params, "overlay")])
    row = comparison.loc[comparison.model.eq("selected_v6")].iloc[0]
    for metric in ["cagr", "sharpe", "max_drawdown"]:
        assert row[metric] == pytest.approx(costs.iloc[0][metric])
        assert row[metric] == pytest.approx(wf.iloc[0][metric])
        assert row[metric] == pytest.approx(sig.loc[sig.model.eq("selected_v6"), "observed_" + metric].iloc[0])
    assert curve["signal"].eq(0.5).all()
    assert sig.loc[sig.model.eq("selected_v6"), "n_obs"].iloc[0] == len(prices.loc[start:])
