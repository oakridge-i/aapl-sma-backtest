"""Independent economic expectations, not copies of the engine algorithm."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_backtest.continuous import continuous_oos
from quant_backtest.costs import BpsCost
from quant_backtest.data_quality import price_returns
from quant_backtest.engine import EngineConfig, run_weight_backtest
from quant_backtest.ensemble_research import run_nested_ensemble_walk_forward
from quant_backtest.strategies import SmaParameters
from test_m2_signal_families import m2_config


def test_flat_cash_without_orders_preserves_every_dollar():
    dates = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.DataFrame({"AAPL": 0.0}, index=dates)
    result = run_weight_backtest(returns, returns.copy(), EngineConfig(1000, BpsCost(50)))
    assert result.curve.strategy_equity.tolist() == [1000] * 8
    assert result.curve.transaction_cost.sum() == 0
    assert result.curve.turnover.sum() == 0


def test_entry_and_exit_fees_land_on_execution_closes():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": [100, 110, 121, 133.1, 133.1]}, index=dates)
    target = pd.DataFrame({"AAPL": [1, 1, 0, 0, 0]}, index=dates)
    result = run_weight_backtest(price_returns(prices), target, EngineConfig(100, BpsCost(100)))
    # Buy at the second close, earn two 10% moves, then sell; each side costs 1%.
    assert result.curve.strategy_equity.tolist() == pytest.approx(
        [100, 100 / 1.01, 100 / 1.01 * 1.1, 100 / 1.01 * 1.21 * .99, 100 / 1.01 * 1.21 * .99])
    assert result.curve.transaction_cost.tolist() == pytest.approx([0, .01 / 1.01, 0, .011, 0])
    assert result.closing_weights.AAPL.tolist() == [0, 1, 1, 0, 0]


def test_year_boundary_changes_asset_without_restarting_the_account(tmp_path):
    dates = pd.to_datetime(["2023-12-27", "2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"AAPL": [100, 100, 110, 110, 110], "SPY": [100, 100, 100, 100, 120]}, index=dates)
    before = pd.DataFrame({"AAPL": [1, 1, 0], "SPY": [0, 0, 1]}, index=dates[:3])
    after = pd.DataFrame({"SPY": 1.0}, index=dates[3:])
    windows = pd.DataFrame({"test_start": ["2023-12-27", "2024-01-02"], "test_end": ["2023-12-29", "2024-01-03"]})
    config = replace(m2_config(tmp_path), initial_capital=1000, cash_proxy_ticker=None)
    result = continuous_oos(prices, config, windows, [before, after], "manual_boundary")
    # First buy costs 10 bps; replacing A by B sells the old NAV and buys the
    # remaining NAV, so Vnew=Vold*(1-.001)/(1+.001), not two independent entries.
    expected = 1000 / 1.001 * 1.1 * .999 / 1.001 * 1.2
    assert result["curve"].strategy_equity.iloc[-1] == pytest.approx(expected)
    assert result["curve"].transaction_cost.loc[dates[3]] == pytest.approx(.002 / 1.001)
    assert result["curve"].earning_weight_SPY.loc[dates[3]] == 0
    assert result["curve"].earning_weight_SPY.loc[dates[4]] == 1
    assert result["curve"].turnover.loc[dates[4]] == pytest.approx(0)


def small_real_selector_config(tmp_path):
    return replace(m2_config(tmp_path), cash_proxy_ticker=None, enable_overlays=False,
                   parallel_workers=0, ensemble_min_members=2, ensemble_max_members=2,
                   signal_family_grids={"ts_momentum": {"lookbacks": [21, 63]},
                                        "donchian": {"entry_windows": []},
                                        "atr_trend": {"sma_windows": []},
                                        "dual_momentum": {"lookbacks": []},
                                        "high_52w": {"entry_thresholds": []}})


def test_real_nested_selection_and_earlier_orders_ignore_future_prices(tmp_path):
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    steps = np.arange(len(dates))
    prices = pd.DataFrame({"AAPL": 100 * np.exp(.0007 * steps + .035 * np.sin(steps / 15)),
                           "SPY": 100 * np.exp(.0004 * steps)}, index=dates)
    config = small_real_selector_config(tmp_path)
    cutoff = pd.Timestamp("2021-06-01")
    changed = prices.copy()
    changed.loc[cutoff:, "AAPL"] *= np.linspace(.2, 4, len(changed.loc[cutoff:]))
    original = run_nested_ensemble_walk_forward(prices, config)
    perturbed = run_nested_ensemble_walk_forward(changed, config)
    assert len(original["windows"]) == 2
    columns = ["train_start", "train_end", "selected_label", "selected_variant", "selection_status"]
    pd.testing.assert_frame_equal(original["windows"][columns], perturbed["windows"][columns])
    pd.testing.assert_frame_equal(original["target_weights"].loc[:"2021-05-31"], perturbed["target_weights"].loc[:"2021-05-31"])
    pd.testing.assert_frame_equal(original["curve"].loc[:"2021-05-31"], perturbed["curve"].loc[:"2021-05-31"])


def test_real_no_eligible_ensemble_uses_fixed_fallback(tmp_path):
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    prices = pd.DataFrame({"AAPL": 100.0, "SPY": 100.0}, index=dates)
    config = small_real_selector_config(tmp_path)
    result = run_nested_ensemble_walk_forward(prices, config, {"params": SmaParameters(1, 2), "variant": "long_cash"})
    assert result["windows"].selection_status.str.startswith("no_robust_").all()
    assert result["windows"].candidates_evaluated.gt(0).all()
    assert result["windows"].selected_label.eq("sma_20_100").all()
    assert result["curve"].strategy_equity.eq(config.initial_capital).all()
