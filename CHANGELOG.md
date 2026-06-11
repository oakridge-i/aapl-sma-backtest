# Changelog

All notable project changes are documented here.

## Unreleased (0.6.0 M3 - Exit, Sizing, and Regime Overlays)

### Added

- Composable overlays (`quant_backtest.overlays`) that post-process any base
  strategy's target position, applied in a fixed order:
  - regime scaling: cut exposure (default to 50%) when the market trades
    below its long SMA, boost it (capped at 1) when above;
  - volatility targeting: scale exposure down when realized volatility runs
    above target;
  - ATR trailing stop: force the position flat after price falls a multiple
    of ATR from the post-entry peak; re-arm on a new high or when the base
    signal itself resets. The stop runs last so it sees the final sized
    exposure.
- Overlay search (`quant_backtest.overlay_research`) around the selected v6
  model: a small grid (trailing-stop multiples x regime scaling on/off x
  vol targeting on/off) that always contains the identity combination, so an
  overlay must beat the plain model on the train period (with the 20 bps
  stress) to displace it. Output: `overlay_leaderboard.csv`.
- Overlay re-selection inside every nested-ensemble walk-forward window, so
  the stitched OOS scoreboard covers the full M3 procedure.
- The Deflated Sharpe hurdle for `selected_v6` now counts ensemble and
  overlay candidates together.
- `overlays` section in `configs/research_v6.yaml`.

## Unreleased (0.6.0 M2 - Signal Families and Ensemble)

The first genuine model-search expansion since the SMA crossover, built on
the M1 honest-selection machinery.

### Added

- Five long-only signal families (`quant_backtest.signal_families`), each
  registered with the strategy registry:
  - time-series momentum (3/6/12-month absolute momentum with optional
    hold/cooldown rules);
  - Donchian channel breakout (enter on an N-day high, exit on an M-day low,
    channels lagged one day);
  - ATR-scaled trend strength (continuous exposure from 0 at the SMA to 1 at
    `scale` ATRs above it; close-to-close ATR proxy);
  - dual momentum (long only when the asset beats the market and its own
    zero hurdle over the lookback);
  - 52-week-high proximity with entry/exit hysteresis.
- Equal-vote ensemble (`EnsembleVoteStrategy`): exposure is the mean of
  member target positions. The search is over which families participate
  (subset combinations), not over a dense parameter grid, which keeps the
  selection degrees of freedom small. A frozen canonical trend member (the
  published v0.3 parameterization) can join without adding search space.
- Train-only selection pipeline (`quant_backtest.ensemble_research`):
  per-family grids -> one champion per family -> ensemble candidate subsets
  -> 20 bps stress -> selection with retained-baseline fallback. Outputs:
  `family_leaderboard.csv`, `ensemble_leaderboard.csv`, `v06_comparison.csv`,
  `v06_cost_sensitivity.csv`, `v06_selected_curve.csv`.
- Nested ensemble walk-forward: family champions and the ensemble
  composition are re-selected inside every walk-forward window; the stitched
  OOS series is bootstrapped in `significance_results.csv`
  (`nested_ensemble_oos_stitched`). Outputs:
  `nested_ensemble_walk_forward.csv`, `nested_ensemble_summary.csv`.
- `selected_v6` joins the final-model walk-forward and the significance
  table (bootstrap, Deflated Sharpe against the ensemble leaderboard,
  permutation test).
- `signal_families` and `ensemble` sections in `configs/research_v6.yaml`.

### Changed

- `evaluate_strategy` gained a generic dispatch path for registry families
  (including ones that need the market price series), so new families work
  through every sweep without bespoke branches.

### Findings (preview run, data through 2026-06-09)

- The nested-walk-forward ensemble (composition re-selected every window)
  earned a stitched OOS Sharpe of `0.91` with max drawdown `-13.9%` over
  2018-2026, versus Sharpe `0.96` and drawdown `-38.6%` for AAPL
  buy-and-hold on the same windows. The bootstrap 5-95% Sharpe interval is
  `+0.32` to `+1.46`; the probability of a negative true Sharpe is `0.2%` -
  the first fully positive interval in this project.
- The same procedure applied to the v0.3 trend model alone gives Sharpe
  `0.76` (P(negative) `2.6%`), so the ensemble improves on the single-family
  model.
- A frozen ensemble composition selected once on 2015-2020 does not
  generalize (test Sharpe `0.08`), and individual family champions are
  mostly unprofitable on the test split. The value is in the annual
  re-selection procedure, not in any fixed formula.
- Raw CAGR (`11.7%`) still trails buy-and-hold (`27.7%`): this is
  risk-managed participation with one third of the drawdown, not alpha.

## Unreleased (0.6.0 M1 - Research Foundation)

Engineering and methodology groundwork for the 0.6.0 model-improvement cycle.
No trading logic changed; all 0.5.0 results remain reproducible.

### Added

- Strategy family registry (`quant_backtest.registry`): families are
  registered with their parameter and strategy types, and evaluation
  dispatches through the registry, so new signal families can be added
  without touching the sweep machinery.
- Parallel sweep execution (`quant_backtest.parallel`): grids run on a
  process pool (`compute.workers: auto|N` in YAML), with results bit-for-bit
  identical to serial runs. Engages only for grids of 32+ jobs.
- Nested walk-forward selection (`nested_walk_forward.enabled`): the full
  v0.3 selection pipeline re-runs inside every walk-forward window and the
  selected model is evaluated on that window's out-of-sample slice. The
  stitched OOS series is selection-clean by construction, is bootstrapped in
  `significance_results.csv` (`nested_oos_stitched` row), and becomes the
  primary scoreboard for 0.6 model upgrades. Outputs:
  `nested_walk_forward.csv`, `nested_walk_forward_summary.csv`.
- Probability of Backtest Overfitting (`pbo.enabled`): vectorized CSCV
  (Bailey et al.) over the hysteresis grid's daily-return matrix, with
  candidate capping and configurable block count. Output: `pbo_results.csv`.
- `configs/research_v6.yaml`: v5 settings plus parallel workers, nested
  walk-forward, and PBO enabled.

### Changed

- Split the `experiments.py` monolith (~1700 lines) into focused modules:
  `research_config`, `research_data`, `evaluation`, `selection`, `sweeps`,
  `significance`, `parallel`, `registry`. `experiments.py` remains the
  orchestrator and re-exports the public API, so existing imports keep
  working.
- Package version bumped to `0.6.0.dev0`.

## 0.5.0 - Honest Methodology

This release changes how results are produced, not what the model trades. The
goal is that reported out-of-sample numbers can be trusted.

### Fixed

- Removed test-period leakage from model selection. Previously the v0.3
  allocation leaderboard and the v0.4 capture leaderboard were evaluated on
  the test period and the best test performer was selected, which made the
  reported "out-of-sample" metrics optimistically biased. All candidate
  ranking and selection now happens on the train period only; the test period
  is touched once per final model for reporting.
- Closed-trade statistics now segment trades by exposure episodes (entering
  and leaving a flat position) instead of treating any weight decrease as an
  exit. With volatility sizing the old logic counted daily resizing as
  trades, which corrupted win rates and trade counts.

### Added

- Cash yield: an optional cash proxy (default `BIL` in `research_v5.yaml`)
  earns the uninvested weight's return in the engine, and its annualized
  return is used as the risk-free rate in Sharpe/Sortino. Long/cash models
  are no longer penalized as if cash earned zero.
- `quant_backtest.stats` with three significance tools:
  - circular block bootstrap confidence intervals for CAGR, Sharpe, and max
    drawdown;
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado), using the number and
    dispersion of train-period candidates as the multiple-testing hurdle;
  - a circular-shift permutation test for timing skill that preserves the
    exposure profile and cost structure.
- Walk-forward evaluation of the final fixed models
  (`final_model_walk_forward.csv` and chart), so the selected models are
  judged across every test window, not one favorable split.
- Reproducibility artifacts per run: `data_snapshot.csv` (prices with SHA256
  content hash) and `run_manifest.json` (config, data hash, package versions,
  git commit). `research.py --data-snapshot` reruns on saved data.
- `configs/research_v5.yaml` as the default research configuration.
- New outputs: `significance_results.csv`, `final_model_walk_forward.csv`,
  plus `Significance` and `Final Walk Forward` workbook sheets.
- `pyproject.toml`; the package can be installed with `pip install -e .`.

### Changed

- `research.py` now defaults to `configs/research_v5.yaml`.
- `model_leaderboard.csv` and `allocation_leaderboard.csv` now contain
  train-period metrics (labels `leaderboard_train`, `allocation_train`).
  Comparison tables (`v03_comparison.csv`, `v04_comparison.csv`,
  `benchmark_comparison.csv`) remain test-period.
- The v0.4 capture filter hurdle now compares candidates against the v0.3
  model's train-period metrics instead of test-period metrics.
- `trades` now counts exposure episodes rather than days with turnover.
- Older configs (`research_v2/v3/v4.yaml`) remain runnable, but selection is
  always train-only now; their results will differ from the 0.3/0.4 reports
  because the leak is gone.

### Findings

- With honest selection the framework picks `SMA 5/50` hysteresis (entry
  `0.0%`, exit `-0.5%`, 20-day hold, 5-day cooldown); v0.4 again retains the
  v0.3 model.
- On the test period through `2026-06-09` the selected model earns CAGR
  `5.89%`, Sharpe `0.24` (excess over BIL), max drawdown `-29.10%`, turnover
  `7.21`, versus AAPL buy-and-hold CAGR `17.50%` and Sharpe `0.61`.
- Significance: bootstrap Sharpe interval `-0.57` to `+1.01`, probability of
  negative Sharpe `33.9%`, Deflated Sharpe Ratio `0.54`, permutation p-value
  `0.71`. There is no statistical evidence of timing skill.
- Conclusion: the v0.3/v0.4 reported edge was a selection artifact. The
  honest value of the current rules is drawdown cushioning in bear regimes
  (2022: `-8.6%` vs `-28.5%` for AAPL), not alpha.

## 0.4.0 - Capture-Aware Risk Model

### Added

- `CaptureAwareTrendStrategy`, combining trend allocation, downside risk
  filters, volatility-based sizing, and fallback allocation.
- Capture-aware selection fields, including `capture_spread` and a selection
  score that rewards upside/downside capture separation.
- Downside filters for price below SMA, market below SMA, rolling drawdown,
  realized volatility, and sharp 20-day losses.
- Long-only volatility sizing with exposure capped at `100%`.
- Fallback variants that only allocate to SPY or QQQ in a positive market
  regime, with fallback weights, minimum hold, and cooldown settings.
- Regime classification for bull, correction, bear, recovery, and sideways
  periods.
- Trade log with entry/exit dates, holding days, trade return, MFE, MAE, entry
  regime, entry volatility, and trend spread.
- `configs/research_v4.yaml` as the default research configuration.
- New research outputs:
  - `capture_leaderboard.csv`;
  - `risk_filter_sweep.csv`;
  - `regime_results.csv`;
  - `trade_log.csv`;
  - `benchmark_comparison.csv`;
  - `v04_comparison.csv`;
  - `v04_cost_sensitivity.csv`;
  - `v04_selected_curve.csv`.
- New charts for v0.4 equity/drawdown, entries/exits, capture profile, regime
  performance, turnover vs capture spread, and exposure sizing.

### Changed

- `research.py` now defaults to `configs/research_v4.yaml`.
- The report workbook now includes v0.4 sheets for capture, risk filters,
  regimes, trade logs, benchmarks, and v0.4 cost sensitivity.
- Model selection can explicitly retain the v0.3 baseline when no v0.4
  candidate passes the robustness filters.
- `research_v2.yaml` and `research_v3.yaml` remain runnable for compatibility.

### Findings

- The v0.4 search did not produce a robust upgrade under the configured hard
  filters.
- The best capture-aware candidates improved test CAGR, Sharpe, and drawdown,
  but required high turnover and did not reach the upside capture target.
- The selected output is therefore `no_robust_upgrade_baseline_retained`, using
  the v0.3 `SMA 5/200` long/cash hysteresis model.
- On the test period through `2026-05-19`, the retained model has CAGR `11.00%`,
  Sharpe `0.66`, max drawdown `-23.62%`, turnover `1.68`, upside capture
  `54.48%`, downside capture `53.61%`, and capture spread `0.87` percentage
  points.

## 0.3.0 - Turnover-Aware Trend Allocation

### Added

- `TrendAllocationStrategy` with SMA hysteresis, minimum holding periods, and
  cooldown periods.
- Regime-aware fallback variants for SPY and QQQ.
- Hybrid allocation variants that can split weak-trend exposure between AAPL and
  a market fallback asset.
- Turnover-aware `selection_score` that penalizes excessive trading and unstable
  parameter choices.
- `configs/research_v3.yaml` as the default research configuration.
- New research outputs:
  - `hysteresis_sweep.csv`;
  - `allocation_leaderboard.csv`;
  - `capture_analysis.csv`;
  - `turnover_analysis.csv`;
  - `v03_comparison.csv`;
  - `v03_cost_sensitivity.csv`;
  - `v03_selected_curve.csv`.
- New charts for v0.3 equity/drawdown, turnover vs Sharpe, capture ratios,
  allocation exposure, selected-model cost sensitivity, and entry/exit signals.
- Additional metrics: upside capture, downside capture, missed return while
  underweight AAPL, holding-period stats, trade frequency, and fallback exposure.

### Changed

- `research.py` now defaults to `configs/research_v3.yaml`.
- The model selection process now prefers robust low-turnover candidates over
  marginally higher Sharpe candidates.
- Yahoo Finance timezone cache is stored under the project-local `.cache/`
  directory to avoid Windows cache permission failures.
- Market-regime window settings from the YAML config are now passed through to
  the fallback and hybrid allocation evaluators.
- `research_v2.yaml` remains runnable for compatibility.

### Findings

- The selected v0.3 model is `SMA 5/200` with `1%` entry threshold, `-1%` exit
  threshold, 10-day minimum hold, and 5-day cooldown.
- On the test period, the selected model improves versus v2 `SMA 5/50`
  long/cash: higher CAGR, higher Sharpe, lower turnover, lower cost drag, and
  similar drawdown control.
- The aggressive SPY/QQQ fallback variants still produce higher raw CAGR, but
  they fail the current turnover filter.
- The selected v0.3 model remains below AAPL buy-and-hold by raw CAGR, so it is
  a risk-managed trend baseline rather than a proven alpha model.

## 0.2.0 - Robustness Research Framework

### Added

- Research CLI: `research.py --config configs/research_v2.yaml`.
- Config-driven research workflow for repeatable experiments.
- Transaction cost sensitivity across `0`, `5`, `10`, `20`, and `50` bps.
- SMA parameter sweep with Sharpe/CAGR heatmaps and local stability fields.
- Train/test validation and walk-forward testing.
- Multi-asset validation across AAPL, MSFT, NVDA, AMZN, META, GOOGL, SPY, and QQQ.
- Long-only return enhancement variants:
  - long/cash baseline;
  - SPY fallback;
  - QQQ fallback;
  - partial exposure;
  - SMA spread threshold;
  - 3-month momentum filter;
  - 6-month momentum filter.
- Expanded metrics: Sortino, Calmar, exposure, turnover, trade distribution,
  cost drag, excess CAGR, and drawdown improvement versus benchmark.
- Research outputs:
  - `research_report.xlsx`;
  - cost sensitivity table and chart;
  - parameter sweep table and heatmaps;
  - train/test and walk-forward result tables;
  - multi-asset comparison;
  - model leaderboard.
- GitHub Actions workflow for running the test suite.

### Changed

- Split the original single-function backtest into strategy, cost, engine,
  experiment, and reporting modules.
- Kept the original `main.py` command compatible with the first version.
- Updated generated sample outputs to include the expanded metric set.
- Improved project documentation and result interpretation.

### Findings

- The original `SMA 20/100` strategy was profitable, but underperformed AAPL
  buy-and-hold by a wide margin.
- The faster `SMA 5/50` region improved the full-sample strategy profile:
  higher CAGR, better Sharpe, and lower max drawdown than `SMA 20/100`.
- Out-of-sample results are still weaker than buy-and-hold by CAGR and Sharpe.
- Fallback variants using SPY or QQQ improve raw return but currently trade too
  much to pass the robustness filter.

## 0.1.0 - Initial AAPL SMA Backtest

### Added

- AAPL daily price download via Yahoo Finance.
- Long-only SMA crossover strategy.
- Basic backtest metrics and buy-and-hold comparison.
- CSV, PNG, and Excel report outputs.
- Unit tests for signal shifting, transaction costs, metrics, and smoke runs.
