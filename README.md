# AAPL SMA Robustness Research

Educational quantitative analysis project for Apple Inc. (`AAPL`). It downloads
daily market data, tests a long-only SMA 20 / SMA 100 crossover strategy, applies
10 bps transaction costs on position changes, and compares the strategy with a
buy-and-hold benchmark.

The current research version adds capture-aware risk controls: downside filters,
volatility-based position sizing, fallback stress tests, regime diagnostics,
trade logs, and a model selection rule that can retain the previous baseline
when no robust upgrade is found.

This project is for research and education only. It is not investment advice.

## Current Status

The project started as a single AAPL SMA crossover backtest. It now includes a
research workflow that stress-tests the strategy across transaction costs,
parameter choices, train/test periods, walk-forward windows, and a small
multi-asset universe.

The latest run keeps the v0.3 low-turnover model as the selected baseline. v0.4
candidates improved some raw risk metrics, but the best candidates failed the
capture and turnover filters. The project is currently more useful as a
risk-management research framework than as a finished trading model.

## Project Contents

- `main.py` - CLI entrypoint for downloading data and running the backtest.
- `research.py` - CLI entrypoint for the robustness research workflow.
- `configs/research_v4.yaml` - default capture-aware research configuration.
- `configs/research_v3.yaml` - turnover-aware configuration, kept compatible.
- `configs/research_v2.yaml` - previous robustness configuration, kept compatible.
- `src/quant_backtest/` - data loading, strategy, metrics, and charting code.
- `scripts/create_visual_report.py` - creates the model forecast PNG and Excel-ready CSV.
- `scripts/create_excel_report.py` - creates the Excel dashboard from generated CSV files.
- `outputs/` - generated reports and sample output from the AAPL run.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python main.py --ticker AAPL --start 2015-01-01 --short-window 20 --long-window 100 --cost-bps 10 --initial-capital 10000
```

The command writes:

- `outputs/equity_curve.csv`
- `outputs/metrics.csv`
- `outputs/aapl_sma_backtest.png`

## Run Robustness Research

Full run using Yahoo Finance data:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v4.yaml
```

Legacy v0.3 run:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v3.yaml
```

Offline smoke run using deterministic fixture data:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v4.yaml --no-download --output-dir outputs_fixture_v4
```

```powershell
.\.venv\Scripts\python research.py --config configs\research_v3.yaml --no-download --output-dir outputs_fixture_v3
```

The research command writes CSV tables, charts, and an Excel workbook:

- `outputs/base_backtest.csv`
- `outputs/cost_sensitivity.csv`
- `outputs/parameter_sweep.csv`
- `outputs/train_test_results.csv`
- `outputs/walk_forward_results.csv`
- `outputs/multi_asset_results.csv`
- `outputs/model_leaderboard.csv`
- `outputs/hysteresis_sweep.csv`
- `outputs/allocation_leaderboard.csv`
- `outputs/capture_analysis.csv`
- `outputs/turnover_analysis.csv`
- `outputs/v03_comparison.csv`
- `outputs/v03_selected_curve.csv`
- `outputs/capture_leaderboard.csv`
- `outputs/risk_filter_sweep.csv`
- `outputs/regime_results.csv`
- `outputs/trade_log.csv`
- `outputs/benchmark_comparison.csv`
- `outputs/v04_comparison.csv`
- `outputs/v04_selected_curve.csv`
- `outputs/research_report.xlsx`
- PNG charts for baseline, costs, heatmaps, train/test, multi-asset,
  leaderboard, v0.3 equity/drawdown, turnover, capture ratios, allocation
  exposure, cost sensitivity, entry/exit signals, v0.4 capture diagnostics,
  regime performance, and exposure sizing.

## Research Notes

The latest research run found:

- the original `SMA 20/100` strategy is profitable but materially lags AAPL
  buy-and-hold;
- the v2 parameter sweep selected a faster `SMA 5/50` region, but its turnover
  was high out of sample;
- the v0.3 selected model is `SMA 5/200` with hysteresis, a 10-day minimum hold,
  and a 5-day cooldown;
- the selected v0.3 model lowers annualized turnover from `8.40` to `1.68` on
  the test period;
- `outputs/v03_entry_exit_signals.png` shows the selected model's entries and
  exits on top of the AAPL price/SMA chart;
- SPY/QQQ fallback variants still improve raw return, but their turnover remains
  too high for the current robustness filter.
- v0.4 did not select a new model. The capture-aware candidates with better
  Sharpe and drawdown still failed the turnover and upside-capture filters, so
  the framework retained the v0.3 `SMA 5/200` long/cash hysteresis model.

See `docs/research_summary.md` for a fuller interpretation.

## Create Reports

```powershell
.\.venv\Scripts\python scripts\create_visual_report.py
.\.venv\Scripts\python scripts\create_excel_report.py
```

The report scripts write:

- `outputs/aapl_model_forecast_visual.png`
- `outputs/excel_model_data.csv`
- `outputs/aapl_model_forecast_report.xlsx`

## Test

```powershell
.\.venv\Scripts\python -m pytest
```

## Latest Sample Result

For the AAPL run from `2015-01-02` through `2026-05-19`, starting with
`$10,000`:

- SMA strategy ending equity: `$34,832.56`
- SMA strategy total return: `248.33%`
- Buy-and-hold ending equity: `$123,579.06`
- Buy-and-hold total return: `1135.79%`

The strategy made money historically, but it did not outperform buy-and-hold for
this AAPL period.

## Notes

- The strategy uses adjusted close when available.
- Signals are shifted by one day to avoid lookahead bias.
- The `--end` argument is exclusive because Yahoo Finance treats it that way.
- The current research intentionally avoids shorts; it focuses on validating and improving long-only signals first.
