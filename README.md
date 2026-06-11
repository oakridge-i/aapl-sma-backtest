# AAPL SMA Robustness Research

Educational quantitative analysis project for Apple Inc. (`AAPL`). It downloads
daily market data, tests a long-only SMA 20 / SMA 100 crossover strategy, applies
10 bps transaction costs on position changes, and compares the strategy with a
buy-and-hold benchmark.

The current research version (0.5.0) focuses on honest methodology: all model
selection happens on the train period only, cash earns a T-bill proxy yield,
Sharpe uses a real risk-free rate, and every reported result comes with
significance diagnostics (block bootstrap intervals, Deflated Sharpe Ratio,
and a timing permutation test). Each run also writes a data snapshot and a
manifest so results can be reproduced exactly.

This project is for research and education only. It is not investment advice.

## Current Status

The project started as a single AAPL SMA crossover backtest. It now includes a
research workflow that stress-tests the strategy across transaction costs,
parameter choices, train/test periods, walk-forward windows, and a small
multi-asset universe.

Version 0.5.0 removed a test-period leak from model selection: earlier
versions picked the final model by its test-period score, which biased the
reported out-of-sample numbers upward. Selection is now train-only, the test
period is touched once per final model, and the final models are additionally
evaluated across every walk-forward window. Pre-0.5 reported metrics should be
treated as optimistic; the 0.5 reports supersede them.

## Project Contents

- `main.py` - CLI entrypoint for downloading data and running the backtest.
- `research.py` - CLI entrypoint for the robustness research workflow.
- `configs/research_v6.yaml` - 0.6-dev configuration: v5 plus parallel sweep
  execution, nested walk-forward selection, PBO (probability of backtest
  overfitting) diagnostics, five extra signal families (time-series momentum,
  Donchian breakout, ATR trend strength, dual momentum, 52-week-high), and
  equal-vote ensemble selection.
- `configs/research_v5.yaml` - default configuration: cash yield, train-only
  selection, and significance testing enabled.
- `configs/research_v4.yaml` - capture-aware configuration, kept compatible.
- `configs/research_v3.yaml` - turnover-aware configuration, kept compatible.
- `configs/research_v2.yaml` - previous robustness configuration, kept compatible.
- `src/quant_backtest/` - data loading, strategy, metrics, stats, and charting code.
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
.\.venv\Scripts\python research.py --config configs\research_v5.yaml
```

Reproduce a previous run exactly from its saved data snapshot:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v5.yaml --data-snapshot outputs\data_snapshot.csv
```

Legacy v0.3/v0.4 configs remain runnable:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v4.yaml
```

Offline smoke run using deterministic fixture data:

```powershell
.\.venv\Scripts\python research.py --config configs\research_v5.yaml --no-download --output-dir outputs_fixture_v5
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
- `outputs/final_model_walk_forward.csv`
- `outputs/significance_results.csv`
- `outputs/data_snapshot.csv` and `outputs/run_manifest.json` (reproducibility)
- `outputs/research_report.xlsx`
- PNG charts for baseline, costs, heatmaps, train/test, multi-asset,
  leaderboard, v0.3 equity/drawdown, turnover, capture ratios, allocation
  exposure, cost sensitivity, entry/exit signals, v0.4 capture diagnostics,
  regime performance, exposure sizing, and final-model walk-forward CAGR.

Note on leaderboards: since 0.5.0, `model_leaderboard.csv` and
`allocation_leaderboard.csv` report **train-period** metrics, because that is
the data selection is allowed to see. Test-period numbers live in the
comparison tables and `significance_results.csv`.

## Research Notes

Methodology since 0.5.0:

- candidate ranking and model selection use the train period only; the test
  period is evaluated once per final model;
- cash earns the `BIL` proxy return, and Sharpe/Sortino subtract the
  corresponding risk-free rate;
- every selected model is re-checked across walk-forward windows with frozen
  parameters (`final_model_walk_forward.csv`);
- `significance_results.csv` reports bootstrap confidence intervals, the
  Deflated Sharpe Ratio against the number of candidates tried, and a
  permutation p-value for timing skill.

The first honest run (June 2026) found that with train-only selection the
model picks `SMA 5/50` hysteresis and earns test CAGR `5.89%` with Sharpe
`0.24` versus AAPL buy-and-hold CAGR `17.50%` with Sharpe `0.61`. The
permutation test gives p = `0.71` and the bootstrap puts a `33.9%` chance on a
negative true Sharpe: the previously reported v0.3 numbers were an artifact of
the selection leak, and the current rules show no demonstrated timing alpha.
The proven benefit is limited to drawdown cushioning in bear regimes (2022:
`-8.6%` vs `-28.5%` for AAPL).

Earlier findings (v0.2-v0.4) are kept in `docs/research_summary.md`, with the
caveat that pre-0.5 selection leaked test data, so those numbers are
optimistic.

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

For the AAPL run from `2015-01-02` through `2026-06-08`, starting with
`$10,000`:

- SMA strategy ending equity: `$35,132.01`
- SMA strategy total return: `251.32%`
- Buy-and-hold ending equity: `$124,641.40`
- Buy-and-hold total return: `1146.41%`

The strategy made money historically, but it did not outperform buy-and-hold for
this AAPL period.

## Notes

- The strategy uses adjusted close when available.
- Signals are shifted by one day to avoid lookahead bias.
- The `--end` argument is exclusive because Yahoo Finance treats it that way.
- The current research intentionally avoids shorts; it focuses on validating and improving long-only signals first.
- The package can be installed in editable mode with
  `.\.venv\Scripts\python -m pip install -e .` (see `pyproject.toml`); the CLI
  scripts also work without installation via the bundled `src` path.
