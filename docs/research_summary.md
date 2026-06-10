# Research Summary

This project evaluates a long-only SMA timing model for AAPL and related assets.
The current objective is not to force a profitable-looking chart, but to test
whether the signal remains useful across reasonable assumptions.

> **Methodology caveat (added in 0.5.0).** Versions 0.2-0.4 selected the final
> model from leaderboards evaluated on the *test* period. That is a selection
> leak: with enough candidates, picking the best test score inflates the
> reported out-of-sample metrics. The numbers in the historical sections below
> are kept for the record but should be read as optimistic. Section "v0.5
> Honest Methodology" describes the fix.

## Baseline

The first version used `SMA 20/100` on AAPL with 10 bps transaction costs.

| Metric | Strategy | Buy and Hold |
| --- | ---: | ---: |
| Total return | 249.79% | 1141.00% |
| CAGR | 11.65% | 24.81% |
| Sharpe | 0.62 | 0.92 |
| Max drawdown | -35.64% | -38.52% |

The baseline was profitable and slightly reduced drawdown, but it gave up too
much upside.

## Parameter Sweep Result

The robustness workflow found a better full-sample SMA region around `SMA 5/50`.
At 10 bps, this selected setup improved the model profile:

| Metric | SMA 20/100 | SMA 5/50 |
| --- | ---: | ---: |
| Total return | 249.79% | 467.47% |
| CAGR | 11.65% | 16.50% |
| Sharpe | 0.62 | 0.91 |
| Max drawdown | -35.64% | -25.04% |

This is a real improvement over the original rule, but it comes with higher
turnover.

## Cost Sensitivity

The selected `SMA 5/50` strategy remains profitable across the tested cost range.

| Cost | CAGR | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0 bps | 17.42% | 0.95 | -24.59% |
| 5 bps | 16.96% | 0.93 | -24.82% |
| 10 bps | 16.50% | 0.91 | -25.04% |
| 20 bps | 15.60% | 0.87 | -25.49% |
| 50 bps | 12.91% | 0.74 | -28.64% |

The model does not depend on exactly 10 bps, but transaction costs matter because
the faster rule trades more often.

## Out-of-Sample Check

The selected parameters perform worse out of sample than on the training period.

| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Train | 26.41% | 32.19% | 1.29 | 1.09 |
| Test | 7.39% | 17.64% | 0.51 | 0.73 |

This is the main limitation. The strategy improves risk control, but it has not
yet proven persistent alpha versus buy-and-hold.

## Variant Review

The SPY and QQQ fallback variants improve raw return, especially on the test
period, but they also raise turnover. They are promising research directions,
not finished models.

## v0.3 Turnover-Aware Model

The next research pass added hysteresis, minimum holding periods, cooldown
periods, regime-aware fallback variants, and a turnover-aware selection score.
The selected model is:

| Field | Value |
| --- | --- |
| Variant | `long_cash_hysteresis` |
| SMA windows | `5 / 200` |
| Entry threshold | `1.0%` |
| Exit threshold | `-1.0%` |
| Minimum hold | `10 trading days` |
| Cooldown | `5 trading days` |

On the test period, the selected model improves the v2 long/cash baseline:

| Metric | v2 SMA 5/50 | v0.3 selected |
| --- | ---: | ---: |
| CAGR | 7.39% | 11.11% |
| Sharpe | 0.51 | 0.66 |
| Max drawdown | -23.08% | -23.62% |
| Annualized turnover | 8.41 | 1.68 |
| Cost drag | 6.75% | 1.58% |

This is a cleaner model: it trades less, pays less cost, and captures more AAPL
upside than the v2 `SMA 5/50` long/cash strategy. It still does not beat AAPL
buy-and-hold by raw CAGR, so it should be treated as a stronger risk-managed
baseline, not as a finished alpha model.

The best current interpretation:

- `SMA 5/50 long/cash` is a stronger baseline than `SMA 20/100`.
- `SMA 5/200` with hysteresis is a stronger low-turnover test-period candidate.
- Fallback assets may reduce cash drag, but only if their turnover is controlled.
- Turnover control is now built into model selection, not just reviewed after the fact.
- Shorts should still wait until the long-only signal is stronger.

## v0.4 Capture-Aware Risk Model

The v0.4 pass tested whether the model could improve behavior quality, not just
raw return. The target was higher upside capture, lower downside capture, a
positive capture spread, controlled turnover, and no degradation versus the
v0.3 low-turnover baseline.

The selected result is:

| Field | Value |
| --- | --- |
| Selection status | `no_robust_upgrade_baseline_retained` |
| Retained model | v0.3 `SMA 5/200 long_cash_hysteresis` |
| Reason | No v0.4 candidate passed the full capture and turnover filter set |

On the test period from `2021-01-04` through `2026-05-19`:

| Metric | v2 SMA 5/50 | v0.3 retained | v0.4 selected output |
| --- | ---: | ---: | ---: |
| CAGR | 7.29% | 11.00% | 11.00% |
| Sharpe | 0.51 | 0.66 | 0.66 |
| Max drawdown | -23.08% | -23.62% | -23.62% |
| Annualized turnover | 8.40 | 1.68 | 1.68 |
| Upside capture | 46.56% | 54.48% | 54.48% |
| Downside capture | 47.14% | 53.61% | 53.61% |
| Capture spread | -0.58 pp | 0.87 pp | 0.87 pp |

Some v0.4 candidates had higher Sharpe and lower drawdown than the retained
model. The problem is why they improved: they leaned on QQQ fallback exposure,
pushed annualized turnover into the `7.8-10.6` range, and still captured only
about `30-36%` of AAPL upside. That does not satisfy the model objective. Under
the configured rules, retaining v0.3 is the correct outcome.

The useful work from v0.4 is the new diagnostics:

- `capture_leaderboard.csv` shows which candidates failed and why.
- `risk_filter_sweep.csv` separates risk-filter behavior from fallback behavior.
- `regime_results.csv` shows where the retained model still struggles.
- `trade_log.csv` makes entries, exits, MFE, and MAE inspectable.
- `v04_entry_exit_signals.png` and `exposure_sizing.png` show the selected
  exposure path visually.

## v0.5 Honest Methodology

v0.5 changes how results are produced rather than what the model trades:

1. **No selection on test data.** All leaderboards (v0.2 variants, v0.3
   allocation, v0.4 capture) are now evaluated and ranked on the train period
   only. The test period is evaluated once per final model. The v0.4 hurdle
   ("beat the v0.3 model") also uses v0.3's train-period metrics.
2. **Cash earns yield.** The uninvested weight earns the `BIL` T-bill proxy
   return, and the same yield is subtracted as the risk-free rate in
   Sharpe/Sortino. This removes the structural penalty against long/cash
   models that sat in cash during 2022-2026 while rates were 4-5%.
3. **Significance testing.** `significance_results.csv` reports, for each
   final model on the test period: block-bootstrap 5/50/95 percentiles for
   CAGR, Sharpe, and max drawdown; the probability of a negative Sharpe; the
   Deflated Sharpe Ratio given how many candidates were tried during
   selection; and a permutation p-value that asks whether the model's timing
   beats random alignment of the same exposure profile with identical costs.
4. **Final-model walk-forward.** The selected models are re-evaluated with
   frozen parameters across every walk-forward test window
   (`final_model_walk_forward.csv`), so one favorable train/test split cannot
   carry the conclusion.
5. **Reproducibility.** Each run writes `data_snapshot.csv` (hashed prices)
   and `run_manifest.json` (config, data hash, package versions, git commit).
   `research.py --data-snapshot` reruns on the saved data, which matters
   because Yahoo Finance revises adjusted prices retroactively.
6. **Honest trade stats.** Closed trades are segmented by exposure episodes,
   so volatility-sizing weight changes no longer count as trades.

Because the selection leak is gone, the v0.5 selected model and its metrics
may differ from the v0.3/v0.4 numbers above. The next real-data run should be
read as the new baseline, and the interesting question is whether the
permutation p-value and Deflated Sharpe Ratio support any timing skill at all.

### First Honest Run (2026-06-10, data through 2026-06-09)

Selecting on the train period only (2015-2020), the framework picked
`SMA 5/50` hysteresis (entry `0.0%`, exit `-0.5%`, 20-day hold, 5-day
cooldown) as the v0.3 model; v0.4 again retained it with
`no_robust_upgrade_baseline_retained`. On the test period (2021-2026), with
cash earning the BIL yield and Sharpe measured against that risk-free rate:

| Metric | Selected model (test) | AAPL buy-and-hold (test) |
| --- | ---: | ---: |
| CAGR | 5.89% | 17.50% |
| Sharpe (excess) | 0.24 | 0.61 |
| Max drawdown | -29.10% | -33.36% |
| Annualized turnover | 7.21 | - |

Significance diagnostics for the selected model on the test period:

| Diagnostic | Value |
| --- | ---: |
| Bootstrap Sharpe 5-95% interval | -0.57 to +1.01 |
| Probability of negative Sharpe | 33.9% |
| Deflated Sharpe Ratio | 0.54 |
| Permutation p-value (timing skill) | 0.71 |

The interpretation is blunt: the previously reported v0.3 numbers (CAGR
`11.00%`, Sharpe `0.66`, turnover `1.68`) were an artifact of selecting the
best test-period candidate. Selected honestly, the model underperforms both
the benchmark and the plain `SMA 20/100` baseline on the test period, and the
permutation test finds no evidence that its timing beats random alignment of
the same exposure profile (p = 0.71). The per-window walk-forward shows the
expected trend-following shape - it cushioned 2022 (-8.6% vs -28.5% for AAPL)
and lagged every strong up year.

The framework is working as intended: it now correctly reports that this
family of long-only SMA timing rules on AAPL has no demonstrated alpha. The
honest baseline to beat going forward is AAPL buy-and-hold and the SMA200
filter (test CAGR `9.21%`, Sharpe `0.40`), with risk management as the only
proven benefit (drawdown cushioning in bear regimes).

## Next Research Steps

1. Re-run the full workflow on fresh data and re-baseline the conclusions on
   the honest pipeline (train-only selection, cash yield, significance).
2. Make fallback allocation slower and more selective so it does not recreate
   the v0.2 turnover problem.
3. Improve upside capture before adding shorts. The model still exits too much
   upside during strong AAPL regimes.
4. Use regime diagnostics to tune risk filters by failure mode, especially bear
   and recovery periods.
5. Consider multi-signal confirmation only after the current long-only rules
   produce a better capture spread.
6. Address survivorship bias in the multi-asset universe (the current tickers
   are known winners chosen in hindsight).
