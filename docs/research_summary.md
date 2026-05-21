# Research Summary

This project evaluates a long-only SMA timing model for AAPL and related assets.
The current objective is not to force a profitable-looking chart, but to test
whether the signal remains useful across reasonable assumptions.

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

## Next Research Steps

1. Make fallback allocation slower and more selective so it does not recreate
   the v0.2 turnover problem.
2. Add an explicit cash-yield or Treasury-bill proxy, because long/cash models
   are currently penalized as if cash earns zero.
3. Improve upside capture before adding shorts. The model still exits too much
   upside during strong AAPL regimes.
4. Use regime diagnostics to tune risk filters by failure mode, especially bear
   and recovery periods.
5. Consider multi-signal confirmation only after the current long-only rules
   produce a better capture spread.
