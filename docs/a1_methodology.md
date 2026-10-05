# A1: accounting and validation contract

This document supersedes the execution/statistical interpretation in archived
v0.2–v0.6 results. No corrected real-data performance claim is made here; the
controlled historical rerun belongs to A3.

## Execution and financing

- A signal uses information available after close t. With the default lag of
  one session it trades at close t+1. Its new holdings earn returns from t+1
  to t+2. Price movement into the execution close belongs to the old holdings.
- Timing comparison with zero fees: prices 100, 200, 220, 220 and an always-on
  signal from the first close produced 220 from capital 100 under the earlier
  assumption of exposure into the next close. A1 buys at 200 and ends at 110.
  This synthetic comparison isolates timing; the historical comparison is A3.
- Each evaluation begins with cash at its first supplied close. Warm history
  initializes indicators/stops only; it does not contribute training P&L.
- Targets are post-cost closing proportions. Risky holdings are marked to
  market before rebalancing. Daily constant fractional targets therefore
  generate trades when asset/cash returns differ.
- One-way costs apply to risky traded notional: replacing A with B incurs both
  sale and purchase. The financing equation is solved for post-cost NAV;
  a full-investment entry cannot borrow money to pay its fee.
- Turnover and transaction_cost are fractions of the previous closing NAV.
  Gross and net daily returns differ by transaction_cost. Gross equity is the
  no-fee path for the same closing target proportions.
- Fractions are allowed; execution at the next close is an idealized daily
  allocation model. It does not simulate auctions, share rounding or market
  impact. No terminal liquidation is assumed. An extra lag can be tested via
  EngineConfig; that scenario is not selected for a prettier result.
- EngineResult.executed_weights earn the current bar's return;
  closing_weights are the holdings after that bar's trades. Evaluation curve
  position is the closing position for trade logs; earning_position is the
  position receiving the price move. Exposure episodes are portfolio episodes,
  not security-lot P&L attribution.

## Cash and return reference

The residual leg is a synthetic remunerated cash account. BIL total returns
are a proxy for its yield, not a claim that BIL is risk-free and not a simulated
ETF purchase. Consequently the model does not charge BIL purchase/sale fees.
An actually traded cash ETF must instead be an explicit risky asset column.

By default reports subtract the aligned daily cash-proxy return to compute
excess returns. Their standard deviation is used for Sharpe; Sortino uses the
root mean square of negative excess returns over all observations. No daily
proxy is silently replaced by its period-average yield. evaluate_strategy
also accepts an independent daily or scalar annual reference_returns argument.
The legacy risk_free_rate output is a descriptive annual mean, not proof of
a risk-free instrument. Manifests record the convention.

All performance reports and bootstrap annualize with 252 supplied sessions.
The initial zero-return valuation row is included consistently. Initial NAV
is included in drawdown peaks, so an immediate loss does not disappear.
The standalone legacy cagr(equity) helper remains a calendar-span utility;
report CAGR is explicitly computed from the full return series instead.

Buy-and-hold and static-blend benchmark curves are frictionless references;
they are not described as fully executable portfolios. The controlled A3
comparison must distinguish these references from costed allocation policies.

## Walk-forward and warm history

All selected-model comparisons, cost tables and significance calculations use
pre-test history for signals, with the same evaluation start. Individual
frozen-model walk-forward rows are retrospective diagnostics; if the model was
chosen using later data they are not independent OOS evidence.

Nested selection generates one target schedule over disjoint, adjacent test
windows. That schedule is run once through the engine. Holdings, fees and
close-to-close returns carry across boundaries. A pending instruction from
the preceding close is not erased at a model-change boundary; the new model's
first OOS-close instruction executes the following close. There is no implicit
year-end liquidation/re-entry. Overlaps and uncovered supplied sessions fail
explicitly instead of being silently deduplicated or omitted.

Main window metrics are sliced from this continuous account; legacy independent
cash-start statistics are retained only as independent_window_* diagnostics.
The old nested_*_stitched labels remain for CSV compatibility but their
accounting field identifies continuous_account. The returned research object
also exposes the account curve, targets, earning weights and closing weights.
The full workflow exports nested_oos_curve.csv and nested_ensemble_oos_curve.csv
with daily NAV, fees and per-asset target/earning/closing weights for audit.

The ensemble fallback is fixed SMA 20/100 long/cash. A globally fitted model
is never a fallback in an earlier window. The optional legacy function
argument is ignored for compatibility. This fixes that code path but does
not erase earlier human strategy-design choices.

## Data failures

Downloads require Adj Close and consistent OHLC. Evaluations and CSV snapshots
require unique increasing dates, unique columns and complete finite positive
prices. Missing cash, requested fallback or market-context series fail.
Only the first return of a valid price series is explicitly zero as an initial
valuation. Gaps are not forward-filled and missing returns are not set to zero.

The calendar here is the supplied joint daily index. This validation cannot
detect a trading session omitted from every instrument or independently
certify Yahoo corporate-action adjustments. Historical snapshot provenance and
cross-source/session checks remain explicit A3 data-audit limitations. A new
source or an incomplete universe must be reconciled before running this model.

## Statistical scope

- Bootstrap describes resamples of the selected return path, with its daily
  reference resampled using the same indices. It does not repeat selection.
- bootstrap_fraction_negative_* gives fractions of negative replicates.
  The old prob_negative_* aliases remain for file compatibility; neither is
  a posterior probability of alpha or a forecast probability of loss.
- DSR is an approximate diagnostic of the supplied candidates under an
  independence approximation. Family, ensemble and overlay rows are included
  for v6, but correlated variants and the incomplete history of manual
  research prevent a calibrated whole-project claim.
- PBO covers the hysteresis grid (possibly a capped subset), with raw-return
  Sharpe as its scoring rule. It does not validate the whole v6 pipeline.
- Circular-shift permutation diagnostics reprice each schedule with the same
  drift-aware, financed-cost engine. Their observed Sharpe matches evaluation.
  Costs need not remain identical after a shift because price-induced drift
  changes turnover. This is conditional on the supplied schedule, not a full
  experiment-selection test.
- A prospective or otherwise genuinely uninspected test is still needed for
  new evidence. Nested historical selection does not undo repeated human
  inspection of the same history.

## Verification

The A1 regression suite checks next-close execution, a manual 50/50 rebalance,
an independent dollar ledger, continuous window boundaries, overlap/gap
rejection, daily-reference alignment, summary/significance agreement, invalid
prices, missing adjusted/cash data and independence from a global fallback.
Existing strategy, parallelism and fixture-report tests remain part of the gate.
