# A2: accounting verification

Verified on 2026-10-05, base commit `5aabf02`, Python 3.14.

`python -m pytest -q --basetemp=.pytest_tmp/a2_full`: **141 passed** in 195.81 seconds.

Five additional tests exercise actual accounting and selectors:

- Flat prices, no orders, zero cash yield: NAV remains exactly 1,000; no fees.
- Full investment with a 1% fee: entry NAV is `100 / 1.01`. After two 10% returns and an exit it is `100 / 1.01 * 1.21 * .99`. Instructions after close execute at the following close.
- A year-boundary replacement of AAPL by SPY preserves one account. Final NAV is `1000 / 1.001 * 1.1 * .999 / 1.001 * 1.2`; both sides of the replacement incur costs, without a reset or duplicate entry.
- Changing future AAPL prices leaves prior decisions, target weights, selected models and NAV unchanged, using real nested selectors.
- A real grid with no eligible ensemble uses the fixed SMA 20/100 fallback, independent of a globally supplied candidate.

Existing A1 tests independently verify 50/50 portfolio drift, a 40-day dollar ledger, daily cash-reference metrics and warmed legacy model evaluation. M3 tests cover causal signal and stop-state behavior.

These checks confirm behavior already implemented in A1; no production accounting change was required. Initial fixture issues (missing SPY and an incorrect expected status string) were corrected in the new tests before the final full run. This is verification evidence, not a claim that already-correct production behavior initially failed.

Scope: unlevered daily close execution, proportional one-way costs, a synthetic cash account remunerated with the BIL return proxy, no terminal liquidation. It does not verify market impact, executable quotes, taxes or point-in-time vendor data.
