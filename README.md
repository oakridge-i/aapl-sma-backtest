# AAPL SMA Robustness Research — 0.6.0

Completed educational research into long-only AAPL trend strategies, selection,
execution costs and overfitting. The final controlled reruns **do not establish
alpha or an economic advantage over simple AAPL/cash allocations**.

## Final result

On the preserved 2021-01-04–2026-06-10 calendar, at 10 bps one-way costs:

| Policy | CAGR | Sharpe above daily BIL proxy | Max drawdown |
| --- | ---: | ---: | ---: |
| 50% AAPL / 50% synthetic cash | 10.58% | 0.573 | -17.50% |
| Selected v6 | 5.95% | 0.257 | -15.67% |
| Annually selected nested ensemble | 2.47% | 0.037 | -29.40% |

The separate updated run through 2026-10-05 gives 11.43% CAGR for 50% AAPL,
versus 4.74% for the nested ensemble. These are simulated results, not future
return forecasts. The narrower drawdown of some models must be compared with
simple reduced equity exposure, not only with 100% AAPL.

- [Final report, conclusions and figures (Russian)](docs/final/research_report.md)
- [Historical comparison](docs/final/historical/comparison.csv)
- [Updated comparison](docs/final/updated/comparison.csv)
- [Archive reconciliation](docs/final/archive_comparison.csv)
- [Accounting verification](docs/final/accounting_checks.md)
- [Reproduction instructions and data requirements](docs/final/reproduction.md)

## Recommended use

Start with the final report and reproduce the preserved accounting run in a
clean environment. Use Python 3.14.0 for the verified lock:

```powershell
python -m venv .venv-closeout
.\.venv-closeout\Scripts\python.exe -X utf8 -m pip --disable-pip-version-check install --no-cache-dir -r requirements-lock.txt
.\.venv-closeout\Scripts\python.exe -X utf8 -m pip --disable-pip-version-check install --no-deps --no-build-isolation -e .
New-Item -ItemType Directory -Force .pytest_tmp | Out-Null
.\.venv-closeout\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp/reproduction
```

Then follow the [offline replay command](docs/final/reproduction.md#recommended-path-powershell-from-repository-root).
Exact replay requires the local vintage snapshot and hashed target CSVs. They
are kept in ignored `outputs_closeout/`; a fresh clone alone does not contain the
market data needed to reproduce vintage numbers. A full search is the same CLI
without `--replay`, using one of `configs/closeout_*.yaml` and the declared input hash.

The generic entry points `main.py` and `research.py`, earlier configs and APIs
remain available for learning. The final closeout route is
`scripts/finalize_research.py`; it preserves outputs instead of overwriting runs.

## Financial contract

Signals after close t trade at close t+1 and earn risky returns afterward.
Each comparison begins in cash, warms indicators with earlier prices, finances
fees from NAV and rebalances against drifted holdings. Nested annual selections
share one continuous account. Fixed allocations also pay the same costs.

Cash is a synthetic account remunerated by adjusted BIL returns; it is not an
executed ETF holding or a guaranteed risk-free rate. Sharpe subtracts its daily
return. Metrics use 252 supplied sessions per year. No terminal liquidation,
leverage, shorting, market impact or taxes are modeled.

## Research limits

The grids were frozen for closeout. Cost/lag sensitivity replays the same target
schedules. Bootstrap, DSR and PBO are conditional diagnostics with disclosed
scope; they are not probabilities that alpha exists. Prior manual experiments
and retrospectively adjusted data prevent an untouched-holdout claim.

Raw vendor prices are retained locally. Rights to redistribute them have not
been established; the data provider's conditions apply independently of the
Python library's license. See the report's sources and reproduction limits.

## Project status

A1 accounting corrections and A2–A5 closeout finish this study. Further indicator
search is not justified by the current evidence. A separate research project
should begin with an economic hypothesis, a defined historical universe,
a complete attempt log and a simple baseline.

The [pre-closeout README](docs/archive/README_pre_closeout.md), legacy documents
and locally retained `outputs/` are historical artifacts. Their optimistic preview
numbers are superseded by `docs/final/` and should not be cited as current results.
Generated legacy exports were removed from the current tracked tree while kept
locally. Earlier Git commits still contain those exports; history was not rewritten.
