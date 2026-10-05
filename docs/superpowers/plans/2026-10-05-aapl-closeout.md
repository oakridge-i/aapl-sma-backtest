# AAPL A2–A5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Close the existing AAPL research with verified accounting, controlled reruns, a final report and reproducible release.
**Architecture:** Preserve existing selectors and frozen grids. Add one closeout runner that reuses their results and target schedules with the common engine; keep detailed runs/raw prices local and commit compact evidence.
**Tech Stack:** Python, pandas/numpy, matplotlib, pytest, existing YAML and Git.
**Spec:** `docs/closeout_scope.md` (approved MASTER_PLAN A2–A5).

## Global Constraints

- No new signal families, enlarged grids, chosen test periods or profit targets.
- Preserve the 2026-06-10 snapshot; updated downloads are a separate run.
- 10 bps baseline, 20 bps stress, 0/50 diagnostic costs; lag 1 baseline, lag 2 sensitivity.
- Synthetic BIL-remunerated cash, 252 supplied sessions; no risk-free claim.
- No GitHub reactions; user's later instruction: local commits/tag only, no push/PR/publication/remote merge.
- Keep nonredistributable raw prices local; publish hashes, sources and instructions.

## Review Focus

1. Late future changes must not alter earlier selected models or instructions.
2. Empty eligible ensemble grids must use the fixed fallback with real selectors.
3. Nested and fixed-model comparisons must share dates; show both common 2021+ and full nested history.
4. Cost/lag sensitivity must replay the same targets, never refit for favorable outcomes.
5. Clean/offline reproduction must fail explicitly on missing or wrong snapshots; adjusted-price revisions are separate from new observations.

### Task 1: A2 financial checks

**Files:** `tests/test_a2_financial_invariants.py`, `docs/final/accounting_checks.md`.
**Interfaces:** Consumes `run_weight_backtest`, `continuous_oos`, actual nested selectors. Produces independently checked invariants and full-suite evidence.

- [x] Verify flat cash, financed entry/exit, year-boundary asset replacement with literal/manual expected NAV and fees.
- [x] Exercise real train-only selectors on perturbed future prices and an empty eligible ensemble case; compare earlier windows and target schedules.
- [x] Reuse A1 daily-reference/report checks and run `python -m pytest -q`; Expected: all tests pass. New checks of existing correct behavior may pass without production changes; record that distinction.
- [x] Commit accounting tests and evidence.

### Task 2: A3 controlled closeout runner and reruns

**Files:** `src/quant_backtest/closeout.py`, `scripts/finalize_research.py`, `tests/test_closeout.py`, `src/quant_backtest/experiments.py`, `configs/closeout_*.yaml`.
**Interfaces:** Consumes `ResearchResult`, daily targets, prices, config and archive tables. Produces `comparison.csv`, `daily_audit.csv`, `sensitivity.csv`, `period_results.csv`, `archive_comparison.csv`, data audit and hashed run manifests.

- [x] Write tests for manual flat-price allocation fees, identical costed calendars, frozen targets under cost/lag changes and absent/mismatched snapshots. Run first; Expected: fail because closeout behavior is unavailable.
- [x] Implement `build_policy_comparison(prices, config, schedules, costs=(0,10,20,50), lags=(1,2))` using the common engine; expose selected model descriptors/targets from the existing research result without changing selection.
- [x] Add the CLI with explicit snapshot/config/output/archive arguments and immutable data validation; Expected: targeted tests pass.
- [x] Audit old snapshot hash/session coverage. Run unchanged old search once, then separately updated data; preserve logs/parameter choices and distinguish historical revisions. Expected: complete outputs with reconciled NAV/fees and honest source status.
- [x] Commit runner, tests, configs and compact manifests/tables (not raw prices).

### Task 3: A4 final report

**Files:** `docs/final/research_report.md`, `docs/final/figures/`, `docs/final/*.csv`.
**Interfaces:** Consumes verified Task 2 evidence; produces the final narrative and figures, source/data-method receipt and reproducible commands.

- [ ] Create NAV, drawdown, earning exposure and turnover charts on the common calendar, plus period/cost/lag evidence.
- [ ] Explain archive differences, selected model changes, uncertainty and manual-search limits; report whether evidence supports an advantage without claiming calibrated alpha.
- [ ] Check every quoted number against CSV, inspect all exported charts and links; Expected: source-backed internally consistent report.
- [ ] Commit report and figures.

### Task 4: A5 reproducible release

**Files:** `requirements-lock.txt`, `README.md`, `CHANGELOG.md`, `docs/final/reproduction.md`, `docs/final/release_manifest.json`.
**Interfaces:** Consumes runner/configs/report; produces a clean-environment verification and compact local release on the completion branch.

- [ ] Pin installed compatible dependencies, install in a fresh venv; run the full suite and deterministic offline replay. Expected: same accounting/metrics from preserved targets and prices.
- [ ] Update README with one recommended reproduction path; document external data access/redistribution conditions and snapshot hashes.
- [ ] Obtain one independent whole-branch review; fix Important/Critical findings with RED→GREEN tests and a green suite. Expected: no unaddressed material finding.
- [ ] Check existing tags, choose release name, commit locally and prepare a local tag; verify local SHA, tag, clean status and completion checklist. No push, PR, remote release, merge or reaction tools.


