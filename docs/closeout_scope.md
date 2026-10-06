# Approved AAPL closeout scope (A2–A5)

The user requested execution on 2026-10-05 of A2–A5 from the existing
`quant-research-plan/MASTER_PLAN.md`, following completion of A1 in PR #2.
The controlling requirements are reproduced here so the repository is sufficient
without the original chat. This is completion of an educational research project.

- A2: independently check flat cash, entry/exit fees, drift rebalance, model-change
  continuity, delayed execution, future-price invariance (including selection),
  forced fallback and identical report/significance metrics; run the full suite.
- A3: first rerun the unchanged search on the immutable snapshot ending
  2026-06-10, then separately on an updated snapshot. Compare executable AAPL
  buy-and-hold, 25/75, 50/50, 75/25 allocations, fixed SMA 20/100, selected v3,
  selected v6 and continuous nested policies. Use a common cash model, calendar,
  warm history and execution, with 10 bps baseline, 20 bps stress, 0/50 diagnostic
  costs. Show archived → corrected numbers and explain material differences.
- A4: report the question/history, data/actions/limits, rules and selection,
  comparable-risk results, NAV/drawdown/exposure/turnover figures, period/cost/lag
  sensitivity, uncertainty, conclusions and exact reproduction commands.
- A5: verify a clean pinned environment, update README/changelog, retain compact
  reports/configs/manifests, review data redistribution terms, commit and push
  the completion branch and select a nonconflicting release version/tag.

Constraints: do not add signal families or enlarge grids; do not tune to test
performance; negative evidence is a valid outcome. Historical manual research
means neither nested selection nor new downloads create an untouched holdout.
The A1 cash leg is synthetic remuneration using BIL total returns, not an ETF
holding or a claim of a risk-free rate. Apply that same leg to every costed policy.
Distinguish the frictionless price reference from the costed buy-and-hold policy.
Use the documented next-close execution, initially cash and no terminal sale.
Keep raw external prices local unless redistribution rights are established.
Do not add GitHub reactions. The user's later instruction on 2026-10-05 is to
keep all work local: do not push, create PRs, publish releases or merge remotely.

Completion: one reproducible, auditable research result; all material limits
explicit; no new indicator search; reader need not access the chat.
