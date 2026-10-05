# SDD ledger — plan: docs/superpowers/plans/2026-10-05-aapl-closeout.md

BASE: 5aabf02cf23a4ca4818e31c2c9708652c4f1f8bc; branch codex/aapl-final-release.
Spec: docs/closeout_scope.md; approved user MASTER_PLAN A2–A5.
Tasks: 1 A2 invariants; 2 A3 controlled reruns; 3 A4 report; 4 A5 local release.

Pre-flight: Task 1 verifies engine/evaluation consumed by Task 2; common next-close,
financed costs and initially-cash semantics match A1. Task 2 daily targets and
CSV/manifests feed Task 3; same calendars/cash reference must be retained. Task 4
consumes Task 2 reproduction CLI and Task 3 compact report; raw prices remain local.

Ruling: Execute the previously approved MASTER_PLAN without another approval round
— user explicitly requested A2–A5; this plan only makes execution reviewable — cost
if wrong: user can correct implementation choices while work proceeds.
Ruling: Native worktree tool targets the chat's Internship repository, not AAPL;
use git worktree for the actual AAPL repository in the writable root — cost if
wrong: manual worktree lifecycle rather than app-managed archive.
Ruling: Superpowers Bash workspace helper failed under Windows sandbox mkdir and
looped over slug suffixes; interrupted it and reproduce its ledger/brief contracts
with native file tools — cost if wrong: helper compatibility has to be maintained.
Environment: initial baseline invocation failed (93 pass, 43 fixture setup errors)
because pytest basetemp's parent did not exist; created .pytest_tmp and reran unchanged
code. This was a runner setup error, not a production test failure.
User steering: all A2–A5 local; no GitHub push, PR, remote release, merge or reactions.

Task 1: in progress; BASE 5aabf02. Check existing correct behaviors with independent
expectations; production changes only if a real invariant failure is found.
Task 1 complete: five new real invariant tests, full suite 141 passed in 195.81s. Production mechanics unchanged. Evidence docs/final/accounting_checks.md. Task 2 starts after local Task 1 commit.
Task 2 BASE 61466e9: closeout tests observed RED missing module; implementing frozen common-calendar replay and selected_models result descriptors.
Environment: installing the calendar auditor into the original read-only venv failed permission checks; moved installation to fresh workspace .venv-closeout. UTF-8 required for pip diagnostics with Cyrillic paths; rerunning pinned install with -X utf8 and no cache. No project accounting change.
Task 2: historical full frozen-grid run complete, original hash verified; source run timing retained. XNYS audit has zero missing/unexpected sessions. Full nested ensemble archive Sharpe 1.195945 -> 0.590986; DD -0.142274 -> -0.293993. Common 2021+ fixed allocations outperform nested strategies. Fresh pinned env installed successfully after disabling corrupt cache and UTF-8 diagnostics. Updated snapshot downloaded only after old reconciliation: cutoff 2026-10-05, hash 87b0315bc31e916696d6261c6db8b66088653bc21343edb0e3223c87b318fea8. Task 2 in progress; commit runner before updated run.
Task 2: complete (61466e9..current), tests full clean suite 148 passed in 87.04s plus two real full searches and two clean offline replays. Verified frozen grids against v6 except cutoffs/workers; zero missing/unexpected sessions in both snapshots. Financial identities checked in 98,280 + 104,040 daily rows, all 72 scenarios per dataset, NAV residual <=1.46e-11; all summary columns replayed within rtol1e-12/atol1e-10. Config semantic hash addition observed RED then GREEN; receipts upgraded metadata-only, preserving original raw-config hash and unchanged targets. Task 3 starts: source-backed Russian report and all four exported plots inspected.
Task 4: Ruling: legacy tracked outputs contain raw vendor prices; untrack generated outputs with --cached, preserving all local files and prior Git history — satisfies compact release/data scope — cost if wrong: generated attachments are absent in fresh clones, recoverable from earlier commits.
Task 3: complete (318752d..current), report rows 18/18 checked against CSV, other quoted archive/sensitivity/statistical values checked in source tables; all report/README/reproduction local links exist. Four plot exports inspected visually. Final Russian report covers sources, accounting, archived changes, periods, both datasets and conditional uncertainty without an alpha claim. Task 4: clean environment 49 exact pins, package import+installed metadata both0.6.0, pip check green, suite148 passed; two deterministic replays and daily identities green. Local+remote tag query confirms no v0.6.0 tag. Remaining: independent review, final metadata/commit and local tag.
Task 1: complete (5aabf02..61466e9), full suite141 passed; earlier completion entry without colon normalized for ledger readers. Task 2 commits61466e9..318752d; Task 3 commits318752d..d94ad05. Task 4 implementation commitf47ba77, independent review dispatched on fresh gpt-6-astra context; no final tag until review gate.
Final review: fresh gpt-6-astra reviewed5aabf02..f47ba77; independently checked202,320 daily rows,72 scenarios per dataset and both replay tables; directly reran5 real A2 tests. Critical0, Important1, Minor0. Important snapshot crop mismatch reproduced with real full-run integration RED (hash mismatch), fixed by using result.prices and separate input hash; GREEN8 closeout tests. Full post-fix suite running. No second review.
Final: Ruling: execution/liquidity/tax/impact set aside by reviewer — disclosed educational simulator assumptions stand — cost if wrong: real execution diverges.
Final: Ruling: independent vendor adjustment/point-in-time certification set aside — audit discloses uncertified retrospective snapshots — cost if wrong: hidden vendor revisions/errors remain.
Final: Ruling: new families/grids and calibrated pipeline alpha set aside — closeout freezes search and makes scoped negative conclusion — cost if wrong: alternative hypotheses remain untested.
Final: Ruling: other platforms/interpreters set aside — certify Python3.14.0 with49 exact pins only — cost if wrong: other environments may require changes or differ numerically.
Final: Ruling: publication/final metadata/tag set aside — local user constraint honored, final tag/metadata follow review gate and direct Git checks — cost if wrong: remote integration outstanding and final metadata lacks second review.
Final: fixed effective snapshot/target receipt mismatch — test_full_run_with_extra_history_replays_its_effective_snapshot RED hash mismatch -> GREEN real full run plus replay, closeout8/8 and full suite149/149 passed in96.44s; published exact-period inputs unaffected. No deferred minor findings.
