# Independent closeout review

2026-10-06: a fresh read-only reviewer (`gpt-6-astra`) inspected the whole branch
from `5aabf02` to `f47ba77`, the approved scope, execution ledger and five review
focus points. No reviewer changes were made to checkout, index or HEAD.

## Evidence independently checked

- 98,280 historical and 104,040 updated daily rows, 72 scenarios each.
- Preserved versus replayed tables, daily BIL references, execution delays,
  snapshot/config/target hashes and compact-versus-local outputs.
- The five real A2 tests, including an actually selected ensemble and forced fallback.
- Frozen grids and the distinction between common/full periods, vendor revisions
  and conditional statistical diagnostics.

The reviewer did not repeat the expensive searches or the entire test suite.
Those were run by the implementer, as recorded in the verification receipts.

## Finding and correction

**One Important finding; no Critical or Minor findings.**

For an input extending beyond the configured research period, `run_research`
saved cropped prices while the closeout runner generated targets and hashed the
uncropped input. A bundle could therefore fail to replay its own saved snapshot;
extra preceding history could also give the fixed targets a different warmup.

The implementer reproduced this with
`test_full_run_with_extra_history_replays_its_effective_snapshot`: a real small
research run on 2018–2022 input configured for 2019–2021 failed on the saved-frame
versus target-manifest hash. The runner now consistently uses `result.prices` for
saved data, target generation and comparisons, and retains the original input
hash separately. The same regression then passed through a full run and a replay
of the produced bundle; all eight closeout tests passed.

The published historical and updated frames already matched their configured
periods, so their decisions and results are unchanged. This was one fix pass,
verified by the implementer with RED→GREEN and the full suite, rather than a second
review of the same diff. See `logs/final_tests.log` for the post-fix suite result.

## Behaviors the reviewer declined to judge, and executor rulings

| Area | Ruling / reasonable reader expectation | Cost if this boundary is wrong |
| --- | --- | --- |
| Executable trading, liquidity, tax and market impact | Educational daily-close simulation; these are disclosed assumptions, not execution promises. | Real trading results can diverge. |
| Independent adjustment certification / point-in-time data | Vendor snapshot audit checks supplied data and sessions; no independent certification is claimed. | Vendor errors or revisions can remain. |
| New families, larger grids, calibrated pipeline alpha | Approved closeout freezes the existing search and gives a scoped negative conclusion. | Other hypotheses and a fully calibrated selection experiment remain untested. |
| Other platforms / Python versions | Certified environment is Python 3.14.0 with exact recorded pins; advertised package compatibility is broader. | Other environments may need work and may differ numerically. |
| Remote publication, final tag and metadata | User requested local completion; tag and final metadata follow the review/fix gate and receive direct Git checks. | Remote integration remains a separate user decision; final metadata has not had a second reviewer. |

The independent pre-fix verdict was **ready for local finalization with fixes**.
The implementer completed the identified fix and its verification before tagging.
