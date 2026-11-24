# 0019 Live tracking: log before kickoff, grade after, post only strong calls

Status: accepted, 2026-11-23.

Harsh's goal: better prediction accuracy that could later be sold, and an Instagram page as a
prediction expert. Nothing manual: the desk's own slips are evaluated, not just saved ones.

1. **Every prediction is logged before kickoff, automatically** (`tracking/snapshot.py`): every
   leg of every upcoming match from the daily run and whenever the desk is opened. A match is
   re-logged only when its numbers change; the last block before kickoff is the one graded.
2. **Append-only and hash-chained** (`tracking/ledger.py`), committed under `records/`. The chain
   detects edits and deletions; GitHub push times date the blocks. ~~We do not auto-commit or
   auto-push.~~ **Amended 25 Nov 2026:** the daily run commits `records/` (only that folder,
   `git commit -- records`) and pushes, so the public timestamp no longer depends on
   remembering (`tracking.auto_push`, off with `daily --no-push`). Every logged row also
   carries `model_commit`, `params_sha` and `data_sha` (tracking/version.py) and the raw,
   uncalibrated chance `p_raw` (ADR 0020).
3. **Accuracy is measured with proper scores, not win rate**: Brier score per market against two
   fixed benchmarks, the league's own 3-year rate and the closing market without margin. Win
   rate alone rewards picking only heavy favourites. Each yes/no question counts once.
4. **The bar for "sellable" is the market.** ADRs 0004, 0011-0015 found no reliable edge over
   prices; the live scorecard is where that can change, market by market. On the 2026 spring
   walk-forward (400 matches) the result model beat the league baseline by 6% and was slightly
   less accurate than the closing market (0.201 against 0.198).
5. **Strong calls come from the track record, not from feel and not from the confidence label.**
   The label measures whether past chances at that level were accurate, not whether the call is
   likely: favourites at 70%+ are labelled Medium or Low (small samples, and they came in *more*
   often than predicted), so the video picker's "High + 70%" rule almost never fires. Rule: the
   lowest level from which every past level (30+ calls) came true 65%+, with 100+ calls. On 23
   Nov that was favourites at 60%+; since calibration (ADR 0020), favourites at 65%+ and over/
   under 2.5 at 65%+.
6. **Posted means counted.** Marking a call as posted is permanent and only possible before
   kickoff; the weekly recap shows every posted call, misses included.
7. **Desk slips are logged privately** (`bets/desk_log/`, never committed): they are trading
   output, not the public record.
