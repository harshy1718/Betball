# 0029 Best markets only: pre-registered

Status: pre-registered 2026-11-25, before the replay below was run. Result appended.

## Why

Harsh: show and pick only the markets our model predicts well. ADR 0028 measured every new
market against always predicting the overall rate; the same measure on the older markets
(25 Nov) splits them in two: bets about who is stronger (result without the draw, handicaps, team
goals, clean sheets, win to nil) beat the average by 4-16%, bets about how many goals (over/under,
both score, totals, exact goals, halves' totals) by 0-3%. The draw is about 1%.

## Rule (fixed now)

A selection (market + side + line) is a **best market** when, on walk-forward predictions:
1. its log loss is at least **5% lower** than always predicting the overall rate;
2. its average chance is within **1.5 points** of how often it came in;
3. at least 1,000 matches.
Live: measured on 2020/21-2025/26. Correct scores and corners are never best (no skill shown).
Cards lines are measured the same way.

## Use (fixed now)

* Desk "All markets" shows best markets by default, the rest one click away.
* The board (good singles, safe and risky parlays, top of the board) is built only from best
  markets, old or new (ADR 0028's "not on the board" rule is replaced by this one), refund bets as
  singles only, if the replay passes.

## Replay test (fixed now)

The desk replay (2020/21-2025/26, every input walk-forward: best-market list, labels, half shares
and lifts for season s from seasons before s) with the best-only board. It passes if no slip type
lands significantly below its predicted chance (the 95% interval of landed minus predicted does
not lie entirely below zero). Fail: the board stays as it is and only the All markets filter
changes.

## Result

Run 2026-11-25. **Best markets (live, 2020/21-2025/26): 62 of 156 selections**
(reports/confidence/best_markets.md): home and away win, home-or-draw and away-or-draw, every
handicap (1.5, 2.5, Asian whole lines, draw no bet), team goals (most lines), win to nil, winning by
2 or 3+, a team to score 3+ or 0, clean sheet of the away side, result & both score / result & over
2.5 for a winning side, half-time / full-time home-home and away-away, first-half result for
either side, second-half home win, the home side to score in both halves. Not best: the draw,
over/under and both score (gain 1-2%), every total-goals line, exact goals and ranges, half totals,
cards (3-4%), odd/even, correct score, corners.

Replay (`replay-desk --best-only`, reports/replay/best_only/verdict.md), landed minus predicted:

| Slip | Slips | Predicted | Landed | Gap (95%) |
|---|---|---|---|---|
| Good singles | 38,706 | 75.6% | 75.8% | +0.2 (-0.3 to +0.7) |
| Top single per match | 10,517 | 80.2% | 80.9% | +0.7 (-0.1 to +1.5) |
| Safe parlay (same match) | 10,134 | 61.5% | 62.1% | +0.6 (-0.4 to +1.6) |
| Risky parlay (same match) | 9,030 | 34.8% | 34.9% | +0.1 (-0.9 to +1.1) |
| Safe parlay across matches | 888 | 62.8% | 66.3% | +3.6 (+0.5 to +6.4) |
| Risky parlay across matches | 902 | 42.1% | 46.0% | +3.9 (+0.7 to +7.1) |

**PASS:** no slip type lands below its predicted chance.

**Decision (Harsh, 25 Nov 2026):** the board uses its old markets **plus** every best market,
new ones included (`trader.add_best: true`). Replayed the same way first
(`replay-desk --add-best`, reports/replay/add_best/verdict.md), landed minus predicted:

| Slip | Slips | Predicted | Landed | Gap (95%) |
|---|---|---|---|---|
| Good singles | 42,185 | 78.2% | 78.6% | +0.5 (+0.0 to +0.9) |
| Top single per match | 10,628 | 81.8% | 83.0% | +1.2 (+0.5 to +1.9) |
| Safe parlay (same match) | 10,687 | 60.5% | 61.5% | +0.9 (-0.1 to +1.9) |
| Risky parlay (same match) | 10,365 | 33.7% | 34.1% | +0.4 (-0.4 to +1.3) |
| Safe parlay across matches | 918 | 63.2% | 66.2% | +3.1 (-0.1 to +6.4) |
| Risky parlay across matches | 943 | 44.7% | 47.4% | +2.7 (-0.3 to +5.8) |

PASS by the same rule. The desk's "Build your own slip" panel offers best markets only. Every
leg, new or old, is logged and graded daily on the scorecard. Walk-forward
best lists changed by season (cards were best for 2021-2023 on the data then available, not now).
This says the chances are honest; it is not evidence of an edge over the bookmakers.
