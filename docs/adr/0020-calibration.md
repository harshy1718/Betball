# 0020 Calibration of the goals and cards probabilities

Status: pre-registered 2026-11-24, before the confirmation runs below. Result appended after.

## Why

The audit (24 Nov 2026) found Nox's raw probabilities are reliable on average but bent at the
edges: over/under 2.5 and both-teams-score spread too wide, clear favourites win more often than
predicted (64% said, 70% happened), cards over-spread at 60-70%. An outside review proposed a
calibration layer per market. A layer per market would break the desk: every goal leg (double
chance, handicaps, team goals, correct score) and every same-match parlay is read off one score
grid, so the grid itself must be calibrated, or single legs and parlays stop agreeing.

## What was already looked at (disclosed, not hidden)

On 24 Nov, using the saved walk-forward predictions of 2018/19-2025/26 only (2026/27 untouched):

* per market, fitted on earlier seasons, tested 2020/21-2025/26: temperature scaling on match
  result 0.98295 -> 0.98240; Platt on over/under 0.68298 -> 0.68191, on both-score
  0.68920 -> 0.68714; isotonic was worse than Platt in every market;
* three score-grid maps, same design. The one chosen below scored match result
  0.98306 -> 0.98157, over/under 0.68298 -> 0.68193, both-score 0.68931 -> 0.68692, correct
  score 2.86595 -> 2.86379 (grid rebuilt with rho -0.0251).

Those seasons are therefore not a clean test of the chosen method. The clean test is 2026/27.

## The method (fixed now)

**Goals.** With s = log lambda - log mu (who is stronger) and t = log lambda + log mu (how many
goals), the calibrated expected goals use s' = a*s + b and t' = t_ref + d*(t - t_ref) + c,
where t_ref is the mean of t in the calibration data; then the score grid's draw cells are
multiplied by exp(delta) and the grid renormalised. Five numbers (a, b, d, c, delta), fitted by
minimising the summed log loss of match result + over/under 2.5 + both-score on out-of-sample
predictions of earlier seasons only. One map for all three leagues.

**Cards.** The match total's mean is pulled towards the league-average model's mean:
m' = base^(1-d) * m^d, and the negative-binomial dispersion is scaled, k' = f*k. Two numbers,
fitted by minimising the summed log loss of over 3.5, 4.5, 5.5 and 6.5 cards on earlier seasons.

Every calibrated walk-forward prediction uses a map fitted only on seasons before it. The live
map is fitted on every out-of-sample season available.

## Decision rule (fixed now)

Goals calibration is adopted only if **both**:

1. on 2020/21-2025/26 (expanding window, map from earlier seasons) the summed log loss of the
   three markets is lower than raw, and no one of them is worse by more than 0.001;
2. on 2026/27 so far (329 matches, never used above; map fitted on 2018/19-2025/26) the summed
   log loss of the three markets is lower than raw.

Cards calibration: the same two conditions on the summed log loss of the four lines
(2020/21-2025/26 expanding window, then 2026/27; the count model's walk-forward is extended to
2018/19-2026/27 for this).

A 95% block-bootstrap interval (league x week) is reported with each result. 329 matches cannot
prove a small gain, so condition 2 is a guard against a method that only fitted the past, not
a significance test.

**Live check.** The prediction log keeps the raw chance next to the calibrated one. At 300
graded live matches, if the calibrated Brier score summed over match result, over/under and
both-score is worse than the raw one, calibration is switched off.

**Confidence labels** are rebuilt from the calibrated walk-forward predictions, because the
labels describe the numbers the desk actually shows.

## Result (run 2026-11-25, `football-edge calibrate`, reports/calibration/verdict.md)

**Goals: ADOPTED.** Rebuilt grids use rho -0.0351 (current fit) and the edge-folded grid, so the
raw numbers differ slightly from the exploration above.

| 2020/21-2025/26 (walk-forward) | raw | calibrated |
|---|---|---|
| Match result | 0.98288 | 0.98168 |
| Over/under 2.5 | 0.68298 | 0.68197 |
| Both score | 0.68920 | 0.68691 |
| Correct score | 2.86592 | 2.86412 |

* Condition 1 PASS: the three markets together -0.00450 per match (95% interval -0.00767 to
  -0.00132); none got worse. Match result got worse in one season (2022/23), better in five.
* Condition 2 PASS: 2026/27 so far (329 matches) 2.32718 -> 2.31567, every market better
  (match result 0.9548 -> 0.9513, over/under 0.6827 -> 0.6794, both score 0.6896 -> 0.6850);
  interval -0.0261 to +0.0021, as expected with 329 matches.
* The fitted numbers are stable season to season: a 1.07-1.10 (the gap between the sides is
  stretched ~10%: shrinkage made the model too cautious about clear favourites), d 0.59-0.67
  (each match's goal total is pulled ~35-40% towards the average: totals were too spread),
  delta +0.07 to +0.11 (draws ~10% more likely than Poisson says).

**Cards: ADOPTED.** Four lines summed, 2020/21-2025/26: 2.30230 -> 2.29148. 2026/27:
2.24472 -> 2.23954 (interval -0.0298 to +0.0187). d 0.56-0.66: expected card totals are pulled
~35-45% towards the league-average model.

Live maps, fitted on 2018/19-2026/27 (9,449 matches): goals a 1.111, b -0.009, d 0.650,
c +0.059, delta +0.116; cards d 0.658, f 0.679.

**Consequences.**
* Confidence labels rebuilt from the calibrated walk-forward predictions. More predictions are
  now High (main markets: 171,708 against 160,262 before).
* The strong-call rule (ADR 0019) now gives **favourites at 65%+** (74% of 1,116 came true) and,
  for the first time, **over/under 2.5 at 65%+** (66% of 332).
* Corners are no longer forecast anywhere (site, desk, log): the team model never beat the
  league average (ADR 0008), and showing its numbers implied a skill it does not have.
* The prediction log keeps each leg's raw chance (`p_raw`); the scorecard's "Is the calibration
  helping?" card is the live check. Decision at 300 graded matches.

**Desk replay with nothing from the future** (`football-edge replay-desk`, reports/replay/):
calibrated chances, labels for each season built only from earlier seasons, corners/cards
lift from matches before that week. 2020/21-2025/26:

| Slip | Slips | Predicted | Landed | Gap (95% interval) |
|---|---|---|---|---|
| Good singles | 25,870 | 75.4% | 75.8% | +0.4 (-0.2 to +0.9) |
| Top single per match | 6,826 | 80.2% | 80.7% | +0.5 (-0.4 to +1.4) |
| Safe parlay (same match) | 6,672 | 60.8% | 62.3% | +1.5 (+0.3 to +2.7) |
| Risky parlay (same match) | 6,012 | 33.2% | 33.4% | +0.2 (-1.0 to +1.4) |
| Safe parlay across matches | 842 | 63.0% | 65.4% | +2.4 (-1.0 to +5.9) |
| Risky parlay across matches | 887 | 42.1% | 44.8% | +2.7 (-0.4 to +5.8) |

The audit's conclusion holds without the leaks: the desk's chances are honest. Safe parlays are
slightly conservative (they land ~1.5 points more than promised). Intervals are blocked by
league x week, so 25,870 singles count as 640 independent weeks, not 25,870 trials.
