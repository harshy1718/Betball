# 0023 A model per league: pre-registered test against the combined model

Status: pre-registered 2026-11-25, before any tuning or test result. Result appended.

## Hypothesis

Harsh: every league behaves differently. Today one model is fitted on all three leagues: team
ratings, scoring level, home advantage and goals-per-shot are already per league, but time
decay, ridge, shots weight, the low-score correction (rho), the empty-stadium term and the
calibration map are shared. Letting each league have its own may fit each league better.
The risk: each league has a third of the data, so its own settings may chase noise.

## Stage A: the goals model (`dixon_coles_per_league`)

* Three separate Dixon-Coles models, one per league, each fitted only on its own league's
  matches: own rho, own empty-stadium term, own promoted-team prior.
* Each league's `xi`, `ridge` and `shots_weight` tuned on that league's 2018/19-2019/20
  matches only, on the same 45-combination grid the combined model was tuned on, judged by
  that league's match-result log loss. The best per league is locked.
* Same weekly folds as every other test (trained on everything before Monday; each league's
  model sees only its league's part of the training data).

## Stage B: the calibration (ADR 0020)

* One calibration map per league, each fitted only on that league's earlier seasons, against
  the shared map; both applied to the walk-forward predictions of whichever model wins Stage
  A (the combined model if Stage A fails).

## Decision rule (fixed now, for each stage separately)

The per-league version replaces the combined one only if **both**:

1. on 2020/21-2025/26, the per-match sum of match result + over/under 2.5 + both-score log
   loss improves, with the 95% block-bootstrap interval (league x week) entirely below zero;
2. on 2026/27 so far, the same sum improves (point estimate).

Also reported, not deciding: each league on its own, so a league that does better alone is
visible even if the total fails. A per-league result does not reopen the decision for that
league alone: that would be choosing after seeing results.

## Result (run 2026-11-25)

**Stage A tuning** (`football-edge tune-per-league`, reports/tuning/dixon_coles_per_league.csv),
best per league on its own 2018/19-2019/20: Premier League xi 0.0019, ridge 1, shots 0.25;
Serie A xi 0.0035, ridge 1, shots 0.25; La Liga xi 0.0019, ridge 2, shots 0.25. Locked.

**Stage A test** (`football-edge challenge --model dixon_coles_per_league`): FAIL.

| 2020/21-2025/26 | combined | per league |
|---|---|---|
| Match result | 0.98295 | 0.98296 |
| Over/under 2.5 | 0.68298 | 0.68278 |
| Both score | 0.68920 | 0.68975 |
| Sum | 2.35512 | 2.35550 |

Difference +0.00037 per match (95% interval -0.00134 to +0.00202): no real difference either
way. 2026/27: worse, +0.00232. By league (sum, 2020-26 / 2026-27): Premier League -0.0001 /
+0.0036, Serie A +0.0025 / +0.0029, La Liga -0.0012 / +0.0008.

**Stage B** (`football-edge per-league-calibration`, reports/calibration/per_league_verdict.md),
on the combined model: FAIL. A map per league was worse in every league (2020-26 sum 2.35056
-> 2.35227, interval -0.00059 to +0.00403; 2026/27 2.31567 -> 2.31876).

**The combined model and the shared calibration stay.** The leagues do differ, but the parts
that differ (team strength, scoring level, home advantage, goals per shot) were already per
league. What a separate model adds is its own decay, shrinkage and low-score correction, and
with a third of the data each, those extra settings fit noise as much as signal. The live
scorecard now shows each league on its own; if one drifts away from the others on live
matches, that is new evidence and a reason to test again.
