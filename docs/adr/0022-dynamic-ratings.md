# 0022 Dynamic team ratings: pre-registered challenger test

Status: pre-registered 2026-11-25, before any tuning or test result. Result appended.

## Hypothesis

The locked model gives each team one attack and one defence rating, learned from all past
matches with older ones counting less (time decay, half-life ~9 months). Teams really change
during a season (injuries, a new manager, form), so ratings that move over time
should predict better. The outside review proposed this (alpha_t = alpha_{t-1} + noise).

## The challenger (`dixon_coles_dynamic`)

Identical to the locked model (same goals model, shots blend 0.25, low-score correction, league
scoring and home advantage, no-crowd term, ridge 1.0, promoted-team prior), except:

* each team has an attack and a defence rating **per period of `dynamic_weeks` weeks** in which it
  played;
* consecutive periods are tied by a random walk: the penalty `rw * (change)^2 / gap` keeps a
  rating from jumping unless the results insist; the gap (in periods) lets a rating move more
  over a summer break;
* the ridge towards the prior applies to the first period of each spell in the league (a team
  absent the season before starts a new spell from the promoted-team average);
* a prediction uses the team's latest period rating (the random walk's forecast).

Fitted by the same penalised maximum likelihood (L-BFGS-B, exact gradients). A full hierarchical
Bayesian version is not tested: the locked model's ridge already is a Gaussian prior whose
strength was tuned, which is what a hierarchy would estimate.

## Tuning (fixed now)

Grid on 2018/19-2019/20 only, judged by match-result log loss, exactly as the locked model was
tuned (ADR 0002 folds, reports/tuning/dixon_coles.csv):
`dynamic_weeks` in {4, 8}, `rw` in {10, 40, 160}, `xi` in {0, 0.0025}. The best is locked.

## Decision rule (fixed now)

The challenger replaces the locked model only if **both**:

1. on 2020/21-2025/26 (weekly walk-forward, same folds), the per-match sum of match result +
   over/under 2.5 + both-score log loss improves, with the 95% block-bootstrap interval
   (league x week) entirely below zero;
2. on 2026/27 so far, the same sum improves (point estimate).

Raw models are compared (no calibration on either). If adopted, calibration (ADR 0020) and the
confidence labels are refitted on the new model's walk-forward predictions before going live.

## Result (run 2026-11-25)

**Tuning** (`football-edge tune --model dixon_coles_dynamic`, reports/tuning/dixon_coles_dynamic.csv):
best `dynamic_weeks` 8, `rw` 160, `xi` 0, match-result log loss 0.96874 on 2018/19-2019/20
(the locked model scored 0.97037 there). Locked. The winner sits at the slow edge of the grid:
the data preferred ratings that change slowly. Note for later: a larger `rw` moves the model
back towards one rating per team, so there is little room left on that side.

**Test** (`football-edge challenge`, reports/challenger/dixon_coles_dynamic/verdict.md), both
models refitted on every weekly fold:

| 2020/21-2025/26 | locked | dynamic |
|---|---|---|
| Match result | 0.98295 | 0.98237 |
| Over/under 2.5 | 0.68298 | 0.68516 |
| Both score | 0.68920 | 0.69082 |
| Sum of the three | 2.35512 | 2.35835 |

Condition 1 FAIL: the sum got worse by +0.00322 per match (95% interval +0.00093 to +0.00542),
worse in five of six seasons. Condition 2 (2026/27, 329 matches): slightly better, -0.00411
(interval -0.0168 to +0.0091), not enough to matter.

**FAIL: the locked model stays.** Moving ratings helped a little on who wins (-0.0006) but
hurt goal totals and both-score more: ratings fitted to short periods chase noise in how many
goals a team scores and concedes, and the time-decayed single rating already follows form.
The option (`dynamic_weeks`, `rw`) stays in the code, off, with its tests.
