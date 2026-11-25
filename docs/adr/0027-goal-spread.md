# 0027 Less spread in goal counts: pre-registered test

Status: pre-registered 2026-11-25, before the model was fitted. Result appended.

## Why

Diagnostics (reports/diagnostics/residuals.md): with the live calibration, goal counts are less
spread out than the Poisson score grid assumes (squared error over predicted variance 0.94 home,
0.95 away; chance about +-0.025). Fewer 0-0s (6.4% vs 6.9% predicted) and fewer 6+ goal matches
(6.8% vs 7.4%), more 1-goal and 3-goal matches. Negative binomial (more spread) and
zero-inflation (more 0-0s) point the wrong way, so they are not tested.

## Challenger (fixed now)

Each side's goal count follows a Conway-Maxwell-Poisson distribution with dispersion `nu`
(1 = Poisson, above 1 = less spread), with its mean kept at the calibrated expected goals; the
Dixon-Coles low-score correction (current rho) and the draw adjustment are applied on top as now.
`nu` is one number shared by all matches, fitted together with the five calibration numbers on
earlier seasons only, same objective (match result + over/under 2.5 + both-score log loss),
bounded 0.7-1.6.

## Decision rule (fixed now)

As ADR 0026: against the live model (five-number calibration), 2020/21-2025/26 interval entirely
below zero at 98.3% (Bonferroni for 3 tests) and 2026/27 better. Secondary: the spread ratio and
0-0 / 6+ frequencies after the change.

## Result

Run 2026-11-25 (`promoted-spread`, reports/challenger/promoted/verdict.md).

| Challenger | Difference | 98.3% interval | 2026/27 | Verdict |
|---|---|---|---|---|
| Less spread (Conway-Maxwell-Poisson) | -0.00007 | -0.00020 to +0.00004 | -0.00003 | FAIL |

**FAIL: the live model stays.** The fitted `nu` is 1.04-1.07 in recent seasons (slightly less
spread than Poisson, as the diagnostics said) and the spread check moves from 0.94-0.95 to 0.98;
predicted 0-0 falls from 6.9% to 6.6% (observed 6.4%). The shape is real, but fixing it improves
the three markets by 0.00007 per match, inside luck. The draw adjustment and the Dixon-Coles
low-score correction already absorb most of it where it matters (0-0, 1-0, 1-1).
