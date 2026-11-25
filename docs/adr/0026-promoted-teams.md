# 0026 Promoted teams: pre-registered fix

Status: pre-registered 2026-11-25, before either variant was fitted. Result appended.

## Why

Diagnostics (reports/diagnostics/residuals.md): with the live calibration, promoted teams beat
the model by +0.126 goals per match over their first 19 matches (95%: +0.055 to +0.196); it
fades in the second half of the season. A follow-up check before this ADR: in the *raw* model
the gap is about half as big and mostly inside noise (+0.05 in matches 1-5, +0.01 in 6-10,
+0.09 in 11-19). The calibration stretches every match's expected goal difference by about 8%
(a = 1.08). Promoted teams are usually underdogs, so the stretch makes them bigger underdogs:
it fits established teams but over-stretches teams whose strength is still uncertain.

## Variants (fixed now)

* **A. Promoted stretch in the calibration.** The calibration gets one more number, `a_p`: the
  stretch of the expected goal difference in matches where either side is a promoted team in its
  first 19 league matches of the season (match number counted before kickoff). Every other match
  keeps `a`. Fitted with the other five numbers on earlier seasons only, same objective.
* **B. Weaker pull towards the promoted prior.** In the Dixon-Coles fit, a team promoted this
  season is pulled towards the average-promoted prior with half the usual strength
  (`promoted_pull` 0.5; others 1.0), so its own results take over sooner. Refitted on every weekly
  fold 2018/19-2026/27; then the usual calibration (five numbers) refitted on its predictions.

## Decision rule (fixed now)

Champion: the live model, Dixon-Coles weekly walk-forward with the five-number calibration fitted
season by season on earlier seasons. Score per match: match result + over/under 2.5 + both-score
log loss, same matches. A variant passes only if both:

1. 2020/21-2025/26: the difference is below zero with the block-bootstrap interval (league x week)
   entirely below zero at **98.3%** (95% Bonferroni for 3 tests: A, B and ADR 0027);
2. 2026/27 so far: the difference is below zero (point estimate).

If both pass, the better on 2020-26 is taken; combining them is a separate test. Secondary, not
part of the rule: the promoted-team residual for matches 1-19 after the change.

## Result

Run 2026-11-25 (`promoted-spread`, reports/challenger/promoted/verdict.md). Live model 2.34941
per match on 10,734 matches 2020/21-2025/26 (reproduces ADR 0020's 2.34942).

| Variant | Difference | 98.3% interval | 2026/27 | Verdict |
|---|---|---|---|---|
| A: promoted stretch in the calibration | -0.00013 | -0.00086 to +0.00057 | -0.00100 | FAIL |
| B: weaker pull to the promoted prior | +0.00080 | +0.00043 to +0.00119 | +0.00244 | FAIL (worse) |

**FAIL: the live model stays.** Notes:

* A does remove the bias: the promoted-side residual over matches 1-19 drops from +0.103 to
  +0.016 goals per match, and the fitted `a_p` is about 0.97 against `a` about 1.10 every season
  (promoted-early matches should not be stretched). But the accuracy gain is too small to tell
  from luck: a bias of 0.1 goals moves a win chance by 2-3 points, and log loss only grows with
  the square of such small errors.
* B makes it worse: the prior is not the problem; loosening it lets early results add noise.
* A stays a watch item. It may be re-tested once, on 2026/27 and 2027/28 matches only (data not
  used here), with the same rule; it is not adopted now.
