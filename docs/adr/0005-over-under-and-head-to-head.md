# 0005 Over/under 2.5 and head-to-head: pre-registered tests

Status: accepted, 2026-11-20. Written before any over/under or head-to-head result existed.

## Why
Harsh bets on goals markets (over/under, BTTS, Asian totals) using head-to-head history, team records and goals scored. ADR 0003 only tested the 1X2 market with a model that ignores head-to-head. This ADR tests his approach directly.

## Test A: does the model beat the over/under 2.5 market?
- **Model:** `dixon_coles` with its locked settings (no retuning). P(over 2.5) = sum of its scoreline probabilities with 3+ goals.
- **Market:** closing over/under 2.5 odds, Pinnacle (`PC>2.5`, `PC<2.5`), falling back to the market average (`AvgC>2.5`) where Pinnacle is missing; margin removed (proportional).
- **Method and criteria:** identical to the ADR 0003 addendum (stack test, β per season fitted on earlier seasons only, Δ ≥ 0.005 with interval above 0, β interval above 0, Brier not worse, positive in ≥ 4 of 6 seasons and every league). Evaluation seasons 2020/21-2025/26. Over/under odds start in 2019/20, so 2019/20 only serves to fit β.

## Test B: does head-to-head history improve the model?
- **Feature (fixed in advance, not tuned):** for a match between home team H and away team A, take up to the **6 most recent previous meetings** between them (either venue) in the **5 years** before the match, using only matches the model is allowed to see. From H's point of view:
  - `h2h_gd` = sum of goal differences ÷ (meetings + 2)
  - `h2h_tot` = sum of (total goals − league average total) ÷ (meetings + 2)
  - The "+2" shrinks the feature towards zero when there are few meetings.
- **Model `dixon_coles_h2h`:** `dixon_coles` plus two coefficients, γ_gd and γ_tot (expected goals: home × exp(γ_gd·h2h_gd + γ_tot·h2h_tot), away × exp(−γ_gd·h2h_gd + γ_tot·h2h_tot)), estimated in each fit with the same ridge penalty.
- **Comparison:** paired per-match log-loss of `dixon_coles_h2h` minus `dixon_coles` on 2020/21-2025/26, for 1X2 and for over/under 2.5 separately. **Head-to-head helps** if the improvement is positive and its 95% interval (block bootstrap, league × week) is above 0.
- If it helps, `dixon_coles_h2h` is also stack-tested against the market (1X2 and over/under).

## Multiple testing
Market tests in this ADR: Test A, plus up to two for `dixon_coles_h2h`. Together with ADR 0003's test, intervals for market tests here are **98.75%** (Bonferroni for 4 market tests).

## BTTS
No historical BTTS odds exist in the data, so BTTS cannot be tested against the market. Its log-loss is reported for information only.
