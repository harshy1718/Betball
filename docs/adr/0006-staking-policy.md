# 0006 Probability app and confidence-based staking

Status: accepted, 2026-11-20.

## Context
Four pre-registered tests (ADR 0003, 0004, 0005) found no edge over the market. The model's probabilities are well calibrated: when it says 74%, the event happened about 75% of the time. Harsh's goal is now a tool that gives the probability of each event and stakes more when the model is more confident, within a fixed budget, accepting the risk.

## Decision
`football-edge predict` forecasts upcoming fixtures with the evaluated model (`dixon_coles`, locked settings, trained on every match before the first fixture) and applies this policy (all numbers live in `configs/default.yaml` under `staking:`):

1. **Selection:** the outcome the model rates most likely in each market (1X2, over/under 2.5, BTTS).
2. **Confidence tiers:** the selection's probability sets a tier, and the tier sets the stake as a share of the **monthly budget** (not of the money left, so a losing run does not make stakes chase losses). Two-way markets have higher thresholds because 55% on a two-way market is barely better than a coin flip.
3. **Price guard:** skip when expected value `p × odds − 1` is below `−price_tolerance` (6%). That is worse than a normal bookmaker margin, so the price is poor even by the standard of a no-edge bettor. `min_odds = (1 − tolerance) / p` is shown for every pick.
4. **Caps:** at most 4% of the budget on one match (its markets are correlated); never more than the money left this month; **stop-loss** when the month's settled result reaches −30% of the budget.
5. **Doubles:** at most two legs, from different matches, both high-tier and priced. Same-match legs are excluded because their outcomes are correlated and the combined price misleads.
6. **Record:** placed bets go to `bets/log.csv` (`--log`), and `football-edge settle` marks them won or lost from real results. The monthly summary compares wins with the wins the model expected, which is the live check that the model is still calibrated.

## What this does not claim
This does not create an edge. On a site with an ~8% margin, the backtest of confidence staking lost about 5–16% of the amount staked, and doubles lost 10.6%. The expected cost is roughly the margin times the total staked. The policy controls how fast that can happen and avoids the worst prices; it is an entertainment and risk budget, not an income.

## Revisit when
- A pre-registered test finds an edge (staking would then move to fractional Kelly on the edge).
- The live log shows wins far from expected wins over 100+ settled bets (calibration has drifted).

## Amendment, 2026-11-20: confidence uses the market's fair probability
ADR 0007 measured the margin-free market as far more accurate than any model here (log-loss 2020–25: Pinnacle pre-match 0.967, closing 0.964; best model 0.981). So the probability that sets the selection, tier, fair odds, min odds and expected value is now the **market's fair probability** whenever prices are given: Pinnacle (`ref_*` columns, or `PS*` in football-data files) first, else the site's own prices, margin removed proportionally. The model's probability is still shown beside it, and picks where the two differ by 8+ points are flagged (`disagree_at`). `confidence: model` in the config restores the old behaviour.

Consequence to keep in mind: when the only prices are the site's own, the market probability *is* the site's view, so expected value is about minus the site's margin and the price guard will skip more bets. That is the honest number.

