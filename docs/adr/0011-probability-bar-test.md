# 0011 A probability bar per market: pre-registered rules

Status: accepted, 2026-11-22. Written before the test was first run.

## Question
Harsh: model only, no market prices in the decision. For each market, bet the model's most likely
outcome only when its probability is above a bar; each market has its own bar. Is there a bar that
makes a profit?

## Markets and data
- **Match winner (1X2)** and **over/under 2.5**: the model's walk-forward probabilities (`dixon_coles`,
  locked settings), seasons 2020/21-2025/26. Bets are settled at the football-data.co.uk pre-match
  prices: **Bet365** (primary, a real bookmaker) and the **market average** (reported, close to a
  typical site). The price only settles the bet; it plays no part in choosing it.
- **Both teams to score**: no historical prices exist, so profit cannot be measured. Reported: how often
  the pick came true at each bar, and the price you would need to break even.

## Bars
Match winner: 0.40, 0.45, ..., 0.85. Over/under 2.5: 0.50, 0.52, ..., 0.70. BTTS: 0.50, 0.52, ..., 0.66.
Flat stake 100 per bet, at most one bet per match per market.

## Choosing the bar without cheating
Walk-forward: for each season, the bar is the one with the highest ROI on **all earlier seasons**
(predictions start 2018/19) with at least 100 bets there; if no bar was profitable, that season
places no bets in that market. The season is then settled with that bar. The full in-sample table
(every bar on 2020-2025) is reported for information only: picking the best row afterwards is
hindsight, not a strategy.

## Verdict (per market with prices)
**Profitable** if walk-forward ROI > 0 with the lower end of its 97.5% block-bootstrap interval
(league x week; Bonferroni for 2 markets) above 0, and ROI > 0 in at least 4 of 6 seasons.
