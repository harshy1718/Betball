# 0014 Golden bar per bet type and per league: pre-registered rules

Status: accepted, 2026-11-22. Written before the search was run.

## Question
Harsh: look for a golden bar (a probability window where betting the model makes money) for each
bet type on its own (match winner; over/under 2.5) and, for each, over all three leagues and then
each league separately.

## Method
Same as ADR 0012 (model only, `dixon_coles` walk-forward, Bet365 pre-match prices, flat 100;
windows lo 0.20-0.90 x width 0.02-0.30, steps of 0.01). Match winner: home, draw and away all
count; over/under: over and under both count.
- 8 searches: {match winner, over/under} x {all leagues, Premier League, Serie A, La Liga}.
- Minimum bets in the search seasons: 200 for all leagues, 100 for one league.
- Search seasons 2018/19-2022/23; the best window of each search is tested on 2023/24-2025/26,
  which the search never saw.
- **Golden bar found** only if the unseen ROI > 0 and the lower end of its block-bootstrap interval
  (league x week, 2,000 resamples) is above 0 at **99.375%** (Bonferroni for 8 searches).
- Placebo for each search (model probabilities shuffled within league-season): count of
  "profitable" windows reported beside the real count.
- Also reported, not part of the verdict: the existing match-winner bar (60-64%) on unseen seasons
  in each league.
