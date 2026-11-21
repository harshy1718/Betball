# 0010 Head-to-head with a chosen weight: pre-registered rules

Status: accepted, 2026-11-22. Written before the test was first run.

## Question
Harsh: keep the team-strength model, but before each fixture also use the two teams' head-to-head
results from the **past 3 years**, give them a weight, and pick the weight that predicts final scores
best. Does it change anything? (ADR 0005 tested a learned head-to-head term, 6 meetings / 5 years,
and it did not help.)

## Method
For each fixture, with the model's walk-forward expected goals (lambda_home, lambda_away) from
`dixon_coles` (locked settings, never trained on the week it predicts):
1. Meetings: every earlier match between the two teams in the 3 years before the fixture, either
   venue, from our data only.
2. Venue adjustment: goals scored as the away side are scaled by the league's home/away goals
   ratio (from matches before the fixture), so a meeting at the other ground counts fairly.
3. Head-to-head rate per team, with one pseudo-meeting at the model's own rate so 1 meeting cannot
   dominate: rate = (adjusted goals in meetings + lambda) / (meetings + 1).
4. Blend, weight w: lambda' = lambda^(1 - w) * rate^w. w = 0 is the current model; w = 1 uses only the
   head-to-head rate. Fixtures with no meeting are unchanged.
5. Probabilities from lambda' with independent Poisson goals (scores 0-10). The w = 0 baseline is
   computed the same way, so the comparison is like for like.

## Choosing w without cheating
Grid w = 0, 0.05, ..., 1. Walk-forward: for each season from 2020/21 to 2025/26, w is the grid value
with the best exact-score log-loss on **all earlier seasons only** (predictions start 2018/19). The
season is then scored with that w. The full in-sample curve is also reported, for information only.

## Measures and verdict
- Primary: **exact final score** log-loss (the probability given to the score that happened).
- Also reported: 1X2 and over/under 2.5 log-loss.
- **Head-to-head helps** if the primary improvement (walk-forward w vs w = 0) is positive and its 95%
  block-bootstrap interval (league x week, 2,000 resamples) is above 0. Otherwise it does not help.
