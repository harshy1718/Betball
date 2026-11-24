# 0021 Referee effect on Premier League cards: pre-registered test

Status: pre-registered 2026-11-25, before any referee result was looked at. Result appended.

## Hypothesis

Some referees show many more cards than others. Adding a per-referee effect to the cards model
makes Premier League card probabilities more accurate. The outside review (24 Nov 2026) ranked
this the most promising improvement for cards; ADR 0008 parked it for lack of data.

## Data

Only the Premier League files carry the referee (football-data.co.uk `Referee`, every season
2015/16-2026/27, no gaps). Serie A and La Liga have none, so the test and any use are Premier
League only. The referee is announced before kickoff, so it is fair pre-match information.

## Method (fixed now)

* Two cards models, identical except one adds a referee effect: log rate of each side +=
  ref[referee], with the same ridge penalty (1.0) as the team effects; an unseen referee gets 0.
  Both are trained on Premier League matches only, so the comparison isolates the referee.
* Weekly walk-forward folds (same as every other test): train on everything before Monday,
  predict that week. Test seasons 2020/21-2025/26 (2,280 matches).
* Score: summed log loss of over 3.5, 4.5, 5.5 and 6.5 cards, per match, raw (uncalibrated)
  models.

## Decision rule (fixed now)

Adopt the referee effect only if the mean per-match improvement (without referee minus with)
is above zero with its 95% block-bootstrap interval (blocks = weeks) entirely above zero on
2020/21-2025/26, **and** the 2026/27 matches so far also improve (point estimate). Otherwise
the cards model stays as it is and the referee stays out.

## Result

(appended after the run)

**Run 2026-11-25 (`football-edge referee-test`, reports/referee_cards/verdict.md): FAIL.**

| Season | without referee | with referee | gain |
|---|---|---|---|
| 2020/21 | 1.5594 | 1.5183 | +0.0411 |
| 2021/22 | 2.0334 | 2.0320 | +0.0013 |
| 2022/23 | 2.0323 | 2.0608 | -0.0284 |
| 2023/24 | 2.4235 | 2.4770 | -0.0536 |
| 2024/25 | 2.3172 | 2.3591 | -0.0420 |
| 2025/26 | 2.1612 | 2.2020 | -0.0407 |

2020/21-2025/26: the referee made cards *less* accurate, -0.0204 per match (95% interval
-0.0452 to +0.0032). 2026/27 so far (100 matches): -0.0057. The referee stays out. The likely
reason: each referee takes only 15-25 games a season, so his effect is mostly noise, and
referee styles drift as the league's card policy changes year to year (2023/24's crackdown on
dissent). The code (`CountParams.referee_effects`, off) and the `referee` column stay so the
test can be re-run.
