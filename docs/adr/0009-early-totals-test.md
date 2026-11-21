# 0009 Early over/under prices: pre-registered rules

Status: accepted, 2026-11-21. Written before the test was first run.

## Question
ADR 0005 found the model adds nothing to the **closing** over/under 2.5 market (pooled beta 0.000).
Harsh asks about the **early** prices, set before line-ups and team news. Are they soft enough for the
model's total-goals probabilities to make money?

## Data
- football-data.co.uk "pre-match" over/under 2.5 odds, collected on Friday afternoons for weekend games
  and on Tuesdays for midweek games: Pinnacle (`P>2.5`, `P<2.5`), Bet365 (`B365>2.5`), best price across
  bookmakers (`Max>2.5`). These are the earliest prices in the data, not the first prices ever posted.
- Pinnacle closing (`PC>2.5`) for closing line value.
- Model: `dixon_coles` walk-forward `p_over25` with its locked settings (no retuning), seasons 2020/21 to
  2025/26 (E0, I1, SP1).

## Test A (information): does the model add to the early sharp price?
Paired per-match log-loss, early Pinnacle fair probability (margin removed, proportional) against the
model, and the blend logit(p) = logit(market) + beta * (logit(model) - logit(market)) with beta fitted on
earlier seasons only. Reported, with a 98.75% block-bootstrap interval; it explains Test B but does not
by itself pass or fail anything.

## Test B (money): betting the model against early prices
- **Edge of a side** (over or under) = model probability x price - 1. Per match, take the side with the
  larger edge; bet if edge >= threshold. Flat stake 100, at most one bet per match.

| Variant | Bet at | Threshold |
|---|---|---|
| b365_3 | Bet365 early price | 3% |
| b365_6 | Bet365 early price | 6% |
| max_3 | best early price across bookmakers (`Max`) | 3% |
| max_6 | best early price across bookmakers | 6% |

- **Interval:** block bootstrap (league x week), 2,000 resamples, Bonferroni for 4 variants: **98.75%**.
- **Pass (per variant):** ROI > 0 with the lower end of its interval > 0; ROI > 0 in at least 4 of the
  6 seasons; at least 200 bets.
- **Closing line value:** mean of bet price x Pinnacle closing fair probability - 1. Reported for all.

## Known limits (stated in advance)
- `Max` is an upper bound: it picks the best of many bookmakers at that moment, some of which limit
  winning accounts. A pass would be an upper bound on a real edge; a fail is strong evidence of none.
- Only the 2.5 line has historical prices; other Asian total lines (2.25, 2.75, 3.0) cannot be tested.
- Thresholds and variants are fixed here and are not changed after seeing results.
