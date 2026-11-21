# 0004 Price-comparison ("soft-book lag") test: pre-registered rules

Status: accepted, 2026-11-20. Written before the test was first run.

## Question
Without any prediction model: if we only bet when the **best available price across bookmakers** beats the **sharp (Pinnacle) fair price**, do we make money?

## Definitions
- **Fair probability:** Pinnacle odds for the same moment, margin removed (proportional method, `remove_margin`).
- **Best price:** the maximum decimal odds across bookmakers in the football-data.co.uk file (`Max*` columns).
- **Edge of an outcome:** best price × fair probability − 1.
- **Bet rule:** per match, take the outcome with the largest edge; bet if edge ≥ threshold. Flat stake of 100. At most one bet per match.

## Variants (all four reported; none chosen after seeing results)
| Variant | Fair price from | Bet at | Threshold |
|---|---|---|---|
| pre_2 | Pinnacle pre-match (`PS*`) | best pre-match (`Max*`) | 2% |
| pre_5 | Pinnacle pre-match | best pre-match | 5% |
| close_2 | Pinnacle closing (`PSC*`) | best closing (`MaxC*`) | 2% |
| close_5 | Pinnacle closing | best closing | 5% |

## Data
Seasons 2019/20-2025/26 for E0, I1, SP1 (the `Max*` columns start in 2019/20). Only matches where the Pinnacle prices for that variant exist (about half of 2025/26 is missing them).

## Measures
- **ROI** = profit ÷ total staked.
- **Interval:** block bootstrap (blocks = league × week), 2,000 resamples. Four variants are tested, so intervals are Bonferroni-adjusted: **98.75%** (quantiles 0.625% and 99.375%).
- **Closing line value** (pre-match variants only): average of bet price × Pinnacle *closing* fair probability − 1. Positive means we consistently got better prices than where the sharp market ended up.

## Pass criteria (per variant)
1. ROI > 0 and the lower end of its 98.75% interval > 0.
2. ROI > 0 in at least 4 of the 7 seasons.
3. At least 200 bets.

## Known limits (stated in advance)
- `Max*` is the best price across many bookmakers at the moment the file was collected; some are soft books that limit winners or void obvious errors, and the stake actually available is unknown. **A pass is an upper bound on a real-world edge, not proof of one. A fail is strong evidence that no edge exists here.**
- Pre-match prices are snapshots collected a day or more before kick-off, not continuous.
