# 0008 Corners, cards and confidence labels

Status: accepted, 2026-11-20.

## Corners and cards
Football-data files carry corners (HC/AC), yellow cards (HY/AY) and red cards (HR/AR) for every match since 2015; they are now loaded and treated as after-the-match information (hidden from models for the match predicted). `models/counts.py` rates each team for what it racks up and what it allows (same time decay and shrinkage as the goals model); match totals are negative binomial with a per-league dispersion.

Walk-forward test, 2020/21–2025/26 (6,840 matches), log-loss against the league-average model:

| Line | Team model | League average | Improvement |
|---|---|---|---|
| Corners over 8.5 | 0.6572 | 0.6589 | +0.0017 |
| Corners over 9.5 | 0.6874 | 0.6875 | +0.0001 |
| Corners over 10.5 | 0.6627 | 0.6609 | −0.0018 |
| Corners over 11.5 | 0.5834 | 0.5839 | +0.0005 |
| Yellows over 3.5 | 0.6438 | 0.6558 | **+0.0121** |
| Yellows over 4.5 | 0.6491 | 0.6605 | **+0.0114** |
| Yellows over 5.5 | 0.5451 | 0.5503 | +0.0052 |
| Any red card | 0.4379 | 0.4352 | −0.0027 |
| Total cards over 3.5 | 0.6371 | 0.6484 | **+0.0113** |
| Total cards over 4.5 | 0.6552 | 0.6675 | **+0.0123** |
| Total cards over 5.5 | 0.5717 | 0.5782 | +0.0065 |
| Total cards over 6.5 | 0.4382 | 0.4433 | +0.0051 |

Total cards = yellows + reds, each card counting 1 (added 2026-11-20 at Harsh's request; this is what "cards" means on the card). Bookmakers differ on second yellows and some use booking points (10 per yellow, 25 per red); check your site's rule.

Decisions: team ratings are kept for corners, yellow cards and total cards; **red cards use the league average** (team effects scored worse). The corners model is no better than the league average and is over-confident at the extremes (when it said 67% for over 9.5, 61% happened). No bookmaker prices exist for these markets in the data, so they are shown as probabilities and never staked. Known missing factor: the referee (strongly linked to cards), parked until live referee data is available.

## Confidence labels
Each probability on the card carries a confidence label from its own track record (`predict/confidence.py`, `football-edge build-confidence`): per source (model or market), market and 5-point probability band, how often such predictions came true in 2020/21–2025/26. Rules fixed in advance: Low if no skill over a plain base rate, fewer than 200 past cases, or more than 6 points off; High if skilled, 500+ cases and within 3 points; else Medium. The label says how trustworthy the number is, not how likely the outcome is, and not that a bet is good value.
