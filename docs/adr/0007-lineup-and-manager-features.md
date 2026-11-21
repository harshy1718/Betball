# 0007 Line-up strength and manager change: pre-registered tests

Status: accepted, 2026-11-20. Written before any line-up or manager result existed.

## Why
Harsh considers player ratings critical, and wants the manager included. Team ratings (Dixon-Coles) learn each team's level from results but cannot see who plays today or that a manager has just changed. These tests ask whether that information makes the probabilities better.

## Data
`transfermarkt-datasets` (github.com/dcaribou/transfermarkt-datasets, CC0). Tables `games` (with each side's manager), `game_lineups` (starting XI) and `player_valuations` (market value history). The dataset stopped updating on 6 July 2026, so it covers the test seasons 2020/21–2025/26 but not the current one.

Clubs are linked to our team names automatically, from games with the same league, date and score (`reports/features/team_links.csv`); games are linked by league, home and away club and nearest date within 3 days.

## Features (fixed in advance, not tuned)
- **lineup** (per side): `log(XI value / median XI value in the team's league games of the previous 365 days)`, clipped to [−1.5, 0.5]. XI value = sum of the starters' latest market value published strictly before the match day (values older than 2 years ignored), scaled to 11 players; needs 9+ valued starters and 5+ earlier games, else unknown (0).
- **new_manager** (per side): 1 in a team's first 6 league games under a manager who replaced someone else; else 0.

## Models
- `dixon_coles_lineup` = `dixon_coles` (locked settings) + two coefficients: own XI strength on own goals, opponent XI strength on own goals conceded.
- `dixon_coles_manager` = `dixon_coles` + one coefficient on (new_home − new_away).
Coefficients are fitted inside every weekly fit with the same ridge penalty. Nothing is retuned.

## Test
Paired per-match log-loss of each model minus `dixon_coles`, on 2020/21–2025/26, for 1X2 and over/under 2.5 (`football-edge compare --level 0.9875`). **A feature helps** if the improvement is positive and its interval (block bootstrap over league × week) is above 0.

Multiple testing: 2 features × 2 markets = 4 comparisons, so **98.75%** intervals (Bonferroni). These seasons have already been used by ADR 0003–0005, which is why the correction is required rather than optional.

**Data sufficiency:** if the line-up feature is known for both sides in less than 80% of evaluation matches, the line-up result is reported as "not enough data", not as a pass or a fail.

## If a feature helps
1. Stack test against the closing market at **99.17%** (Bonferroni over 6 market tests so far). The closing price already knows the line-ups, so this is the real question.
2. Only then: wire it into `predict`. That needs live line-ups (API-Football, about an hour before kick-off) and current market values.

## If it does not help
The feature stays out of `predict`. The code stays, documented, for fresh seasons later.

## Honest caveat (stated now, before the numbers)
Using the actual starting XI assumes betting after line-ups are announced. If line-ups add nothing even then, a guessed XI will not either.

## Amendment, 2026-11-20 (before any result existed)
Harsh pointed out that market values are inflated by age, potential, club and league, so a team resting a cheap but central veteran would look barely weakened. Added a third, price-free feature:
- **trust** (per side): each player's share of the team's league starts in the previous 365 days (the coach's revealed opinion). `trust = log(sum of today's starters' shares / sum of the 11 highest shares)`, clipped to [−1.5, 0.5]; needs 5+ earlier games, else unknown (0). Its own blind spot: new signings have no starts yet.
- **Model** `dixon_coles_trust`: same form as `dixon_coles_lineup` (two coefficients).

Multiple testing is updated accordingly: 3 features × 2 markets = **6 comparisons, 99.17% intervals** (`compare --level 0.9917`). The 80% data-sufficiency rule applies to `trust` too. Parked, not tested: referee effects and a full manager-quality rating (see the conversation of 2026-11-20).

## Results, 2026-11-20
Features built from transfermarkt-datasets: 100 teams linked automatically, every match 2015/16–2025/26 linked to its game, line-up strength known for both sides in 96% of matches (sufficiency rule met). Evaluation 2020/21–2025/26, 6,840 matches, 99.17% intervals.

| Model vs `dixon_coles` | 1X2 improvement (interval) | Over/under 2.5 | Learned effect |
|---|---|---|---|
| lineup (market value) | **+0.00196** (+0.00041 to +0.00351): **helps** | −0.00001: no | own XI +0.110, opponent XI +0.033 |
| trust (coach's choices) | **+0.00189** (+0.00037 to +0.00342): **helps** | −0.00008: no | own XI +0.235, opponent XI +0.091 |
| manager | −0.00016 (−0.00056 to +0.00023): no | −0.00001: no | +0.010 |

Stack test against the closing 1X2 market (99.17%): **both FAIL.** Market 0.9643; blend with lineup 0.9641 (+0.00018, interval −0.00019 to +0.00054), with trust 0.9640 (+0.00023, interval −0.00018 to +0.00066); β intervals include 0.

**Conclusion.** Knowing who starts makes our probabilities measurably better (closing about a tenth of the gap to the market), but the closing price already contains that information. As pre-registered, neither feature is wired into `predict`, so no live line-up pipeline is built and the news module (which would estimate line-ups before they are announced) stays parked. The manager flag adds nothing, as expected.

Practical note for the probability app: the margin-free closing market (0.9643) is a far better probability than any model here (0.981). Where a fair price is available, it should be shown beside the model.
