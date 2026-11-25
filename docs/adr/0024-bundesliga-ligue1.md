# 0024 Adding the Bundesliga and Ligue 1: pre-registered check

Status: pre-registered 2026-11-25, before the new leagues' data was loaded. Result appended.

## Why a check is needed

Harsh asked to add the Bundesliga (D1) and Ligue 1 (F1). The goals model is fitted on all
leagues together: team ratings, scoring level, home advantage and goals per shot are per league,
but the low-score correction, the empty-stadium term and the promoted-team prior's fit are
shared. So adding leagues changes the existing three leagues' predictions too. Two questions:
does the model work in the new leagues, and does it stay as good in the old ones?

## Data

football-data.co.uk, same format as the other leagues, 2015/16-2026/27 (downloaded by Harsh;
the source is not reachable from the build machine). Schedule from football-data.org (BL1, FL1).
Both leagues have 18 teams (306 matches a season); Ligue 1 had 20 until 2022/23, and its
2019/20 season stopped in March 2020.

## Test (fixed now)

Weekly walk-forward, 2020/21-2025/26 and 2026/27 so far, with the locked settings:
* "3-league fit": the model trained on E0, I1, SP1 only (today's model);
* "5-league fit": the same model trained on all five leagues.
Score per match: match result + over/under 2.5 + both-score log loss.

## Decision rule (fixed now)

The two leagues go live only if **all** of:

1. **Old leagues not hurt:** on E0, I1 and SP1, 2020/21-2025/26, the 5-league fit is not
   worse than the 3-league fit by more than 0.001 per match (point estimate).
2. **New leagues have skill:** in each new league, on 2020/21-2025/26, the 5-league fit beats
   that league's own baseline (its rates over the previous 3 seasons for result, over 2.5 and
   both-score), with the 95% block-bootstrap interval (week blocks) entirely below zero.
3. **Still true this season:** both hold as point estimates on 2026/27 so far.

If they pass, the calibration map (ADR 0020), the cards model's track record and the
confidence labels are refitted on all five leagues, and the new leagues appear on the site,
the desk, the log and the scorecard. Golden bars are not extended to the new leagues (they
are unproven where they exist). If a new league fails, it stays out.

## Result (run 2026-11-25, `football-edge league-check`, reports/new_leagues/verdict.md): PASS

Sum of result + over/under 2.5 + both-score log loss per match, 2020/21-2025/26:

| League | Matches | 5-league fit | 3-league fit | League baseline |
|---|---|---|---|---|
| Premier League | 2,280 | 2.3517 | 2.3522 | 2.4518 |
| Serie A | 2,280 | 2.3591 | 2.3600 | 2.4746 |
| La Liga | 2,280 | 2.3525 | 2.3532 | 2.4538 |
| Bundesliga | 1,836 | 2.3181 | - | 2.4184 |
| Ligue 1 | 2,058 | 2.3828 | - | 2.4606 |

1. Old leagues: the 5-league fit is slightly *better* (-0.00068 per match); 2026/27 -0.00037.
2. Bundesliga beats its baseline by 0.100 per match (95% interval -0.122 to -0.078); Ligue 1
   by 0.078 (-0.096 to -0.060). Both about as far ahead as in the old leagues.
3. 2026/27 so far: Bundesliga -0.090, Ligue 1 -0.094 against their baselines (90 and 99 matches).

**Both leagues are live.** Then, on all five leagues:
* walk-forward predictions rebuilt (2018/19-2025/26 and 2026/27), goals and cards;
* calibration re-tested with the ADR 0020 rule and adopted again: result + over/under +
  both-score 2.35413 -> 2.34942 per match (interval -0.00716 to -0.00226), 2026/27
  2.31737 -> 2.31300; cards 2.27889 -> 2.26607. Live maps from 14,833 matches: goals a 1.084,
  d 0.672, delta +0.102; cards d 0.605, f 0.706;
* confidence labels rebuilt; strong calls are still favourites at 65%+ (74% of 1,658 came
  true) and over/under 2.5 at 65%+ (69% of 772);
* desk replay (no future data): singles 77.2% predicted, 77.7% landed; safe parlays 60.6%,
  61.9%; risky parlays 31.2%, 31.7%;
* golden bars stay in the Premier League, Serie A and La Liga only (`leagues:` on each bar);
* schedule names: aliases added for the football-data.org names that do not match on their own
  (Bayern München, Eintracht Frankfurt, Mönchengladbach, Hamburger SV, Paris Saint-Germain,
  which would otherwise have matched Paris FC, Lyonnais, Rennais, Brestois, Saint-Étienne).
