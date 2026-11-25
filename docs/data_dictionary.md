# Processed matches table (`data/processed/matches.parquet`)

| Column | Meaning |
|---|---|
| league | E0 = EPL, I1 = Serie A, SP1 = La Liga, D1 = Bundesliga, F1 = Ligue 1 (the last two since 25 Nov 2026, ADR 0024) |
| season | start year (2018 = 2018/19) |
| date | match date (no kickoff time in source) |
| home, away | team names as in source, trimmed |
| FTHG, FTAG, FTR | full-time goals and result (H/D/A) |
| HTHG, HTAG | half-time goals |
| HS, AS | shots (home, away) |
| HST, AST | shots on target (home, away) |
| close_h/d/a | decimal closing odds (1X2); source in `close_src` (PS = Pinnacle, Avg = market average) |
| pre_h/d/a | earlier snapshot odds; source in `pre_src` |
| HC, AC / HY, AY / HR, AR | corners, yellow cards, red cards per side |
| referee | the match referee (Premier League only; empty for Serie A and La Liga). Named before kickoff, so it is pre-match information. Tested for cards and rejected (ADR 0021) |

Caveats: BTTS has no historical odds. Over/Under 2.5 closing odds exist only from 2019/20 (not yet in this table). `pre_*` is not a true opening price. 2025/26 and 2026/27 mix Pinnacle and average closes; check `close_src` before comparing seasons.

In-match stats (goals, shots) describe a match after it is played. Models may learn from them for past matches; the runner strips them from the rows being predicted.
