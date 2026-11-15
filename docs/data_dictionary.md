# Processed matches table (`data/processed/matches.parquet`)

| Column | Meaning |
|---|---|
| league | E0 = EPL, I1 = Serie A, SP1 = La Liga |
| season | start year (2018 = 2018/19) |
| date | match date (no kickoff time in source) |
| home, away | team names as in source, trimmed |
| FTHG, FTAG, FTR | full-time goals and result (H/D/A) |
| HTHG, HTAG | half-time goals |
| close_h/d/a | decimal closing odds (1X2); source in `close_src` (PS = Pinnacle, Avg = market average) |
| pre_h/d/a | earlier snapshot odds; source in `pre_src` |

Caveats: BTTS has no historical odds. Over/Under 2.5 closing odds exist only from 2019/20 (not yet in this table). `pre_*` is not a true opening price. 2025/26 and 2026/27 mix Pinnacle and average closes; check `close_src` before comparing seasons.
