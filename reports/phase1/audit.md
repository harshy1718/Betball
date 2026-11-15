# Phase 1 data audit

Reproduce: `python scripts/audit_raw.py "raw data" reports/phase1`
(each file's season was read from its Date column; file names are not season-coded).

- 12,869 matches = 11 full seasons (2015/16-2025/26) x 3 leagues x 380 = 12,540, plus 329 from the partial 2026/27 season (to 2026-11-08). 2014/15 is absent.
- Integrity: no missing results, no duplicate fixtures, result agrees with goals in every row, team names consistent within each league.
- Closing 1X2: Pinnacle (PSC*) for every season from 2015/16; average close (AvgC*) only from 2019/20.
- Closing Over/Under 2.5: only from 2019/20.
- 2025/26 Pinnacle columns ~50% filled; 2026/27 Pinnacle absent. Loader falls back to average close per row and records the source (`close_src`).
- 2019/20 restart runs to Aug 2020, so some weeks straddle the restart.
