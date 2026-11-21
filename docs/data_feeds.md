# Data feeds (all free)

| Feed | What it gives | Key | Command |
|---|---|---|---|
| football-data.co.uk results | Every played match this season: score, shots, corners, cards, odds. Updated after each round. | none | `football-edge update` |
| football-data.co.uk fixtures.csv | Next round's matches with pre-match odds (incl. Pinnacle) | none | `football-edge update` |
| football-data.org v4 | Whole season schedule: dates, kick-offs, matchday, status, referee when announced | free token in `FOOTBALL_DATA_ORG_KEY` | `football-edge schedule` (also run by `update` when the token is set) |

API-Football was dropped: its free plan does not cover the current season (checked 2026-11-20). Its paid plan ($19/month) is only worth it for live line-ups or referees; our tests found line-ups do not beat the closing price (ADR 0007).

## Match-day routine (PowerShell, in the Betball folder)
1. After a round finishes: `football-edge update`
   Downloads the new results into `raw data` (dated file names, older downloads kept), refuses a broken or shrunken file, loads everything, refreshes upcoming odds and the schedule.
2. Before the next round: `football-edge predict --next 4`
   Every scheduled match in the next 4 days, with prices from fixtures.csv where football-data has them. The model retrains on all results first (a few seconds), so picks always use the latest data.
3. Once a week, optionally: `football-edge settle` for logged bets.

## Notes
- football-data.co.uk updates after each round, usually within a day or two; `update` prints the latest result date per league so you can see if it has caught up.
- Team names from football-data.org are mapped automatically; a name it cannot place is printed as a warning; add it to `configs/team_aliases.yaml`.
- Downloads come from your own PC (this needs internet access there). Keys are only read from environment variables, never from files.
