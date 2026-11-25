# Using `predict` and `settle`

## Weekly routine (PowerShell, from the project folder, venv active)

1. **Refresh results.** Download the current-season CSVs for E0, I1 and SP1 from football-data.co.uk into the `raw data` folder, then:
   `football-edge ingest --src "raw data"`
2. **Get fixtures.** Either:
   - download `https://www.football-data.co.uk/fixtures.csv` (team names already match), or
   - type your site's prices into a CSV like `fixtures/my_site.csv` (format below).
3. **Forecast:** `football-edge predict --fixtures fixtures/my_site.csv`
   The card is printed and saved to `reports/predict/<date>/card.md`, with `picks.csv`, `doubles.csv` and `forecasts.csv` beside it.
4. **Bet only what you placed.** If you place the staked picks, run the same command with `--log` to record them in `bets/log.csv`. Edit that file if you placed something different.
5. **After the matches:** ingest again, then `football-edge settle`. It marks bets won or lost and prints the month: bets, staked, profit, return, and wins vs wins expected.

## One match, no file
```
football-edge predict --home Arsenal --away "Man United"
football-edge predict --home Inter --away Lecce --date 2026-11-23 --odds 1.40 4.80 8.00 --ou-odds 1.70 2.15
```
- Team names: our names or your site's (aliases in `configs/team_aliases.yaml`); the league is found from the names (`--league E0/I1/SP1` if ambiguous). A typo stops with the closest names.
- `--odds H D A` and `--ou-odds OVER UNDER` are optional: without them you get the model's probabilities only.
- `--date` defaults to today; the model trains on every result before that date.

Corners, yellow cards and red cards appear under each match as probabilities with their confidence label. There are no prices for these markets in our data, so they are never staked.

## Your-site CSV format
```
league,date,home,away,odds_h,odds_d,odds_a,odds_over25,odds_under25,odds_btts_yes,odds_btts_no
EPL,22/11/2026,Arsenal,Manchester United,1.55,4.2,5.5,1.65,2.2,1.75,2.0
Serie A,23/11/2026,Inter,Lecce,,,,,,,
```
- `league`: E0/I1/SP1/D1/F1, or EPL, Premier League, Serie A, La Liga, Bundesliga, Ligue 1.
- Optional `ref_h,ref_d,ref_a,ref_over25,ref_under25,ref_btts_yes,ref_btts_no`: the same match at a sharp bookmaker (Pinnacle). They set the market's fair probability; without them your site's own prices are used for that. football-data's `fixtures.csv` already includes Pinnacle.
- Odds columns are optional; leave blank if you don't have a price. Without a price the card says "bet if price >= min_odds".
- Site names like "Manchester United" are mapped by `configs/team_aliases.yaml`. An unknown name stops the run and suggests the closest names; add the alias and run again.

## Reading the card
| Column | Meaning |
|---|---|
| Prob | The probability used: the market's fair probability (margin removed) when prices were given, else the model's |
| Model / Market | Both probabilities side by side; ⚠ = they differ by 8+ points |
| Fair odds | 1 ÷ Prob: the price at which the bet breaks even on average |
| Min odds | Lowest price worth taking (fair odds less a 6% tolerance) |
| Exp. value | Prob × site odds − 1: average gain per 1 staked if the model is right; negative means the price is worse than fair |
| Confidence | How far to trust the probability: High / Medium / Low from its track record in 2020/21–2025/26 (`football-edge build-confidence`), with the record itself, e.g. "came true 63% of 1,204" |
| Stake tier | low / normal / high, set by the probability, which sets the stake |
| Decision | bet, or why it was skipped |

Settings (budget, tiers, caps, stop-loss) are in `configs/default.yaml` under `staking:`. The policy and its honest limits are in `docs/adr/0006-staking-policy.md`.
