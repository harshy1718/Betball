# Architecture

```
raw CSV (immutable) --ingest--> validated matches table (parquet)
                                   |
                     walk-forward folds (evaluation.folds)
                                   |
             model fit per fold (models/, Phase 2) -> probabilities
                                   |
        margin-removed closing odds (markets/, Phase 4) -> stack test
                                   |
                              reports/<phase>/
```
Layers depend only downward: `cli` -> `evaluation`/`models` -> `data` -> `config`. Data code never imports model code.

## Models (`models/`)
Every model implements the `Model` protocol in `models/base.py`: `fit(train)` then `predict_proba(test)` returning an `(n, 3)` array of H/D/A probabilities. The runner fits a **fresh** model per fold and strips result columns from the test rows before calling `predict_proba`, so a model cannot read the answers.

| Model | File | Status |
|---|---|---|
| Team-blind (league H/D/A rates) | `team_blind.py` | done |
| Elo + multinomial logistic mapping | `elo.py` | done |
| Dixon-Coles (goals model, time decay, promoted-team prior) | `dixon_coles.py` | done, tuned and locked |
| Corners/cards count models (negative binomial totals) | `counts.py` | cards live; corners not used |
| Calibration of goals and cards (ADR 0020) | `calibration.py` | live; maps in `reports/calibration/` |

Calibration sits between the models and everything that reads them: the raw model is what every
walk-forward test measures; `football-edge calibrate` fits the maps on out-of-sample seasons,
tests them as pre-registered and marks them adopted; forecasts then attach them
(`predict/forecast.py: load_calibration`). The prediction model never knows how its chances
are used: the desk builds slips from them, the tracking log grades them.

Research commands added 25 Nov 2026: `calibrate` (ADR 0020), `replay-desk` (the desk on past
seasons with nothing from the future), `referee-test` (ADR 0021), `challenge --model <name>`
(the locked goals model against a challenger on the same weekly folds; ADR 0022 tested
dynamic ratings with it and kept the locked model), `tune-per-league` and
`per-league-calibration` (ADR 0023: a model and a calibration per league; both kept the
combined versions).

Run one: `football-edge evaluate --model elo`. Output in `reports/evaluation/<model>/`: `summary.csv` (kept in git), `predictions.parquet` (rebuilt, ignored), `run.json` (git SHA, parameters, data hash).

## Markets (`markets/`)
`margin.py`: odds to fair probabilities (Harsh). `stack.py`: the pre-registered stack test; `blend` by Harsh. Run: `football-edge stack-test --model dixon_coles`; report in `reports/stack_test/<model>/`.

Planned, not yet created (added when the phase starts, to avoid empty scaffolding):
`live/` (API-Football client with rate limiting and caching; separate tables with a source tag), `serving/` (read-only results view).

## Productionisation notes
- Storage moves SQLite -> Postgres only if concurrent writers appear; parquet + SQLite suffices for research.
- Live predictions go through an append-only table keyed by (fixture_id, model_version, made_at); no UPDATE or DELETE.
- Model artefacts are versioned by git SHA + config hash, recorded in every report.
