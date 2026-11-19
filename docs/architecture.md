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
| Dixon-Coles (goals model, time decay, promoted-team prior) | `dixon_coles.py` | done, untuned |

Run one: `football-edge evaluate --model elo`. Output in `reports/evaluation/<model>/`: `summary.csv` (kept in git), `predictions.parquet` (rebuilt, ignored), `run.json` (git SHA, parameters, data hash).

Planned, not yet created (added when the phase starts, to avoid empty scaffolding):
`markets/` (margin removal, stack test), `live/` (API-Football client with rate limiting and caching; separate tables with a source tag), `serving/` (read-only results view).

## Productionisation notes
- Storage moves SQLite -> Postgres only if concurrent writers appear; parquet + SQLite suffices for research.
- Live predictions go through an append-only table keyed by (fixture_id, model_version, made_at); no UPDATE or DELETE.
- Model artefacts are versioned by git SHA + config hash, recorded in every report.
