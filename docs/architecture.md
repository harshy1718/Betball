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

Planned, not yet created (added when the phase starts, to avoid empty scaffolding):
`models/` (team-blind baseline, Elo, Dixon-Coles), `markets/` (margin removal, log-loss, stack test), `live/` (API-Football client with rate limiting and caching; separate tables with a source tag), `serving/` (read-only results view).

## Productionisation notes
- Storage moves SQLite -> Postgres only if concurrent writers appear; parquet + SQLite suffices for research.
- Live predictions go through an append-only table keyed by (fixture_id, model_version, made_at); no UPDATE or DELETE.
- Model artefacts are versioned by git SHA + config hash, recorded in every report.
