# Football Edge

Research platform that tests one question: **does a goals model (Dixon-Coles) add information beyond bookmaker closing odds?** Leagues: EPL, Serie A, La Liga. If the pre-registered test fails, the project stops; the threshold is never lowered afterwards (docs/adr/0003).

## Quick start (Windows PowerShell, no make needed)
The virtual environment lives outside the project folder so OneDrive doesn't sync thousands of package files.
```
py -3.12 -m venv C:\venvs\betball
C:\venvs\betball\Scripts\Activate.ps1
pip install -e ".[dev]"
football-edge ingest --src "raw data"     # validates CSVs, writes data/processed/matches.parquet
ruff check src tests; mypy src; pytest -m "not integration"     # the full check
```
On Linux/macOS the same steps are `make install`, `make ingest`, `make check`.

## Layout
| Path | Purpose |
|---|---|
| `src/football_edge/` | the package (config, data, evaluation; models and markets arrive in later phases) |
| `configs/` | all parameters; no magic numbers in code |
| `data/raw/` | immutable season-coded CSVs + `manifest.json` with SHA-256 hashes (versioned) |
| `data/processed/` | generated tables (git-ignored, rebuildable with `ingest`) |
| `tests/unit`, `tests/integration` | fast tests / tests needing processed data |
| `docs/` | architecture, data dictionary, specs, decision records (`adr/`) |
| `reports/` | generated evaluation output, one folder per phase |
| `scripts/` | one-off exploratory scripts (not imported by the package) |

## Engineering rules
1. Raw data is never edited. Everything downstream is rebuilt from raw + code + config.
2. No post-match information or closing odds as model inputs.
3. Predictions are timestamped before kickoff and never overwritten; live and historical results live in separate tables.
4. Every change passes the full check (lint, types, tests). Typed code (mypy strict), formatted with ruff.
5. Parameters live in `configs/`, secrets in `.env` (never committed).
