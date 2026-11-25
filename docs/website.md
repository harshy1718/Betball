# The Nox website

**Nox** is the public face of the model: a small owl who "says" each prediction. The site has four
pages, all built from the same numbers as `football-edge predict`:

| Page | Address | What it shows |
|---|---|---|
| Home | `/` | the big match coming up (most big clubs, then the earliest), and Nox's next 5 |
| Fixtures | `/fixtures` | every match in the next 7 days, filter by league; pick one for a preview |
| Match | `/match/<id>` | win/draw/win, confidence and why, goals, corners, cards, every score's chance, form, meetings, golden calls |
| Golden calls | `/golden` | upcoming golden calls, the rules behind them, and the tracker |
| Ask Nox | `/ask` | chat with Nox: the match analyst, in Nox's voice (match pages link here with ready questions) |

The design is on the canvas "Nox website design" (Claude artifact), with Nox's character sheet.

## How it fits together

```
data/processed/schedule.parquet ─┐
data/processed/matches.parquet ──┼─> NoxService (src/football_edge/web/service.py)
reports/confidence/track_record ─┘        │  one Dixon-Coles fit for the week, cached
                                          ▼
                         FastAPI (web/app.py)  /api/meta /api/next /api/matches /api/golden
                                          ▼
                         React app (web/)  Home, Fixtures, Match, Golden calls
```

- **Same numbers as the CLI.** The service uses `from_schedule` + `forecast_with_model`, the locked
  settings, our own probabilities only (`staking.confidence: model`) and the confidence track
  record. `tests/integration/test_web_numbers.py` checks the site against the CLI.
- **No prices, odds or bookmakers.** The API never returns them. Golden calls are shown as
  predictions with their rule; your prices stay in `football-edge price` and `bets/`.
- **Nox's mood** is set by code, not by an LLM: *sure* when one outcome is more likely than not and
  the track record at that level is not Low; otherwise *unsure* (a close call or a shaky level).
  His one-liner quotes only the pick's own probability.
- **Caching.** The week is forecast once (about 2 seconds) and rebuilt when the schedule or the
  results change, or at most once an hour. Matches stay listed for 2 hours after kick-off.

## Ask Nox

The analyst from `football-edge chat` (docs/analyst.md), on the site. Same loop: the LLM picks
tools, our code runs them, and every number in the answer is checked against the tool results
(one rewrite, then flagged). What the site adds (`src/football_edge/web/ask.py`, ADR 0017):

- **Nox's voice**, first person, 2 to 6 sentences.
- **Predictions only.** Tools: fixtures, match prediction, team form, head to head. No
  `price_check` and no golden scoreboard (it reports profit); golden lines lose their minimum price.
- **Betting words are banned.** A question about odds, bets, stakes or prices gets a polite
  refusal at once (no LLM time). An answer that uses a betting word is sent back once; if it
  still does, the refusal replaces it.
- **Live progress.** `POST /api/ask` streams JSON lines: one per tool Nox uses ("Looking at
  Liverpool v Manchester United…"), then the answer with its check result.
- **One question at a time** (the LLM runs on this PC); chats are kept per browser tab, for
  follow-up questions. Questions and answers are logged to `reports/analyst/web_chat_log.jsonl`.
- Needs Ollama running with the analyst model (`analyst.model` in the config, qwen3:8b).
  `football-edge serve` says at start-up whether Ask Nox is ready; the page shows a notice if not.

## Run it on your PC

One-time setup (PowerShell, in the project folder, venv active):

```
pip install -e ".[web,dev]"
cd web
npm install
npm run build
cd ..
```

Then each time:

```
football-edge update        # fresh results (Mondays)
football-edge schedule      # fresh fixtures
ollama serve                # only if Ollama isn't already running (it usually starts itself)
football-edge serve         # open http://127.0.0.1:8000
```

`npm run build` is needed again only after the React code changes. Node.js 20 or newer is needed
for the `npm` steps (https://nodejs.org, the LTS installer).

### Working on the React code

Run the API and the React dev server side by side; the page reloads on every save:

```
football-edge serve              # terminal 1: API on :8000
cd web; npm run dev              # terminal 2: open http://localhost:5173
```

## The API

| Route | Returns |
|---|---|
| `GET /api/meta` | model trained up to, number of matches, a warning when results are stale |
| `GET /api/next?n=5` | the next n matches that have not kicked off |
| `GET /api/matches?league=E0` | every match in the window (league optional: E0, I1, SP1, D1, F1) |
| `GET /api/matches/{id}` | one match in full, including the 6x6 score grid |
| `GET /api/golden` | matches with golden calls, the bars, the tracker (counts only) |
| `GET /api/ask/status` | is Ollama up with the analyst model |
| `POST /api/ask` | `{"session", "question"}`; streams `tool` events, then `answer` or `error` |
| `GET /api/desk/...` | the private trader desk (docs/trader.md); this PC only |
| `GET /api/desk/scorecard` | live accuracy of every logged prediction (docs/tracking.md); this PC only |
| `GET /api/posts`, `POST /api/posts/mark`, `GET /api/posts/recap` | Instagram strong calls, posted log, weekly recap; this PC only |

Interactive docs: http://127.0.0.1:8000/api/docs while `serve` is running.

## Private pages

Desk, Scorecard and Posts (tabs under the 🔒 Desk pill) answer only on this PC. The fonts are
bundled with the site (no Google Fonts call), which also lets the Posts page export images.
After pulling new web code, run `npm install` once in `web/`, then `npm run build`.

## Not built yet

- Hosting it on the internet (it runs on this PC only).
- Before publishing: check the name "Nox" and a domain; other products use the name.
