# Video scripts (LLM part)

## Flow
1. `football-edge update` - latest results, odds and schedule.
2. `football-edge script --next 4` - the picker chooses this round's video matches, builds a
   fact sheet for each, and the local LLM writes a 45-60 second script.
   One match instead: `football-edge script --home "Real Madrid" --away Celta`.
   Fact sheets only (no LLM): add `--dry-run`.
3. Output per match in `reports/content/<date>/<Home>_v_<Away>/`:
   `facts.txt` (all the LLM may use), `script.txt` (passed every check), `drafts.txt`
   (every attempt and why earlier ones failed).

## The checks (content/script.py)
Every draft is checked in code; a failing draft goes back to the LLM with the problems listed,
up to `writer.max_retries` rewrites, then the match is skipped rather than published wrong.
- every number in the script must appear in the fact sheet (no invented stats);
- no betting words (bet, odds, tips, stake, bookmaker, guaranteed ...);
- none of the stock AI phrases (delve, showdown, buckle up, stay tuned ...);
- 90-170 words (45-60 seconds spoken), both teams named, plain text only.

## Which matches get a video (content/picker.py)
All three: a High confidence label on the main call; a clear story (favourite 70%+, goals
call 65%+, or 6+ expected cards); at least one club from `content.big_clubs` in the config.
Top `content.videos_per_round` by story strength.

## Setting up the model
1. Install Ollama (ollama.com), then pull the writer model, e.g. Gemma 4 12B.
2. Run `ollama list` and copy the exact name into `configs/default.yaml` under `writer: model:`.
3. Close heavy apps while it writes; on a CPU expect a few minutes per script.

How the presenter says a club: automatic (first plain alias in team_aliases.yaml), with
overrides under `content.display_names`.
