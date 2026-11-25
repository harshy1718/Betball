# Live tracking: the prediction log, the scorecard and the posts

The question this answers: **how good are the predictions on matches the model has never seen?**
Backtests can flatter a model; the live log cannot. Decisions: ADR 0019.

## The daily run

`football-edge daily` does, in order:
1. `update`: the latest results, odds and schedule (skipped with `--no-update`);
2. logs every prediction for every match not yet kicked off, and the desk's slips;
3. settles the golden-call logs;
4. prints the scorecard;
5. commits `records/` (only that folder) and pushes it to GitHub (`tracking.auto_push`; skip
   with `--no-push`). A failed push is reported in the log and retried by the next run.

Set it to run every morning (once, from the project folder with the venv active):

    football-edge schedule-daily --time 10:30

That registers a Windows scheduled task, "Nox daily refresh". It runs hidden (no console
window, so it cannot be stopped by closing one); if the PC is off at 10:30, it runs as soon as
the PC is back on. Running the command again replaces the task. Its output goes to `data/interim/daily/daily.log`. Opening
the trader desk while `football-edge serve` runs also logs (at most every 10 minutes), so a
missed morning still gets caught.

## The logs

| Log | Where | In git? | What |
|---|---|---|---|
| Predictions | `records/predictions/` | yes: the public proof | every leg of every match: chance, raw chance, confidence, model commit, settings hash, data hash |
| Posted calls | `records/posted/` | yes | each strong call marked as posted on Instagram |
| Desk slips | `bets/desk_log/` | no (`bets/` is private) | the board Nox generated: singles and parlays |

Rules:
- **Append-only.** A match gets a new block only when its numbers change; old rows are never
  edited. Grading always uses the **last block before kickoff**; anything logged after kickoff
  never counts.
- **Hash-chained.** Each block's SHA-256 and a chain hash go into `chain.csv` next to the files.
  `football-edge records-verify` checks nothing was changed or deleted. The scorecard shows
  "✓ Log untouched" or a warning.
- **Proof needs a push.** The chain shows the log was not edited, but anyone could rebuild a
  whole chain. GitHub's record of when each commit arrived is what proves a prediction existed
  before the match. The daily run pushes `records/` itself; by hand it is
  `git add records`, `git commit -m "Prediction log" -- records`, `git push`.
- **Every row says what made it.** `model_commit` (the code's git commit, `+dirty` if src/ or
  configs/ had uncommitted changes), `params_sha` (the locked model settings plus the
  calibration file) and `data_sha` (the results table the model saw). `p_raw` is the chance
  before calibration (ADR 0020). Rows logged before 25 Nov 2026 lack these columns; later rows
  of that month went to `2026-11_v2.csv` so the old file is never rewritten.

## The scorecard (`/scorecard`, private)

- **Accuracy by market**: graded predictions, the average chance we gave, how often they came
  in, and the **accuracy score** (Brier score: the average squared gap between the chance and
  what happened; lower is better) compared with:
  - **the league baseline**: the league's own rate for that outcome over the previous 3 years.
    "+6%" means 6% more accurate than just knowing the league.
  - **the market**: the bookmakers' closing prices with their margin removed (match result and
    over/under 2.5 only; no closing prices exist for the other markets in our data). This is the
    bar a paying customer would set: odds are free to read.
- **League by league**: the overall accuracy of each league on its own, against that league's
  usual rates (ADR 0023), so a league that drifts shows up.
- **Calibration**: of everything called at 60-70%, how much came in.
- **Is the calibration helping?**: the calibrated chances against the raw ones logged next to
  them, same matches. At 300 graded matches, if raw is more accurate, calibration goes off.
- **By confidence label**, **the desk's slips** by type (against their promise: safe parlays 60%+,
  risky 25%+), and **the calls posted on Instagram**.
- Each yes/no question counts once (over 2.5, not also under 2.5; double chance mirrors the
  result and is left out). Below 30 graded predictions a figure says "too few to judge yet".

## Posts (`/posts`, private)

Instagram posts for **strong calls only**, chosen by a fixed rule from the track record:
- for the match result and over/under 2.5, the lowest chance level from which every past 5% level
  (with 30+ past calls) came true **65%+** of the time, with 100+ past calls in total
  (`tracking.strong`). Today: **favourites at 65%+** (74% of 1,116 came true in 2020-26) and
  **over/under 2.5 at 65%+** (66% of 332), both on the calibrated numbers (ADR 0020);
- a draw is never a strong call.

For each: a 1080x1350 image (Instagram's portrait size) to download, a caption to copy, and
**Mark as posted**. That logs the call to `records/posted/`; it cannot be undone and only works
before kickoff. The **weekly recap** image grades every call posted in the last 7 days, misses
included.

No betting words (checked in code, the same list as the video scripts), no prices, no club
badges. Before charging anyone for predictions, check the law (India's Online Gaming Act 2025)
with a lawyer.
