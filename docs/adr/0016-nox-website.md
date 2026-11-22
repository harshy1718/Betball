# 0016 The Nox website: FastAPI over the existing pipeline, React front end

Status: accepted, 2026-11-22.

Harsh chose the "Nox" brand (an owl mascot) and asked for a React dashboard: next 5 predictions,
a fixtures list to pick a match from, a match page and golden calls.

Decisions:

1. **The API reuses the CLI pipeline, unchanged.** `NoxService` calls `from_schedule` and
   `forecast_with_model` with the locked model settings and builds every line with
   `build_fact_sheet`, so the site cannot drift from `football-edge predict`. An integration test
   compares the two.
2. **Model probabilities only, no prices.** The site is public-facing; it shows predictions and
   never odds, prices, stakes or bookmakers (India Online Gaming Act 2025, and Harsh's rule that
   the model runs on our data only). Golden calls appear as predictions with their rule.
3. **Nox's words come from code.** His mood and one-liner are fixed templates over the pick's
   probability and confidence label; no LLM writes on the site yet, so nothing needs a number
   check there. "Ask Nox" (the analyst chat) will bring the LLM in, with its existing guardrails.
4. **FastAPI + Vite/React/TypeScript, one process to run.** `football-edge serve` serves the API
   and, once `npm run build` has run, the built site on the same port. FastAPI and uvicorn are an
   optional extra (`.[web]`) so the research install stays small.
5. **One fit per week of fixtures, cached.** About 2 seconds; rebuilt when the schedule or results
   files change, or hourly.
