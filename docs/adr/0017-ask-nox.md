# 0017 Ask Nox: the analyst on the public site

Status: accepted, 2026-11-22.

Harsh asked for the analyst chat on the website. It reuses `Analyst` (analyst/agent.py) with
three additions, all enforced in code rather than trusted to the prompt:

1. **A different tool set.** Only prediction tools (fixtures, match prediction, team form, head
   to head). `price_check` and `golden_scoreboard` stay CLI-only: one is about prices, the other
   reports profit. Golden lines from `predict_match` lose their "take at" price.
2. **Banned words.** `Analyst` gained an optional `banned` list. A banned word in the answer
   triggers the same single rewrite as an unverified number; if it survives, the site shows a
   fixed refusal instead. Questions that ask about odds, bets, stakes or prices are refused
   before the LLM runs. The CLI analyst is unchanged (no banned list).
3. **Progress events and one question at a time.** `Analyst.ask` takes an optional `on_event`
   callback; the API streams those as JSON lines so the page can say what Nox is checking during
   a 30 to 60 second answer. A lock allows one question at a time, because the LLM runs on one
   CPU.

The evaluation set (evals/analyst_questions.yaml) still scores the CLI analyst; the site's prompt
should get its own run before the site is public.
