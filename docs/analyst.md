# Match-analyst chat

`football-edge chat` (or `football-edge chat "who wins Liverpool v United?"` for one question).

## How it works (tool calling)
1. Your question goes to the local LLM (Ollama, `analyst:` in configs/default.yaml) together with
   descriptions of six tools.
2. The LLM does not answer from memory: it asks for tools. Our code runs them and sends the
   results back. It may call several (e.g. `predict_match` then `price_check`), up to `max_steps`.
3. The LLM writes the answer from the tool results.
4. **Number check:** every number in the answer must appear in a tool result or your question.
   If one does not, the LLM gets one rewrite with the problem named; if it still fails, the
   answer is shown with a `[check]` warning instead of being hidden.

| Tool | What it returns |
|---|---|
| upcoming_matches | fixtures in the next days, kick-off in IST |
| predict_match | model probabilities, confidence + reason, xG, scores, cards, corners, golden flags |
| team_form | last N results |
| head_to_head | past meetings |
| price_check | fair price and expected return for a probability and a price (code does the maths) |
| golden_scoreboard | how both golden bars are doing |

Team names: our names, the names people say ("Manchester United"), nicknames (Spurs, Barca,
Atleti) and small typos work; anything unclear returns "unknown team, did you mean ...".

Every chat turn is logged to `reports/analyst/chat_log.jsonl` (question, tools, answer, time).

## Evaluation
`football-edge analyst-eval` asks the 20 questions in `evals/analyst_questions.yaml` and reports
the share with the right tools called, fully grounded answers, required/banned words, and speed.
Compare models: `football-edge analyst-eval --model qwen3:8b`. Reports: reports/analyst/eval_*.md.
Add a question whenever you find a weak spot; never edit one to make a model pass.

If Ollama says the model "does not support tools", pull one that does (e.g. `ollama pull qwen3:8b`)
and set `analyst: model:`.
