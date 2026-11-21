# Team news module (step 3)

## What it does
1. **Extract (LLM):** an open model reads one article and fills a fixed form: which players are out, doubtful, suspended or returning, plus manager changes and expected rotation. Every fact must quote its sentence from the article; facts without a matching quote are thrown away as made up.
2. **Convert (plain code):** `news/impact.py` turns the facts into the same line-up number the model uses (`lineup`/`trust` scale): the expected share of the usual XI's strength that will be missing. A doubtful player counts as half missing.
3. **Measure before trusting:** `football-edge news-eval` scores the extractor on hand-labelled articles (precision, recall, failure rate, latency, tokens).

The LLM never outputs probabilities or stakes. It only reads.

## Which model
Any server with the OpenAI-compatible chat API; set it in `configs/default.yaml` under `news:`.
- Free and private on your PC: install Ollama, run `ollama pull qwen2.5:7b-instruct`, keep the default settings. It needs about 8 GB of RAM and takes a few seconds per article on a CPU.
- Hosted: point `base_url` at a Hugging Face Inference endpoint (or similar) and set `api_key_env: HF_TOKEN`; put the token in your environment, never in a file.

Specialised "football news" models on Hugging Face are rare and untested; a good general instruction model with a strict form is the standard approach.

## Gate before use
Label at least **100 real articles** in the format of `data/news/labels_example.jsonl` (example uses fictional teams). The extractor is used only if, on those:
- precision ≥ 0.90 (a wrong "out" is the costly mistake),
- recall ≥ 0.70,
- valid answers ≥ 95%.

## When it feeds the model
Only if ADR 0007 finds that line-up strength improves the probabilities. News is how we would estimate line-up strength *before* line-ups are announced; if knowing the actual XI does not help, guessing it from news will not either.

## Commands
- `football-edge news-extract --articles <folder of .txt> --out reports/news/facts.json`
- `football-edge news-eval --labels data/news/labels.jsonl`
