"""Run the analyst evaluation set against the live LLM (evals/analyst_questions.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from football_edge.analyst.agent import Analyst, ChatClient
from football_edge.analyst.tools import AnalystTools


def run(
    client: ChatClient, tools: AnalystTools, path: Path, max_steps: int
) -> tuple[str, list[dict[str, Any]]]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = []
    for i, c in enumerate(cases, start=1):
        a = Analyst(client, tools, max_steps).ask(c["q"])  # fresh chat per question
        called = {n for n, _, _ in a.tool_calls}
        want = set(c.get("tools") or [])
        low = a.text.lower()
        say_ok = all(w.lower() in low for w in c.get("say", []))
        not_ok = not any(w.lower() in low for w in c.get("not_say", []))
        rows.append({
            "n": i, "q": c["q"], "want": sorted(want), "called": sorted(called),
            "tools_ok": want <= called,
            "grounded": not a.unverified, "words_ok": say_ok and not_ok,
            "seconds": a.seconds, "answer": a.text, "unverified": a.unverified,
        })  # fmt: skip
        print(f"  {i}/{len(cases)} {'ok ' if rows[-1]['tools_ok'] else 'MISS'} "
              f"{a.seconds:5.0f}s  {c['q']}", flush=True)  # fmt: skip
    n = len(rows)

    def pct(k: str) -> float:
        return sum(bool(r[k]) for r in rows) / n

    secs = sorted(r["seconds"] for r in rows)
    lines = [
        "# Analyst evaluation", "",
        f"{n} questions. Right tools called: **{pct('tools_ok'):.0%}**. Answers with every "
        f"number traced to a tool: **{pct('grounded'):.0%}**. Required words present / banned "
        f"words absent: **{pct('words_ok'):.0%}**. Median time {secs[n // 2]:.0f}s, "
        f"slowest {secs[-1]:.0f}s.", "",
        "| # | Question | Wanted tools | Called | Tools | Grounded | Words | Seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['q']} | {', '.join(r['want']) or '-'} "
            f"| {', '.join(r['called']) or '-'} "
            f"| {'ok' if r['tools_ok'] else 'MISS'} | {'ok' if r['grounded'] else 'NO'} "
            f"| {'ok' if r['words_ok'] else 'NO'} | {r['seconds']:.0f} |"
        )
    lines += ["", "## Answers", ""]
    for r in rows:
        extra = f" (numbers not in tools: {', '.join(r['unverified'])})" if r["unverified"] else ""
        lines += [f"**{r['n']}. {r['q']}**{extra}", "", r["answer"] or "(empty)", ""]
    return "\n".join(lines) + "\n", rows
