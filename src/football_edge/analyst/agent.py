"""The analyst loop: the LLM chooses tools, our code runs them, the LLM explains. Every number
in the final answer is checked against the tool results (and the question)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from football_edge.analyst.tools import SCHEMAS, AnalystTools
from football_edge.content.facts import normalise_number

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

SYSTEM = """You are the match analyst for Betball, a football prediction model for the Premier \
League, Serie A and La Liga. Answer in plain, friendly English, in 2 to 8 short lines.

Rules:
1. For ANY fact about a match, team, schedule or the golden bars, call a tool first. Never \
answer football facts from memory.
2. Never calculate or estimate a number yourself. Quote numbers exactly as the tools give them. \
For "is this price worth it", call price_check with the teams, the bet and the price.
3. When you give a prediction, give its confidence and the reason from the tool's "Why that \
confidence" line.
4. If a tool returns an error, say so simply and suggest what to try (for example the right \
team name).
5. Golden-bar picks are always "unproven, being tracked". Never say guaranteed, sure, lock or \
banker, and never suggest a stake other than the flat 100 the golden bars track.
6. No tables or markdown headings: plain sentences and short lists only.
7. When you use price_check, give its verdict and its note."""


class ChatClient(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass
class Answer:
    text: str
    tool_calls: list[tuple[str, dict[str, Any], float]] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    seconds: float = 0.0
    llm_seconds: float = 0.0


def _numbers(text: str) -> set[str]:
    return {normalise_number(n.replace(",", "")) for n in NUMBER.findall(text)}


class Analyst:
    def __init__(self, client: ChatClient, tools: AnalystTools, max_steps: int = 4) -> None:
        self.client, self.tools, self.max_steps = client, tools, max_steps
        self.history: list[dict[str, Any]] = []  # previous turns, for follow-up questions
        self.seen: set[str] = set()  # numbers from tool results so far in this chat

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        ans = Answer(text="")
        messages = [{"role": "system", "content": SYSTEM}, *self.history[-12:],
                    {"role": "user", "content": question}]  # fmt: skip
        allowed = set(self.seen) | _numbers(question)
        tools = [{"type": "function", "function": s} for s in SCHEMAS]
        text = ""
        for step in range(self.max_steps + 1):
            final_only = step == self.max_steps  # last round: no more tools, answer now
            t1 = time.perf_counter()
            msg, _ = self.client.chat(messages, [] if final_only else tools)
            ans.llm_seconds += time.perf_counter() - t1
            calls = msg.get("tool_calls") or []
            messages.append(
                {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
            )
            if not calls:
                text = str(msg.get("content") or "").strip()
                break
            for c in calls:
                fn = c.get("function", {})
                name, args = str(fn.get("name", "")), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                t2 = time.perf_counter()
                result = self.tools.call(name, args)
                ans.tool_calls.append((name, args, time.perf_counter() - t2))
                allowed |= _numbers(result)
                messages.append({"role": "tool", "tool_name": name, "content": result})
        bad = sorted(_numbers(text) - allowed)
        if bad:  # one rewrite with the problem spelled out
            fix = (
                f"Check: these numbers are not in any tool result: {', '.join(bad)}. Rewrite "
                "your answer using only numbers the tools returned. Answer only."
            )
            messages.append({"role": "user", "content": fix})
            t1 = time.perf_counter()
            msg, _ = self.client.chat(messages, [])
            ans.llm_seconds += time.perf_counter() - t1
            text = str(msg.get("content") or "").strip() or text
            bad = sorted(_numbers(text) - allowed)
        # Guarantee, in code, the one caveat both models dropped: a "worth taking" from our model
        # always comes with the reminder that the market was usually more accurate in testing.
        if (
            any(n == "price_check" for n, _, _ in ans.tool_calls)
            and "worth taking"
            in "".join(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
            and "market" not in text.lower()
        ):
            text += ("\n(Our model's own view: in testing, the market's price was usually more "
                     "accurate than our model.)")  # fmt: skip
        ans.text, ans.unverified = text, bad
        self.seen |= allowed
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": text}]  # fmt: skip
        ans.seconds = time.perf_counter() - t0
        return ans
