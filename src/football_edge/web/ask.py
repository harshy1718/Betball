"""Ask Nox: the match analyst (analyst/agent.py) as Nox, on the website.

Same loop and guardrails as `football-edge chat` (tools for every fact, every number checked
against tool results, one rewrite), plus what a public page needs:
  * Nox's voice, first person, short answers;
  * prediction tools only: no price_check and no golden scoreboard (profit), and golden lines
    lose their minimum price;
  * betting words are banned: an answer that uses one is sent back once, and replaced by a
    plain refusal if it still does;
  * one question at a time (the LLM runs on this PC), chats kept per browser tab.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections import OrderedDict
from typing import Any

import pandas as pd

from football_edge.analyst.agent import Analyst
from football_edge.analyst.tools import SCHEMAS, AnalystTools
from football_edge.config import Config
from football_edge.content.script import BETTING_WORDS
from football_edge.news.extract import LLMError, NewsConfig, OllamaClient

WEB_TOOLS = ("upcoming_matches", "predict_match", "team_form", "head_to_head")
BANNED = (*BETTING_WORDS, "price", "prices", "roi", "profit", "bookmakers", "gamble", "gambling")
# A question about these gets the refusal at once, without a minute of LLM time.
ASKS_FOR_BETS = ("bet", "bets", "betting", "odds", "stake", "stakes", "bookie", "bookies",
                 "bookmaker", "bookmakers", "wager", "accumulator", "acca", "parlay", "price",
                 "prices", "punt")  # fmt: skip
MAX_QUESTION = 300
MAX_SESSIONS = 100
REFUSAL = (
    "I only do predictions, not betting. Ask me who I think wins a match, how many goals to "
    "expect, or how a team has been playing."
)

NOX_SYSTEM = """You are Nox, a small owl who stays up for every Premier League, Serie A and \
La Liga match and reads our football prediction model. Speak as Nox, in the first person, warm \
and plain. Answer in 2 to 6 short sentences.

Rules:
1. For ANY fact about a match, team or the fixtures, call a tool first. Never answer football \
facts from memory.
2. Never calculate or estimate a number yourself. Quote numbers exactly as the tools give them.
3. When you give a prediction, give its confidence and the reason from the tool's "Why that \
confidence" line, in your own words.
4. If a tool returns an error, say so simply and suggest what to try (for example the right \
team name).
5. Golden calls are always "unproven, still being tracked". Never say guaranteed, sure thing, \
lock or banker.
6. This site is about predictions only. Never talk about betting, odds, prices, stakes, tips \
or bookmakers. If asked, say you only make predictions.
7. No tables, headings or markdown: plain sentences only."""

_PRICE = re.compile(r"\s*\((?:take|back) at [^)]*\)", re.I)


class WebTools(AnalystTools):
    """The analyst's tools without anything about prices."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.fns = {k: v for k, v in self.fns.items() if k in WEB_TOOLS}

    def _golden(self, f: pd.Series) -> list[str]:
        return [_PRICE.sub("", line) for line in super()._golden(f)]


WEB_SCHEMAS = [s for s in SCHEMAS if s["name"] in WEB_TOOLS]


class AskNox:
    def __init__(self, cfg: Config, client: Any = None, tools: AnalystTools | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._tools = tools
        self._sessions: OrderedDict[str, Analyst] = OrderedDict()
        self._busy = threading.Lock()
        self._setup = threading.Lock()
        self.log_path = cfg.reports_dir / "analyst" / "web_chat_log.jsonl"

    @property
    def model(self) -> str:
        return str(self.cfg.analyst.get("model", ""))

    def status(self) -> dict[str, Any]:
        """Is the local LLM server up and does it have the model? Checked in under 2 seconds."""
        if self._client is not None and not isinstance(self._client, OllamaClient):
            return {"ready": True, "model": self.model, "problem": None}
        base = str(self.cfg.analyst.get("base_url", "http://localhost:11434")).rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as r:
                names = {m.get("name", "") for m in json.load(r).get("models", [])}
        except OSError:
            down = "Ollama isn't running on this PC. Start Ollama, then try again."
            return {"ready": False, "model": self.model, "problem": down}
        if self.model not in names and f"{self.model}:latest" not in names:
            return {"ready": False, "model": self.model,
                    "problem": f"Ollama doesn't have {self.model} yet: run `ollama pull "
                               f"{self.model}`."}  # fmt: skip
        return {"ready": True, "model": self.model, "problem": None}

    def client(self) -> Any:
        """The LLM client (created on first use), shared with the trader's review."""
        with self._setup:
            if self._client is None:
                self._client = OllamaClient(NewsConfig.from_dict(self.cfg.analyst))
            return self._client

    def _analyst(self, session: str) -> Analyst:
        with self._setup:
            if self._tools is None:
                self._tools = WebTools(self.cfg)
            if self._client is None:
                self._client = OllamaClient(NewsConfig.from_dict(self.cfg.analyst))
            if session in self._sessions:
                self._sessions.move_to_end(session)
                return self._sessions[session]
            a = Analyst(self._client, self._tools, int(self.cfg.analyst.get("max_steps", 4)),
                        system=NOX_SYSTEM, schemas=WEB_SCHEMAS, banned=BANNED)  # fmt: skip
            self._sessions[session] = a
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
            return a

    def ask(self, session: str, question: str, on_event: Any = None) -> dict[str, Any]:
        """One question. Raises BusyError when another question is being answered."""
        q = " ".join(question.split())[:MAX_QUESTION]
        if not q:
            raise ValueError("ask Nox a question")
        low = q.lower()
        if any(re.search(rf"\b{w}\b", low) for w in ASKS_FOR_BETS):
            out = {"answer": REFUSAL, "tools": [], "checked": True, "unverified": [],
                   "refused": True, "seconds": 0.0}  # fmt: skip
            self._log(q, out)
            return out
        if not self._busy.acquire(blocking=False):
            raise BusyError("Nox is answering another question. Try again in a moment.")
        try:
            analyst = self._analyst(session[:64] or "default")
            ans = analyst.ask(q, on_event=on_event)
        finally:
            self._busy.release()
        text, refused = ans.text, False
        if ans.banned or not text:
            text, refused = REFUSAL, bool(ans.banned)
            if analyst.history:
                analyst.history[-1] = {"role": "assistant", "content": text}
        out = {
            "answer": text,
            "tools": [n for n, _, _ in ans.tool_calls],
            "checked": not ans.unverified,
            "unverified": ans.unverified,
            "refused": refused,
            "seconds": round(ans.seconds, 1),
        }
        self._log(q, out)
        return out

    def _log(self, q: str, out: dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": q, **out}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # the log is a convenience; never fail an answer over it


class BusyError(RuntimeError):
    """Another question is being answered."""


__all__ = ["BANNED", "REFUSAL", "AskNox", "BusyError", "LLMError", "WebTools"]
