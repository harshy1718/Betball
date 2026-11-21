"""Write a 45-60 second video script from a fact sheet, with checks the LLM cannot skip.

The LLM (any OpenAI-compatible server, e.g. Ollama running Gemma on this PC) only writes
prose. After every draft, plain code checks it:
  * every number in the script appears in the fact sheet (no invented stats);
  * no betting language (Indian law and our own framing: predictions, not tips);
  * none of the stock phrases that make text read as machine-written;
  * the length fits a 45-60 second voice-over, and both teams are named.
A draft that fails goes back with the exact problems listed; after `max_retries` rewrites
the match is skipped rather than published with an error.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from football_edge.content.facts import NUMBER, FactSheet, normalise_number
from football_edge.news.extract import ChatClient

MAX_NUMBERS = 12  # a scoreline like 2-0 counts as one
MIN_WORDS, MAX_WORDS = 90, 170  # about 45-60 seconds spoken

BETTING_WORDS = (
    "bet", "bets", "betting", "odds", "bookie", "bookies", "bookmaker", "stake", "stakes",
    "wager", "punt", "tip", "tips", "tipster", "accumulator", "acca", "parlay", "banker",
    "lock", "sure thing", "guaranteed", "free money",
)  # fmt: skip
BANNED_PHRASES = (
    "delve", "in conclusion", "it's worth noting", "it is worth noting", "clash of titans",
    "titanic clash", "showdown", "buckle up", "without further ado", "in the world of",
    "when it comes to", "a testament to", "navigate", "tapestry", "game-changer",
    "let's dive", "dive into", "brace yourselves", "nail-biting", "edge-of-your-seat",
    "only time will tell", "all eyes", "stay tuned", "the main focus here", "the data points",
    "the data suggests",
)  # fmt: skip

SYSTEM_PROMPT = """You write the voice-over for a 45-60 second football prediction video.
You will get a fact sheet. Write the script a presenter reads aloud.

Hard rules:
- Use ONLY facts from the fact sheet. Every number you say must appear in it exactly.
  Write numbers as digits (64%, 2.5, 1-0), never as words.
- This is a prediction and analysis channel. Never mention betting, odds, tips, stakes,
  bookmakers or money.
- 110 to 150 words. Plain text only: no headings, emojis, hashtags or stage directions.

Style:
- Talk like a knowledgeable friend, not a commentator. Short and long sentences mixed.
- Build the whole script around THE STORY given above the fact sheet: open with a hook
  about it and come back to it at the end. Everything else supports it.
- Choose the 5 or 6 most telling numbers. Do not recite every line of the fact sheet.
- Say the confidence of the story's call and its reason using only the "Why that
  confidence" line. Confidence comes from our past track record, never from this
  match's history, so never link it to form, results or head to head.
- Never explain why something will happen unless the fact sheet says so.
- End with one natural line asking viewers to follow for the next prediction.
- Avoid cliches like "clash", "showdown", "buckle up", "delve", "stay tuned"."""

EXAMPLE = """Example of the style (fictional teams, do not reuse its facts):
Northbridge have won 4 of their last 5, and Eastvale come in having conceded 9 in that \
run. Our model gives Northbridge a 58% chance, with a draw at 24%. The most likely \
score is 2-1. Goals look likely too: there is a 61% chance of over 2.5, and both teams \
have scored in 4 of the last 6 meetings. It could get feisty, with 5.2 cards expected and \
a 19% chance of a red. Confidence in the Northbridge call is medium: calls like this \
have been fairly close in the past, but not reliable enough to be sure. Follow for the \
next one."""


@dataclass(frozen=True)
class ScriptResult:
    text: str | None  # None when every attempt failed the checks
    attempts: int
    problems: tuple[str, ...]  # problems of the last failed attempt
    seconds: float
    history: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'%.\-]+", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def check_script(text: str, facts: FactSheet, story: str | None = None) -> list[str]:
    """Every reason this script may not be published (empty list = OK)."""
    problems = []
    allowed = facts.numbers()
    unknown = sorted({n for n in (normalise_number(t) for t in NUMBER.findall(text))
                      if n not in allowed}, key=float)  # fmt: skip
    if unknown:
        problems.append(f"numbers not in the fact sheet: {', '.join(unknown)}")
    low = text.lower()
    bad = [w for w in BETTING_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
    if bad:
        problems.append(f"betting language: {', '.join(bad)}")
    cliches = [p for p in BANNED_PHRASES if p in low]
    if cliches:
        problems.append(f"phrases to avoid: {', '.join(cliches)}")
    n = len(_words(text))
    if not MIN_WORDS <= n <= MAX_WORDS:
        problems.append(f"{n} words; needs {MIN_WORDS}-{MAX_WORDS}")
    for team in (facts.home, facts.away):
        if team.lower() not in low:
            problems.append(f"team not named: {team}")
    if re.search(r"(^|\n)\s*(#|\*\*|\[)", text):
        problems.append("formatting found; plain spoken text only")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        s = sentence.lower()
        if "confiden" in s and (
            re.search(r"\b(19|20)\d\d\b|\d+\s*-\s*\d+", s)
            or re.search(r"\b(form|head to head|meetings?|last \d+|results?)\b", s)
        ):
            problems.append(
                "confidence linked to results or form; confidence comes only from our track "
                f'record, use the "Why that confidence" line: "{sentence.strip()}"'
            )
    # The hook: the story's own number must come in the first two sentences.
    story_pct = re.findall(r"(\d+)%", story or "")
    if story_pct and f"{story_pct[-1]}%" not in " ".join(_sentences(text)[:2]):
        problems.append(
            f"the opening must be about THE STORY: say {story_pct[-1]}% in the first two sentences"
        )
    # A call we rate Low may be mentioned only together with its low confidence.
    pick = facts.data.get("pick", {})
    for p_key, c_key in (("p", "confidence"), ("goals_p", "goals_confidence")):
        if pick.get(c_key) != "Low" or p_key not in pick:
            continue
        pct = f"{round(float(pick[p_key]) * 100)}%"
        for s in _sentences(text):
            if pct in s and "low" not in s.lower():
                problems.append(
                    f"{pct} is a call we rate Low confidence: drop it or say so in that "
                    f'sentence: "{s.strip()}"'
                )
    # Distinct numbers only (repeating the story's number is good), and the required
    # confidence sentence does not count against the limit.
    body = " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if "confiden" not in s.lower())
    body = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1:\2", body)  # a scoreline counts as one
    used = sorted({normalise_number(t) for t in re.findall(r"\d+(?::\d+|\.\d+)?", body)})
    if len(used) > MAX_NUMBERS:
        problems.append(
            f"{len(used)} different numbers ({', '.join(used)}); at most {MAX_NUMBERS}: drop "
            "the least telling ones (the confidence sentence does not count)"
        )
    return problems


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(script|voice-?over)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+\n", "\n", text)


def write_script(
    client: ChatClient,
    facts: FactSheet,
    max_retries: int = 2,
    on_attempt: Callable[[int, float, list[str]], None] | None = None,
    story: str | None = None,
) -> ScriptResult:
    start = time.perf_counter()
    lead = story or "no single strong story: lead with the call that has the highest confidence"
    base = f"{EXAMPLE}\n\nTHE STORY: {lead}\n\nFACT SHEET:\n{facts.text()}\n\nWrite the script now."
    user = base
    history: list[tuple[str, tuple[str, ...]]] = []
    problems: list[str] = []
    for attempt in range(1, max_retries + 2):
        t0 = time.perf_counter()
        text, usage = client.complete(SYSTEM_PROMPT, user)
        text = _clean(text)
        if text:
            problems = check_script(text, facts, story)
        else:
            problems = [
                "the model returned an empty answer (stop reason: "
                f"{usage.get('done_reason') or 'unknown'}, hidden thinking: "
                f"{usage.get('thinking_chars', 'unknown')} characters)"
            ]
        history.append((text, tuple(problems)))
        if on_attempt is not None:
            on_attempt(attempt, time.perf_counter() - t0, problems)
        if not problems:
            return ScriptResult(text, attempt, (), time.perf_counter() - start, history)
        user = (
            f"{base}\n\nYour previous draft was:\n{text}\n\nIt broke these rules: "
            + "; ".join(problems)
            + ". Rewrite it, fixing every problem, and keep everything else natural."
        )
    return ScriptResult(None, len(history), tuple(problems), time.perf_counter() - start, history)
