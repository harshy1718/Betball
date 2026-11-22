"""The trader's review: the LLM explains a checked slip like a risk manager.

One LLM call, no tools: the code has already done every number (slip.check) and the desk's
call (desk.desk_view), so the model only explains. That makes it fast on a CPU and keeps the
same guarantee as the analyst: every number in the review must appear in the facts, with one
rewrite, then the leftovers are flagged.
"""

from __future__ import annotations

import time
from typing import Any

from football_edge.analyst.agent import _banned, _numbers

SYSTEM = """You are the trader on Harsh's private betting desk. You think like a risk manager: \
protect the bankroll first, profit second. You are given the desk's facts about one bet slip; \
the numbers and the desk's call are already worked out by code.

Write 4 to 7 short plain sentences:
1. Start with the desk's call and why, in your own words.
2. Name the weakest leg and what it costs (use the "without it" chances), and whether to drop it.
3. Give the suggested stake exactly as the facts give it.
4. If a price was entered, say whether it is above or below our fair price and pass on the note. \
If no price was entered, say the value can't be judged without the site's price.
Rules: use only numbers that appear in the facts; never calculate new ones. Never say guaranteed, \
lock, banker or sure thing. No markdown, no lists."""

BANNED = ("guaranteed", "lock", "banker", "sure thing", "can't lose", "cannot lose")


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def facts(c: dict[str, Any]) -> str:
    """The checked slip as plain lines, with the rounded numbers the review may quote."""
    lines = [f"Slip: {c['kind']}, {len(c['legs'])} leg{'s' if len(c['legs']) > 1 else ''}"]
    for i, x in enumerate(c["legs"], start=1):
        skill = "yes" if x["skill"] else "no (no better than a plain average)"
        lines.append(
            f"Leg {i}: {x['label']} in {x['home']} v {x['away']}: chance {_pct(x['p'])}, "
            f"confidence {x['confidence']}, model has skill on this kind of bet: {skill}"
        )
        if x["market"] == "cards":
            lines.append(f"Leg {i} note: sites count cards differently (some use booking points)")
    lines.append(f"Chance the whole slip wins: {_pct(c['p'])}")
    if c["fair_odds"]:
        lines.append(f"Our fair odds for the slip: {c['fair_odds']:.2f}")
    lines.append(f"Risk grade: {c['risk']}")
    lines.append(f"Weak legs: {', '.join(c['weak_legs']) or 'none'}")
    for lc in c["leg_costs"]:
        lines.append(
            f"Without {lc['label']} the slip's chance would be {_pct(lc['without'])} "
            f"(with it: {_pct(lc['with'])})"
        )
    lines.append(f"Suggested stake: {c['stake']:.0f} {c['currency']} ({c['stake_note']})")
    if c["price"]:
        pr = c["price"]
        lines.append(
            f"Price entered: {pr['odds']:.2f}; on average {pr['back_per_100']:.0f} back "
            f"per 100 staked; {pr['verdict']}. Note: {pr['note']}"
        )
    else:
        lines.append("Price entered: none")
    lines.append(f"Desk's call: {c['desk_view']}")
    return "\n".join(lines)


def review(client: Any, c: dict[str, Any]) -> dict[str, Any]:
    """{text, checked, unverified, seconds}. `client.chat(messages, tools)` as OllamaClient."""
    t0 = time.perf_counter()
    text_facts = facts(c)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"The desk's facts:\n{text_facts}\n\nYour review:"},
    ]
    allowed = _numbers(text_facts)
    msg, _ = client.chat(messages, [])
    text = str(msg.get("content") or "").strip()
    bad, words = sorted(_numbers(text) - allowed), _banned(text, BANNED)
    if bad or words or not text:
        problems = []
        if bad:
            problems.append(f"these numbers are not in the facts: {', '.join(bad)}")
        if words:
            problems.append(f"these words are not allowed: {', '.join(words)}")
        if not text:
            problems.append("the review was empty")
        messages += [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": f"Check: {'; '.join(problems)}. Rewrite the review. Review only.",
            },
        ]
        msg, _ = client.chat(messages, [])
        text = str(msg.get("content") or "").strip() or text
        bad, words = sorted(_numbers(text) - allowed), _banned(text, BANNED)
    if words:  # never show a promise the numbers can't back
        note = "(The trader's wording broke the desk's rules, so here is the call alone.)"
        text = f"{c['desk_view']} {note}"
        bad = []
    return {
        "text": text,
        "checked": not bad,
        "unverified": bad,
        "seconds": round(time.perf_counter() - t0, 1),
    }
