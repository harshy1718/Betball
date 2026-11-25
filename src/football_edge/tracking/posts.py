"""Instagram posts: the strong calls only, chosen by a fixed rule before kickoff.

A strong call is decided by the track record (2020-26, walk-forward), never by feel and never
from this week's matches. For the match result and for over/under 2.5, the code finds the
lowest chance level from which *every* past level (with 30+ past calls) came true at least
65% of the time, with 100+ past calls in total (tracking.strong in the config). Today that
is favourites at 60%+; the goals model has no level that qualifies, so goals calls are never
strong until it earns one. A draw is never a strong call.

Harsh picks which strong calls to post. Marking one as posted adds it to the committed
`records/posted` log (append-only: it cannot be taken back), and the weekly recap grades
every posted call, misses included.

Captions are predictions only: no betting words (checked in code, same list as the video
scripts), no prices, no club badges.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from football_edge.config import Config
from football_edge.content.script import BETTING_WORDS
from football_edge.leagues import TAGS
from football_edge.tracking.results import Results
from football_edge.tracking.scorecard import graded_posts
from football_edge.tracking.snapshot import ledgers

LEAGUE_TAGS = TAGS


class NotPostableError(ValueError):
    """Not a strong call, unknown match, or already kicked off."""


def betting_words(text: str) -> list[str]:
    low = text.lower()
    return [w for w in BETTING_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def _tag(name: str) -> str:
    return "#" + re.sub(r"[^A-Za-z0-9]", "", name)


BAND = 0.05
STRONG_MARKETS = {"1x2": "Match result", "ou25": "Over/under 2.5 goals"}


def strong_levels(record: pd.DataFrame | None, rule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per market, the lowest chance from which calls have held up: every past 5% level from
    there up (with `min_band_n`+ calls) came true `min_hit`+ of the time, and there are
    `min_class_n`+ past calls in total. Markets with no such level are left out."""
    if record is None:
        return {}
    min_hit = float(rule.get("min_hit", 0.65))
    band_n, class_n = int(rule.get("min_band_n", 30)), int(rule.get("min_class_n", 100))
    out = {}
    for market in STRONG_MARKETS:
        t = record[(record["source"] == "model") & (record["market"] == market)]
        t = t[t["band"] >= 0.5].sort_values("band")
        for b in t["band"]:
            up = t[t["band"] >= b]
            solid = up[up["n"] >= band_n]
            n = int(up["n"].sum())
            if n >= class_n and len(solid) and bool((solid["hit_rate"] >= min_hit).all()):
                hit = float((up["n"] * up["hit_rate"]).sum() / n)
                out[market] = {"market": market, "name": STRONG_MARKETS[market],
                               "min_p": float(b), "n": n, "hit": hit}  # fmt: skip
                break
    return out


def strong_calls(legs: list[Any], levels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The strong calls of one match (0, 1 or 2), strongest first."""
    out = []
    lv = levels.get("1x2")
    res = [x for x in legs if x.market == "1x2" and x.sel != "draw"]
    if lv and res:
        x = max(res, key=lambda x: x.p)
        if x.p >= lv["min_p"]:
            team = x.home if x.sel == "home" else x.away
            out.append({"kind": "result", "leg": x, "call": f"{team} to win", "level": lv})
    lv = levels.get("ou25")
    ou = [x for x in legs if x.market == "ou25"]
    if lv and ou:
        x = max(ou, key=lambda x: x.p)
        if x.p >= lv["min_p"]:
            call = f"{x.sel.capitalize()} 2.5 goals"
            out.append({"kind": "goals", "leg": x, "call": call, "level": lv})
    return sorted(out, key=lambda c: -c["leg"].p)


def caption(m: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    lines = [
        f"{m['home']} v {m['away']} · {m['league_name']} · {m['day_ist']}, {m['time_ist']} IST",
        "",
    ]
    for i, c in enumerate(calls):
        head = "Nox's call" if i == 0 else "Also strong"
        lines.append(f"{head}: {c['call']} ({_pct(c['leg'].p)})")
    pr = m["probs"]
    lines += [
        "",
        f"Win chances: {m['home']} {_pct(pr['home'])} · Draw {_pct(pr['draw'])} · "
        f"{m['away']} {_pct(pr['away'])}",
        f"Most likely score: {m['top_score']['score']} · Expected goals "
        f"{m['exp_goals']['home']:.1f}-{m['exp_goals']['away']:.1f}",
        "",
        "Model predictions for football fans. Every call is logged before kickoff and "
        "graded after the match, hits and misses.",
        "",
        " ".join(
            [
                "#nox",
                "#footballpredictions",
                "#football",
                _tag(LEAGUE_TAGS.get(m["league"], m["league_name"])),
                _tag(m["home"]),
                _tag(m["away"]),
            ]
        ),
    ]
    text = "\n".join(lines)
    bad = betting_words(text)
    if bad:  # team names never contain these; a hit means the template is wrong
        raise ValueError(f"betting language in a caption: {bad}")
    return text


def posted_keys(cfg: Config) -> set[tuple[str, str]]:
    df = ledgers(cfg)["posted"].read()
    return set(zip(df["match_id"], df["kind"], strict=True)) if len(df) else set()


def candidates(cfg: Config, desk: Any, now: pd.Timestamp | None = None) -> list[dict[str, Any]]:
    """Every match not yet kicked off with at least one strong call, soonest first."""
    now = now or pd.Timestamp.now(tz="UTC")
    matches, legs = desk.legs_by_match()
    levels = strong_levels(desk.svc.record, cfg.tracking.get("strong") or {})
    done = posted_keys(cfg)
    out = []
    for m in matches:
        if pd.Timestamp(m["kickoff_utc"]) <= now:
            continue
        calls = strong_calls(legs.get(m["id"], []), levels)
        if not calls:
            continue
        out.append({
            "match": m,
            "calls": [{"kind": c["kind"], "call": c["call"], "p": c["leg"].p,
                       "confidence": c["leg"].confidence,
                       "posted": (m["id"], c["kind"]) in done} for c in calls],
            "big": m["big"] > 0,
            "caption": caption(m, calls),
        })  # fmt: skip
    return out


def rule_text(cfg: Config, desk: Any) -> list[dict[str, Any]]:
    """What counts as strong right now, with the evidence behind it."""
    desk.legs_by_match()  # loads the track record
    levels = strong_levels(desk.svc.record, cfg.tracking.get("strong") or {})
    out = []
    for market, name in STRONG_MARKETS.items():
        lv = levels.get(market)
        if lv:
            what = "Favourite to win" if market == "1x2" else "Over or under 2.5 goals"
            text = (
                f"{what} at {lv['min_p']:.0%}+: in 2020-26 these came true "
                f"{lv['hit']:.0%} of the time ({lv['n']:,} calls)."
            )
        else:
            bar = float((cfg.tracking.get("strong") or {}).get("min_hit", 0.65))
            text = f"{name}: no chance level has held up at {bar:.0%}+ yet, so never a strong call."
        out.append({"market": market, "text": text})
    return out


def mark_posted(cfg: Config, desk: Any, match_id: str, kind: str,
                now: pd.Timestamp | None = None) -> dict[str, Any]:  # fmt: skip
    """Log a strong call as posted (once; never removed)."""
    now = now or pd.Timestamp.now(tz="UTC")
    matches, legs = desk.legs_by_match()
    m = next((x for x in matches if x["id"] == match_id), None)
    if m is None:
        raise NotPostableError("unknown match")
    if pd.Timestamp(m["kickoff_utc"]) <= now:
        raise NotPostableError("this match has kicked off: only calls made before kickoff count")
    levels = strong_levels(desk.svc.record, cfg.tracking.get("strong") or {})
    calls = {c["kind"]: c for c in strong_calls(legs.get(match_id, []), levels)}
    if kind not in calls:
        raise NotPostableError("not a strong call")
    if (match_id, kind) in posted_keys(cfg):
        return {"already": True}
    x = calls[kind]["leg"]
    row = pd.DataFrame([{
        "match_id": match_id, "league": x.league, "kickoff_utc": x.kickoff_utc,
        "home_key": x.home_key, "away_key": x.away_key, "home": x.home, "away": x.away,
        "kind": kind, "market": x.market, "sel": x.sel,
        "line": "" if x.line is None else f"{x.line:g}", "call": calls[kind]["call"],
        "p": f"{x.p:.4f}",
    }])  # fmt: skip
    ledgers(cfg)["posted"].append(row, now.isoformat(timespec="seconds").replace("+00:00", "Z"))
    return {"already": False}


def recap(cfg: Config, matches: pd.DataFrame, now: pd.Timestamp | None = None,
          days: int = 7) -> dict[str, Any]:  # fmt: skip
    """Every call posted for matches in the last `days`, graded: hits and misses."""
    now = now or pd.Timestamp.now(tz="UTC")
    posts = graded_posts(ledgers(cfg)["posted"].read(), Results(matches))
    if posts.empty:
        return {"from": None, "to": None, "calls": [], "settled": 0, "won": 0, "pending": 0}
    ko = pd.to_datetime(posts["kickoff_utc"], utc=True, format="ISO8601")
    week = posts[(ko > now - pd.Timedelta(days=days)) & (ko <= now)].copy()
    week = week.assign(_ko=ko[week.index]).sort_values("_ko")
    calls = [{
        "match": f"{r['home']} v {r['away']}", "call": r["call"], "p": float(r["p"]),
        "score": r["score"], "won": None if pd.isna(r["won"]) else bool(r["won"]),
        "day": f"{r['_ko'].tz_convert('Asia/Kolkata'):%a %d %b}",
    } for r in week.to_dict("records")]  # fmt: skip
    settled = [c for c in calls if c["won"] is not None]
    return {
        "from": f"{now - pd.Timedelta(days=days):%d %b}",
        "to": f"{now:%d %b %Y}",
        "calls": calls,
        "settled": len(settled),
        "won": sum(c["won"] for c in settled),
        "pending": len(calls) - len(settled),
    }
