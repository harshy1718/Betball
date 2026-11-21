"""Turn extracted facts into the model's line-up number, before line-ups are announced.

Same scale as the `lineup` feature (features/lineup.py): log of expected XI value over the
usual XI value. For a team whose usual starters have market values v_i:

    expected XI value = usual value - sum(P(miss) * v_i for starters reported missing)
                        + sum(v_i for usual starters reported returning)  [capped at usual]

A player reported missing who is not a usual starter changes nothing (a squad player
being out does not weaken the XI). P(miss): out/suspended 1.0, doubtful 0.5.
Replacements are not modelled: the drop is an upper bound on the loss, which is why
the result is clipped like the historical feature.
"""

from __future__ import annotations

import difflib
import math
import unicodedata
from dataclasses import dataclass

from football_edge.features.lineup import CLIP
from football_edge.news.schema import MISS_PROB, ArticleFacts


def _norm(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(plain.lower().replace("-", " ").split())


def match_player(name: str, squad: list[str]) -> str | None:
    """Find a reported name in the squad: exact, then same surname if unique, then fuzzy."""
    by_norm = {_norm(p): p for p in squad}
    n = _norm(name)
    if n in by_norm:
        return by_norm[n]
    surname = n.split()[-1] if n else ""
    same = [p for k, p in by_norm.items() if k.split()[-1] == surname]
    if len(same) == 1:
        return same[0]
    close = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.85)
    return by_norm[close[0]] if close else None


@dataclass(frozen=True)
class TeamImpact:
    team: str
    lineup_estimate: float  # 0 = usual XI expected
    missing: tuple[str, ...]  # matched starters and their miss probability, for the card
    unmatched: tuple[str, ...]  # reported names we could not find in the usual XI or squad


def team_impact(
    team: str, facts: list[ArticleFacts], usual_xi: dict[str, float], squad: list[str]
) -> TeamImpact:
    """`usual_xi`: usual starters -> market value; `squad`: every registered player name."""
    usual = sum(usual_xi.values())
    if usual <= 0:
        return TeamImpact(team, 0.0, (), ())
    miss: dict[str, float] = {}
    back: set[str] = set()
    unmatched: list[str] = []
    for f in facts:
        for p in f.players:
            if _norm(p.team) != _norm(team):
                continue
            who = match_player(p.player, list(usual_xi) + squad)
            if who is None:
                unmatched.append(p.player)
                continue
            if who not in usual_xi:
                continue  # not a usual starter
            if p.status == "returning":
                back.add(who)
            else:  # several reports: keep the most pessimistic
                miss[who] = max(miss.get(who, 0.0), MISS_PROB[p.status])
    for who in back:
        miss.pop(who, None)  # a return report cancels an older absence report
    lost = sum(prob * usual_xi[who] for who, prob in miss.items())
    est = math.log(max(usual - lost, 1e-9) / usual)
    est = min(max(est, CLIP[0]), CLIP[1])
    shown = tuple(f"{who} ({prob:.0%})" for who, prob in sorted(miss.items()))
    return TeamImpact(team, est, shown, tuple(unmatched))
