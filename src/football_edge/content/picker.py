"""Which matches get a video: a fixed rule, not a gut feeling (decided 2026-11-20).

A match qualifies when all three hold:
  1. Trusted numbers: the main call (result or goals) has a High confidence label.
  2. A clear story: a strong favourite (70%+), a goals game (over 2.5 at 65%+ or under
     2.5 at 65%+), or a card-heavy fixture (6+ expected cards).
  3. People search for it: at least one club from the configured big-club list.
Qualifying matches are ranked by the strength of their story; the top N are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

from football_edge.content.facts import FactSheet


@dataclass(frozen=True)
class Pick:
    facts: FactSheet
    story: str
    strength: float


def best_story(fs: FactSheet, exp_cards: float) -> tuple[float, str] | None:
    """The strongest story whose own call we trust (High confidence), or None.
    A story built on a call we rate Low would have the video contradict itself."""
    p = fs.data["pick"]
    stories = []
    if p["confidence"] == "High" and p["p"] >= 0.70 and p["result"] != "a draw":
        stories.append((p["p"], f"strong favourite: {p['result']} {p['p']:.0%}"))
    if p["goals_confidence"] == "High" and p["goals_p"] >= 0.65:
        stories.append((p["goals_p"], f"goals: {p['goals']} {p['goals_p']:.0%}"))
    if exp_cards >= 6:  # cards beat the league average in testing (counts evaluation)
        stories.append((exp_cards / 9, f"card-heavy: {exp_cards:.1f} expected cards"))
    return max(stories) if stories else None


def pick_matches(
    sheets: list[tuple[FactSheet, str, str, float]],
    big_clubs: dict[str, list[str]],
    n: int,
) -> list[Pick]:
    """`sheets`: (fact sheet, our home name, our away name, expected cards). A match needs
    a big club and a story whose call is rated High; the strongest stories win."""
    picks = []
    for fs, home, away, exp_cards in sheets:
        big = bool({home, away} & set(big_clubs.get(fs.league, [])))
        story = best_story(fs, exp_cards)
        if big and story is not None:
            picks.append(Pick(fs, story[1], story[0]))
    return sorted(picks, key=lambda x: -x.strength)[:n]
