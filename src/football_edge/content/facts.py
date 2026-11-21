"""The fact sheet: everything a script may say about one match, and nothing else.

Built only from our own numbers: the forecast (market fair probabilities where prices were
available, else the model's), corners and cards, confidence labels, and each team's recent
record from the results data. The sheet is rendered as plain text for the LLM; the set of
numbers in that text is what the number check allows a script to contain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from football_edge.predict.confidence import BAND, LOW_N, lookup

LEAGUE_NAMES = {"E0": "Premier League", "I1": "Serie A", "SP1": "La Liga"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


_CLUB_WORDS = {"fc", "cf", "afc", "ac", "acf", "as", "ss", "ssc", "us", "ud", "cd", "rcd",
               "ca", "rc", "bc", "calcio", "club", "de"}  # fmt: skip


def display_names(
    aliases_path: Path | None, overrides: dict[str, dict[str, str]] | None = None
) -> dict[tuple[str, str], str]:
    """Our short names -> the name a presenter would say ("Man United" -> "Manchester
    United"). Uses the first alias with no digits and no club-type words (FC, Calcio...),
    else our own name; `overrides` (config content.display_names) wins over both."""
    out: dict[tuple[str, str], str] = {}
    if aliases_path is not None and aliases_path.exists():
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
        for league, mapping in raw.items():
            for long_name, ours in (mapping or {}).items():
                words = {w.lower() for w in long_name.replace(".", " ").split()}
                plain = not any(ch.isdigit() for ch in long_name) and not words & _CLUB_WORDS
                if (league, ours) not in out and plain:
                    out[(league, ours)] = long_name
    for league, mapping in (overrides or {}).items():
        for ours, shown in (mapping or {}).items():
            out[(league, ours)] = shown
    return out


@dataclass(frozen=True)
class FactSheet:
    league: str
    date: pd.Timestamp
    home: str  # display names
    away: str
    sections: dict[str, list[str]]
    data: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        parts = []
        for title, lines in self.sections.items():
            parts.append(f"{title}:")
            parts += [f"- {line}" for line in lines]
        return "\n".join(parts)

    def numbers(self) -> set[str]:
        return {normalise_number(n) for n in NUMBER.findall(self.text())}

    def to_json(self) -> str:
        return json.dumps(
            {"league": self.league, "date": f"{self.date:%Y-%m-%d}", "home": self.home,
             "away": self.away, "sections": self.sections, "data": self.data},
            indent=2, ensure_ascii=False, default=str,
        )  # fmt: skip


def normalise_number(token: str) -> str:
    """'1.50' -> '1.5', '64.0' -> '64'; so formatting differences never fail the check."""
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token.lstrip("0") or "0"


def _n(count: int, word: str) -> str:
    """'1 win', '2 wins', '1 draw', '0 losses'."""
    if count == 1:
        return f"1 {word}"
    return f"{count} {word}es" if word.endswith("s") else f"{count} {word}s"


def _why(record: pd.DataFrame | None, source: str, market: str, p: float, label: str) -> str:
    """The real reason behind a confidence label, in words a viewer understands. Confidence
    comes from our track record at this probability level, never from this fixture."""
    if record is None:
        return "no track record yet"
    band = round(float(np.floor(p / BAND) * BAND), 2)
    row = record[(record["source"] == source) & (record["market"] == market)
                 & (record["band"] == band)]  # fmt: skip
    if row.empty:
        return "we have no past predictions at this level"
    r = row.iloc[0]
    n, hit, lo = int(r["n"]), float(r["hit_rate"]), round(band * 100)
    past = (
        f"in the past, calls like this at {lo} to {lo + 5}% came true {_pct(hit)} of the "
        f"time across {n:,} matches"
    )
    if label == "High":
        return past + ", so this level of prediction has been reliable"
    if not bool(r["skill"]):
        return past + ", but they were no better than a simple average"
    if n < LOW_N:
        return past + ", which is too few matches to be sure"
    if label == "Low":
        return past + ", which is far from the prediction"
    return past + ", fairly close but not reliable enough for high confidence"


def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def _recent(history: pd.DataFrame, team: str, before: pd.Timestamp, n: int) -> pd.DataFrame:
    played = history[
        ((history["home"] == team) | (history["away"] == team)) & (history["date"] < before)
    ]
    return played.sort_values("date").tail(n)


def _form(history: pd.DataFrame, team: str, before: pd.Timestamp) -> tuple[str, int, int]:
    """Last 5 results as letters (oldest first), goals scored, goals conceded."""
    last = _recent(history, team, before, 5)
    at_home = (last["home"] == team).to_numpy()
    hg, ag = last["FTHG"].to_numpy(dtype=int), last["FTAG"].to_numpy(dtype=int)
    mine, theirs = np.where(at_home, hg, ag), np.where(at_home, ag, hg)
    letters = [
        "W" if m > t else ("D" if m == t else "L") for m, t in zip(mine, theirs, strict=True)
    ]
    return " ".join(letters), int(mine.sum()), int(theirs.sum())


def _per_game(history: pd.DataFrame, team: str, before: pd.Timestamp, h: str, a: str) -> float:
    """Average of a per-side stat (e.g. cards) for this team over its last 10 matches."""
    last = _recent(history, team, before, 10)
    if last.empty or h not in last.columns:
        return float("nan")
    vals = np.where(last["home"] == team, last[h], last[a]).astype(float)
    return float(np.nanmean(vals))


def build_fact_sheet(
    f: pd.Series,
    history: pd.DataFrame,
    record: pd.DataFrame | None,
    names: dict[tuple[str, str], str] | None = None,
) -> FactSheet:
    """`f`: one row of the forecast table (forecast + market_view + count_forecasts)."""
    names = names or {}
    lg, date = str(f["league"]), pd.Timestamp(f["date"])
    home = names.get((lg, f["home"]), str(f["home"]))
    away = names.get((lg, f["away"]), str(f["away"]))

    def prob(col: str) -> tuple[float, str]:
        m = f.get("mkt_" + col[2:], np.nan)
        return (float(m), "market") if pd.notna(m) else (float(f[col]), "model")

    (ph, src), (pd_, _), (pa, _) = prob("p_h"), prob("p_d"), prob("p_a")
    (po, src_ou) = prob("p_over25")
    pick_i = int(np.argmax([ph, pd_, pa]))
    pick_p = [ph, pd_, pa][pick_i]
    pick_txt = [f"{home} win", "a draw", f"{away} win"][pick_i]
    res_label = lookup(record, src, "1x2", pick_p)[0]
    ou_txt, ou_p = ("over 2.5 goals", po) if po >= 0.5 else ("under 2.5 goals", 1 - po)
    ou_label = lookup(record, src_ou, "ou25", ou_p)[0]

    sections: dict[str, list[str]] = {
        "Match": [f"{home} v {away}, {LEAGUE_NAMES.get(lg, lg)}, {date:%A %d %B %Y}"],
        "Prediction": [
            f"Chance of {home} win: {_pct(ph)}",
            f"Chance of a draw: {_pct(pd_)}",
            f"Chance of {away} win: {_pct(pa)}",
            f"Most likely outcome: {pick_txt} ({_pct(pick_p)}), confidence {res_label}",
            f"Why that confidence: {_why(record, src, '1x2', pick_p, res_label)}",
            f"Most likely scores: {f['top_scores']}",
            f"Expected goals: {home} {f['exp_home']:.1f}, {away} {f['exp_away']:.1f}",
        ],
        "Goals": [
            f"Chance of over 2.5 goals: {_pct(po)}",
            f"Goals call: {ou_txt} ({_pct(ou_p)}), confidence {ou_label}",
            f"Why that confidence: {_why(record, src_ou, 'ou25', ou_p, ou_label)}",
            f"Chance both teams score: {_pct(float(f['p_btts_yes']))}",
        ],
    }
    cc = []
    if "exp_cards" in f.index:
        cc += [
            f"Expected total cards in the match: {f['exp_cards']:.1f}",
            f"Chance of over 4.5 cards: {_pct(float(f['p_cards_over_4.5']))}",
        ]
    if "p_reds_over_0.5" in f.index:
        cc.append(f"Chance of a red card: {_pct(float(f['p_reds_over_0.5']))}")
    if "exp_corners" in f.index:
        cc += [
            f"Expected corners in the match: {f['exp_corners']:.1f}",
            f"Chance of over 9.5 corners: {_pct(float(f['p_corners_over_9.5']))}",
        ]
    if cc:
        sections["Cards and corners"] = cc

    form = []
    with_cards = history.assign(
        HCARD=history["HY"] + history["HR"] if {"HY", "HR"} <= set(history.columns) else np.nan,
        ACARD=history["AY"] + history["AR"] if {"AY", "AR"} <= set(history.columns) else np.nan,
    )
    for team, ours in ((home, f["home"]), (away, f["away"])):
        letters, gf, ga = _form(history, ours, date)
        if letters:
            res = letters.split()
            form.append(
                f"{team} last {len(res)} matches (oldest first): {letters}; "
                f"{_n(res.count('W'), 'win')}, {_n(res.count('D'), 'draw')}, "
                f"{_n(res.count('L'), 'loss')}; "
                f"scored {gf}, conceded {ga}"
            )
        cards = _per_game(with_cards, ours, date, "HCARD", "ACARD")
        corners = _per_game(history, ours, date, "HC", "AC")
        if not np.isnan(cards):
            form.append(f"{team} average cards per match (last 10): {cards:.1f}")
        if not np.isnan(corners):
            form.append(f"{team} average corners won per match (last 10): {corners:.1f}")
    if form:
        sections["Recent form"] = form

    h2h = history[
        (((history["home"] == f["home"]) & (history["away"] == f["away"]))
         | ((history["home"] == f["away"]) & (history["away"] == f["home"])))
        & (history["date"] < date)
    ].sort_values("date").tail(6)  # fmt: skip
    if not h2h.empty:
        lines = []
        for r in h2h.itertuples(index=False):
            hn = names.get((lg, r.home), r.home)
            an = names.get((lg, r.away), r.away)
            lines.append(f"{r.date:%Y}: {hn} {int(r.FTHG)}-{int(r.FTAG)} {an}")
        mine = np.where(h2h["home"] == f["home"], h2h["FTHG"], h2h["FTAG"])
        theirs = np.where(h2h["home"] == f["home"], h2h["FTAG"], h2h["FTHG"])
        lines.append(
            f"In the last {len(h2h)} meetings: {home} won {int((mine > theirs).sum())}, "
            f"{_n(int((mine == theirs).sum()), 'draw')}, {away} won {int((mine < theirs).sum())}; "
            f"both teams scored in {int(((mine > 0) & (theirs > 0)).sum())}; "
            f"over 2.5 goals in {int(((mine + theirs) >= 3).sum())}"
        )
        sections["Head to head (most recent last)"] = lines

    data = {
        "probabilities": {"home": ph, "draw": pd_, "away": pa, "over25": po, "source": src},
        "pick": {"result": pick_txt, "p": pick_p, "confidence": res_label,
                 "goals": ou_txt, "goals_p": ou_p, "goals_confidence": ou_label},
    }  # fmt: skip
    return FactSheet(lg, date, home, away, sections, data)
