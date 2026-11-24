"""Everything the Nox website shows, as plain dictionaries (no web framework in here).

The numbers come from the same pipeline as `football-edge predict --next`: the schedule from
football-data.org, the locked Dixon-Coles settings, our own probabilities only
(`staking.confidence: model`), and the confidence track record. Nothing is re-tuned for the
site and no prices or odds are returned: the site is public and shows predictions only.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_edge.betting import golden
from football_edge.config import Config
from football_edge.content.facts import _form, build_fact_sheet, display_names
from football_edge.data.loader import load_matches
from football_edge.predict.confidence import load_track_record
from football_edge.predict.fixtures import from_schedule
from football_edge.predict.forecast import TrainingInfo
from football_edge.predict.pipeline import forecast_with_model

LEAGUES = {"E0": "Premier League", "I1": "Serie A", "SP1": "La Liga"}
IST = "Asia/Kolkata"
GRID = 6  # scoreline grid 0..5 goals each side


class MatchNotFoundError(KeyError):
    """No upcoming match with that id."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def match_id(league: str, kickoff: pd.Timestamp, home: str, away: str) -> str:
    return f"{league.lower()}-{kickoff:%Y%m%d}-{_slug(home)}-{_slug(away)}"


def mood_of(p: float, confidence: str) -> str:
    """Nox's face. 'sure' when one outcome is more likely than not and the track record at
    that level is not Low; otherwise 'unsure' (a close call). 'shock' is kept for results."""
    return "sure" if p >= 0.5 and confidence not in ("Low", "-") else "unsure"


def nox_line(team: str, p: float, mood: str, confidence: str) -> str:
    """Nox's one-liner. Every number in it is the pick's own probability."""
    pct = round(100 * p)
    who = "A draw" if team == "Draw" else team
    if mood == "sure":
        tail = "Confident on this one." if confidence == "High" else "That's my lean."
        return f"{who}, {pct}%. {tail}"
    if p >= 0.5:  # more likely than not, but the track record at this level is weak
        return f"{who} {pct}%, but I'm not sure of this one."
    return f"{who} {pct}%, but this could go any way."


@dataclass
class Slate:
    """One forecast of every upcoming match in the window, plus what built it."""

    key: tuple[Any, ...]
    fc: pd.DataFrame  # forecast rows with kickoff_utc and id
    info: TrainingInfo | None
    model: Any


class NoxService:
    def __init__(self, cfg: Config, days: int = 7) -> None:
        self.cfg = cfg
        self.days = days
        self.aliases = cfg.root / cfg.predict.get("aliases", "configs/team_aliases.yaml")
        self.names = display_names(self.aliases, cfg.content.get("display_names"))
        self.source = str(cfg.staking.get("confidence", "model"))
        self._lock = threading.Lock()
        self._slate: Slate | None = None
        self._history: pd.DataFrame | None = None
        self._history_key: float | None = None
        self.record: pd.DataFrame | None = None

    # -- data --------------------------------------------------------------------------
    @property
    def schedule_path(self) -> Path:
        return self.cfg.processed_dir / "schedule.parquet"

    @property
    def matches_path(self) -> Path:
        return self.cfg.processed_dir / "matches.parquet"

    def history(self) -> pd.DataFrame:
        mtime = self.matches_path.stat().st_mtime if self.matches_path.exists() else 0.0
        if self._history is None or mtime != self._history_key:
            self._history = load_matches(self.cfg)
            self._history_key = mtime
            self.record = load_track_record(
                self.cfg.reports_dir / "confidence" / "track_record.csv"
            )
        return self._history

    def _show(self, league: str, team: str) -> str:
        return self.names.get((league, team), team)

    def slate(self, now: pd.Timestamp | None = None) -> Slate:
        """Forecast every match kicking off in the next `days` (or started < 2 hours ago).
        Rebuilt when the schedule or results change, or at most once an hour."""
        now = now or pd.Timestamp.now(tz="UTC")
        if not self.schedule_path.exists():
            raise FileNotFoundError("no schedule yet: run `football-edge schedule`")
        key = (
            self.schedule_path.stat().st_mtime,
            self.matches_path.stat().st_mtime if self.matches_path.exists() else 0.0,
            now.floor("h"),
        )
        with self._lock:
            if self._slate is not None and self._slate.key == key:
                return self._slate
            history = self.history()
            s = pd.read_parquet(self.schedule_path)
            sel = (
                s[
                    (s["kickoff_utc"] >= now - pd.Timedelta(hours=2))
                    & (s["kickoff_utc"] <= now + pd.Timedelta(days=self.days))
                    & (s["status"] != "FINISHED")
                    & s["home"].notna()
                    & s["away"].notna()
                ]
                .sort_values(["kickoff_utc", "league", "home"])
                .reset_index(drop=True)
            )
            if sel.empty:
                self._slate = Slate(key, pd.DataFrame(), None, None)
                return self._slate
            fixtures = from_schedule(sel, None)
            fc, info, model = forecast_with_model(self.cfg, history, fixtures)
            fc["kickoff_utc"] = sel["kickoff_utc"].to_numpy()
            fc["id"] = [match_id(str(lg), pd.Timestamp(k), str(h), str(a))
                        for lg, k, h, a in zip(fc["league"], sel["kickoff_utc"], fc["home"],
                                               fc["away"], strict=True)]  # fmt: skip
            self._slate = Slate(key, fc.reset_index(drop=True), info, model)
            return self._slate

    # -- one match ---------------------------------------------------------------------
    def _golden(self, f: pd.Series) -> list[dict[str, Any]]:
        """Golden calls for one match. Price-free: a call is golden whatever the odds."""
        st = self.cfg.staking
        row = f.to_frame().T
        lg = str(f["league"])
        league = LEAGUES.get(lg, lg)
        out: list[dict[str, Any]] = []
        gb, go, cs = (st.get(k) or {} for k in ("golden_bar", "golden_ou", "golden_cs"))
        if gb:
            g = golden.golden_picks(row, gb)
            if len(g):
                lo, hi = golden.winner_window(gb, lg)
                pick = str(g.iloc[0]["pick"])
                team = self._show(lg, str(f[pick])) if pick in ("home", "away") else "Draw"
                out.append({
                    "kind": "win", "title": "Match winner",
                    "call": f"{team} win" if team != "Draw" else "Draw",
                    "p": float(g.iloc[0]["prob"]),
                    "rule": f"{float(g.iloc[0]['prob']):.0%} sits inside {league}'s "
                            f"{lo:.0%}-{hi:.0%} window",
                })  # fmt: skip
        if go:
            g = golden.ou_picks(row, go)
            if len(g):
                lo, hi = float(go.get("lo", 0.51)), float(go.get("hi", 0.56))
                out.append({
                    "kind": "ou", "title": "Over / under 2.5",
                    "call": str(g.iloc[0]["label"]).capitalize(), "p": float(g.iloc[0]["prob"]),
                    "rule": f"{float(g.iloc[0]['prob']):.0%} sits inside the {lo:.0%}-{hi:.0%} "
                            "window",
                })  # fmt: skip
        if cs and float(f["top_score_p"]) >= float(cs.get("lo", 0.12)):
            out.append({
                "kind": "cs", "title": "Correct score", "call": str(f["top_score"]),
                "p": float(f["top_score_p"]),
                "rule": f"the likeliest score clears the {float(cs.get('lo', 0.12)):.0%} bar",
            })  # fmt: skip
        return out

    def summary(self, f: pd.Series) -> dict[str, Any]:
        lg = str(f["league"])
        fs = build_fact_sheet(f, self.history(), self.record, self.names, self.source)
        probs = fs.data["probabilities"]
        pick = fs.data["pick"]
        trio = [float(probs["home"]), float(probs["draw"]), float(probs["away"])]
        side = ("home", "draw", "away")[int(np.argmax(trio))]
        team = {"home": fs.home, "draw": "Draw", "away": fs.away}[side]
        p, conf = float(pick["p"]), str(pick["confidence"])
        why = next((x.split(": ", 1)[1] for x in fs.sections["Prediction"]
                    if x.startswith("Why that confidence")), "")  # fmt: skip
        mood = mood_of(p, conf)
        big = set((self.cfg.content.get("big_clubs") or {}).get(lg, []))
        ko = pd.Timestamp(f["kickoff_utc"])
        ist = ko.tz_convert(IST)
        return {
            "id": str(f["id"]),
            "league": lg,
            "league_name": LEAGUES.get(lg, lg),
            "kickoff_utc": ko.isoformat(),
            "day_ist": f"{ist:%a %d %b}",
            "time_ist": f"{ist:%H:%M}",
            "home": fs.home,
            "away": fs.away,
            "probs": {"home": trio[0], "draw": trio[1], "away": trio[2]},
            "pick": {
                "side": side,
                "team": team,
                "text": str(pick["result"]),
                "p": p,
                "confidence": conf,
                "why": why,
            },
            "mood": mood,
            "line": nox_line(team, p, mood, conf),
            "exp_goals": {"home": float(f["exp_home"]), "away": float(f["exp_away"])},
            "top_score": {"score": str(f["top_score"]), "p": float(f["top_score_p"])},
            "over25": float(f["p_over25"]),
            "btts": float(f["p_btts_yes"]),
            "golden": self._golden(f),
            "big": sum(str(t) in big for t in (f["home"], f["away"])),
        }

    def detail(self, f: pd.Series, model: Any) -> dict[str, Any]:
        out = self.summary(f)
        history = self.history()
        lg = str(f["league"])
        fs = build_fact_sheet(f, history, self.record, self.names, self.source)
        goals = fs.data["pick"]
        why_goals = next((x.split(": ", 1)[1] for x in fs.sections["Goals"]
                          if x.startswith("Why that confidence")), "")  # fmt: skip
        m = model.score_matrix(float(f["exp_home"]), float(f["exp_away"]))
        grid = [[float(m[i, j]) if i < m.shape[0] and j < m.shape[1] else 0.0
                 for j in range(GRID)] for i in range(GRID)]  # fmt: skip
        order = np.argsort(m, axis=None)[::-1][:3]
        top = [{"score": f"{i}-{j}", "p": float(m[i, j])}
               for i, j in zip(*np.unravel_index(order, m.shape), strict=True)]  # fmt: skip
        date = pd.Timestamp(f["date"])

        def form(team: str) -> dict[str, Any]:
            letters, gf, ga = _form(history, team, date)
            return {"results": letters.split(), "scored": gf, "conceded": ga}

        h2h = history[
            (((history["home"] == f["home"]) & (history["away"] == f["away"]))
             | ((history["home"] == f["away"]) & (history["away"] == f["home"])))
            & (history["date"] < date)
        ].sort_values("date").tail(4)  # fmt: skip
        meetings = [
            {"date": f"{pd.Timestamp(r['date']):%b %Y}", "home": self._show(lg, r["home"]),
             "away": self._show(lg, r["away"]), "score": f"{int(r['FTHG'])}-{int(r['FTAG'])}"}
            for r in h2h.to_dict("records")
        ][::-1]  # fmt: skip

        def get(col: str) -> float | None:
            v = f.get(col)
            return None if v is None or pd.isna(v) else float(v)

        out.update({
            "goals_call": {"text": str(goals["goals"]), "p": float(goals["goals_p"]),
                           "confidence": str(goals["goals_confidence"]), "why": why_goals},
            "top_scores": top,
            "grid": grid,
            "cards": {"expected": get("exp_cards"), "over_3_5": get("p_cards_over_3.5"),
                      "over_4_5": get("p_cards_over_4.5"), "red": get("p_reds_over_0.5")},
            "form": {"home": form(str(f["home"])), "away": form(str(f["away"]))},
            "meetings": meetings,
        })  # fmt: skip
        return out

    # -- endpoints ---------------------------------------------------------------------
    def meta(self, now: pd.Timestamp | None = None) -> dict[str, Any]:
        sl = self.slate(now)
        trained = sl.info.last_match if sl.info else pd.Timestamp(self.history()["date"].max())
        warning = None
        if len(sl.fc):
            gap = (pd.Timestamp(sl.fc["date"].min()) - trained).days
            if gap > int(self.cfg.predict.get("max_data_age_days", 10)):
                warning = (f"Latest result in our data is {trained:%d %b %Y}: run "
                           "`football-edge update` for fresher predictions.")  # fmt: skip
        return {"trained_up_to": f"{trained:%d %b %Y}", "matches": len(sl.fc),
                "days": self.days, "warning": warning}  # fmt: skip

    def matches(self, now: pd.Timestamp | None = None, league: str | None = None
                ) -> list[dict[str, Any]]:  # fmt: skip
        fc = self.slate(now).fc
        if fc.empty:
            return []
        if league:
            fc = fc[fc["league"] == league]
        return [self.summary(f) for _, f in fc.iterrows()]

    def next_matches(self, n: int = 5, now: pd.Timestamp | None = None) -> list[dict[str, Any]]:
        now = now or pd.Timestamp.now(tz="UTC")
        fc = self.slate(now).fc
        if fc.empty:
            return []
        fut = fc[pd.to_datetime(fc["kickoff_utc"], utc=True) >= now].head(n)
        return [self.summary(f) for _, f in fut.iterrows()]

    def match(self, mid: str, now: pd.Timestamp | None = None) -> dict[str, Any]:
        sl = self.slate(now)
        hit = sl.fc[sl.fc["id"] == mid] if len(sl.fc) else sl.fc
        if hit.empty:
            raise MatchNotFoundError(mid)
        return self.detail(hit.iloc[0], sl.model)

    def golden_calls(self, now: pd.Timestamp | None = None) -> dict[str, Any]:
        calls = [m for m in self.matches(now) if m["golden"]]
        st = self.cfg.staking
        bars = []
        gb = st.get("golden_bar") or {}
        if gb:
            wins = " · ".join(
                f"{LEAGUES.get(lg, lg)} {golden.winner_window(gb, lg)[0]:.0%}-"
                f"{golden.winner_window(gb, lg)[1]:.0%}"
                for lg in self.cfg.leagues
            )
            bars.append(
                {
                    "kind": "win",
                    "title": "Match winner",
                    "text": f"The favourite's chance is inside its league's window: {wins}.",
                }
            )
        go = st.get("golden_ou") or {}
        if go:
            lgs = ", ".join(LEAGUES.get(x, x) for x in go.get("leagues") or []) or "all leagues"
            lo, hi = float(go.get("lo", 0.51)), float(go.get("hi", 0.56))
            bars.append(
                {
                    "kind": "ou",
                    "title": "Over / under 2.5",
                    "text": f"{lgs} only, when the goals call is {lo:.0%}-{hi:.0%}.",
                }
            )
        cs = st.get("golden_cs") or {}
        if cs:
            bars.append(
                {
                    "kind": "cs",
                    "title": "Correct score",
                    "text": f"When the most likely score is at {float(cs.get('lo', 0.12)):.0%}"
                    " or more. In 2020-26 those came true 14.7% of the time.",
                }
            )
        tracker = []
        for key, default, label in golden.BARS:
            conf = st.get(key) or {}
            if not conf:
                continue
            log = golden.read(self.cfg.root / conf.get("log_path", default))
            done = log[log["result"].notna()]
            tracker.append(
                {
                    "title": label,
                    "logged": len(log),
                    "settled": len(done),
                    "came_true": int(done["won"].astype(bool).sum()) if len(done) else 0,
                }
            )
        return {"calls": calls, "bars": bars, "tracker": tracker}
