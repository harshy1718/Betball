"""The analyst's tools. Every number the LLM may quote comes from one of these functions;
the LLM only decides which to call and explains the result."""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from football_edge.config import Config
from football_edge.content.facts import build_fact_sheet, display_names
from football_edge.data.loader import load_matches
from football_edge.leagues import NAMES
from football_edge.predict.confidence import load_track_record
from football_edge.predict.fixtures import FixtureError, single_fixture
from football_edge.predict.pipeline import forecast_with_model

LEAGUES = NAMES
NICKNAMES = {
    "spurs": "Tottenham", "barca": "Barcelona", "barça": "Barcelona", "atleti": "Ath Madrid",
    "atletico": "Ath Madrid", "athletic": "Ath Bilbao", "bilbao": "Ath Bilbao",
    "juve": "Juventus", "man utd": "Man United", "united": "Man United", "city": "Man City",
    "villa": "Aston Villa", "forest": "Nott'm Forest", "wolves": "Wolves",
    "palace": "Crystal Palace",
    "real": "Real Madrid", "madrid": "Real Madrid", "rayo": "Vallecano", "sociedad": "Sociedad",
    "real sociedad": "Sociedad", "inter milan": "Inter", "ac milan": "Milan",
}  # fmt: skip

SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "upcoming_matches",
        "description": "Matches still to be played in the next few days, with kick-off times "
        "in India time (IST). Use for 'what games are on', 'next matches', 'tonight'.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "how many days ahead, default 2"},
                "league": {
                    "type": "string",
                    "description": "optional: EPL, Serie A, La Liga, Bundesliga or Ligue 1",
                },
            },
        },
    },
    {
        "name": "predict_match",
        "description": "Our model's full prediction for one match: win/draw/loss, over/under "
        "2.5, both teams to score, expected goals, likeliest scores, cards, corners, the "
        "confidence of each call with its reason, and golden-bar flags. Use for any question "
        "about who will win, goals, cards, corners, scores or how sure we are.",
        "parameters": {
            "type": "object",
            "properties": {
                "home": {"type": "string", "description": "home team"},
                "away": {"type": "string", "description": "away team"},
            },
            "required": ["home", "away"],
        },
    },
    {
        "name": "team_form",
        "description": "A team's most recent league results with scores and dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string"},
                "matches": {"type": "integer", "description": "how many, default 5"},
            },
            "required": ["team"],
        },
    },
    {
        "name": "head_to_head",
        "description": "Past league meetings between two teams with scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string"},
                "team_b": {"type": "string"},
                "matches": {"type": "integer", "description": "how many, default 6"},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "price_check",
        "description": "Is a bookmaker price worth taking for one bet in one match? Give the "
        "teams, the bet and the decimal price. The tool looks up our model's probability itself "
        "and returns the fair price, expected return per 100 staked, whether it is a golden-bar "
        "pick, and a verdict. Bets: 'home win', 'draw', 'away win', 'over 2.5', 'under 2.5', "
        "'btts yes', 'btts no', or an exact score like '1-1'.",
        "parameters": {
            "type": "object",
            "properties": {
                "home": {"type": "string"},
                "away": {"type": "string"},
                "bet": {"type": "string", "description": "e.g. 'home win', 'over 2.5', '1-1'"},
                "price": {"type": "number", "description": "decimal odds, e.g. 8.4"},
            },
            "required": ["home", "away", "bet", "price"],
        },
    },
    {
        "name": "golden_scoreboard",
        "description": "How the two golden bars are doing so far (match winner 60-64% and "
        "likeliest score 12%+): picks logged, settled, hit rate, profit.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class ToolError(ValueError):
    """A tool could not answer (unknown team, no schedule...): shown to the LLM as the result."""


class AnalystTools:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.history = load_matches(cfg)
        self.known = {str(lg): set(g["home"]) | set(g["away"])
                      for lg, g in self.history.groupby("league")}  # fmt: skip
        self.aliases = cfg.root / cfg.predict.get("aliases", "configs/team_aliases.yaml")
        self.record = load_track_record(cfg.reports_dir / "confidence" / "track_record.csv")
        self.names = display_names(self.aliases, cfg.content.get("display_names"))
        self._reverse = {v.lower(): k[1] for k, v in self.names.items()}
        self._cache: dict[tuple[str, str], tuple[pd.Series, Any, Any]] = {}
        self.fns: dict[str, Callable[..., dict[str, Any]]] = {
            "upcoming_matches": self.upcoming_matches,
            "predict_match": self.predict_match,
            "team_form": self.team_form,
            "head_to_head": self.head_to_head,
            "golden_scoreboard": self.golden_scoreboard,
            "price_check": self.price_check,
        }

    # -- helpers ---------------------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool; errors come back as a result the LLM can explain, never a crash."""
        fn = self.fns.get(name)
        if fn is None:
            return json.dumps({"error": f"no tool called {name}"})
        try:
            return json.dumps(fn(**(args or {})), ensure_ascii=False, default=str)
        except (ToolError, FixtureError, TypeError) as e:
            return json.dumps({"error": str(e)})

    def _team(self, name: str) -> str:
        """User's name for a team -> our data's name ('Manchester United' -> 'Man United')."""
        all_teams: set[str] = {str(t) for ts in self.known.values() for t in ts}
        if name in all_teams:
            return name
        low = name.lower().strip()
        nick = NICKNAMES.get(low)
        if nick and nick in all_teams:
            return nick
        if low in self._reverse:
            return self._reverse[low]
        for t in all_teams:
            if t.lower() == low:
                return t
        pool = sorted(all_teams) + sorted(self._reverse)
        close = difflib.get_close_matches(name.lower(), [p.lower() for p in pool], n=3, cutoff=0.8)
        if close:  # a small typo ("Villareal"): take the nearest
            hit = close[0]
            return self._reverse.get(hit) or next(t for t in all_teams if t.lower() == hit)
        maybe = difflib.get_close_matches(name.lower(), [p.lower() for p in pool], n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(maybe)}?" if maybe else ""
        raise ToolError(
            f"unknown team '{name}'{hint} (EPL, Serie A, La Liga, Bundesliga and Ligue 1 only)"
        )

    def _show(self, league: str, team: str) -> str:
        return self.names.get((league, team), team)

    # -- tools -----------------------------------------------------------------------------
    def upcoming_matches(self, days: int = 2, league: str | None = None) -> dict[str, Any]:
        path = self.cfg.processed_dir / "schedule.parquet"
        if not path.exists():
            raise ToolError("no schedule saved yet: run `football-edge schedule`")
        s = pd.read_parquet(path)
        now = pd.Timestamp.now(tz="UTC")
        end = now + pd.Timedelta(days=int(days))
        sel = s[(s["kickoff_utc"] >= now) & (s["kickoff_utc"] <= end)
                & (s["status"] != "FINISHED")].sort_values("kickoff_utc")  # fmt: skip
        if league:
            want = {v.lower(): k for k, v in LEAGUES.items()} | {"epl": "E0", "pl": "E0"}
            code = want.get(league.lower().strip(), league)
            sel = sel[sel["league"] == code]
        return {"matches": [
            {"league": LEAGUES.get(r["league"], r["league"]),
             "kickoff_ist": r["kickoff_utc"].tz_convert("Asia/Kolkata").strftime("%a %d %b %H:%M"),
             "home": self._show(r["league"], r["home"]), "away": self._show(r["league"], r["away"])}
            for r in sel.to_dict("records")
        ]}  # fmt: skip

    def _forecast(self, home: str, away: str) -> tuple[pd.Series, Any, Any]:
        """(forecast row, training info, fitted model) for one match, cached per chat."""
        h, a = self._team(home), self._team(away)
        if (h, a) not in self._cache:
            fx = single_fixture(h, a, self.known, self.cfg.leagues, aliases_path=self.aliases)
            fc, info, model = forecast_with_model(self.cfg, self.history, fx)
            self._cache[(h, a)] = (fc.iloc[0], info, model)
        return self._cache[(h, a)]

    def predict_match(self, home: str, away: str) -> dict[str, Any]:
        f, info, _ = self._forecast(home, away)
        fs = build_fact_sheet(f, self.history, self.record, self.names, source="model")
        out: dict[str, Any] = {
            "match": f"{fs.home} v {fs.away}",
            "league": LEAGUES.get(fs.league, fs.league),
            "model_trained_up_to": f"{info.last_match:%d %b %Y}",
            **{k: v for k, v in fs.sections.items() if k != "Match"},
            "More corners and cards lines": [
                f"Chance of over 8.5 corners: {f['p_corners_over_8.5']:.0%}",
                f"Chance of over 10.5 corners: {f['p_corners_over_10.5']:.0%}",
                f"Chance of over 3.5 cards: {f['p_cards_over_3.5']:.0%}",
                f"Chance of over 5.5 cards: {f['p_cards_over_5.5']:.0%}",
            ],
        }
        golden = self._golden(f)
        if golden:
            out["Golden bars"] = golden
        return out

    def _golden(self, f: pd.Series) -> list[str]:
        from football_edge.betting.golden import flags

        names = {"win": "match winner", "ou": "over/under 2.5", "cs": "correct score"}
        return [f"Golden {names[k]} pick: {v.replace('⭐ ', '')} (unproven, being tracked)"
                for k, v in flags(f, self.cfg.staking).items() if v]  # fmt: skip

    def team_form(self, team: str, matches: int = 5) -> dict[str, Any]:
        t = self._team(team)
        h = self.history
        g = h[(h["home"] == t) | (h["away"] == t)].sort_values("date").tail(int(matches))
        if g.empty:
            raise ToolError(f"no matches for {team}")
        lg = str(g["league"].iloc[-1])
        rows = []
        for r in g.to_dict("records"):
            home = r["home"] == t
            mine, theirs = (r["FTHG"], r["FTAG"]) if home else (r["FTAG"], r["FTHG"])
            res = "W" if mine > theirs else ("D" if mine == theirs else "L")
            opp = self._show(lg, r["away"] if home else r["home"])
            rows.append(f"{pd.Timestamp(r['date']):%d %b %Y}: {res} {int(mine)}-{int(theirs)} "
                        f"{'v' if home else 'at'} {opp}")  # fmt: skip
        return {"team": self._show(lg, t), "results_oldest_first": rows}

    def head_to_head(self, team_a: str, team_b: str, matches: int = 6) -> dict[str, Any]:
        a, b = self._team(team_a), self._team(team_b)
        h = self.history
        g = h[((h["home"] == a) & (h["away"] == b)) | ((h["home"] == b) & (h["away"] == a))]
        g = g.sort_values("date").tail(int(matches))
        if g.empty:
            return {"meetings": [], "note": "no meetings in our data"}
        lg = str(g["league"].iloc[-1])
        return {"meetings_oldest_first": [
            f"{pd.Timestamp(r['date']):%d %b %Y}: {self._show(lg, r['home'])} {int(r['FTHG'])}-"
            f"{int(r['FTAG'])} {self._show(lg, r['away'])}" for r in g.to_dict("records")
        ]}  # fmt: skip

    def price_check(self, home: str, away: str, bet: str, price: float) -> dict[str, Any]:
        """The probability comes from our model, never from the LLM."""
        import re as _re

        from football_edge.betting.golden import flags, golden_picks, ou_picks

        odds = float(price)
        if odds <= 1:
            raise ToolError("the price must be decimal odds above 1, e.g. 1.85")
        f, _, model = self._forecast(home, away)
        lg = str(f["league"])
        hn, an = self._show(lg, str(f["home"])), self._show(lg, str(f["away"]))
        b = bet.lower().strip()
        gold = flags(f, self.cfg.staking)
        score = _re.fullmatch(r"(\d+)\s*-\s*(\d+)", b)
        golden = ""
        if score:
            i, j = int(score.group(1)), int(score.group(2))
            m = model.score_matrix(float(f["exp_home"]), float(f["exp_away"]))
            p = float(m[i, j]) if i < m.shape[0] and j < m.shape[1] else 0.0
            label = f"{i}-{j}"
            if gold["cs"] and str(f["top_score"]) == label:
                golden = "correct-score golden pick"
        else:
            choices = {
                "home win": float(f["p_h"]), "draw": float(f["p_d"]), "away win": float(f["p_a"]),
                "over 2.5": float(f["p_over25"]), "under 2.5": float(f["p_under25"]),
                "btts yes": float(f["p_btts_yes"]), "btts no": float(f["p_btts_no"]),
            }  # fmt: skip
            alias = {"home": "home win", "away": "away win", "over": "over 2.5",
                     "under": "under 2.5", "btts": "btts yes", "both teams to score": "btts yes",
                     f"{hn.lower()} win": "home win", f"{an.lower()} win": "away win",
                     f"{str(f['home']).lower()} win": "home win",
                     f"{str(f['away']).lower()} win": "away win"}  # fmt: skip
            key = alias.get(b, b)
            if key not in choices:
                raise ToolError(f"unknown bet '{bet}'; use one of: {', '.join(choices)} or a score")
            p, label = choices[key], key
            side = {"home win": "home", "draw": "draw", "away win": "away"}.get(key)
            row = f.to_frame().T
            gb = self.cfg.staking.get("golden_bar") or {}
            go = self.cfg.staking.get("golden_ou") or {}
            if side and gb:
                g = golden_picks(row, gb)
                if len(g) and g.iloc[0]["pick"] == side:
                    golden = "match-winner golden pick"
            if key in ("over 2.5", "under 2.5") and go:
                g = ou_picks(row, go)
                if len(g) and g.iloc[0]["pick"] == key.split()[0]:
                    golden = "over/under golden pick"
        if p <= 0:
            raise ToolError(f"the model gives {label} no chance; check the bet")
        use = 0.147 if golden == "correct-score golden pick" else p
        line = 1 / use
        note = "our model's own view; tests found the market is usually more accurate"
        if golden == "correct-score golden pick":
            note = ("uses the correct-score bar's 2020-26 hit rate (14.7%) instead of the "
                    f"model's {p:.1%} for this score. " + note)  # fmt: skip
        return {
            "match": f"{hn} v {an}",
            "bet": label,
            "model_probability": f"{p:.1%}",
            "fair_price": f"{line:.2f}",
            "expected_back_per_100": f"{use * odds * 100:.0f}",
            "golden": golden or "no: this bet is not in any golden bar",
            "verdict": (
                "worth taking: above the fair price"
                if odds >= line
                else "not worth it: below the fair price"
            ),
            "note": note,
        }

    def golden_scoreboard(self) -> dict[str, Any]:
        from football_edge.betting import golden

        out = {}
        for key, default, label in golden.BARS:
            conf = self.cfg.staking.get(key) or {}
            if conf:
                path = self.cfg.root / conf.get("log_path", default)
                out[label] = golden.scoreboard(golden.read(path), label)
        return out
