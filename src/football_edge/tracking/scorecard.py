"""How good were the predictions? Graded from final results, on matches the model had not
seen: every block was logged before kickoff, and only the last one before kickoff counts.

Accuracy is measured three ways:
  * calibration: of everything called at 60-70%, did 60-70% come in?
  * Brier score (mean squared error of the chance, lower is better), against
      - a plain baseline: the league's own rate for that outcome over the previous
        `baseline_years` (does the model beat simply knowing the league?), and
      - the market: the bookmakers' closing prices with their margin taken out, where we have
        them (match result and over/under 2.5 only). Prices are used here as a benchmark of
        accuracy, nothing else.
  * how the desk's slips and the posted calls did against the chance we gave them.

Each binary question is counted once (over 2.5, not also under 2.5; double chance is the
mirror of the match result, so it is left out of the accuracy figures).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_edge.config import Config
from football_edge.leagues import NAMES as LEAGUE_NAMES
from football_edge.tracking.results import COUNT_MARKETS, Results, leg_won, parse_line
from football_edge.tracking.snapshot import ledgers
from football_edge.trader.legs import count_test, goal_test

NAMES = {
    "1x2": "Match result", "ou25": "Over/under 2.5 goals", "btts": "Both teams score",
    "goals": "Total goals (other lines)", "home_goals": "Team goals", "away_goals": "Team goals",
    "hcp": "Handicap", "wtn": "Win to nil", "oddeven": "Odd/even goals",
    "score": "Correct score", "cards": "Cards", "corners": "Corners",
}  # fmt: skip
ORDER = ["1x2", "ou25", "btts", "goals", "team_goals", "hcp", "wtn", "cards", "corners",
         "oddeven", "score"]  # fmt: skip
BANDS = [(i / 10, (i + 1) / 10) for i in range(10)]
SLIP_ORDER = ["Safest single", "Single", "Golden single", "Safe parlay", "Risky parlay",
              "Safe parlay across matches", "Risky parlay across matches"]  # fmt: skip
FLOORS = {"Safe parlay": 0.60, "Risky parlay": 0.25, "Safe parlay across matches": 0.60,
          "Risky parlay across matches": 0.25}  # fmt: skip


def family(market: str) -> str:
    return "team_goals" if market in ("home_goals", "away_goals") else market


def canonical(market: str, sel: str) -> bool:
    """One side of each yes/no question, so nothing is counted twice."""
    if market in ("1x2", "score", "wtn"):
        return True
    if market == "dc":
        return False
    if market == "btts":
        return sel == "yes"
    if market == "oddeven":
        return sel == "odd"
    if market == "hcp":
        return "-" in sel
    return sel == "over"  # ou25, goals, team goals, cards, corners


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601")


def last_before_kickoff(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Per group, the rows of the last block logged before kickoff."""
    if df.empty:
        return df
    df = df[_utc(df["taken_at"]) < _utc(df["kickoff_utc"])]
    if df.empty:
        return df
    return df[df["taken_at"] == df.groupby(group)["taken_at"].transform("max")]


def grade(rows: pd.DataFrame, results: Results) -> pd.DataFrame:
    """Add `won` (True/False) and the match's result row index; drop rows not yet played."""
    if rows.empty:
        return rows.assign(won=pd.Series(dtype=bool), score=pd.Series(dtype=str))
    found: dict[str, pd.Series | None] = {}
    won, score, keep, res = [], [], [], []
    for i, r in rows.iterrows():
        mid = str(r["match_id"])
        if mid not in found:
            found[mid] = results.find(str(r["league"]), str(r["home_key"]), str(r["away_key"]),
                                      str(r["kickoff_utc"]))  # fmt: skip
        hit = found[mid]
        if hit is None:
            continue
        w = leg_won(hit, str(r["market"]), str(r["sel"]), parse_line(r["line"]))
        if w is None:
            continue
        keep.append(i)
        won.append(w)
        score.append(f"{int(hit['FTHG'])}-{int(hit['FTAG'])}")
        res.append(hit)
    out = rows.loc[keep].copy()
    out["won"] = won
    out["score"] = score
    out["_res"] = pd.Series(res, index=out.index, dtype=object)
    return out


class Baseline:
    """The league's own rate for an outcome over the `years` before a date."""

    def __init__(self, matches: pd.DataFrame, years: int = 3) -> None:
        self.years = years
        m = matches.dropna(subset=["FTHG", "FTAG"]).sort_values("date")
        self.by = {str(lg): g.reset_index(drop=True) for lg, g in m.groupby("league")}
        self._cache: dict[tuple[str, str, str, float | None], tuple[np.ndarray, ...]] = {}

    def _series(self, lg: str, market: str, sel: str, line: float | None) -> tuple[np.ndarray, ...]:
        key = (lg, market, sel, line)
        if key not in self._cache:
            g = self.by[lg]
            if market in COUNT_MARKETS:
                cols = ["HC", "AC"] if market == "corners" else ["HY", "AY", "HR", "AR"]
                tot = g[cols].sum(axis=1, min_count=len(cols)).to_numpy(dtype=float)
                ok = ~np.isnan(tot)
                hit = np.where(ok, count_test(sel, float(line or 0))(np.nan_to_num(tot)), False)
            else:
                h, a = g["FTHG"].to_numpy(int), g["FTAG"].to_numpy(int)
                ok = np.ones(len(g), bool)
                hit = goal_test(market, sel, line)(h, a)
            dates = g["date"].to_numpy(dtype="datetime64[ns]")
            self._cache[key] = (dates, np.cumsum(ok), np.cumsum(hit & ok))
        return self._cache[key]

    def rate(self, lg: str, market: str, sel: str, line: float | None, date: pd.Timestamp) -> float:
        if lg not in self.by:
            return float("nan")
        dates, n, k = self._series(lg, market, sel, line)
        day = (date.tz_convert(None) if date.tzinfo else date).normalize()  # not the same day
        d = np.datetime64(day, "ns")
        lo = np.searchsorted(dates, d - np.timedelta64(int(365.25 * self.years), "D"))
        hi = np.searchsorted(dates, d)
        cnt = (n[hi - 1] if hi else 0) - (n[lo - 1] if lo else 0)
        win = (k[hi - 1] if hi else 0) - (k[lo - 1] if lo else 0)
        return float(win / cnt) if cnt else float("nan")


def market_prob(res: pd.Series, market: str, sel: str) -> float:
    """The closing prices' chance with the margin taken out (result and over/under 2.5)."""
    cols = {"1x2": ("close_h", "close_d", "close_a"), "ou25": ("ou_close_over", "ou_close_under")}
    if market not in cols:
        return float("nan")
    odds: list[Any] = [res.get(c) for c in cols[market]]
    if any(o is None or pd.isna(o) or float(o) <= 1 for o in odds):
        return float("nan")
    inv = np.array([1 / float(o) for o in odds])
    inv /= inv.sum()
    i = {"home": 0, "draw": 1, "away": 2, "over": 0, "under": 1}[sel]
    return float(inv[i])


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def _num(x: float) -> float | None:
    return None if x is None or np.isnan(x) else round(float(x), 4)


def accuracy(g: pd.DataFrame, base: Baseline, min_n: int) -> dict[str, Any]:
    """Accuracy figures for graded canonical predictions."""
    q = g[[canonical(m, s) for m, s in zip(g["market"], g["sel"], strict=True)]].copy()
    q["pf"] = q["p"].astype(float)
    q["y"] = q["won"].astype(float)
    q["base"] = [
        base.rate(lg, m, s, parse_line(ln), pd.Timestamp(k))
        for lg, m, s, ln, k in zip(q["league"], q["market"], q["sel"], q["line"], q["kickoff_utc"],
                                   strict=True)
    ]  # fmt: skip
    q["mkt"] = [
        market_prob(r, m, s) for r, m, s in zip(q["_res"], q["market"], q["sel"], strict=True)
    ]
    q["fam"] = q["market"].map(family)

    def row(d: pd.DataFrame, name: str, fam: str) -> dict[str, Any]:
        b = d.dropna(subset=["base"])
        mk = d.dropna(subset=["mkt"])
        bm, bb = (
            _brier(b["pf"].to_numpy(), b["y"].to_numpy()),
            _brier(b["base"].to_numpy(), b["y"].to_numpy()),
        )
        skill = 1 - bm / bb if bb and not np.isnan(bb) and bb > 0 else float("nan")
        y_mk = mk["y"].to_numpy()
        out = {
            "family": fam, "name": name, "n": len(d), "matches": int(d["match_id"].nunique()),
            "avg_p": _num(d["pf"].mean()), "hit": _num(d["y"].mean()),
            "brier": _num(_brier(d["pf"].to_numpy(), d["y"].to_numpy())),
            "base_brier": _num(bb), "skill": _num(skill),
            "market_n": len(mk),
            "market_brier": _num(_brier(mk["mkt"].to_numpy(), y_mk)) if len(mk) else None,
            "model_brier_same": _num(_brier(mk["pf"].to_numpy(), y_mk)) if len(mk) else None,
        }  # fmt: skip
        out["verdict"] = verdict(out, min_n)
        return out

    markets = [row(q[q["fam"] == f], NAMES.get(f, "Team goals"), f)
               for f in ORDER if (q["fam"] == f).any()]  # fmt: skip
    bands = []
    for lo, hi in BANDS:
        d = q[(q["pf"] >= lo) & ((q["pf"] < hi) | (hi == 1.0))]
        if len(d):
            bands.append({"band": f"{lo:.0%}-{hi:.0%}", "lo": lo, "hi": hi, "n": len(d),
                          "avg_p": _num(d["pf"].mean()), "hit": _num(d["y"].mean())})  # fmt: skip
    conf = []
    for label in ("High", "Medium", "Low"):
        d = q[(q["confidence"] == label) & (q["market"] != "score")]
        if len(d):
            conf.append({"label": label, "n": len(d), "avg_p": _num(d["pf"].mean()),
                         "hit": _num(d["y"].mean()),
                         "gap": _num(d["y"].mean() - d["pf"].mean())})  # fmt: skip
    return {"overall": row(q, "All predictions", "all") if len(q) else None,
            "markets": markets, "calibration": bands, "confidence": conf}  # fmt: skip


CAL_MARKETS = ("1x2", "ou25", "btts")
CAL_REVIEW_AT = 300  # ADR 0020: graded matches before the live calibration check decides


def calibration_check(graded: pd.DataFrame) -> dict[str, Any] | None:
    """ADR 0020 live check: Brier of the calibrated chances against the raw ones logged next to
    them, summed over match result, over/under 2.5 and both-score, on the same predictions."""
    if graded.empty or "p_raw" not in graded.columns:
        return None
    q = graded[graded["market"].isin(CAL_MARKETS) & (graded["p_raw"].astype(str) != "")]
    q = q[[canonical(m, x) for m, x in zip(q["market"], q["sel"], strict=True)]]
    if q.empty:
        return None
    y = q["won"].astype(float).to_numpy()
    cal_b, raw_b = 0.0, 0.0
    per = []
    for m in CAL_MARKETS:
        d = q["market"].to_numpy() == m
        if not d.any():
            continue
        c = _brier(q["p"].astype(float).to_numpy()[d], y[d])
        r = _brier(q["p_raw"].astype(float).to_numpy()[d], y[d])
        cal_b, raw_b = cal_b + c, raw_b + r
        per.append({"market": NAMES[m], "calibrated": _num(c), "raw": _num(r)})
    n = int(q["match_id"].nunique())
    if n < CAL_REVIEW_AT:
        status = f"Collecting: {n} of {CAL_REVIEW_AT} graded matches before the check decides"
    elif cal_b <= raw_b:
        status = "Calibration is helping: keep it"
    else:
        status = "Calibration is not helping on live matches: switch it off (ADR 0020)"
    return {"matches": n, "calibrated": _num(cal_b), "raw": _num(raw_b), "markets": per,
            "review_at": CAL_REVIEW_AT, "status": status}  # fmt: skip




def by_league(graded: pd.DataFrame, base: Baseline, min_n: int) -> list[dict[str, Any]]:
    """The overall accuracy row and the match-result row for each league on its own, so a
    league that behaves differently shows up (ADR 0023)."""
    out: list[dict[str, Any]] = []
    if graded.empty:
        return out
    for lg, g in graded.groupby("league"):
        acc = accuracy(g, base, min_n)
        if acc["overall"] is None:
            continue
        res = next((r for r in acc["markets"] if r["family"] == "1x2"), None)
        out.append({"league": str(lg), "name": LEAGUE_NAMES.get(str(lg), str(lg)),
                    "overall": acc["overall"], "result": res,
                    "calibration": acc["calibration"]})  # fmt: skip
    return out


def verdict(r: dict[str, Any], min_n: int) -> str:
    if r["n"] < min_n:
        return f"Too few to judge yet ({r['n']} of {min_n} needed)"
    parts = []
    s = r["skill"]
    if s is not None:
        parts.append("beats the league baseline" if s > 0.02 else
                     "worse than the league baseline" if s < -0.02 else
                     "no better than the league baseline")  # fmt: skip
    if r["market_brier"] is not None and r["market_n"] >= min_n:
        diff = r["model_brier_same"] - r["market_brier"]
        parts.append("more accurate than the market" if diff < -0.002 else
                     "less accurate than the market" if diff > 0.002 else
                     "level with the market")  # fmt: skip
    return "; ".join(parts).capitalize() if parts else ""


def slip_record(slips: pd.DataFrame, preds: pd.DataFrame, results: Results) -> list[dict[str, Any]]:
    """Every desk slip, graded: per match the last version before kickoff; the top of the
    board, every distinct slip it showed."""
    if slips.empty or preds.empty:
        return []
    slips = slips[_utc(slips["taken_at"]) < _utc(slips["kickoff_utc"])]
    per_match = last_before_kickoff(slips[slips["group"] != "top"], "group")
    top = slips[slips["group"] == "top"].drop_duplicates(["title", "keys"], keep="last")
    info = preds.drop_duplicates("match_id").set_index("match_id")
    rows = []
    for s in pd.concat([per_match, top]).itertuples(index=False):
        wins = []
        for key in str(s.keys).split(";"):
            mid, market, sel, line = key.split("|")
            if mid not in info.index:
                break
            m = info.loc[mid]
            hit = results.find(str(m["league"]), str(m["home_key"]), str(m["away_key"]),
                               str(m["kickoff_utc"]))  # fmt: skip
            w = None if hit is None else leg_won(hit, market, sel, parse_line(line))
            if w is None:
                break
            wins.append(w)
        else:
            rows.append({"title": s.title, "p": float(str(s.p)), "won": all(wins)})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for t in [x for x in SLIP_ORDER if x in set(df["title"])]:
        d = df[df["title"] == t]
        out.append({"title": t, "n": len(d), "expected": _num(d["p"].mean()),
                    "hit": _num(d["won"].mean()), "won": int(d["won"].sum()),
                    "floor": FLOORS.get(t)})  # fmt: skip
    return out


def graded_posts(posted: pd.DataFrame, results: Results) -> pd.DataFrame:
    if posted.empty:
        return posted.assign(won=pd.Series(dtype=object), score=pd.Series(dtype=str))
    rows = []
    for r in posted.drop_duplicates(["match_id", "kind"], keep="first").to_dict("records"):
        hit = results.find(r["league"], r["home_key"], r["away_key"], r["kickoff_utc"])
        w = None if hit is None else leg_won(hit, r["market"], r["sel"], parse_line(r["line"]))
        rows.append(
            {
                **r,
                "won": w,
                "score": None if hit is None else f"{int(hit['FTHG'])}-{int(hit['FTAG'])}",
            }
        )
    return pd.DataFrame(rows)


def recent_matches(graded: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """The latest graded matches with Nox's main calls: the most likely result and the
    over/under 2.5 side, each with the chance given and whether it came in."""
    if graded.empty:
        return []
    out = []
    for mid, g in graded.groupby("match_id"):
        g = g.assign(pf=g["p"].astype(float))
        r0 = g.iloc[0]
        row: dict[str, Any] = {"id": str(mid), "kickoff_utc": str(r0["kickoff_utc"]),
                               "league": str(r0["league"]), "home": str(r0["home_key"]),
                               "away": str(r0["away_key"]), "score": str(r0["score"])}  # fmt: skip
        res = g[g["market"] == "1x2"]
        if len(res):
            x = res.sort_values("pf").iloc[-1]
            name = {"home": f"{r0['home_key']} win", "away": f"{r0['away_key']} win"}
            row["result"] = {"call": name.get(str(x["sel"]), "Draw"), "p": float(x["pf"]),
                             "won": bool(x["won"])}  # fmt: skip
        ou = g[g["market"] == "ou25"]
        if len(ou):
            x = ou.sort_values("pf").iloc[-1]
            row["goals"] = {"call": f"{str(x['sel']).capitalize()} 2.5 goals",
                            "p": float(x["pf"]), "won": bool(x["won"])}  # fmt: skip
        out.append(row)
    return sorted(out, key=lambda r: r["kickoff_utc"], reverse=True)[:n]


def build(cfg: Config, matches: pd.DataFrame, now: pd.Timestamp | None = None) -> dict[str, Any]:
    now = now or pd.Timestamp.now(tz="UTC")
    t = cfg.tracking
    books = ledgers(cfg)
    preds = books["predictions"].read()
    min_n = int(t.get("min_sample", 30))
    results = Results(matches)
    latest = last_before_kickoff(preds, "match_id")
    graded = grade(latest, results)
    base = Baseline(matches, int(t.get("baseline_years", 3)))
    acc = accuracy(graded, base, min_n) if len(graded) else {
        "overall": None, "markets": [], "calibration": [], "confidence": []}  # fmt: skip
    posts = graded_posts(books["posted"].read(), results)
    done = posts[posts["won"].notna()] if len(posts) else posts
    problems = [p.__dict__ for name in ("predictions", "posted") for p in books[name].verify()]
    open_ = latest[~latest["match_id"].isin(set(graded["match_id"]))] if len(latest) else latest
    ko = _utc(open_.drop_duplicates("match_id")["kickoff_utc"]) if len(open_) else pd.Series([])
    return {
        "since": str(preds["taken_at"].min())[:10] if len(preds) else None,
        "logged_matches": int(latest["match_id"].nunique()) if len(latest) else 0,
        "graded_matches": int(graded["match_id"].nunique()) if len(graded) else 0,
        "upcoming_matches": int((ko > now).sum()),
        "awaiting_results": int((ko <= now).sum()),
        "next_kickoff": str(ko[ko > now].min()) if (ko > now).any() else None,
        "min_sample": min_n,
        "recent": recent_matches(graded),
        **acc,
        "slips": slip_record(books["slips"].read(), preds, results),
        "calibration_check": calibration_check(graded),
        "by_league": by_league(graded, base, min_n),
        "posted": {
            "n": len(posts),
            "settled": len(done),
            "won": int(done["won"].astype(bool).sum()) if len(done) else 0,
        },
        "log_ok": not problems,
        "log_problems": problems[:10],
    }
