"""The trader desk: the legs of every upcoming match, slips Nox builds himself, checking any
slip, and the saved-slip book with its track record. Private: Harsh's own PC only.

The board (fixed recipe, `trader:` in configs/default.yaml), generated, never hand-picked:
  * strong legs: not weak (model has skill, confidence not Low), match not kicked off and
    within `days`; corners only if `include_corners`; correct scores only as singles;
  * per match, Good singles: the best leg of each bet family (result, total goals, team goals,
    both score, cards...) with a chance inside `leg_range` (55-85%), High confidence first;
  * per match, two parlays, each the highest-paying combination of strong legs (up to
    `parlay_leg_max`) above a floor, where every leg cuts the chance by `parlay_min_cost`+
    (5 points: no passengers) and one leg per family (`trader.parlays`):
      - Safe: comes in 60%+ of the time, 2-4 legs;
      - Risky: comes in 25%+ of the time, 2-5 legs, High-confidence legs only (the maths
        behind them has held up), so the risk is in the length, not in shaky legs;
  * top: the safest single on the board, and a safe and a risky parlay across matches (one
    strong leg per match, same rules); golden calls appear as golden singles under their match.
Desk view (code, not the LLM): skip / trim / play from the risk grade, weak legs and price.
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_edge.betting.ledger import month_status, read_log
from football_edge.config import Config
from football_edge.trader import markets as more
from football_edge.trader.legs import COUNT_LINES, Leg, LegFactory, count_test, goal_test
from football_edge.trader.slip import Book, History, check
from football_edge.web.service import NoxService

SLIP_COLS = [
    "saved_on",
    "slip_id",
    "kind",
    "legs",
    "labels",
    "prob",
    "risk",
    "price",
    "stake",
    "result",
    "won",
    "profit",
]


FAMILIES = {"1x2": "result", "dc": "result", "hcp": "result", "wtn": "result",
            "ou25": "total", "goals": "total",
            **{m.market: m.family for m in more.CATALOGUE if m.market != "goals"}}  # fmt: skip


GROUPS = [  # how the All markets list is grouped on the desk page
    ("Result", ("1x2", "dc", "dnb", "margin")),
    ("Handicaps", ("hcp", "ahc")),
    ("Goals", ("ou25", "goals", "atot", "exact", "range", "oddeven")),
    ("Team goals", ("home_goals", "away_goals", "home_exact", "away_exact", "cs_home",
                    "cs_away", "wtn")),
    ("Both score & combos", ("btts", "res_btts", "res_ou", "ou_btts")),
    ("Correct score", ("score",)),
    ("1st half", ("fh1x2", "fhgoals", "fhbtts")),
    ("2nd half", ("sh1x2", "shgoals", "shbtts")),
    ("Half-time / full-time", ("htft", "hsh", "bh_home", "bh_away")),
    ("Cards", ("cards",)),
    ("Corners", ("corners",)),
]  # fmt: skip
_GROUP = {m: name for name, ms in GROUPS for m in ms}


def group(market: str) -> str:
    return _GROUP.get(market, "Other")


def family(market: str) -> str:
    """Bets that answer the same question (who wins, how many goals...) share a family; a
    slip uses at most one leg per family in each match."""
    return FAMILIES.get(market, market)


def desk_view(c: dict[str, Any]) -> str:
    """The desk's call on a checked slip, from fixed rules."""
    if c["price"] and c["price"]["verdict"].startswith("below"):
        return "Skip at this price: it is below our fair price."
    if c["risk"] == "Very high":
        return "Too risky as it stands: cut legs or skip."
    if c["weak_legs"] and c["kind"] == "single":
        return "A weak bet: our track record at this level, or on this kind of bet, is poor."
    if c["weak_legs"]:
        return (
            f"Drop the weak leg{'s' if len(c['weak_legs']) > 1 else ''} first: "
            + ", ".join(c["weak_legs"])
            + "."
        )
    if c["risk"] == "High":
        return "Playable only small: the chance of the whole slip is low."
    if not c["price"]:
        return "The risk is reasonable; enter your site's price to judge the value."
    return "Playable: reasonable risk at a price above our fair price."


class Desk:
    def __init__(self, cfg: Config, svc: NoxService) -> None:
        self.cfg, self.svc = cfg, svc
        self.conf: dict[str, Any] = dict(cfg.trader)
        self.path = cfg.root / str(self.conf.get("slips_path", "bets/slips.csv"))
        self._lock = threading.Lock()
        self._cache: tuple[Any, dict[str, Any]] | None = None
        self._board: tuple[Any, dict[str, Any]] | None = None

    # -- the legs of every upcoming match ----------------------------------------------
    def _state(self) -> dict[str, Any]:
        sl = self.svc.slate()
        with self._lock:
            if self._cache is not None and self._cache[0] == sl.key:
                return self._cache[1]
            history = self.svc.history()
            latest = int(history["season"].max()) if len(history) else None
            shares = more.half_shares(history, before_season=latest)  # ADR 0028
            factory = LegFactory(
                self._record(),
                tuple(self.conf.get("no_skill_markets", ["corners"])),
                self._count_lines(),
                extended=bool(self.conf.get("extended_markets", True)),
                half_shares=shares,
                best=self._best(),
            )
            self._has_best = bool(factory.best)
            matches, legs, grids, counts, by_match = [], {}, {}, {}, {}
            for _, f in sl.fc.iterrows():
                info = self.svc.summary(f)
                ls = factory.for_match(info, f, sl.model)
                matches.append(info)
                legs[info["id"]] = ls
                if str(info["league"]) in shares:
                    by_match[info["id"]] = shares[str(info["league"])]
                grids[info["id"]] = sl.model.score_matrix(
                    float(f["exp_home"]), float(f["exp_away"])
                )
                counts[info["id"]] = {  # every line the count models give, for ranges
                    (c.split("_")[1], float(c.rsplit("_", 1)[1])): float(f[c])
                    for c in f.index
                    if c.startswith(("p_corners_over_", "p_cards_over_")) and pd.notna(f[c])
                }
            book = Book(grids, counts, History.from_matches(history), by_match)
            state = {"matches": matches, "legs": legs, "book": book, "history": history}
            self._cache = (sl.key, state)
            return state

    def legs_by_match(self) -> tuple[list[dict[str, Any]], dict[str, list[Leg]]]:
        """(match summaries, every leg of every match in the slate) for the prediction log."""
        st = self._state()
        return st["matches"], st["legs"]

    def _count_lines(self) -> dict[str, tuple[float, ...]]:
        conf = self.conf.get("count_lines") or {}
        return {s: tuple(float(x) for x in conf.get(s, COUNT_LINES[s])) for s in COUNT_LINES}

    def _record(self) -> pd.DataFrame | None:
        """The main track record plus the extra goal bets' (track_record_goals.csv)."""
        base = self.svc.record
        extra = self.cfg.reports_dir / "confidence" / "track_record_goals.csv"
        if base is None or not extra.exists():
            return base
        return pd.concat([base, pd.read_csv(extra)], ignore_index=True)

    def _best(self) -> set[tuple[str, str, float | None]]:
        """The best markets (ADR 0029), from reports/confidence/market_skill.csv. Without that
        file (run build-confidence) the board falls back to its older markets."""
        from football_edge.trader.record import best_set

        path = self.cfg.reports_dir / "confidence" / "market_skill.csv"
        return best_set(pd.read_csv(path)) if path.exists() else set()

    def _eligible(self, x: Leg, single: bool) -> bool:
        """May the board use this leg? `best_only`: best markets only; `add_best`: the markets the
        board always used plus every best market (ADR 0029); refund bets as singles only."""
        ok_void = single or x.market not in more.VOID_MARKETS
        if bool(self.conf.get("best_only", False)) and getattr(self, "_has_best", True):
            return x.best and ok_void
        if bool(self.conf.get("add_best", False)):  # the old board markets plus every best market
            return x.board or (x.best and ok_void)
        return x.board

    def leg(self, key: str) -> Leg:
        mid = key.split("|", 1)[0]
        legs: list[Leg] = self._state()["legs"].get(mid, [])
        for x in legs:
            if x.key == key:
                return x
        raise KeyError(f"unknown leg {key}")

    def options(self, match_id: str) -> dict[str, Any]:
        st = self._state()
        if match_id not in st["legs"]:
            raise KeyError(match_id)
        info = next(m for m in st["matches"] if m["id"] == match_id)
        legs = [
            {**x.to_dict(), "group": group(x.market), "refund": x.market in more.VOID_MARKETS}
            for x in st["legs"][match_id]
        ]
        return {"match": info, "legs": legs, "groups": [g for g, _ in GROUPS]}

    def check(self, keys: list[str], price: float | None = None) -> dict[str, Any]:
        legs = [self.leg(k) for k in keys]
        log = read_log(self.cfg.root / self.cfg.staking.get("log_path", "bets/log.csv"))
        ms = month_status(log, pd.Timestamp.today())
        c = check(
            legs,
            self._state()["book"],
            self.cfg.staking,
            self.conf,
            ms.settled_profit,
            ms.pending_stake,
            price,
            self.exposure(),
        )
        c["desk_view"] = desk_view(c)
        return c

    # -- slips Nox builds -----------------------------------------------------------------
    def _open(self, now: pd.Timestamp) -> list[dict[str, Any]]:
        days = int(self.conf.get("days", 3))
        return [
            m
            for m in self._state()["matches"]
            if now <= pd.Timestamp(m["kickoff_utc"]) <= now + pd.Timedelta(days=days)
        ]

    def _candidates(self, mid: str) -> list[Leg]:
        lo, hi = (float(x) for x in self.conf.get("leg_range", [0.55, 0.85]))
        corners = bool(self.conf.get("include_corners", False))
        out = [
            x
            for x in self._state()["legs"][mid]
            if lo <= x.p <= hi
            and self._eligible(x, single=True)
            and not x.weak
            and x.market != "score"
            and (corners or x.market != "corners")
        ]
        return sorted(out, key=lambda x: (x.confidence == "High", x.p), reverse=True)

    def _pool(self, mid: str, hi: float) -> list[Leg]:
        """Strong legs of one match for parlays: like the singles, but up to `hi` (a 90%
        leg is fine inside a parlay if it still pulls its weight)."""
        lo = float(self.conf.get("leg_range", [0.55, 0.85])[0])
        corners = bool(self.conf.get("include_corners", False))
        out = [
            x
            for x in self._state()["legs"][mid]
            if lo <= x.p <= hi
            and self._eligible(x, single=False)
            and not x.weak
            and x.market != "score"
            and (corners or x.market != "corners")
        ]
        return sorted(out, key=lambda x: x.p, reverse=True)

    def kinds(self) -> dict[str, dict[str, Any]]:
        """The two parlay types: safe and risky (defaults if the config has none)."""
        default = {
            "safe": {"floor": 0.60, "max_legs": 4, "high_only": False},
            "risky": {"floor": 0.25, "max_legs": 5, "high_only": True},
        }
        conf = self.conf.get("parlays") or {}
        return {k: {**v, **(conf.get(k) or {})} for k, v in default.items()}

    def best_parlay(
        self, pool: list[Leg], floor: float = 0.40, max_legs: int = 4, high_only: bool = False
    ) -> list[Leg] | None:
        """The highest-paying combination of 2..max_legs legs that still comes in at least
        `floor` of the time, with every leg cutting the chance by `parlay_min_cost` or more
        (no passengers), at most one leg per bet family; `high_only`: High-confidence legs
        only. None if nothing qualifies."""
        min_cost = float(self.conf.get("parlay_min_cost", 0.05))
        if high_only:
            pool = [x for x in pool if x.confidence == "High"]
        pool = pool[: int(self.conf.get("parlay_pool", 12))]
        book: Book = self._state()["book"]
        best: tuple[float, int, list[Leg]] | None = None
        for k in range(2, max_legs + 1):
            for combo in itertools.combinations(pool, k):
                fams = [(x.match_id, family(x.market)) for x in combo]
                if len(set(fams)) < k:
                    continue
                p = book.chance(list(combo))
                if p < floor or (best is not None and (1 / p, k) <= best[:2]):
                    continue
                if all(
                    book.chance([y for y in combo if y is not x]) - p >= min_cost for x in combo
                ):
                    best = (1 / p, k, list(combo))
        return best[2] if best else None

    def board(self, now: pd.Timestamp | None = None) -> dict[str, Any]:
        """Everything the desk generates: the top of the board and, per match, the good
        singles and the best parlay from that match. Cached until the forecasts change."""
        now = now or pd.Timestamp.now(tz="UTC")
        key = (self.svc.slate().key, now.floor("10min"))
        if self._board is not None and self._board[0] == key:
            return self._board[1]
        state = self._state()
        open_ = self._open(now)
        n_singles = int(self.conf.get("singles_per_match", 4))

        def slip(title: str, why: str, legs: list[Leg]) -> dict[str, Any]:
            return {"title": title, "why": why, **self.check([x.key for x in legs])}

        kinds = self.kinds()
        titles = {"safe": "Safe parlay", "risky": "Risky parlay"}
        whys = {
            "safe": f"the highest-paying mix that still comes in "
            f"{kinds['safe']['floor']:.0%}+ of the time",
            "risky": f"a bigger payout that comes in {kinds['risky']['floor']:.0%}+ of the time, "
            "built only from High-confidence legs that each pull their weight",
        }

        matches, firsts = [], []
        all_golden: list[dict[str, Any]] = []
        for m in open_:
            singles, seen = [], set()
            for x in self._candidates(m["id"]):  # best leg per bet family
                if family(x.market) not in seen:
                    seen.add(family(x.market))
                    singles.append(x)
            if singles:
                firsts.append(singles[0])
            golden: dict[str, str] = {}  # leg key -> why it is golden
            for g in m["golden"]:
                legs = state["legs"][m["id"]]
                if g["kind"] == "win":
                    leg = max((x for x in legs if x.market == "1x2"), key=lambda x: x.p)
                elif g["kind"] == "ou":
                    leg = max((x for x in legs if x.market == "ou25"), key=lambda x: x.p)
                else:
                    leg = next(x for x in legs if x.market == "score" and x.sel == g["call"])
                golden[leg.key] = f"golden call: {g['rule']}"
            # Good singles first (a golden one keeps its star), then golden calls not already
            # among them: every golden bet is always listed as a single, weak or not.
            chosen = singles[:n_singles]
            chosen += [x for x in state["legs"][m["id"]]
                       if x.key in golden and x not in chosen]  # fmt: skip
            single_slips = [
                slip("Golden single", golden[x.key], [x]) if x.key in golden
                else slip("Single", "a strong leg", [x])
                for x in chosen
            ]  # fmt: skip
            all_golden += [x for x in single_slips if x["title"] == "Golden single"]
            hi = float(self.conf.get("parlay_leg_max", 0.95))
            pool = self._pool(m["id"], hi)
            parlays = {}
            for kind, rule in kinds.items():
                combo = self.best_parlay(pool, rule["floor"], rule["max_legs"], rule["high_only"])
                parlays[kind] = slip(titles[kind], whys[kind], combo) if combo else None
            matches.append({"match": m, "singles": single_slips, "parlays": parlays})  # fmt: skip
        top = []
        firsts.sort(key=lambda x: (x.confidence == "High", x.p), reverse=True)
        if firsts:
            top.append(slip("Safest single", "the strongest leg on the board", [firsts[0]]))
        firsts.sort(key=lambda x: x.p, reverse=True)
        for kind, rule in kinds.items():
            across = self.best_parlay(firsts, rule["floor"], rule["max_legs"], rule["high_only"])
            if across:
                top.append(slip(f"{titles[kind]} across matches",
                                f"one strong leg per match: {whys[kind]}", across))  # fmt: skip
        out = {"top": top, "golden": all_golden, "matches": matches}
        self._board = (key, out)
        return out

    # -- the saved-slip book ---------------------------------------------------------------
    def exposure(self) -> dict[str, float]:
        """Stake riding on each match in saved slips not yet settled (match id -> amount)."""
        return open_exposure(self._read())

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=SLIP_COLS)
        return pd.read_csv(self.path, dtype={"result": object, "won": object, "price": float})

    def save(self, keys: list[str], price: float | None, stake: float | None) -> dict[str, Any]:
        c = self.check(keys, price)
        log = self._read()
        legs = [
            {
                k: x[k]
                for k in (
                    "key",
                    "league",
                    "kickoff_utc",
                    "home_key",
                    "away_key",
                    "market",
                    "sel",
                    "line",
                    "label",
                    "home",
                    "away",
                )
            }
            for x in c["legs"]
        ]
        row = {
            "saved_on": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "slip_id": f"s{len(log) + 1:04d}",
            "kind": c["kind"],
            "legs": json.dumps(legs),
            "labels": " + ".join(f"{x['label']} ({x['home']} v {x['away']})" for x in legs),
            "prob": round(c["p"], 4),
            "risk": c["risk"],
            "price": price,
            "stake": c["stake"] if stake is None else float(stake),
            "result": None,
            "won": None,
            "profit": None,
        }
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(self.path, index=False)
        return row

    def history(self) -> dict[str, Any]:
        log = settle(self._read(), self._state()["history"])
        if len(log):
            log.to_csv(self.path, index=False)
        done = log[log["result"].notna() & log["won"].astype(str).str.lower().ne("void")]
        record = []
        for risk, g in done.groupby("risk"):
            won = g["won"].astype(str).str.lower().eq("true")
            priced = g[g["profit"].notna()]
            record.append(
                {
                    "risk": risk,
                    "settled": len(g),
                    "won": int(won.sum()),
                    "expected": float(g["prob"].astype(float).mean()),
                    "profit": float(priced["profit"].astype(float).sum()) if len(priced) else None,
                }
            )
        rows = log.where(log.notna(), None).to_dict("records")
        return {"slips": rows[::-1], "record": record}


def open_exposure(log: pd.DataFrame) -> dict[str, float]:
    """Per match id, the stakes of open (unsettled) saved slips that include it."""
    out: dict[str, float] = {}
    if log.empty:
        return out
    for r in log[log["result"].isna()].to_dict("records"):
        stake = pd.to_numeric(pd.Series([r.get("stake")]), errors="coerce").fillna(0).iloc[0]
        mids = {str(x["key"]).split("|", 1)[0] for x in json.loads(str(r["legs"]))}
        for m in mids:
            out[m] = out.get(m, 0.0) + float(stake)
    return out


def settle(log: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Mark saved slips won or lost once every leg's match is in the results."""
    if log.empty:
        return log
    m = matches.dropna(subset=["FTHG", "FTAG"])
    for k in log.index[log["result"].isna()]:
        legs = json.loads(str(log.at[k, "legs"]))
        wins, scores = [], []
        for x in legs:
            day = pd.Timestamp(x["kickoff_utc"]).tz_convert(None).normalize()
            hit = m[
                (m["league"] == x["league"])
                & (m["home"] == x["home_key"])
                & (m["away"] == x["away_key"])
                & ((m["date"] - day).abs() <= pd.Timedelta(days=1))
            ]
            if hit.empty:
                break
            r = hit.iloc[0]
            h, a = np.array([int(r["FTHG"])]), np.array([int(r["FTAG"])])
            mk = more.find(x["market"], x["sel"], x.get("line"))
            if mk is not None:  # ADR 0028 markets: may need half-time, may be refunded
                if mk.half and (pd.isna(r.get("HTHG")) or pd.isna(r.get("HTAG"))):
                    break
                res = more.outcome(mk, int(r["FTHG"]), int(r["FTAG"]), r.get("HTHG"), r.get("HTAG"))
                wins.append(res)  # None = refunded
            elif x["market"] in ("corners", "cards"):
                tot = (
                    (r["HC"] + r["AC"])
                    if x["market"] == "corners"
                    else (r["HY"] + r["AY"] + r["HR"] + r["AR"])
                )
                if pd.isna(tot):
                    break
                wins.append(bool(count_test(x["sel"], float(x["line"]))(np.array([tot]))[0]))
            else:
                wins.append(bool(goal_test(x["market"], x["sel"], x.get("line"))(h, a)[0]))
            scores.append(f"{int(r['FTHG'])}-{int(r['FTAG'])}")
        else:
            log.at[k, "result"] = ", ".join(scores)
            if any(w is None for w in wins):  # a refund bet came back: no win, no loss
                log.at[k, "won"] = "void"
                log.at[k, "profit"] = 0.0
                continue
            won = all(wins)
            log.at[k, "won"] = won
            price = pd.to_numeric(pd.Series([log.at[k, "price"]]), errors="coerce").iloc[0]
            stake = float(
                pd.to_numeric(pd.Series([log.at[k, "stake"]]), errors="coerce").fillna(0).iloc[0]
            )
            log.at[k, "profit"] = (
                None if pd.isna(price) else (stake * (float(price) - 1) if won else -stake)
            )
    return log


def slips_path(root: Path, conf: dict[str, Any]) -> Path:
    return root / str(conf.get("slips_path", "bets/slips.csv"))
