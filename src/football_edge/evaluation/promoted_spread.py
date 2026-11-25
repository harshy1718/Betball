"""ADR 0026 (promoted teams) and ADR 0027 (less spread in goal counts).

Both are tested as changes to the live chain: Dixon-Coles weekly walk-forward predictions, then a
goal calibration fitted season by season on earlier seasons only. The champion is the live
five-number calibration. Challengers:

* A: the calibration gets a sixth number `a_p`, the goal-difference stretch in matches with a
  promoted side in its first 19 league matches of the season;
* B: Dixon-Coles refitted with a weaker pull towards the promoted prior (ADR 0026), then the
  usual five-number calibration;
* spread: the calibration gets a sixth number `nu`, a Conway-Maxwell-Poisson dispersion for each
  side's goals (mean kept), under the low-score correction and draw adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from football_edge.models.calibration import adjust_draws, goal_losses, goal_markets
from football_edge.models.dixon_coles import score_grids

FloatArr = npt.NDArray[np.float64]
RHO = -0.035
EARLY = 19  # "first 19 league matches" (ADR 0026)
KMAX = 30


# -- promoted flag ----------------------------------------------------------------------------
def promoted_sides(matches: pd.DataFrame) -> pd.DataFrame:
    """Per match (index of `matches`), columns home / away: is that side a promoted team in its
    first EARLY league matches of the season? Known before kickoff."""
    m = matches
    sides = []
    for side in ("home", "away"):
        sides.append(
            pd.DataFrame(
                {
                    "idx": m.index,
                    "side": side,
                    "league": m["league"],
                    "season": m["season"],
                    "date": m["date"],
                    "team": m[side],
                }
            )
        )
    t = pd.concat(sides).sort_values(["date", "idx"], kind="stable")
    t["n"] = t.groupby(["league", "season", "team"]).cumcount() + 1
    present = set(map(tuple, t[["league", "season", "team"]].drop_duplicates().to_numpy()))
    first = m.groupby("league")["season"].min().to_dict()
    t["promoted"] = [
        se > first[lg] and (lg, se - 1, tm) not in present
        for lg, se, tm in zip(t["league"], t["season"], t["team"], strict=True)
    ]
    t["early"] = t["promoted"] & (t["n"] <= EARLY)
    wide = t.pivot(index="idx", columns="side", values="early")
    return wide.reindex(m.index).fillna(False).astype(bool)[["home", "away"]]


def promoted_early(matches: pd.DataFrame) -> pd.Series:
    """Per match: is either side a promoted team in its first EARLY matches?"""
    return promoted_sides(matches).any(axis=1)


# -- Conway-Maxwell-Poisson with a given mean -------------------------------------------------
def cmp_pmf(means: FloatArr, nu: float, kmax: int = KMAX) -> FloatArr:
    """P(goals = k), k = 0..kmax, for each mean; nu = 1 is Poisson, above 1 less spread."""
    k = np.arange(kmax + 1)
    lf = gammaln(k + 1)
    zs = np.linspace(-7.0, 4.0, 4401)
    logw = zs[:, None] * k[None, :] - nu * lf[None, :]
    logp = logw - logsumexp(logw, axis=1, keepdims=True)
    mean_z = (np.exp(logp) * k).sum(axis=1)
    z = np.interp(np.asarray(means, dtype=np.float64), mean_z, zs)
    lw = z[:, None] * k[None, :] - nu * lf[None, :]
    out: FloatArr = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
    return out


def cmp_grids(lam: FloatArr, mu: FloatArr, rho: float, nu: float, max_goals: int = 10) -> FloatArr:
    """Score grids with CMP marginals and the Dixon-Coles low-score correction; tail folded."""
    ph, pa = cmp_pmf(lam, nu), cmp_pmf(mu, nu)
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1 - lam * mu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + mu * rho
    m[:, 1, 1] *= 1 - rho
    k = max_goals
    out = np.empty((len(lam), k + 1, k + 1))
    out[:, :k, :k] = m[:, :k, :k]
    out[:, k, :k] = m[:, k:, :k].sum(axis=1)
    out[:, :k, k] = m[:, :k, k:].sum(axis=2)
    out[:, k, k] = m[:, k:, k:].sum(axis=(1, 2))
    grids: FloatArr = out / out.sum(axis=(1, 2), keepdims=True)
    return grids


# -- the calibration with optional extras -----------------------------------------------------
@dataclass(frozen=True)
class Map:
    a: float = 1.0
    b: float = 0.0
    d: float = 1.0
    c: float = 0.0
    delta: float = 0.0
    a_p: float | None = None  # stretch in promoted-early matches (variant A)
    nu: float | None = None  # CMP dispersion (spread test)
    t_ref: float = 0.0

    def grids(self, lam: FloatArr, mu: FloatArr, flag: npt.NDArray[np.bool_]) -> FloatArr:
        s, t = np.log(lam) - np.log(mu), np.log(lam) + np.log(mu)
        a = np.where(flag, self.a_p, self.a) if self.a_p is not None else self.a
        s2 = a * s + self.b
        t2 = self.t_ref + self.d * (t - self.t_ref) + self.c
        lam2, mu2 = np.exp((t2 + s2) / 2), np.exp((t2 - s2) / 2)
        g = score_grids(lam2, mu2, RHO) if self.nu is None else cmp_grids(lam2, mu2, RHO, self.nu)
        return adjust_draws(g, self.delta)


def _loss(
    mp: Map,
    lam: FloatArr,
    mu: FloatArr,
    flag: npt.NDArray[np.bool_],
    h: npt.NDArray[np.intp],
    a: npt.NDArray[np.intp],
) -> float:
    return float(goal_losses(mp.grids(lam, mu, flag), h, a).mean())


def fit(
    lam: FloatArr,
    mu: FloatArr,
    flag: npt.NDArray[np.bool_],
    h: npt.NDArray[np.intp],
    a: npt.NDArray[np.intp],
    extra: str | None,
    start: Map | None = None,
) -> Map:
    """extra: None (the live five numbers), "a_p" or "nu"."""
    t_ref = float(np.mean(np.log(lam) + np.log(mu)))

    def make(th: FloatArr) -> Map:
        kw = dict(a=th[0], b=th[1], d=th[2], c=th[3], delta=th[4], t_ref=t_ref)
        if extra == "a_p":
            kw["a_p"] = th[5]
        if extra == "nu":
            kw["nu"] = th[5]
        return Map(**{k: float(v) for k, v in kw.items()})

    def objective(th: FloatArr) -> float:
        if not (0.3 < th[0] < 3 and 0.05 < th[2] < 3 and abs(th[4]) < 1):
            return 1e9
        if extra == "a_p" and not 0.3 < th[5] < 3:
            return 1e9
        if extra == "nu" and not 0.7 < th[5] < 1.6:
            return 1e9
        return _loss(make(th), lam, mu, flag, h, a)

    s = start or Map()
    x0 = [s.a, s.b, s.d, s.c, s.delta] + ([] if extra is None else [s.a if extra == "a_p" else 1.0])
    res = minimize(
        objective,
        np.array(x0, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000},
    )
    return make(res.x)


def walk_forward(
    preds: pd.DataFrame,
    flag: pd.Series,
    extra: str | None,
    first: int = 2019,
    starts: dict[int, Map] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per test season: map fitted on earlier seasons, applied to the season. Returns per-match
    rows (row, season, loss and markets) and one row per season with the map. `starts`: the
    five-number maps already fitted per season (the extra number starts from them)."""
    p = preds.dropna(subset=["exp_home", "exp_away"]).sort_values("row").reset_index(drop=True)
    lam, mu = p["exp_home"].to_numpy(float), p["exp_away"].to_numpy(float)
    h, a = p["goals_home"].to_numpy(int), p["goals_away"].to_numpy(int)
    fl = flag.reindex(p["row"]).fillna(False).to_numpy(bool)
    rows, maps = [], []
    for s in sorted(int(x) for x in p["season"].unique()):
        if s < first:
            continue
        tr, te = (p["season"] < s).to_numpy(), (p["season"] == s).to_numpy()
        base = (starts or {}).get(s) or fit(lam[tr], mu[tr], fl[tr], h[tr], a[tr], None)
        mp = base if extra is None else fit(lam[tr], mu[tr], fl[tr], h[tr], a[tr], extra, base)
        g = mp.grids(lam[te], mu[te], fl[te])
        mk = goal_markets(g)
        i = np.arange(g.shape[1])
        ph, pa = g.sum(axis=2), g.sum(axis=1)
        eh, ea = (ph * i).sum(1), (pa * i).sum(1)
        var_h, var_a = (ph * i**2).sum(1) - eh**2, (pa * i**2).sum(1) - ea**2
        rows.append(
            pd.DataFrame(
                {
                    "row": p.loc[te, "row"].to_numpy(),
                    "season": s,
                    "loss": goal_losses(g, h[te], a[te]),
                    "flag": fl[te],
                    "eh": eh,
                    "ea": ea,
                    "var_h": var_h,
                    "var_a": var_a,
                    "p00": g[:, 0, 0],
                    "goals_home": h[te],
                    "goals_away": a[te],
                    **mk,
                }
            )
        )
        maps.append({"season": s, **{k: v for k, v in mp.__dict__.items()}})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(maps)


def maps_from(table: pd.DataFrame) -> dict[int, Map]:
    keys = ("a", "b", "d", "c", "delta", "t_ref")
    return {int(r["season"]): Map(**{k: float(r[k]) for k in keys}) for _, r in table.iterrows()}


def run_part(
    part: str,
    matches: pd.DataFrame,
    champ: pd.DataFrame,
    b_preds: pd.DataFrame | None,
    out: Path,
    base_maps: dict[int, Map] | None = None,
) -> pd.DataFrame:
    """One of: champion, A, B, spread. Cached as <out>/<part>.parquet and <part>_maps.csv."""
    path = out / f"{part}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    flag = promoted_early(matches.reset_index(drop=True))
    if part == "B":
        if b_preds is None:
            raise ValueError("part B needs the refitted walk-forward predictions")
        rows, maps = walk_forward(b_preds, flag, None)
    else:
        extra = {"champion": None, "A": "a_p", "spread": "nu"}[part]
        rows, maps = walk_forward(champ, flag, extra, starts=base_maps)
    rows.to_parquet(path)
    maps.to_csv(out / f"{part}_maps.csv", index=False)
    return rows


def compare(
    champ: pd.DataFrame, chall: pd.DataFrame, matches: pd.DataFrame, level: float
) -> dict[str, float]:
    """Paired difference (challenger - champion) of the per-match loss on the same rows."""
    j = champ.merge(chall, on="row", suffixes=("_c", "_x"))
    m = matches.reset_index(drop=True).loc[j["row"]]
    week = m["date"] - pd.to_timedelta(m["date"].dt.dayofweek, unit="D")
    blocks = (m["league"].astype(str) + "|" + week.dt.strftime("%Y-%m-%d")).to_numpy()
    d = (j["loss_x"] - j["loss_c"]).to_numpy()
    main = j["season_c"].between(2020, 2025).to_numpy()
    conf = (j["season_c"] == 2026).to_numpy()
    from football_edge.models.calibration import block_bootstrap

    lo, hi = block_bootstrap(d[main], blocks[main], level=level)
    return {
        "n": int(main.sum()),
        "n_2026": int(conf.sum()),
        "champ": float(j["loss_c"][main].mean()),
        "chall": float(j["loss_x"][main].mean()),
        "diff": float(d[main].mean()),
        "lo": lo,
        "hi": hi,
        "diff_2026": float(d[conf].mean()),
        "pass": bool(hi < 0 and d[conf].mean() < 0),
    }


def secondary(rows: pd.DataFrame, sides: pd.DataFrame) -> dict[str, float]:
    """Promoted-early goal-difference residual, spread ratio and 0-0 rate, 2020/21 onwards."""
    r = rows[rows["season"] >= 2020]
    s = sides.loc[r["row"]]
    h, a = r["goals_home"].to_numpy(float), r["goals_away"].to_numpy(float)
    gd_home = (h - r["eh"].to_numpy()) - (a - r["ea"].to_numpy())
    res = np.r_[gd_home[s["home"].to_numpy()], -gd_home[s["away"].to_numpy()]]
    spread_h = float(((h - r["eh"]) ** 2 / r["var_h"]).mean())
    spread_a = float(((a - r["ea"]) ** 2 / r["var_a"]).mean())
    return {
        "promoted_residual": float(res.mean()),
        "spread_home": spread_h,
        "spread_away": spread_a,
        "p00": float(r["p00"].mean()),
        "obs00": float(((h == 0) & (a == 0)).mean()),
    }
