"""Machine-learning challengers (ADR 0025): LightGBM, XGBoost, CatBoost (direct and goals) and
an MLP, all on features known before kickoff, refitted before every test season.

Every feature of a match uses only matches played before it:
  * the champion's walk-forward chances and expected goals (fitted on matches before that week);
  * Elo and Glicko ratings before the match day;
  * each team's last 5 and 10 matches (goals, shots, shots on target, points, for and against),
    days since its previous match;
  * league and the point in the season.
Champion predictions exist from 2018/19, so models are trained on 2018/19 up to the season
before the one predicted.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from football_edge.models.calibration import goal_markets
from football_edge.models.dixon_coles import score_grids
from football_edge.models.elo import Elo
from football_edge.models.glicko import Glicko

RHO = -0.035  # current fit; for the goals-mode grids
MARKET_COLS = ["p_h", "p_d", "p_a", "p_over25", "p_btts"]


# -- features ---------------------------------------------------------------------------------
def ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Elo gap and Glicko gap for every match, from ratings before its match day."""
    m = matches.sort_values("date", kind="stable")
    elo, gl = Elo(), Glicko()
    r: defaultdict[str, dict[str, float]] = defaultdict(dict)
    last: defaultdict[str, dict[str, int]] = defaultdict(dict)
    k, hfa = elo.params.k, elo.params.home_advantage
    out = {}
    for _, day in m.groupby("date", sort=True):
        rows = [
            (int(i), str(lg), int(s), str(h), str(a), pd.Timestamp(d), str(r_))
            for i, lg, s, h, a, d, r_ in zip(
                day.index,
                day["league"],
                day["season"],
                day["home"],
                day["away"],
                day["date"],
                day["FTR"],
                strict=True,
            )
        ]
        for i, lg, s, h, a, d, _r in rows:  # 1) everything before the day's matches
            rr, ls = r[lg], last[lg]
            elo._ensure(rr, ls, h, s)
            elo._ensure(rr, ls, a, s)
            out[i] = (rr[h] + hfa - rr[a], gl.gap(lg, h, a, d))
        for _i, lg, s, h, a, d, res in rows:  # 2) then the day's results
            rr, ls = r[lg], last[lg]
            gap = rr[h] + hfa - rr[a]
            change = k * ({"H": 1.0, "D": 0.5, "A": 0.0}[res] - 1 / (1 + 10 ** (-gap / 400)))
            rr[h] += change
            rr[a] -= change
            ls[h] = ls[a] = s
            gl.update(lg, h, a, d, res)
    return pd.DataFrame.from_dict(out, orient="index", columns=["elo_gap", "glicko_gap"])


def form(matches: pd.DataFrame) -> pd.DataFrame:
    """Per match: each side's averages over its previous 5 and 10 matches and rest days."""
    m = matches.reset_index().rename(columns={"index": "row"})
    sides = []
    cols_by_side = (
        ("home", "FTHG", "FTAG", "HS", "AS", "HST", "AST"),
        ("away", "FTAG", "FTHG", "AS", "HS", "AST", "HST"),
    )
    for side, gf, ga, sf, sa, stf, sta in cols_by_side:
        t = pd.DataFrame(
            {
                "row": m["row"],
                "side": side,
                "league": m["league"],
                "team": m[side],
                "date": m["date"],
                "gf": m[gf],
                "ga": m[ga],
                "sf": m[sf],
                "sa": m[sa],
                "stf": m[stf],
                "sta": m[sta],
            }
        )
        t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3, 1], 0)
        sides.append(t)
    long = pd.concat(sides).sort_values(["league", "team", "date"], kind="stable")
    grp = long.groupby(["league", "team"], sort=False)
    stats = ["gf", "ga", "sf", "sa", "stf", "sta", "pts"]
    for n in (5, 10):
        prev = grp[stats].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
        for c in stats:
            long[f"{c}{n}"] = prev[c]
    long["rest"] = (long["date"] - grp["date"].shift(1)).dt.days.clip(upper=30)
    feats = [f"{c}{n}" for n in (5, 10) for c in stats] + ["rest"]
    wide = long.pivot(index="row", columns="side", values=feats)
    wide.columns = [f"{pair[1]}_{pair[0]}" for pair in wide.columns]
    return wide


def build(matches: pd.DataFrame, champ: pd.DataFrame) -> pd.DataFrame:
    """One row per match with champion predictions: features and targets."""
    m = matches.reset_index(drop=True)
    base = champ.set_index("row")[["season", "league", "exp_home", "exp_away", *MARKET_COLS]]
    df = base.join(ratings(m)).join(form(m))
    order = m.groupby(["league", "season"])["date"].rank(method="first")
    # a full season's length from the number of teams (known in advance; the current season's
    # played-so-far count would leak how many matches come later)
    teams = m.groupby(["league", "season"])["home"].transform("nunique")
    size = teams * (teams - 1)
    df["season_point"] = (order / size).reindex(df.index)
    codes = {lg: i for i, lg in enumerate(sorted(m["league"].unique()))}
    df["league_code"] = df["league"].map(codes)
    hg, ag = m.loc[df.index, "FTHG"].to_numpy(int), m.loc[df.index, "FTAG"].to_numpy(int)
    df["y_res"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    df["y_over"] = (hg + ag >= 3).astype(int)
    df["y_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    df["goals_home"], df["goals_away"] = hg, ag
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"season", "league", "y_res", "y_over", "y_btts", "goals_home", "goals_away"}
    return [c for c in df.columns if c not in skip]


# -- models -----------------------------------------------------------------------------------
def _models(kind: str, task: str) -> Any:
    """Fixed settings (ADR 0025): 400 trees, rate 0.03, depth 4, 50 matches per leaf, 80% rows
    and columns. task: multi | binary | poisson."""
    if kind == "lightgbm":
        import lightgbm as lgb

        common: dict[str, Any] = dict(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=0,
        )
        if task == "poisson":
            return lgb.LGBMRegressor(objective="poisson", **common)
        return lgb.LGBMClassifier(**common)
    if kind == "xgboost":
        import xgboost as xgb

        common = dict(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
        )
        if task == "poisson":
            return xgb.XGBRegressor(objective="count:poisson", **common)
        obj = "multi:softprob" if task == "multi" else "binary:logistic"
        return xgb.XGBClassifier(objective=obj, **common)
    if kind == "catboost":
        import catboost as cb

        common = dict(
            iterations=400,
            learning_rate=0.03,
            depth=4,
            rsm=0.8,
            bootstrap_type="Bernoulli",
            allow_writing_files=False,
            subsample=0.8,
            verbose=0,
            random_seed=0,
        )
        if task == "poisson":
            return cb.CatBoostRegressor(loss_function="Poisson", **common)
        return cb.CatBoostClassifier(
            loss_function="MultiClass" if task == "multi" else "Logloss", **common
        )
    if kind == "mlp":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=500, random_state=0
            ),
        )
    raise ValueError(kind)


def predict_season(df: pd.DataFrame, season: int, kind: str, mode: str) -> pd.DataFrame:
    """Train on seasons before `season` (from 2018/19), predict `season`."""
    cols = feature_cols(df)
    tr, te = df[df["season"] < season], df[df["season"] == season]
    x_tr, x_te = tr[cols].astype(float), te[cols].astype(float)
    out = pd.DataFrame(index=te.index)
    if mode == "direct":
        m = _models(kind, "multi").fit(x_tr, tr["y_res"])
        p3 = m.predict_proba(x_te)
        out["p_h"], out["p_d"], out["p_a"] = p3[:, 0], p3[:, 1], p3[:, 2]
        out["p_over25"] = _models(kind, "binary").fit(x_tr, tr["y_over"]).predict_proba(x_te)[:, 1]
        out["p_btts"] = _models(kind, "binary").fit(x_tr, tr["y_btts"]).predict_proba(x_te)[:, 1]
    else:
        lam = _models(kind, "poisson").fit(x_tr, tr["goals_home"]).predict(x_te)
        mu = _models(kind, "poisson").fit(x_tr, tr["goals_away"]).predict(x_te)
        mk = goal_markets(score_grids(np.clip(lam, 0.05, 6), np.clip(mu, 0.05, 6), RHO))
        for c in MARKET_COLS:
            out[c] = mk[c]
    return out


CHALLENGERS = [
    ("lightgbm", "direct"),
    ("lightgbm", "goals"),
    ("xgboost", "direct"),
    ("xgboost", "goals"),
    ("catboost", "direct"),
    ("catboost", "goals"),
    ("mlp", "direct"),
]


def run_all(df: pd.DataFrame, seasons: list[int]) -> dict[str, pd.DataFrame]:
    out = {}
    for kind, mode in CHALLENGERS:
        parts = [predict_season(df, s, kind, mode) for s in seasons]
        out[f"{kind} ({mode})"] = pd.concat(parts)
    return out
