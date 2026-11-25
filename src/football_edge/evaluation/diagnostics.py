"""Where does the live model miss? Diagnostics on walk-forward predictions, no model changes.

Uses the champion's weekly walk-forward expected goals with the live calibration applied
season by season (each season's map fitted on earlier seasons only), 2019/20 onwards, so every
gap found here is one the live app still has. Residual = actual - expected; goal-difference
residual = (scored - expected scored) - (conceded - expected conceded).

1. Promoted teams by match number in the season.
2. Season starts: is the model too sure early on (slope of actual on expected goal difference)?
3. Rest days (league matches only: cup and European games are not in the data).
4. Team-specific home advantage: does a team's home-minus-away residual repeat next season?
5. Goal-count spread and low scores: observed vs predicted frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_edge.models.calibration import (
    adjust_draws,
    block_bootstrap,
    calibrate_goals_walk_forward,
)
from football_edge.models.dixon_coles import score_grids

RHO = -0.035


def prepared(matches: pd.DataFrame, champ: pd.DataFrame) -> pd.DataFrame:
    """One row per match: calibrated walk-forward expectations, results and context."""
    cal, _ = calibrate_goals_walk_forward(champ, RHO, first=2019)
    m = matches.reset_index(drop=True)
    p = cal.set_index("row")
    df = m.loc[p.index, ["league", "season", "date", "home", "away", "FTHG", "FTAG"]].copy()
    df["lam"], df["mu"] = p["exp_home"], p["exp_away"]
    df["lam_raw"], df["mu_raw"] = p["exp_home_raw"], p["exp_away_raw"]
    df["delta"] = p["cal_delta"]
    week = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    df["block"] = df["league"].astype(str) + "|" + week.dt.strftime("%Y-%m-%d")
    return df


def sides(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team) with promotion flag, match number and rest days."""
    out = []
    for side, opp, gf, ga, xf, xa in (
        ("home", "away", "FTHG", "FTAG", "lam", "mu"),
        ("away", "home", "FTAG", "FTHG", "mu", "lam"),
    ):
        out.append(
            pd.DataFrame(
                {
                    "row": df.index,
                    "venue": side,
                    "league": df["league"],
                    "season": df["season"],
                    "date": df["date"],
                    "team": df[side],
                    "opp": df[opp],
                    "block": df["block"],
                    "gd_res": (df[gf] - df[xf]) - (df[ga] - df[xa]),
                    "for_res": df[gf] - df[xf],
                    "against_res": df[ga] - df[xa],
                }
            )
        )
    s = pd.concat(out, ignore_index=True)
    # promotion and match number / rest from every league match in the data (2015 onwards)
    m = matches
    teams = pd.concat(
        [
            m[["league", "season", "home"]].rename(columns={"home": "team"}),
            m[["league", "season", "away"]].rename(columns={"away": "team"}),
        ]
    )
    present = set(map(tuple, teams.drop_duplicates().to_numpy()))
    first_season = m.groupby("league")["season"].min().to_dict()
    s["promoted"] = [
        None if se <= first_season[lg] else (lg, se - 1, t) not in present
        for lg, se, t in zip(s["league"], s["season"], s["team"], strict=True)
    ]
    games = pd.concat(
        [
            m[["league", "season", "date", "home"]].rename(columns={"home": "team"}),
            m[["league", "season", "date", "away"]].rename(columns={"away": "team"}),
        ]
    )
    games = games.sort_values("date", kind="stable")
    games["n"] = games.groupby(["league", "season", "team"]).cumcount() + 1
    games["rest"] = games.groupby(["league", "team"])["date"].diff().dt.days
    s = s.merge(games, on=["league", "season", "date", "team"], how="left")
    return s


def _ci(x: pd.Series, blocks: pd.Series) -> tuple[float, float, float, int]:
    v = x.to_numpy(float)
    lo, hi = block_bootstrap(v, blocks.to_numpy(), reps=1000)
    return float(v.mean()), lo, hi, len(v)


def _row(label: str, x: pd.Series, blocks: pd.Series) -> str:
    mean, lo, hi, n = _ci(x, blocks)
    flag = " **gap**" if lo > 0 or hi < 0 else ""
    return f"| {label} | {n:,} | {mean:+.3f} | {lo:+.3f} to {hi:+.3f}{flag} |"


def promoted(s: pd.DataFrame) -> list[str]:
    lines = [
        "## 1. Promoted teams",
        "",
        "Goal-difference residual per match for the team (negative = the model rated it too "
        "high). 95% intervals, blocks = league x week.",
        "",
        "| Group | Team-matches | Residual | 95% interval |",
        "|---|---|---|---|",
    ]
    k = s[s["promoted"].notna()]
    bins = [(1, 5), (6, 10), (11, 19), (20, 99)]
    for flag, name in ((True, "promoted"), (False, "others")):
        for a, b in bins:
            g = k[(k["promoted"] == flag) & k["n"].between(a, b)]
            end = b if b < 99 else "end"
            lines.append(_row(f"{name}, matches {a}-{end}", g["gd_res"], g["block"]))
    g = k[(k["promoted"] == True) & (k["n"] <= 10)]  # noqa: E712
    lines += [
        "",
        f"Promoted, first 10: scored {g['for_res'].mean():+.3f}, conceded "
        f"{g['against_res'].mean():+.3f} vs expected per match.",
        "",
    ]
    return lines


def _slope(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, reps: int = 1000
) -> tuple[float, float, float]:
    def fit(i: np.ndarray) -> float:
        xi, yi = x[i], y[i]
        return float(np.cov(xi, yi)[0, 1] / np.var(xi, ddof=1))

    codes, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(codes))]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, len(codes), len(codes))
        boots.append(fit(np.concatenate([members[j] for j in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return fit(np.arange(len(x))), float(lo), float(hi)


def season_start(df: pd.DataFrame, s: pd.DataFrame) -> list[str]:
    home_n = s[s["venue"] == "home"].set_index("row")["n"]
    away_n = s[s["venue"] == "away"].set_index("row")["n"]
    n = pd.concat([home_n.reindex(df.index), away_n.reindex(df.index)], axis=1).max(axis=1)
    xg = (df["lam"] - df["mu"]).to_numpy(float)
    yg = (df["FTHG"] - df["FTAG"]).to_numpy(float)
    lines = [
        "## 2. Season starts",
        "",
        "Slope of actual on expected goal difference (1.00 = the model's spread between teams "
        "is right; below 1 = too sure, strengths should be pulled toward average).",
        "",
        "| Part of season | Matches | Slope | 95% interval |",
        "|---|---|---|---|",
    ]
    for a, b in ((1, 6), (7, 15), (16, 99)):
        k = n.between(a, b).to_numpy()
        sl, lo, hi = _slope(xg[k], yg[k], df["block"].to_numpy()[k])
        flag = " **gap**" if lo > 1 or hi < 1 else ""
        lines.append(
            f"| team match {a}-{b if b < 99 else 'end'} | {k.sum():,} | {sl:.3f} | "
            f"{lo:.3f} to {hi:.3f}{flag} |"
        )
    return [*lines, ""]


def rest(s: pd.DataFrame) -> list[str]:
    lines = [
        "## 3. Rest days",
        "",
        "Days since the team's previous league match (cup and European games are not in the "
        "data, so a 'long' rest can hide a midweek game). Goal-difference residual.",
        "",
        "| Rest | Team-matches | Residual | 95% interval |",
        "|---|---|---|---|",
    ]
    k = s[s["rest"].notna()]
    for a, b, name in (
        (0, 3, "3 days or less"),
        (4, 5, "4-5 days"),
        (6, 8, "6-8 days"),
        (9, 13, "9-13 days"),
        (14, 400, "14+ days (international break)"),
    ):
        g = k[k["rest"].between(a, b)]
        lines.append(_row(name, g["gd_res"], g["block"]))
    return [*lines, ""]


def venue(s: pd.DataFrame) -> list[str]:
    g = s.groupby(["league", "team", "season", "venue"])["gd_res"].agg(["mean", "size"]).unstack()
    g = g[(g[("size", "home")] >= 15) & (g[("size", "away")] >= 15)]
    gap = (g[("mean", "home")] - g[("mean", "away")]).rename("gap").reset_index()
    nxt = gap.assign(season=gap["season"] - 1).rename(columns={"gap": "gap_next"})
    pair = gap.merge(nxt, on=["league", "team", "season"])
    r = float(np.corrcoef(pair["gap"], pair["gap_next"])[0, 1])
    rng = np.random.default_rng(0)
    rs = [
        np.corrcoef(pair["gap"], rng.permutation(pair["gap_next"].to_numpy()))[0, 1]
        for _ in range(2000)
    ]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return [
        "## 4. Team-specific home advantage",
        "",
        f"Each team-season's home-minus-away residual against the same team's next season: "
        f"correlation **{r:+.3f}** over {len(pair):,} pairs (pure chance: {lo:+.3f} to "
        f"{hi:+.3f}). Spread of the gap: {gap['gap'].std():.3f} goals per match.",
        "",
        "A correlation inside the chance band means the gap does not repeat: it is noise, "
        "and a per-team home advantage would only fit noise.",
        "",
    ]


def spread(df: pd.DataFrame) -> list[str]:
    lam, mu = df["lam"].to_numpy(float), df["mu"].to_numpy(float)
    g = adjust_draws(score_grids(lam, mu, RHO), df["delta"].to_numpy(float))
    h, a = df["FTHG"].to_numpy(int), df["FTAG"].to_numpy(int)
    n = len(df)
    lines = [
        "## 5. Goal-count spread and low scores",
        "",
        "| Event | Predicted | Observed | Observed - predicted | Standard errors |",
        "|---|---|---|---|---|",
    ]

    def add(name: str, p: np.ndarray, hit: np.ndarray) -> None:
        pe, ob = float(p.mean()), float(hit.mean())
        se = float(np.sqrt((p * (1 - p)).sum()) / n)
        z = (ob - pe) / se
        flag = " **gap**" if abs(z) > 2.5 else ""
        lines.append(f"| {name} | {pe:.4f} | {ob:.4f} | {ob - pe:+.4f} | {z:+.1f}{flag} |")

    add("0-0", g[:, 0, 0], (h == 0) & (a == 0))
    add("1-0 or 0-1", g[:, 1, 0] + g[:, 0, 1], (h + a == 1))
    add("1-1", g[:, 1, 1], (h == 1) & (a == 1))
    add("any draw", np.einsum("kii->k", g), h == a)
    tot = np.add.outer(np.arange(g.shape[1]), np.arange(g.shape[2]))
    for t in range(6):
        add(f"{t} goals in the match", (g * (tot == t)).sum(axis=(1, 2)), h + a == t)
    add("6+ goals in the match", (g * (tot >= 6)).sum(axis=(1, 2)), h + a >= 6)
    ph, pa = g.sum(axis=2), g.sum(axis=1)
    add("home team 0 goals", ph[:, 0], h == 0)
    add("away team 0 goals", pa[:, 0], a == 0)
    add("home team 4+ goals", ph[:, 4:].sum(1), h >= 4)
    add("away team 4+ goals", pa[:, 4:].sum(1), a >= 4)
    i = np.arange(g.shape[1])
    eh, ea = (ph * i).sum(1), (pa * i).sum(1)
    vh, va = (ph * i**2).sum(1) - eh**2, (pa * i**2).sum(1) - ea**2
    disp_h, disp_a = float(((h - eh) ** 2 / vh).mean()), float(((a - ea) ** 2 / va).mean())
    lines += [
        "",
        f"Spread check (squared error / predicted variance; 1.00 = the predicted spread "
        f"is right, above 1 = more spread than the model allows): home {disp_h:.3f}, away "
        f"{disp_a:.3f} (about +-{2 * np.sqrt(2 / n):.3f} by chance).",
        "",
    ]
    return lines


def report(matches: pd.DataFrame, champ: pd.DataFrame) -> str:
    df = prepared(matches, champ)
    s = sides(df, matches)
    head = [
        "# Where the live model misses (diagnostics)",
        "",
        f"Walk-forward predictions with the live calibration, {int(df['season'].min())}/"
        f"{str(int(df['season'].min()) + 1)[-2:]} to {int(df['season'].max())}/"
        f"{str(int(df['season'].max()) + 1)[-2:]}, {len(df):,} matches, 5 leagues. "
        "'gap' = the interval excludes no effect.",
        "",
    ]
    body = promoted(s) + season_start(df, s) + rest(s) + venue(s) + spread(df)
    return "\n".join(head + body)
