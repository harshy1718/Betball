"""ADR 0025 challengers: bivariate Poisson, Glicko and the ML feature builder."""

import numpy as np
import pandas as pd
import pytest

from football_edge.evaluation.ml_challenge import build, feature_cols, form, ratings
from football_edge.models.bivariate import bp_grid, bp_pmf
from football_edge.models.dixon_coles import score_grid


def _season(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list("ABCDEF")
    rows, day = [], pd.Timestamp("2018-08-11")
    for season in (2018, 2019):
        for rnd in range(10):
            order = rng.permutation(teams)
            for i in range(0, 6, 2):
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                rows.append({
                    "league": "E0", "season": season, "date": day + pd.Timedelta(days=7 * rnd),
                    "home": order[i], "away": order[i + 1], "FTHG": hg, "FTAG": ag,
                    "FTR": "H" if hg > ag else "D" if hg == ag else "A",
                    "HS": hg * 4 + 5, "AS": ag * 4 + 4, "HST": hg + 2, "AST": ag + 1,
                })  # fmt: skip
        day += pd.Timedelta(days=365)
    return pd.DataFrame(rows)


def test_bivariate_grid_sums_to_one_and_keeps_means() -> None:
    g = bp_grid(1.6, 1.1, 0.15)
    assert g.sum() == pytest.approx(1.0, abs=1e-9)
    i = np.arange(g.shape[0])
    assert (g.sum(1) * i).sum() == pytest.approx(1.6, abs=1e-3)
    assert (g.sum(0) * i).sum() == pytest.approx(1.1, abs=1e-3)
    one = bp_pmf(np.array([1]), np.array([2]), np.array([1.6]), np.array([1.1]), np.array([0.15]))
    assert g[1, 2] == pytest.approx(float(one[0]), rel=1e-9)


def test_bivariate_without_covariance_is_independent_poisson() -> None:
    assert np.allclose(bp_grid(1.4, 1.2, 0.0), score_grid(1.4, 1.2, 0.0), atol=1e-12)


def test_form_uses_only_earlier_matches() -> None:
    m = _season()
    f = form(m)
    # the opening round: nobody has played yet
    opening = m.index[m["date"] == m["date"].min()]
    assert f.loc[opening, ["home_gf5", "away_gf5", "home_rest"]].isna().all().all()
    # changing a match's own score must not change its features
    m2 = m.copy()
    m2.loc[30, ["FTHG", "FTAG"]] = [9, 9]
    assert form(m2).loc[30].equals(f.loc[30])


def test_ratings_are_from_before_the_match_day() -> None:
    m = _season()
    r = ratings(m)
    m2 = m.copy()
    last_day = m["date"].max()
    m2.loc[m2["date"] == last_day, ["FTHG", "FTAG", "FTR"]] = [0, 7, "A"]
    assert ratings(m2).loc[m["date"] == last_day].equals(r.loc[m["date"] == last_day])


def test_build_has_targets_and_no_target_in_features() -> None:
    m = _season()
    champ = pd.DataFrame({"row": m.index, "season": m["season"], "league": m["league"],
                          "exp_home": 1.4, "exp_away": 1.1, "p_h": 0.45, "p_d": 0.27,
                          "p_a": 0.28, "p_over25": 0.5, "p_btts": 0.5})  # fmt: skip
    df = build(m, champ)
    cols = feature_cols(df)
    assert not {"y_res", "y_over", "y_btts", "goals_home", "goals_away"} & set(cols)
    assert df["season_point"].between(0, 1).all()
    assert (df["y_res"] == m["FTR"].map({"H": 0, "D": 1, "A": 2})).all()


def test_diagnostic_sides_flag_promoted_and_rest() -> None:
    from football_edge.evaluation.diagnostics import sides

    m = pd.DataFrame({
        "league": "E0", "season": [2018, 2018, 2019, 2019],
        "date": pd.to_datetime(["2018-08-11", "2018-08-18", "2019-08-10", "2019-08-14"]),
        "home": ["A", "B", "A", "C"], "away": ["B", "A", "C", "A"],
        "FTHG": [1, 2, 0, 1], "FTAG": [0, 2, 0, 3],
    })  # fmt: skip
    df = m.assign(lam=1.0, mu=1.0, block="x").loc[[2, 3]]
    s = sides(df, m).set_index(["row", "venue"])
    assert bool(s.loc[(2, "away"), "promoted"])  # C was not in the league in 2018
    assert not bool(s.loc[(2, "home"), "promoted"])
    assert s.loc[(3, "away"), "rest"] == 4 and s.loc[(3, "away"), "n"] == 2
    assert s.loc[(3, "away"), "gd_res"] == pytest.approx(2.0)


def test_cmp_keeps_the_mean_and_nu_one_is_poisson() -> None:
    from scipy.stats import poisson

    from football_edge.evaluation.promoted_spread import cmp_grids, cmp_pmf

    lam = np.array([0.4, 1.3, 2.9])
    k = np.arange(31)
    assert np.allclose(cmp_pmf(lam, 1.0), poisson.pmf(k[None], lam[:, None]), atol=1e-6)
    p = cmp_pmf(lam, 1.3)
    mean = (p * k).sum(1)
    assert np.allclose(mean, lam, atol=1e-4)
    assert (((p * k**2).sum(1) - mean**2) < lam).all()  # less spread than Poisson
    assert np.allclose(cmp_grids(lam, lam[::-1], -0.035, 1.0), score_grid_many(lam), atol=1e-6)


def score_grid_many(lam: np.ndarray) -> np.ndarray:
    from football_edge.models.dixon_coles import score_grids

    return score_grids(lam, lam[::-1], -0.035)


def test_promoted_sides_flag_first_19_matches_only() -> None:
    from football_edge.evaluation.promoted_spread import EARLY, promoted_sides

    m = _season()
    m.loc[m["season"] == 2019, ["home", "away"]] = m.loc[
        m["season"] == 2019, ["home", "away"]
    ].replace("F", "G")
    s = promoted_sides(m)
    g_home = (m["home"] == "G").to_numpy()
    assert s["home"].to_numpy()[g_home].all() and len(m[m["home"] == "G"]) <= EARLY
    assert not s.loc[m["season"] == 2018].to_numpy().any()  # first season in the data


def test_promoted_pull_one_is_the_locked_model() -> None:
    from football_edge.models.dixon_coles import DixonColes, DixonColesParams

    m = _season()
    a, b = (
        DixonColes(DixonColesParams(ridge=1.0)),
        DixonColes(DixonColesParams(ridge=1.0, promoted_pull=1.0)),
    )
    a.fit(m)
    b.fit(m)
    assert np.allclose(a.attack, b.attack) and np.allclose(a.defence, b.defence)
    c = DixonColes(DixonColesParams(ridge=1.0, promoted_pull=0.5))
    c.fit(m)
    assert np.isfinite(c.attack).all()
