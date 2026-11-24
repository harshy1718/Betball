"""The pre-registered calibration test (docs/adr/0020-calibration.md) and the live maps.

Goals and cards, each:
  1. walk-forward over 2019/20-2025/26: each season calibrated by a map fitted only on the
     seasons before it; condition 1 is judged on 2020/21-2025/26;
  2. confirmation on 2026/27 so far, which nothing above has seen: map fitted on 2018-2025;
  3. if both pass, the live map is fitted on every out-of-sample season (2018-2026) and saved
     to reports/calibration/calibration.json with adopted = true.
Calibrated walk-forward predictions are saved next to the raw ones, for the confidence labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from football_edge.models import calibration as cal


@dataclass
class Check:
    name: str
    raw: float
    new: float
    lo: float
    hi: float
    n: int

    @property
    def better(self) -> bool:
        return self.new < self.raw


def _weeks(preds: pd.DataFrame) -> np.ndarray:
    d = pd.to_datetime(preds["date"])
    week = (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    out: np.ndarray = (preds["league"].astype(str) + "|" + week).to_numpy()
    return out


def goals_study(
    preds: pd.DataFrame, p26: pd.DataFrame | None, rho: float
) -> tuple[dict[str, object], pd.DataFrame]:
    wf, report = cal.calibrate_goals_walk_forward(preds, rho, first=2019)
    r = report[report["season"] >= 2020]
    w = r["n"].to_numpy(float)
    avg = {
        k: float(np.average(r[k], weights=w)) for k in r.columns if k.startswith(("raw_", "cal_"))
    }
    worst = max(avg[f"cal_{m}"] - avg[f"raw_{m}"] for m in ("1x2", "ou25", "btts"))
    cond1 = (
        sum(avg[f"cal_{m}"] for m in ("1x2", "ou25", "btts"))
        < sum(avg[f"raw_{m}"] for m in ("1x2", "ou25", "btts"))
        and worst <= 0.001
    )
    sub = wf[wf["season"] >= 2020]
    h, a = sub["goals_home"].to_numpy(int), sub["goals_away"].to_numpy(int)
    from football_edge.models.dixon_coles import score_grids

    def col(name: str) -> np.ndarray:
        out: np.ndarray = sub[name].to_numpy(float)
        return out

    raw_l = cal.goal_losses(score_grids(col("exp_home_raw"), col("exp_away_raw"), rho), h, a)
    new_l = cal.goal_losses(
        cal.adjust_draws(score_grids(col("exp_home"), col("exp_away"), rho), col("cal_delta")),
        h,
        a,
    )
    lo, hi = cal.block_bootstrap(new_l - raw_l, _weeks(sub))
    wf_check = Check(
        "2020/21-2025/26 (walk-forward)", float(raw_l.mean()), float(new_l.mean()), lo, hi, len(sub)
    )

    out: dict[str, object] = {
        "report": report,
        "avg": avg,
        "cond1": cond1,
        "worst": worst,
        "wf": wf_check,
    }
    conf = None
    if p26 is not None and len(p26):
        hist = preds.dropna(subset=["exp_home", "exp_away"])
        m = cal.fit_goals(
            hist["exp_home"].to_numpy(float),
            hist["exp_away"].to_numpy(float),
            hist["goals_home"].to_numpy(int),
            hist["goals_away"].to_numpy(int),
            rho,
            cal._span(hist["season"]),
        )
        q = p26.dropna(subset=["exp_home", "exp_away"])
        h, a = q["goals_home"].to_numpy(int), q["goals_away"].to_numpy(int)
        lam, mu = q["exp_home"].to_numpy(float), q["exp_away"].to_numpy(float)
        g_raw = score_grids(lam, mu, rho)
        lam2, mu2 = m.expected(lam, mu)
        g_new = cal.adjust_draws(score_grids(lam2, mu2, rho), m.delta)
        raw_l, new_l = cal.goal_losses(g_raw, h, a), cal.goal_losses(g_new, h, a)
        lo, hi = cal.block_bootstrap(new_l - raw_l, _weeks(q))
        conf = Check(
            "2026/27 so far (confirmation)",
            float(raw_l.mean()),
            float(new_l.mean()),
            lo,
            hi,
            len(q),
        )
        out["per_market_26"] = (cal.goal_scores(g_raw, h, a), cal.goal_scores(g_new, h, a))
    out["conf"] = conf
    out["adopt"] = bool(cond1 and conf is not None and conf.better)
    return out, wf


def cards_study(
    preds: pd.DataFrame, p26: pd.DataFrame | None, dates: pd.Series
) -> tuple[dict[str, object], pd.DataFrame]:
    """`dates`: match dates by row of the matches table (count predictions carry `row`)."""
    wf, report = cal.calibrate_cards_walk_forward(preds, first=2019)
    r = report[report["season"] >= 2020]
    w = r["n"].to_numpy(float)
    raw, new = (
        float(np.average(r["raw_sum"], weights=w)),
        float(np.average(r["cal_sum"], weights=w)),
    )
    out: dict[str, object] = {"report": report, "raw": raw, "new": new, "cond1": new < raw}
    conf = None
    if p26 is not None and len(p26):
        c = preds[(preds["stat"] == "cards") & preds["total"].notna()]
        m = cal.fit_cards(
            c["exp_team"].to_numpy(float),
            c["exp_base"].to_numpy(float),
            c["k_team"].to_numpy(float),
            c["total"].to_numpy(float),
            cal._span(c["season"]),
        )
        q = p26[(p26["stat"] == "cards") & p26["total"].notna()]
        mm, base = q["exp_team"].to_numpy(float), q["exp_base"].to_numpy(float)
        k, tot = q["k_team"].to_numpy(float), q["total"].to_numpy(float)
        raw_l = cal.card_losses(mm, k, tot)
        new_l = cal.card_losses(m.mean(mm, base), m.k(k), tot)
        conf = Check(
            "2026/27 so far (confirmation)",
            float(raw_l.mean()),
            float(new_l.mean()),
            *cal.block_bootstrap(
                new_l - raw_l, _weeks(q.assign(date=dates.iloc[q["row"]].to_numpy()))
            ),
            len(q),
        )
    out["conf"] = conf
    out["adopt"] = bool(out["cond1"] and conf is not None and conf.better)
    return out, wf


def live_maps(
    goals: pd.DataFrame, counts: pd.DataFrame, rho: float
) -> tuple[cal.GoalCalibration, cal.CardCalibration]:
    """Maps for live use, fitted on every out-of-sample season available."""
    g = goals.dropna(subset=["exp_home", "exp_away"])
    gm = cal.fit_goals(
        g["exp_home"].to_numpy(float),
        g["exp_away"].to_numpy(float),
        g["goals_home"].to_numpy(int),
        g["goals_away"].to_numpy(int),
        rho,
        cal._span(g["season"]),
    )
    c = counts[(counts["stat"] == "cards") & counts["total"].notna()]
    cm = cal.fit_cards(
        c["exp_team"].to_numpy(float),
        c["exp_base"].to_numpy(float),
        c["k_team"].to_numpy(float),
        c["total"].to_numpy(float),
        cal._span(c["season"]),
    )
    return gm, cm


def _fmt_check(c: Check | None) -> str:
    if c is None:
        return "no 2026/27 predictions: confirmation not run"
    sign = "better" if c.better else "WORSE"
    return (
        f"{c.name}: raw {c.raw:.5f}, calibrated {c.new:.5f} ({sign}; difference "
        f"{c.new - c.raw:+.5f}, 95% interval {c.lo:+.5f} to {c.hi:+.5f}; {c.n:,} matches)"
    )


def verdict(g: dict[str, object], c: dict[str, object]) -> str:
    rep = g["report"]
    assert isinstance(rep, pd.DataFrame)
    avg = g["avg"]
    assert isinstance(avg, dict)
    lines = [
        "# Calibration test (ADR 0020)",
        "",
        "## Goals",
        "",
        "Per season, map fitted only on earlier seasons (log loss, lower is better):",
        "",
        "| Season | a | b | d | c | delta | result raw/cal | O/U raw/cal | BTTS raw/cal "
        "| score raw/cal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep.itertuples():
        lines.append(
            f"| {r.season} | {r.a:.3f} | {r.b:+.3f} | {r.d:.3f} | {r.c:+.3f} | {r.delta:+.3f} | "
            f"{r.raw_1x2:.4f} / {r.cal_1x2:.4f} | {r.raw_ou25:.4f} / {r.cal_ou25:.4f} | "
            f"{r.raw_btts:.4f} / {r.cal_btts:.4f} | {r.raw_score:.4f} / {r.cal_score:.4f} |"
        )
    lines += [
        "",
        "2020/21-2025/26 average: "
        + ", ".join(
            f"{m} {avg[f'raw_{m}']:.5f} -> {avg[f'cal_{m}']:.5f}"
            for m in ("1x2", "ou25", "btts", "score")
        )
        + ".",
        "",
        "* Condition 1 (sum lower, no market worse by more than 0.001): "
        + ("PASS" if g["cond1"] else "FAIL"),
        "* " + _fmt_check(g["wf"]),  # type: ignore[arg-type]
        "* Condition 2: " + _fmt_check(g["conf"]),  # type: ignore[arg-type]
    ]
    if "per_market_26" in g:
        pm = g["per_market_26"]
        assert isinstance(pm, tuple)
        raw: dict[str, float] = pm[0]
        new: dict[str, float] = pm[1]
        lines.append(
            "  2026/27 per market: "
            + ", ".join(
                f"{m} {raw[m]:.4f} -> {new[m]:.4f}" for m in ("1x2", "ou25", "btts", "score")
            )
            + "."
        )
    lines += [
        "",
        f"**Goals calibration: {'ADOPTED' if g['adopt'] else 'NOT ADOPTED'}.**",
        "",
        "## Cards",
        "",
        "| Season | d | f | four lines raw | calibrated |",
        "|---|---|---|---|---|",
    ]
    crep = c["report"]
    assert isinstance(crep, pd.DataFrame)
    for r in crep.itertuples():
        lines.append(f"| {r.season} | {r.d:.3f} | {r.f:.3f} | {r.raw_sum:.4f} | {r.cal_sum:.4f} |")
    lines += [
        "",
        f"2020/21-2025/26 average of the four lines summed: {c['raw']:.5f} -> {c['new']:.5f}.",
        "* Condition 1: " + ("PASS" if c["cond1"] else "FAIL"),
        "* Condition 2: " + _fmt_check(c["conf"]),  # type: ignore[arg-type]
        "",
        f"**Cards calibration: {'ADOPTED' if c['adopt'] else 'NOT ADOPTED'}.**",
        "",
    ]
    return "\n".join(lines)


def run(reports: Path, rho: float, matches: pd.DataFrame) -> str:
    ev = reports / "evaluation"
    goals = pd.read_parquet(ev / "dixon_coles" / "predictions.parquet")
    g26p = ev / "dixon_coles" / "predictions_2026.parquet"
    g26 = pd.read_parquet(g26p) if g26p.exists() else None
    counts = pd.read_parquet(ev / "counts" / "predictions.parquet")
    c26p = ev / "counts" / "predictions_2026.parquet"
    c26 = pd.read_parquet(c26p) if c26p.exists() else None
    if "k_team" not in counts.columns:
        raise SystemExit("re-run `football-edge evaluate-counts` (needs the k_team column)")

    g, gwf = goals_study(goals, g26, rho)
    c, cwf = cards_study(counts, c26, matches["date"].reset_index(drop=True))
    gwf.to_parquet(ev / "dixon_coles" / "predictions_calibrated.parquet", index=False)
    cwf.to_parquet(ev / "counts" / "predictions_calibrated.parquet", index=False)

    all_goals = pd.concat([goals] + ([g26] if g26 is not None else []), ignore_index=True)
    all_counts = pd.concat([counts] + ([c26] if c26 is not None else []), ignore_index=True)
    gm, cm = live_maps(all_goals, all_counts, rho)
    out = reports / "calibration"
    cal.save(
        out / "calibration.json", gm, cm, {"goals": bool(g["adopt"]), "cards": bool(c["adopt"])}
    )
    text = verdict(g, c) + (
        f"\nLive maps (fitted on every out-of-sample season, rho {rho:.4f}):\n"
        f"* goals: a {gm.a:.3f}, b {gm.b:+.3f}, d {gm.d:.3f}, c {gm.c:+.3f}, "
        f"delta {gm.delta:+.3f} ({gm.fitted_on})\n"
        f"* cards: d {cm.d:.3f}, f {cm.f:.3f} ({cm.fitted_on})\n"
    )
    (out / "verdict.md").write_text(text, encoding="utf-8")
    return text
