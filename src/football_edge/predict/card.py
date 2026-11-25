"""Render forecasts and stakes as a readable markdown betting card."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from football_edge.betting.golden import cs_min_odds, flags
from football_edge.betting.staking import StakingPolicy
from football_edge.leagues import NAMES
from football_edge.predict.confidence import lookup
from football_edge.predict.forecast import TrainingInfo

_LEAGUE_NAMES = NAMES


def _num(x: float, fmt: str) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def render_card(
    fc: pd.DataFrame,
    singles: pd.DataFrame,
    doubles: pd.DataFrame,
    policy: StakingPolicy,
    info: TrainingInfo,
    model: str,
    notes: list[str],
    record: pd.DataFrame | None = None,
    golden: dict[str, Any] | None = None,
) -> str:
    """`golden`: the `staking` config (golden_bar / golden_ou / golden_cs sections)."""
    g = golden or {}
    has_gold = any(g.get(k) for k in ("golden_bar", "golden_ou", "golden_cs"))
    cur = policy.currency
    staked = float(singles["stake"].sum() + (doubles["stake"].sum() if len(doubles) else 0))
    lines = [
        f"# Betting card: {fc['date'].min():%a %d %b %Y} to {fc['date'].max():%a %d %b %Y}",
        "",
        f"Model `{model}` trained on {info.n_matches:,} matches up to "
        f"{info.last_match:%d %b %Y}. Budget {policy.monthly_budget:,.0f} {cur} a month; "
        f"this card stakes {staked:,.0f} {cur}.",
        "",
        "Read this first: our tests found no edge over the market. Most bets below have a "
        "negative expected value; the stakes are a risk budget, not a money-making system. "
        "`Min odds` is the lowest price worth taking: below it, skip.",
        "",
        PROB_NOTE[policy.confidence if policy.confidence in PROB_NOTE else "market"],
        "",
        "`Confidence` is how far to trust the probability, from its track record in "
        "2020/21-2025/26 (High / Medium / Low; rules in predict/confidence.py): e.g. "
        "'came true 63% of 1,204' = past predictions at this level.",
        "",
    ]
    for n in notes:
        lines += [f"> **{n}**", ""]
    gold = {k: flags(f, g) for k, f in fc.iterrows()} if has_gold else {}
    if has_gold:
        lines += [
            "## ⭐ Golden bets on this card",
            "",
            "Golden = inside a golden bar, whatever the price. All bars are UNPROVEN and tracked "
            "(`football-edge settle`). Correct score: only at the league's minimum price "
            "(`golden_cs.min_odds`: the bar's break-even there).",
            "",
            "| Match | Match winner | Over/under 2.5 | Correct score |",
            "|---|---|---|---|",
        ]
        hits = [(k, fl) for k, fl in gold.items() if any(fl.values())]
        for k, fl in hits:
            home, away = str(fc.at[k, "home"]), str(fc.at[k, "away"])
            lines.append(f"| {home} v {away} | {fl['win'] or '-'} | {fl['ou'] or '-'} "
                         f"| {fl['cs'] or '-'} |")  # fmt: skip
        if not hits:
            lines.append("| none | - | - | - |")
        lines.append("")
    for k, f in fc.iterrows():
        lines += [
            f"## {_LEAGUE_NAMES.get(f['league'], f['league'])}, {f['date']:%a %d %b}: "
            f"{f['home']} v {f['away']}",
            "",
            f"Model: {f['home']} {f['p_h']:.0%} | Draw {f['p_d']:.0%} | "
            f"{f['away']} {f['p_a']:.0%} | "
            f"Over 2.5 {f['p_over25']:.0%} | Both score {f['p_btts_yes']:.0%}",
            "",
            _market_line(f),
            "",
            f"Expected goals: {f['home']} {f['exp_home']:.2f}, {f['away']} {f['exp_away']:.2f}. "
            f"Likeliest scores: {f['top_scores']}.",
            "",
            *_counts_lines(f, record),
            "| Market | Pick | Prob | Confidence | Model | Market | Fair odds | Min odds "
            "| Site odds | Exp. value | Stake tier | Stake | Decision | Golden |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        fl = gold.get(k, {"win": "", "ou": "", "cs": ""})
        cs_gold = fl["cs"]
        for _, s in singles[singles["fixture"] == k].iterrows():
            golden_here = (s["market"] == "1x2" and fl["win"]) or (
                s["market"] == "ou25" and fl["ou"]
            )
            star = "⭐" if golden_here else "-"
            lines.append(
                f"| {s['market']} | {s['selection']}{' ⚠' if s['disagree'] else ''} "
                f"| {s['prob']:.0%} {s['prob_source']} | {_conf(record, s)} "
                f"| {s['model_prob']:.0%} "
                f"| {_num(s['market_prob'], '.0%')} | {s['fair_odds']:.2f} "
                f"| {s['min_odds']:.2f} | {_num(s['odds'], '.2f')} "
                f"| {_num(s['exp_value'], '+.1%')} "
                f"| {s['tier']} | {s['stake']:,.0f} | {s['status']} | {star} |"
            )
        if "top_score" in f:
            p_cs = float(f["top_score_p"])
            cs_low = cs_min_odds(g.get("golden_cs"), str(f["league"]))
            lines.append(
                f"| score | {f['top_score']} | {p_cs:.1%} model | - | {p_cs:.1%} | - "
                f"| {1 / p_cs:.2f} | {cs_low:.2f} | - | - | - | 0 "
                f"| {'golden: take at ' + format(cs_low, '.2f') + '+' if cs_gold else '-'} "
                f"| {'⭐' if cs_gold else '-'} |"
            )
        lines.append("")
    if len(doubles):
        lines += [
            "## Doubles (two legs, different matches)",
            "",
            "| Leg 1 | Leg 2 | Prob | Odds | Exp. value | Stake |",
            "|---|---|---|---|---|---|",
        ]
        for _, d in doubles.iterrows():
            lines.append(
                f"| {d['leg1']} | {d['leg2']} | {d['prob']:.0%} | {d['odds']:.2f} "
                f"| {d['exp_value']:+.1%} | {d['stake']:,.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _market_line(f: pd.Series) -> str:
    def pct(col: str) -> str:
        v = f.get(col, float("nan"))
        return "-" if pd.isna(v) else f"{v:.0%}"

    if all(pd.isna(f.get(c, float("nan"))) for c in ("mkt_h", "mkt_over25", "mkt_btts_yes")):
        return "Market: no prices given."
    return (
        f"Market: {f['home']} {pct('mkt_h')} | Draw {pct('mkt_d')} | {f['away']} {pct('mkt_a')} | "
        f"Over 2.5 {pct('mkt_over25')} | Both score {pct('mkt_btts_yes')}"
    )


PROB_NOTE = {
    "model": "`Prob` is our model's own probability (no market prices used in any decision). "
    "The `Market` column is shown for information only. ⚠ marks picks where model and market "
    "differ by 8+ points: in testing the market was usually closer, so treat those with care.",
    "market": "`Prob` is the market's fair probability (bookmaker margin removed; `sharp` = "
    "Pinnacle, `site` = your site's own prices) when prices were given, otherwise our model's. "
    "The market's is the more accurate of the two. ⚠ marks picks where model and market "
    "differ by 8+ points: usually the market knows something (injury, line-up) we don't.",
}


def _conf(record: pd.DataFrame | None, s: pd.Series) -> str:
    source = "market" if str(s["prob_source"]).startswith("market") else "model"
    label, text = lookup(record, source, str(s["market"]), float(s["prob"]))
    return f"{label} ({text})"


def _counts_lines(f: pd.Series, record: pd.DataFrame | None = None) -> list[str]:
    """Corners and cards: probabilities only (no prices in our data, so no stakes)."""
    out = []
    for stat, title in (("corners", "Corners"), ("cards", "Total cards (yellow + red)")):
        if f"exp_{stat}" not in f.index:
            continue
        overs = []
        for c in f.index:
            if not c.startswith(f"p_{stat}_over_"):
                continue
            line = c.rsplit("_", 1)[1]
            label, _ = lookup(record, "model", f"{stat}_{float(line)}", float(f[c]))
            overs.append(f"over {line} {f[c]:.0%} [{label}]")
        out.append(f"{title}: expect {f[f'exp_{stat}']:.1f} | " + " | ".join(overs))
    if "p_reds_over_0.5" in f.index:
        p = f["p_reds_over_0.5"]
        label, text = lookup(record, "model", "reds_0.5", float(p))
        out.append(f"Red card in the match: {p:.0%} [{label}: {text}]")
    if out:
        out += ["(Corners and cards: model probabilities only, not tested against prices.)", ""]
    return out
