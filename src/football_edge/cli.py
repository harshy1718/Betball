"""Command line entry point: `football-edge <command>`."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from football_edge.config import load_config
from football_edge.data.loader import ingest, load_matches, needs_features
from football_edge.logging_setup import configure_logging

MODELS = [
    "team_blind",
    "elo",
    "dixon_coles",
    "dixon_coles_h2h",
    "dixon_coles_lineup",
    "dixon_coles_trust",
    "dixon_coles_manager",
]


def _ingest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    df = ingest(args.src, cfg)
    print(f"ok: {len(df)} matches, seasons {df.season.min()}-{df.season.max()}")


def _evaluate(args: argparse.Namespace) -> None:
    from football_edge.evaluation.folds import make_folds
    from football_edge.evaluation.report import save_run
    from football_edge.evaluation.runner import summarise, walk_forward
    from football_edge.models.registry import model_factory

    cfg = load_config(args.config)
    ev = cfg.evaluation
    params = cfg.models.get(args.model, {})
    matches = load_matches(cfg, features=needs_features(params))
    folds = make_folds(matches, ev.first_test_season, ev.last_test_season, ev.min_train_matches)
    preds = walk_forward(matches, folds, model_factory(args.model, params))
    summary = summarise(preds)
    out_dir = cfg.reports_dir / "evaluation" / args.model
    save_run(
        out_dir,
        preds,
        summary,
        model=args.model,
        params=params,
        root=cfg.root,
        data_path=cfg.matches_path,
    )
    print(summary.round(4).to_string())
    if needs_features(params):
        _print_coefficients(matches, args.model, params)
    print(f"\nsaved to {out_dir}")


def _print_coefficients(matches: object, model: str, params: dict[str, object]) -> None:
    """Fit once on everything up to the last test season, to show what was learned."""
    from football_edge.models.dixon_coles import DixonColes, DixonColesParams

    m = DixonColes(DixonColesParams.from_dict(dict(params)))
    m.fit(matches)  # type: ignore[arg-type]
    coefs = ", ".join(f"{k} {v:+.3f}" for k, v in m.coefficients.items())
    print(f"\n{model} coefficients on all data: {coefs}")


def _tune(args: argparse.Namespace) -> None:
    from football_edge.evaluation.folds import make_folds
    from football_edge.evaluation.tuning import grid_search

    cfg = load_config(args.config)
    ev = cfg.evaluation
    tuning = cfg.tuning.get(args.model)
    if not tuning:
        raise SystemExit(f"no tuning grid for {args.model!r} in the config")
    matches = load_matches(cfg)
    folds = make_folds(matches, ev.first_test_season, ev.last_test_season, ev.min_train_matches)
    lo, hi = tuning["seasons"]
    table = grid_search(
        matches, folds, args.model, cfg.models.get(args.model, {}), tuning["grid"], (lo, hi)
    )
    out = cfg.reports_dir / "tuning" / f"{args.model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5f")
    print(table.head(10).to_string(index=False))
    print(f"\nbest on seasons {lo}-{hi}: {table.iloc[0].drop(['log_loss', 'n']).to_dict()}")
    print(f"saved to {out}  (copy the best values into models: in the config to lock them)")


def _stack_test(args: argparse.Namespace) -> None:
    import json

    import pandas as pd

    from football_edge.evaluation.compare import market_columns
    from football_edge.markets.margin import remove_margin
    from football_edge.markets.stack import render_report, stack_test

    cfg = load_config(args.config)
    preds_path = cfg.reports_dir / "evaluation" / args.model / "predictions.parquet"
    if not preds_path.exists():
        raise SystemExit(f"run `football-edge evaluate --model {args.model}` first")
    preds = pd.read_parquet(preds_path)
    matches = load_matches(cfg)
    odds_cols = {
        "1x2": ["close_h", "close_d", "close_a"],
        "ou25": ["ou_close_over", "ou_close_under"],
    }[args.market]
    odds = matches.iloc[preds["row"].to_numpy()][odds_cols].to_numpy(dtype=float)
    ok = ~pd.isna(odds).any(axis=1)
    if not ok.all():
        print(f"note: {int((~ok).sum())} matches without closing odds for {args.market} left out")
    preds, odds = preds[ok].reset_index(drop=True), odds[ok]
    p_model, outcome = market_columns(preds, args.market)
    preds = preds.assign(outcome=outcome)
    res = stack_test(
        preds, remove_margin(odds), p_model, eval_seasons=args.eval_seasons, level=args.level
    )
    out = cfg.reports_dir / "stack_test" / args.model / args.market
    if args.market == "1x2":  # keep the original location of the first verdict
        out = cfg.reports_dir / "stack_test" / args.model
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(res, args.model, args.market)
    (out / "verdict.md").write_text(report, encoding="utf-8")
    summary = {k: v for k, v in res.__dict__.items() if k != "criteria"}
    summary["criteria"] = res.criteria
    summary["passed"] = res.passed
    (out / "verdict.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _compare(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.evaluation.compare import paired_compare

    cfg = load_config(args.config)
    ev_dir = cfg.reports_dir / "evaluation"
    a = pd.read_parquet(ev_dir / args.a / "predictions.parquet")
    b = pd.read_parquet(ev_dir / args.b / "predictions.parquet")
    lines = [
        f"# {args.b} vs {args.a}",
        "",
        f"Seasons {args.seasons[0]}-{args.seasons[-1]}; {args.level:.2%} intervals.",
        "",
    ]
    for market in args.markets:
        c = paired_compare(a, b, market, args.seasons, level=args.level)
        lines += [
            f"## {market}: {'HELPS' if c.b_helps else 'DOES NOT HELP'}",
            "",
            f"Log-loss {args.a} {c.log_loss_a:.4f}, {args.b} {c.log_loss_b:.4f} ({c.n} matches).",
            f"Improvement {c.improvement:+.5f} per match "
            f"({args.level:.2%} interval {c.ci[0]:+.5f} to {c.ci[1]:+.5f}).",
            "",
            "| Season | Improvement |",
            "|---|---|",
            *[f"| {s} | {v:+.5f} |" for s, v in c.by_season.items()],
            "",
            "| League | Improvement |",
            "|---|---|",
            *[f"| {lg} | {v:+.5f} |" for lg, v in c.by_league.items()],
            "",
        ]
    report = "\n".join(lines)
    out = cfg.reports_dir / "compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.b}_vs_{args.a}.md").write_text(report, encoding="utf-8")
    print(report)


def _price_test(args: argparse.Namespace) -> None:
    from football_edge.markets.price_edge import VARIANTS, render_report, run_variant

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    lo, hi = args.seasons
    matches = matches[matches["season"].between(lo, hi)].reset_index(drop=True)
    level = 1 - 0.05 / len(VARIANTS)  # Bonferroni
    out = cfg.reports_dir / "price_test"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for v in VARIANTS:
        res, bets = run_variant(matches, v, level)
        results.append(res)
        bets.to_csv(out / f"bets_{v.name}.csv", index=False, float_format="%.4f")
    report = render_report(results, level, (lo, hi))
    (out / "verdict.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _early_totals_test(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.markets.early_totals import (
        VARIANTS,
        join,
        load_early_prices,
        render,
        run_variant,
        test_a,
    )

    cfg = load_config(args.config)
    preds = pd.read_parquet(cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet")
    lo, hi = 2020, 2025
    preds = preds[preds["season"].between(lo, hi)]
    prices = load_early_prices(cfg.raw_dir, list(cfg.leagues), (lo, hi))
    m = join(preds, prices)
    level = 1 - 0.05 / len(VARIANTS)
    out = cfg.reports_dir / "early_totals"
    out.mkdir(parents=True, exist_ok=True)
    results = [run_variant(m, v, level) for v in VARIANTS]
    for r in results:
        r.bets.to_csv(out / f"bets_{r.variant.name}.csv", index=False, float_format="%.4f")
    report = render(test_a(m, level), results, level, len(m))
    (out / "verdict.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}  ({len(preds):,} predictions, {len(m):,} matched to prices)")


def _h2h_weight_test(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.evaluation.h2h_weight import run

    cfg = load_config(args.config)
    preds = pd.read_parquet(cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet")
    report = run(preds, load_matches(cfg), years=args.years)
    out = cfg.reports_dir / "h2h_weight"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _bar_test(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.evaluation.prob_bar import run

    cfg = load_config(args.config)
    preds = pd.read_parquet(cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet")
    report = run(preds, cfg.raw_dir, list(cfg.leagues))
    out = cfg.reports_dir / "prob_bar"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _window_search(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.evaluation import (
        bar_by_league,
        ou_split_search,
        threshold_both,
        window_both,
        window_search,
    )
    from football_edge.evaluation.prob_bar import load_prices

    cfg = load_config(args.config)
    preds = pd.read_parquet(cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"]).dt.normalize()
    prices = load_prices(cfg.raw_dir, list(cfg.leagues), range(2018, 2026))
    m = preds.merge(prices, on=["league", "date", "home", "away"], how="left")
    if args.bars:
        p26 = cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions_2026.parquet"
        m26 = None
        if p26.exists():
            q = pd.read_parquet(p26)
            q["date"] = pd.to_datetime(q["date"]).dt.normalize()
            pr26 = load_prices(cfg.raw_dir, list(cfg.leagues), range(2026, 2027))
            m26 = q.merge(pr26, on=["league", "date", "home", "away"], how="left")
        report = threshold_both.run(m, m26)
    elif args.both:
        report = window_both.run(m, lo_min=args.above or 0.20)
    elif args.by_league:
        report = bar_by_league.run(m, args.above or 0.20)
    elif args.ou_split:
        report = ou_split_search.run(m)
    else:
        report = window_search.run(m)
    out = cfg.reports_dir / (
        "threshold_both" if args.bars
        else ("window_both_above_50" if args.above else "window_both") if args.both
        else ("bar_above_50" if args.above else "bar_by_league") if args.by_league
        else "ou_split_search" if args.ou_split else "window_search"
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _price(args: argparse.Namespace) -> None:
    from football_edge.betting import golden

    cfg = load_config(args.config)
    key, default, _ = {
        "winner": golden.BARS[0], "ou": golden.BARS[1], "score": golden.BARS[2]
    }[args.bar]  # fmt: skip
    path = cfg.root / (cfg.staking.get(key) or {}).get("log_path", default)
    done = golden.set_price(path, args.team, args.odds)
    if not done:
        raise SystemExit(f"no unsettled {args.bar} pick with a team matching '{args.team}'")
    for d in done:
        print(f"saved: {d}")
    if args.bar == "score" and args.odds < 6.80:
        print("note: below 6.80, the break-even price for this bar; history says skip it")


def _analyst(args: argparse.Namespace) -> None:
    import json as _json

    import pandas as pd

    from football_edge.analyst.agent import Analyst
    from football_edge.analyst.tools import AnalystTools
    from football_edge.news.extract import LLMError, NewsConfig, OllamaClient

    cfg = load_config(args.config)
    print("loading data and model...", flush=True)
    analyst = Analyst(
        OllamaClient(NewsConfig.from_dict(cfg.analyst)),
        AnalystTools(cfg),
        max_steps=int(cfg.analyst.get("max_steps", 4)),
    )
    log_path = cfg.reports_dir / "analyst" / "chat_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def one(q: str) -> None:
        try:
            a = analyst.ask(q)
        except LLMError as e:
            print(f"error: {e}")
            return
        for name, targs, sec in a.tool_calls:
            shown = ", ".join(f"{k}={v}" for k, v in targs.items())
            print(f"  [tool] {name}({shown})  {sec:.1f}s")
        print(f"\n{a.text}\n")
        if a.unverified:
            print(f"  [check] numbers not found in any tool result: {', '.join(a.unverified)}")
        print(f"  ({a.seconds:.0f}s, of which LLM {a.llm_seconds:.0f}s)\n", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps({
                "time": pd.Timestamp.now().isoformat(timespec="seconds"), "question": q,
                "tools": [[n, t] for n, t, _ in a.tool_calls], "answer": a.text,
                "unverified": a.unverified, "seconds": round(a.seconds, 1),
            }) + "\n")  # fmt: skip

    if args.question:
        one(" ".join(args.question))
        return
    print("Ask about any EPL, Serie A or La Liga match. Type 'exit' to stop.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            one(q)


def _analyst_eval(args: argparse.Namespace) -> None:
    from football_edge.analyst.evaluate import run
    from football_edge.analyst.tools import AnalystTools
    from football_edge.news.extract import NewsConfig, OllamaClient

    cfg = load_config(args.config)
    conf = dict(cfg.analyst)
    if args.model:
        conf["model"] = args.model
    client = OllamaClient(NewsConfig.from_dict(conf))
    steps = int(conf.get("max_steps", 4))
    make: Callable[[], Any] | None = None
    questions = cfg.root / "evals" / "analyst_questions.yaml"
    if args.site:  # Ask Nox: the website's prompt, tools and banned words
        from football_edge.analyst.agent import Analyst
        from football_edge.web.ask import BANNED, NOX_SYSTEM, WEB_SCHEMAS, WebTools

        tools: AnalystTools = WebTools(cfg)

        def nox() -> Analyst:
            return Analyst(client, tools, steps, system=NOX_SYSTEM, schemas=WEB_SCHEMAS,
                           banned=BANNED)  # fmt: skip

        make, questions = nox, cfg.root / "evals" / "nox_questions.yaml"
    else:
        tools = AnalystTools(cfg)
    print(f"evaluating {'Ask Nox' if args.site else 'the analyst'} with {conf.get('model')}...",
          flush=True)  # fmt: skip
    report, _ = run(client, tools, questions, steps, make)
    out = cfg.reports_dir / "analyst"
    out.mkdir(parents=True, exist_ok=True)
    name = str(conf.get("model")).replace(":", "_").replace("/", "_")
    name = f"nox_{name}" if args.site else name
    (out / f"eval_{name}.md").write_text(report, encoding="utf-8")
    print(report.split("\n\n")[1])
    print(f"saved to {out / f'eval_{name}.md'}")


def _build_features(args: argparse.Namespace) -> None:
    from football_edge.features.build import build_from_folder, coverage

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    feats, links = build_from_folder(matches, args.tm)
    out = cfg.processed_dir / "features.parquet"
    feats.to_parquet(out, index=False)
    report_dir = cfg.reports_dir / "features"
    report_dir.mkdir(parents=True, exist_ok=True)
    links.to_csv(report_dir / "team_links.csv", index=False, float_format="%.3f")
    cov = coverage(feats, matches)
    cov.to_csv(report_dir / "coverage.csv", float_format="%.3f")
    print(f"linked {len(links)} teams (see {report_dir / 'team_links.csv'})")
    print(cov.round(3).to_string())
    print(f"saved to {out}")


def _news_client(cfg_news: dict[str, object]) -> object:
    from football_edge.news.extract import NewsConfig, OpenAICompatibleClient

    return OpenAICompatibleClient(NewsConfig.from_dict(dict(cfg_news)))


def _news_extract(args: argparse.Namespace) -> None:
    import json

    from football_edge.news.extract import OpenAICompatibleClient, extract

    cfg = load_config(args.config)
    client = _news_client(cfg.news)
    assert isinstance(client, OpenAICompatibleClient)
    out = []
    for path in sorted(args.articles.glob("*.txt")):
        res = extract(client, path.read_text(encoding="utf-8"))
        facts = json.loads(res.facts.to_json()) if res.facts else None
        out.append({"file": path.name, "facts": facts, "error": res.error})
        print(f"{path.name}: {len(res.facts.players) if res.facts else 'FAILED'} player facts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {args.out}")


def _news_eval(args: argparse.Namespace) -> None:
    from football_edge.news.evaluate import evaluate, load_labels
    from football_edge.news.extract import OpenAICompatibleClient

    cfg = load_config(args.config)
    client = _news_client(cfg.news)
    assert isinstance(client, OpenAICompatibleClient)
    r = evaluate(client, load_labels(args.labels))
    print(f"model {cfg.news.get('model')} on {r.articles} articles")
    print(f"valid answers {r.valid_rate:.0%} | precision {r.precision:.2f} | recall {r.recall:.2f}")
    print(f"({r.true_pos} right of {r.extracted} extracted, {r.expected} expected)")
    print(f"latency {r.mean_latency_s:.1f}s per article | tokens {r.tokens:,}")
    for e in r.errors:
        print(f"  failed: {e}")


def _evaluate_counts(args: argparse.Namespace) -> None:
    from football_edge.evaluation.counts_eval import (
        calibration,
        summarise_counts,
        walk_forward_counts,
    )
    from football_edge.evaluation.folds import make_folds

    cfg = load_config(args.config)
    lo, hi = args.seasons
    matches = load_matches(cfg)
    folds = make_folds(matches, lo, hi, cfg.evaluation.min_train_matches)
    preds = walk_forward_counts(matches, folds)
    out = cfg.reports_dir / "evaluation" / "counts"
    out.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    summary = summarise_counts(preds)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.5f")
    print(summary.round(4).to_string(index=False))
    for stat, line in (("corners", 9.5), ("yellows", 4.5), ("reds", 0.5)):
        print(f"\ncalibration, {stat} over {line}:")
        print(calibration(preds, stat, line).round(3).to_string())
    print(f"\nsaved to {out}")


def _build_confidence(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.predict.confidence import build_track_record

    cfg = load_config(args.config)
    goals = cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet"
    counts = cfg.reports_dir / "evaluation" / "counts" / "predictions.parquet"
    if not goals.exists():
        raise SystemExit("run `football-edge evaluate --model dixon_coles` first")
    table = build_track_record(
        pd.read_parquet(goals),
        pd.read_parquet(counts) if counts.exists() else None,
        load_matches(cfg),
    )
    out = cfg.reports_dir / "confidence"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "track_record.csv", index=False, float_format="%.4f")
    preds = pd.read_parquet(goals)
    if {"exp_home", "exp_away"} <= set(preds.columns):  # the trader desk's extra goal bets
        from football_edge.models.dixon_coles import DixonColes, DixonColesParams
        from football_edge.trader.record import build_goal_record

        dc = DixonColes(DixonColesParams.from_dict(dict(cfg.models.get("dixon_coles", {}))))
        dc.fit(load_matches(cfg))
        extra = build_goal_record(preds, float(dc.rho))
        extra.to_csv(out / "track_record_goals.csv", index=False, float_format="%.4f")
        print(f"extra goal bets: {extra['market'].nunique()} markets, saved alongside")
    else:
        print("extra goal bets skipped: re-run `football-edge evaluate --model dixon_coles`")
    summary = table.groupby(["source", "market", "label"])["n"].sum().unstack(fill_value=0)
    print(summary.to_string())
    print(f"saved to {out / 'track_record.csv'}")


def _simulate(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.betting.simulate import (
        GridDixonColes,
        by_market,
        candidates,
        read_book_prices,
        simulate,
    )
    from football_edge.betting.staking import StakingPolicy
    from football_edge.evaluation.folds import make_folds
    from football_edge.evaluation.runner import walk_forward
    from football_edge.models.dixon_coles import DixonColesParams

    cfg = load_config(args.config)
    lo, hi = args.seasons
    matches = load_matches(cfg)
    out = cfg.reports_dir / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / f"grid_predictions_{lo}_{hi}.parquet"
    if args.refresh or not grid_path.exists():
        folds = make_folds(matches, lo, hi, cfg.evaluation.min_train_matches)
        params = DixonColesParams.from_dict(cfg.models["dixon_coles"])
        walk_forward(matches, folds, lambda: GridDixonColes(params)).to_parquet(
            grid_path, index=False
        )
    preds = pd.read_parquet(grid_path)
    market = ["close_h", "close_d", "close_a", "ou_close_over", "ou_close_under"]
    preds = preds.merge(matches.reset_index(names="row")[["row", *market]], on="row")
    preds = preds.merge(
        read_book_prices(cfg.raw_dir), on=["league", "date", "home", "away"], how="left"
    )
    policy = StakingPolicy.from_dict(cfg.staking)
    cur = policy.currency
    lines = [
        f"# Bankroll simulation {lo}/{lo + 1} to {hi}/{hi + 1}",
        "",
        f"Start {args.start:,.0f} {cur}. Weekly bets on 1X2, over/under 2.5 and Asian handicap "
        "with the staking policy in configs/default.yaml (tiers as a share of the current "
        "balance, price guard, per-match cap, up to 2 doubles a week, monthly stop-loss). "
        "Prices: Bet365 at kick-off, or re-priced to an 8% margin. BTTS, cards and corners "
        "have no prices in the data and are not included.",
        "",
        "| Picks by | Prices | Final balance | Lowest | Bets | Total staked | Profit | Return |",
        "|---|---|---|---|---|---|---|---|",
    ]
    detail, curves = [], {}
    for conf in ("model", "market"):
        for site, label in ((None, "Bet365"), (0.08, "8% site")):
            res = simulate(candidates(preds, conf, site), policy, args.start)
            b = res.bets
            staked, profit = float(b["stake"].sum()), float(b["profit"].sum())
            lines.append(
                f"| {conf} | {label} | {res.final:,.0f} | {res.lowest:,.0f} | {len(b):,} "
                f"| {staked:,.0f} | {profit:+,.0f} | {profit / staked:+.1%} |"
            )
            name = f"{conf}_{label.replace('% ', 'pct_').lower()}"
            b.to_csv(out / f"bets_{name}.csv", index=False, float_format="%.4f")
            curves[name] = res.weekly.set_index("week")["balance"]
            table = by_market(b)
            detail += [
                f"## Picks by {conf}, {label}",
                "",
                "| Market | Bets | Staked | Profit | Return | Avg confidence | Win rate |",
                "|---|---|---|---|---|---|---|",
                *[
                    f"| {mk} | {int(r.bets):,} | {r.staked:,.0f} | {r.profit:+,.0f} "
                    f"| {r['return']:+.1%} | {r.avg_conf:.0%} | {r.win_rate:.0%} |"
                    for mk, r in table.iterrows()
                ],
                "",
            ]
    pd.DataFrame(curves).to_csv(out / "balance.csv", float_format="%.2f")
    report = "\n".join([*lines, "", *detail])
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def _update(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.data.sources import download_fixtures, download_results
    from football_edge.predict.fixtures import season_of

    cfg = load_config(args.config)
    season = int(season_of(pd.Series([pd.Timestamp.today()])).to_numpy()[0])
    have: dict[str, int] = {}
    if cfg.matches_path.exists():
        m = load_matches(cfg)
        counts = m[m["season"] == season].groupby("league").size()
        have = {str(k): int(v) for k, v in counts.items()}
    for d in download_results(args.src, season, cfg.leagues, have):
        print(f"{d.league} {season}/{season + 1}: {d.note}")
    try:
        n = download_fixtures(cfg.root / "data" / "upcoming" / "fixtures.csv")
        print(f"upcoming fixtures with odds: {n} rows (all leagues)")
    except Exception as e:
        print(f"upcoming fixtures not downloaded: {e}")
    matches = ingest(args.src, cfg)
    for lg, g in matches.groupby("league"):
        print(f"{lg}: latest result {g['date'].max():%a %d %b %Y}")
    from football_edge.data.schedule import TOKEN_ENV

    if os.environ.get(TOKEN_ENV):
        _schedule(args)
    else:
        print(f"(schedule not refreshed: set {TOKEN_ENV} to enable it)")


def _schedule(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.data.schedule import ScheduleError, fetch_schedule, upcoming

    cfg = load_config(args.config)
    history = load_matches(cfg)
    known = {str(lg): set(g["home"]) | set(g["away"]) for lg, g in history.groupby("league")}
    aliases = cfg.root / cfg.predict.get("aliases", "configs/team_aliases.yaml")
    try:
        sched, problems = fetch_schedule(cfg.leagues, known, aliases)
    except ScheduleError as e:
        raise SystemExit(f"error: {e}") from None
    out = cfg.processed_dir / "schedule.parquet"
    sched.to_parquet(out, index=False)
    for p in problems:
        print(f"warning: {p}")
    nxt = upcoming(sched, pd.Timestamp.today().normalize(), days=10)
    print(f"schedule: {len(sched)} matches, {int((sched['status'] == 'FINISHED').sum())} played")
    for lg, g in nxt.groupby("league"):
        first = g[g["matchday"] == g["matchday"].to_numpy()[0]]
        print(
            f"{lg}: next matchday {first['matchday'].to_numpy()[0]}, "
            f"{first['date'].min():%a %d %b} to {first['date'].max():%a %d %b}, "
            f"{len(first)} matches"
        )
    print(f"saved to {out}")


def _script(args: argparse.Namespace) -> None:
    import re as _re

    import pandas as pd

    from football_edge.content.facts import build_fact_sheet, display_names
    from football_edge.content.picker import best_story, pick_matches
    from football_edge.content.script import write_script
    from football_edge.data.schedule import upcoming
    from football_edge.news.extract import LLMError, NewsConfig, make_client
    from football_edge.predict.confidence import load_track_record
    from football_edge.predict.fixtures import FixtureError, from_schedule, single_fixture
    from football_edge.predict.pipeline import attach_prices, forecast_fixtures

    cfg = load_config(args.config)
    history = load_matches(cfg)
    known = {str(lg): set(g["home"]) | set(g["away"]) for lg, g in history.groupby("league")}
    aliases = cfg.root / cfg.predict.get("aliases", "configs/team_aliases.yaml")
    try:
        if args.next is not None:
            sched_path = cfg.processed_dir / "schedule.parquet"
            if not sched_path.exists():
                raise SystemExit("no schedule yet: run `football-edge schedule` first")
            nxt = upcoming(pd.read_parquet(sched_path), pd.Timestamp.today().normalize(),
                           args.next)  # fmt: skip
            if nxt.empty:
                raise SystemExit(f"no scheduled matches in the next {args.next} days")
            fixtures = from_schedule(nxt, None)
        elif args.home and args.away:
            fixtures = single_fixture(
                args.home,
                args.away,
                known,
                cfg.leagues,
                league=args.league,
                date=args.date,
                aliases_path=aliases,
            )
        else:
            raise SystemExit("give --home and --away, or --next DAYS")
    except FixtureError as e:
        raise SystemExit(f"error: {e}") from None
    fixtures = attach_prices(cfg, fixtures, known, aliases)
    fc, _ = forecast_fixtures(cfg, history, fixtures)
    record = load_track_record(cfg.reports_dir / "confidence" / "track_record.csv")
    names = display_names(aliases, cfg.content.get("display_names"))
    source = str(cfg.staking.get("confidence", "model"))
    sheets = [build_fact_sheet(f, history, record, names, source) for _, f in fc.iterrows()]
    if args.next is not None:
        cc = cfg.content
        chosen = pick_matches(
            [(s, f["home"], f["away"], float(f.get("exp_cards", 0.0)))
             for s, (_, f) in zip(sheets, fc.iterrows(), strict=True)],
            cc.get("big_clubs", {}), int(cc.get("videos_per_round", 3)),
        )  # fmt: skip
        print(f"{len(chosen)} of {len(sheets)} matches qualify for a video:")
        for p in chosen:
            print(f"  {p.facts.home} v {p.facts.away}: {p.story}")
        sheets = [p.facts for p in chosen]
        stories = {id(p.facts): p.story for p in chosen}
    else:
        stories = {}
        for s, (_, f) in zip(sheets, fc.iterrows(), strict=True):
            best = best_story(s, float(f.get("exp_cards", 0.0)))
            if best is not None:
                stories[id(s)] = best[1]
    client = None if args.dry_run else make_client(NewsConfig.from_dict(cfg.writer))
    for fs in sheets:
        slug = _re.sub(r"[^A-Za-z0-9]+", "_", f"{fs.home}_v_{fs.away}").strip("_")
        out = cfg.reports_dir / "content" / f"{fs.date:%Y-%m-%d}" / slug
        out.mkdir(parents=True, exist_ok=True)
        (out / "facts.txt").write_text(fs.text(), encoding="utf-8")
        (out / "facts.json").write_text(fs.to_json(), encoding="utf-8")
        print(f"\n=== {fs.home} v {fs.away} ===", flush=True)
        print(f"  story: {stories.get(id(fs), 'none (no High-confidence call)')}", flush=True)
        if client is None:
            print(fs.text())
            continue
        try:
            res = write_script(
                client,
                fs,
                int(cfg.writer.get("max_retries", 2)),
                on_attempt=lambda i, sec, probs: print(
                    f"  attempt {i}: {sec:.0f}s, "
                    + ("passed" if not probs else "failed: " + "; ".join(probs)),
                    flush=True,
                ),
                story=stories.get(id(fs)),
            )
        except LLMError as e:
            raise SystemExit(f"error: {e}") from None
        drafts = "\n\n".join(f"--- attempt {i} ---\n{t}\nproblems: {list(p) or 'none'}"
                               for i, (t, p) in enumerate(res.history, start=1))  # fmt: skip
        (out / "drafts.txt").write_text(drafts, encoding="utf-8")
        if res.text is None:
            print(f"SKIPPED after {res.attempts} attempts: {'; '.join(res.problems)}")
            continue
        (out / "script.txt").write_text(res.text, encoding="utf-8")
        print(res.text)
        print(f"({res.attempts} attempt(s), {res.seconds:.0f}s) saved to {out}")


def _predict(args: argparse.Namespace) -> None:
    import pandas as pd

    from football_edge.betting.ledger import (
        month_status,
        new_entries,
        read_log,
        write_log,
    )
    from football_edge.betting.staking import StakingPolicy, select_bets
    from football_edge.predict.card import render_card
    from football_edge.predict.fixtures import (
        FixtureError,
        from_schedule,
        load_fixtures,
        single_fixture,
    )
    from football_edge.predict.forecast import count_forecasts, fit_for, forecast, market_view

    cfg = load_config(args.config)
    pc, sc = cfg.predict, cfg.staking
    history = load_matches(cfg)
    known = {str(lg): set(g["home"]) | set(g["away"]) for lg, g in history.groupby("league")}
    try:
        aliases_path = cfg.root / pc.get("aliases", "configs/team_aliases.yaml")
        if args.home or args.away:
            if not (args.home and args.away) or args.fixtures:
                raise SystemExit("give --home and --away together, or --fixtures, not both")
            fixtures = single_fixture(
                args.home,
                args.away,
                known,
                cfg.leagues,
                league=args.league,
                date=args.date,
                odds_1x2=tuple(args.odds) if args.odds else None,
                odds_ou25=tuple(args.ou_odds) if args.ou_odds else None,
                aliases_path=aliases_path,
            )
            odds_file = cfg.root / "data" / "upcoming" / "fixtures.csv"
            if not (args.odds or args.ou_odds) and odds_file.exists():
                # No prices typed in: use the market's prices for this match if the odds
                # file lists it, so the fair price is the market's, not the model's.
                priced = load_fixtures(
                    odds_file,
                    known,
                    cfg.leagues,
                    odds_source=args.odds_source or pc.get("odds_source", "Avg"),
                    aliases_path=aliases_path,
                    allow_unknown=True,
                )
                f0 = fixtures.iloc[0]
                hit = priced[(priced["home"] == f0["home"]) & (priced["away"] == f0["away"])]
                if args.date:
                    hit = hit[hit["date"] == f0["date"]]
                if not hit.empty:
                    fixtures = hit.sort_values("date").head(1).reset_index(drop=True)
                    print(f"market prices: {odds_file.name}, {fixtures.iloc[0]['date']:%d %b}")
                else:
                    print(
                        "market prices: this match is not in data/upcoming/fixtures.csv "
                        "(run `football-edge update`); fair prices below are the MODEL's"
                    )
        elif args.next is not None:
            from football_edge.data.schedule import upcoming

            sched_path = cfg.processed_dir / "schedule.parquet"
            if not sched_path.exists():
                raise SystemExit("no schedule yet: run `football-edge schedule` first")
            nxt = upcoming(pd.read_parquet(sched_path), pd.Timestamp.today().normalize(), args.next)
            if nxt.empty:
                raise SystemExit(f"no scheduled matches in the next {args.next} days")
            odds_file = cfg.root / "data" / "upcoming" / "fixtures.csv"
            priced = None
            if odds_file.exists():
                priced = load_fixtures(
                    odds_file,
                    known,
                    cfg.leagues,
                    odds_source=args.odds_source or pc.get("odds_source", "Avg"),
                    aliases_path=aliases_path,
                    allow_unknown=True,
                )
            fixtures = from_schedule(nxt, priced)
        elif args.fixtures:
            fixtures = load_fixtures(
                args.fixtures,
                known,
                cfg.leagues,
                odds_source=args.odds_source or pc.get("odds_source", "Avg"),
                aliases_path=aliases_path,
                allow_unknown=args.allow_unknown,
            )
        else:
            raise SystemExit("give --home and --away, or --fixtures (see docs/predict.md)")
    except FixtureError as e:
        raise SystemExit(f"error: {e}") from None
    model_name = pc.get("model", "dixon_coles")
    model, info = fit_for(history, fixtures, cfg.models.get(model_name, {}))
    gap = (fixtures["date"].min() - info.last_match).days
    warnings = []
    if gap > int(pc.get("max_data_age_days", 10)):
        warnings.append(
            f"latest result in our data is {info.last_match:%d %b %Y}, {gap} days before the "
            "first fixture: download the current-season CSVs and run ingest first"
        )
    fc = market_view(forecast(model, fixtures))
    fc = pd.concat([fc, count_forecasts(history, fixtures)], axis=1)
    policy = StakingPolicy.from_dict(sc)
    today = pd.Timestamp.today().normalize()
    log_path = cfg.root / sc.get("log_path", "bets/log.csv")
    log = read_log(log_path)
    singles, doubles, notes = select_bets(fc, policy, month_status(log, today))

    out = cfg.reports_dir / "predict" / f"{fixtures['date'].min():%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out / "forecasts.csv", index=False, float_format="%.4f")
    singles.drop(columns="fixture").to_csv(out / "picks.csv", index=False, float_format="%.4f")
    doubles.to_csv(out / "doubles.csv", index=False, float_format="%.4f")
    from football_edge.predict.confidence import load_track_record

    record = load_track_record(cfg.reports_dir / "confidence" / "track_record.csv")
    if record is None:
        warnings.append("no confidence track record yet: run `football-edge build-confidence`")
    card = render_card(
        fc, singles, doubles, policy, info, model_name, warnings + notes, record=record,
        golden=cfg.staking,
    )  # fmt: skip
    from football_edge.betting import golden

    track = []
    for key, default, name in golden.BARS:
        conf = cfg.staking.get(key) or {}
        if not conf:
            continue
        if key == "golden_bar":
            picks = golden.golden_picks(fc, conf)
        elif key == "golden_ou":
            picks = golden.ou_picks(fc, conf)
        else:
            picks = golden.cs_picks(fc, float(conf.get("lo", 0.12)))
        path = cfg.root / conf.get("log_path", default)
        n_new = golden.log_picks(picks, path, today)
        track.append(f"- {n_new} new pick(s) logged. {golden.scoreboard(golden.read(path), name)}")
    if track:
        card += (
            "\n## Golden bar tracking\n\n"
            + "\n".join(track)
            + "\n\nRecord your site's price: "
            + "`football-edge price winner|ou|score <team> <odds>`.\n"
        )
    (out / "card.md").write_text(card, encoding="utf-8")
    print(card)
    print(f"saved to {out}")
    if args.log:
        entries = new_entries(singles, doubles, today)
        write_log(pd.concat([log, entries], ignore_index=True), log_path)
        print(f"logged {len(entries)} bets to {log_path}")


def _settle(args: argparse.Namespace) -> None:
    from football_edge.betting.ledger import read_log, settle, summary, write_log

    cfg = load_config(args.config)
    from football_edge.betting import golden

    results = load_matches(cfg)
    for key, default, name in golden.BARS:
        conf = cfg.staking.get(key) or {}
        path = cfg.root / conf.get("log_path", default)
        if conf and path.exists():
            print(golden.scoreboard(golden.settle(path, results), name))
    log_path = cfg.root / cfg.staking.get("log_path", "bets/log.csv")
    log = read_log(log_path)
    if log.empty:
        print(f"no bets logged yet ({log_path})")
        return
    log, n = settle(log, load_matches(cfg))
    write_log(log, log_path)
    pending = int(log["result"].isna().sum())
    print(f"settled {n} bets; {pending} still waiting for results")
    table = summary(log)
    if not table.empty:
        print(table.round(2).to_string())


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn

        from football_edge.web.app import create_app
        from football_edge.web.service import NoxService
    except ImportError:
        raise SystemExit('the website needs FastAPI: pip install -e ".[web]"') from None
    cfg = load_config(args.config)
    svc = NoxService(cfg, days=args.days)
    print("Nox is warming up (fitting the model on the week's fixtures)...", flush=True)
    try:
        meta = svc.meta()
        print(f"{meta['matches']} matches in the next {args.days} days; model trained up to "
              f"{meta['trained_up_to']}.")  # fmt: skip
        if meta["warning"]:
            print(f"warning: {meta['warning']}")
    except FileNotFoundError as e:
        print(f"warning: {e}")
    if not (cfg.root / "web" / "dist" / "index.html").exists():
        print("The React app is not built yet: API only. See docs/website.md.")
    from football_edge.web.ask import AskNox

    nox = AskNox(cfg)
    st = nox.status()
    print(f"Ask Nox: {'ready with ' + st['model'] if st['ready'] else st['problem']}")
    print(f"Open http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(create_app(cfg, svc, nox=nox), host=args.host, port=args.port, log_level="warning")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="football-edge")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="validate raw CSVs and build data/processed/matches.parquet")
    p.add_argument("--src", required=True, type=Path, help="folder with downloaded CSVs")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("evaluate", help="walk-forward evaluation of one model")
    p.add_argument("--model", required=True, choices=MODELS)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("tune", help="grid-search settings on the tuning seasons only")
    p.add_argument("--model", required=True, choices=["dixon_coles"])  # tuning is locked
    p.set_defaults(func=_tune)

    p = sub.add_parser("stack-test", help="pre-registered test: does the model beat the market?")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--market", choices=["1x2", "ou25"], default="1x2")
    p.add_argument("--level", type=float, default=0.95, help="interval level (ADR 0005: 0.9875)")
    p.add_argument(
        "--eval-seasons", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025]
    )
    p.set_defaults(func=_stack_test)

    p = sub.add_parser("compare", help="paired comparison of two models' predictions")
    p.add_argument("--a", required=True, choices=MODELS, help="baseline model")
    p.add_argument("--b", required=True, choices=MODELS, help="challenger model")
    p.add_argument("--markets", nargs="+", choices=["1x2", "ou25"], default=["1x2", "ou25"])
    p.add_argument("--seasons", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    p.add_argument("--level", type=float, default=0.95, help="interval level (ADR 0007: 0.9875)")
    p.set_defaults(func=_compare)

    p = sub.add_parser("price-test", help="pre-registered best-price vs sharp-price test")
    p.add_argument("--seasons", type=int, nargs=2, default=[2019, 2025])
    p.set_defaults(func=_price_test)

    p = sub.add_parser("early-totals-test", help="pre-registered model vs early O/U 2.5 prices")
    p.set_defaults(func=_early_totals_test)

    p = sub.add_parser("h2h-weight-test", help="pre-registered head-to-head weight test")
    p.add_argument("--years", type=int, default=3)
    p.set_defaults(func=_h2h_weight_test)

    p = sub.add_parser("bar-test", help="pre-registered probability-bar test per market")
    p.set_defaults(func=_bar_test)

    p = sub.add_parser("window-search", help="pre-registered search of every probability window")
    p.add_argument("--ou-split", action="store_true", help="over/under by side and league (0013)")
    p.add_argument("--by-league", action="store_true", help="each bet type and league (0014)")
    p.add_argument("--both", action="store_true", help="every window in both periods, best 2")
    p.add_argument("--bars", action="store_true", help="open-ended bars (X%% or more), best 2")
    p.add_argument("--above", type=float, help="with --by-league: windows from this up (0015: 0.5)")
    p.set_defaults(func=_window_search)

    p = sub.add_parser("price", help="record your site's price for a golden pick")
    p.add_argument("bar", choices=["winner", "ou", "score"], help="which golden bar")
    p.add_argument("team", help="any part of either team's name, e.g. Lazio")
    p.add_argument("odds", type=float)
    p.set_defaults(func=_price)

    p = sub.add_parser("chat", help="match-analyst chat (local LLM + our tools)")
    p.add_argument("question", nargs="*", help="ask one question and exit")
    p.set_defaults(func=_analyst)

    p = sub.add_parser("serve", help="the Nox website: API + React app on this PC")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--days", type=int, default=7, help="how many days of fixtures to show")
    p.set_defaults(func=_serve)

    p = sub.add_parser("analyst-eval", help="score the analyst on evals/analyst_questions.yaml")
    p.add_argument("--model", help="try another Ollama model, e.g. qwen3:8b")
    p.add_argument("--site", action="store_true", help="score Ask Nox (evals/nox_questions.yaml)")
    p.set_defaults(func=_analyst_eval)

    p = sub.add_parser("build-features", help="line-up and manager features from Transfermarkt")
    p.add_argument("--tm", required=True, type=Path, help="folder with the extracted dataset")
    p.set_defaults(func=_build_features)

    p = sub.add_parser("news-extract", help="LLM: structured team news from .txt articles")
    p.add_argument("--articles", required=True, type=Path, help="folder of .txt articles")
    p.add_argument("--out", type=Path, default=Path("reports/news/facts.json"))
    p.set_defaults(func=_news_extract)

    p = sub.add_parser("news-eval", help="score the LLM extractor on labelled articles")
    p.add_argument("--labels", required=True, type=Path, help="JSON-lines labelled articles")
    p.set_defaults(func=_news_eval)

    p = sub.add_parser("evaluate-counts", help="walk-forward test of corners and cards models")
    p.add_argument("--seasons", type=int, nargs=2, default=[2020, 2025])
    p.set_defaults(func=_evaluate_counts)

    p = sub.add_parser("update", help="download the latest results and odds, then ingest")
    p.add_argument("--src", type=Path, default=Path("raw data"), help="raw source folder")
    p.set_defaults(func=_update)

    p = sub.add_parser("schedule", help="full-season schedule from football-data.org")
    p.set_defaults(func=_schedule)

    p = sub.add_parser("build-confidence", help="track record behind every confidence label")
    p.set_defaults(func=_build_confidence)

    p = sub.add_parser("simulate", help="bankroll simulation of the staking policy")
    p.add_argument("--start", type=float, default=10000.0, help="starting balance")
    p.add_argument("--seasons", type=int, nargs=2, default=[2020, 2025])
    p.add_argument("--refresh", action="store_true", help="re-run the walk-forward predictions")
    p.set_defaults(func=_simulate)

    p = sub.add_parser("script", help="LLM video script(s) from fact sheets, number-checked")
    p.add_argument("--home")
    p.add_argument("--away")
    p.add_argument("--league")
    p.add_argument("--date")
    p.add_argument("--next", type=int, metavar="DAYS", help="pick this round's video matches")
    p.add_argument("--dry-run", action="store_true", help="fact sheets only, no LLM")
    p.set_defaults(func=_script)

    p = sub.add_parser("predict", help="probabilities and stakes for upcoming fixtures")
    p.add_argument("--fixtures", type=Path, help="fixtures CSV (see docs/predict.md)")
    p.add_argument(
        "--next", type=int, metavar="DAYS", help="every scheduled match in the next DAYS days"
    )
    p.add_argument("--home", help='one match: home team, e.g. "Arsenal"')
    p.add_argument("--away", help='one match: away team, e.g. "Man United"')
    p.add_argument("--league", help="E0, I1 or SP1 (only needed if the names are ambiguous)")
    p.add_argument("--date", help="match date, e.g. 2026-11-22 (default: today)")
    p.add_argument("--odds", type=float, nargs=3, metavar=("H", "D", "A"), help="site 1X2 odds")
    p.add_argument(
        "--ou-odds", type=float, nargs=2, metavar=("OVER", "UNDER"), help="site over/under odds"
    )
    p.add_argument("--odds-source", choices=["Avg", "Max", "B365", "PS"], default=None)
    p.add_argument("--allow-unknown", action="store_true", help="accept team names not in history")
    p.add_argument("--log", action="store_true", help="add the staked bets to the bet log")
    p.set_defaults(func=_predict)

    p = sub.add_parser("settle", help="mark logged bets won/lost from results, show the record")
    p.set_defaults(func=_settle)

    args = parser.parse_args(argv)
    configure_logging()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
