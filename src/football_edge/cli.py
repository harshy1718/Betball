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
    "dixon_coles_dynamic",
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
    out = cfg.reports_dir / (args.out or "prob_bar")
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
    ev = cfg.reports_dir / "evaluation" / "dixon_coles"
    if args.calibrated:  # the chances the site and desk show (ADR 0020): from 2019/20 only
        preds = pd.read_parquet(ev / "predictions_calibrated.parquet")
        window_search.DISCOVERY = (2019, 2022)
    else:
        preds = pd.read_parquet(ev / "predictions.parquet")
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
        report = window_both.run(m, lo_min=args.above or 0.20, shuffles=args.shuffles)
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
    ) if not args.out else cfg.reports_dir / args.out  # fmt: skip
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
    cs = cfg.staking.get("golden_cs") or {}
    for d in done:
        print(f"saved: {d}")
        lg = d.rsplit("(", 1)[1].split(")", 1)[0] if "(" in d else ""
        low = golden.cs_min_odds(cs, lg)
        if args.bar == "score" and args.odds < low:
            print(f"  note: below {low:.2f}, this league's break-even for the bar: skip it")


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
    preds.to_parquet(out / f"{args.out_name}.parquet", index=False)
    summary = summarise_counts(preds)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.5f")
    print(summary.round(4).to_string(index=False))
    for stat, line in (("corners", 9.5), ("yellows", 4.5), ("reds", 0.5)):
        print(f"\ncalibration, {stat} over {line}:")
        print(calibration(preds, stat, line).round(3).to_string())
    print(f"\nsaved to {out}")


def _build_matches(args: argparse.Namespace) -> None:
    """Rebuild data/processed/matches.parquet from the committed raw files (no downloads)."""
    import json

    from football_edge.data.loader import build_matches

    cfg = load_config(args.config)
    manifest = json.loads((cfg.raw_dir / "manifest.json").read_text(encoding="utf-8"))
    matches = build_matches([cfg.raw_dir / str(m["file"]) for m in manifest], cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(cfg.matches_path, index=False)
    print(f"ok: {len(matches)} matches from {len(manifest)} raw files")


def _current_rho(cfg: Any) -> float:
    """rho of the goals model fitted on every match (used to rebuild score grids)."""
    from football_edge.models.dixon_coles import DixonColes, DixonColesParams

    dc = DixonColes(DixonColesParams.from_dict(dict(cfg.models.get("dixon_coles", {}))))
    dc.fit(load_matches(cfg))
    return float(dc.rho)


def _calibrate(args: argparse.Namespace) -> None:
    from football_edge.evaluation.calibration_study import run

    cfg = load_config(args.config)
    print(run(cfg.reports_dir, _current_rho(cfg), load_matches(cfg)))
    print("next: `football-edge build-confidence` so the labels describe the calibrated numbers")


def _confidence_inputs(cfg: Any) -> tuple[Any, Any, bool]:
    """(goals predictions, count predictions, calibrated?) behind the confidence labels: the
    calibrated walk-forward predictions when calibration is on and adopted (ADR 0020), so the
    labels describe the numbers the site and desk actually show."""
    import pandas as pd

    from football_edge.predict.forecast import load_calibration

    ev = cfg.reports_dir / "evaluation"
    goals = pd.read_parquet(ev / "dixon_coles" / "predictions.parquet")
    cpath = ev / "counts" / "predictions.parquet"
    counts = pd.read_parquet(cpath) if cpath.exists() else None
    goals_cal, cards_cal = load_calibration(cfg)
    used = False
    gcal = ev / "dixon_coles" / "predictions_calibrated.parquet"
    if goals_cal is not None and gcal.exists():
        goals, used = pd.read_parquet(gcal), True
    ccal = ev / "counts" / "predictions_calibrated.parquet"
    if cards_cal is not None and counts is not None and ccal.exists():
        cards = pd.read_parquet(ccal)
        counts = pd.concat([counts[counts["stat"] != "cards"], cards[counts.columns]],
                           ignore_index=True)  # fmt: skip
        used = True
    return goals, counts, used


def _replay_desk(args: argparse.Namespace) -> None:
    """Replay the desk's slip rules with walk-forward labels and lifts (no future data)."""
    from football_edge.evaluation.desk_replay import per_season, replay, summarise

    cfg = load_config(args.config)
    goals, counts, calibrated = _confidence_inputs(cfg)
    matches = load_matches(cfg)
    lo, hi = args.seasons
    slips = replay(goals, counts, matches, dict(cfg.trader), _current_rho(cfg), (lo, hi),
                   workers=args.workers)  # fmt: skip
    out = cfg.reports_dir / "replay"
    out.mkdir(parents=True, exist_ok=True)
    slips.to_parquet(out / "slips.parquet", index=False)
    summ, seas = summarise(slips), per_season(slips)
    summ.to_csv(out / "summary.csv", index=False)
    seas.to_csv(out / "per_season.csv", index=False)
    kind = "calibrated" if calibrated else "raw"
    head = (f"Desk replay {lo}/{(lo + 1) % 100:02d}-{hi}/{(hi + 1) % 100:02d}, "
            f"{kind} chances, walk-forward labels and lifts.")  # fmt: skip
    text = head + "\n\n" + summ.to_string(index=False) + "\n\n" + seas.to_string(index=False)
    (out / "verdict.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def _referee_test(args: argparse.Namespace) -> None:
    from football_edge.evaluation.referee_test import run

    cfg = load_config(args.config)
    text, rows = run(load_matches(cfg))
    out = cfg.reports_dir / "referee_cards"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.md").write_text(text + "\n", encoding="utf-8")
    rows.to_parquet(out / "losses.parquet", index=False)
    print(text)


def _tune_per_league(args: argparse.Namespace) -> None:
    """ADR 0023 stage A tuning: the combined model's grid, per league, on 2018-19 only."""
    from football_edge.evaluation.folds import make_folds
    from football_edge.evaluation.per_league import tune

    cfg = load_config(args.config)
    ev = cfg.evaluation
    matches = load_matches(cfg)
    folds = make_folds(matches, ev.first_test_season, ev.last_test_season, ev.min_train_matches)
    t = cfg.tuning["dixon_coles"]
    table = tune(matches, folds, cfg.models["dixon_coles"], t["grid"], tuple(t["seasons"]),
                 workers=args.workers)  # fmt: skip
    out = cfg.reports_dir / "tuning" / "dixon_coles_per_league.csv"
    table.to_csv(out, index=False, float_format="%.5f")
    best = table.groupby("league").head(1)
    print(best.to_string(index=False))
    print(f"saved to {out}")


def _per_league_calibration(args: argparse.Namespace) -> None:
    """ADR 0023 stage B: one calibration map per league against the shared map."""
    import pandas as pd

    from football_edge.evaluation.per_league import (
        calibrate_by_league,
        calibrated_losses,
        confirm_2026,
    )
    from football_edge.models.calibration import block_bootstrap

    cfg = load_config(args.config)
    ev = cfg.reports_dir / "evaluation" / "dixon_coles"
    raw = pd.read_parquet(ev / "predictions.parquet")
    shared = pd.read_parquet(ev / "predictions_calibrated.parquet")
    rho = _current_rho(cfg)
    per = calibrate_by_league(raw, rho)
    key = ["row"]
    both = shared.merge(per[[*key, "exp_home", "exp_away", "cal_delta"]], on=key,
                        suffixes=("", "_pl"))  # fmt: skip
    both = both[both["season"] >= 2020].reset_index(drop=True)
    a = calibrated_losses(both, rho)
    b = calibrated_losses(both.assign(exp_home=both["exp_home_pl"], exp_away=both["exp_away_pl"],
                                      cal_delta=both["cal_delta_pl"]), rho)  # fmt: skip
    snap = pd.to_datetime(both["snapshot"]).dt.strftime("%Y-%m-%d")
    lo, hi = block_bootstrap(b - a, (both["league"].astype(str) + "|" + snap).to_numpy())
    a26, b26 = confirm_2026(raw, pd.read_parquet(ev / "predictions_2026.parquet"), rho)
    lines = ["# One calibration map per league against the shared map (ADR 0023, stage B)", "",
             "Sum of match result + over/under 2.5 + both-score log loss per match.", "",
             "| League | shared map | map per league |", "|---|---|---|"]  # fmt: skip
    for lg, g in both.assign(a=a, b=b).groupby("league"):
        lines.append(f"| {lg} | {g['a'].mean():.5f} | {g['b'].mean():.5f} |")
    passed = hi < 0 and b26.mean() < a26.mean()
    lines += ["", f"2020/21-2025/26: {a.mean():.5f} -> {b.mean():.5f}, difference "
              f"{(b - a).mean():+.5f} (95% interval {lo:+.5f} to {hi:+.5f}; {len(a):,} matches).",
              f"2026/27 so far: {a26.mean():.5f} -> {b26.mean():.5f} ({len(a26)} matches).", "",
              "**" + ("PASS: one map per league" if passed else "FAIL: the shared map stays")
              + ".**"]  # fmt: skip
    text = "\n".join(lines)
    out = cfg.reports_dir / "calibration" / "per_league_verdict.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


def _league_check(args: argparse.Namespace) -> None:
    """ADR 0024: may new leagues join the model?"""
    from football_edge.evaluation.new_leagues import check

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    present = sorted(matches["league"].unique())
    old = [x for x in present if x not in args.new]
    text, _ = check(matches, cfg.models["dixon_coles"], old, list(args.new),
                    cfg.evaluation.min_train_matches)  # fmt: skip
    out = cfg.reports_dir / "new_leagues" / "verdict.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


def _promoted_spread(args: argparse.Namespace) -> None:
    """ADR 0026 (promoted teams) and ADR 0027 (less spread) against the live model."""
    import pandas as pd

    from football_edge.evaluation.challenger import run_models
    from football_edge.evaluation.promoted_spread import (
        compare,
        maps_from,
        promoted_sides,
        run_part,
        secondary,
    )

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    ev = cfg.reports_dir / "evaluation" / "dixon_coles"
    champ = pd.concat(
        [pd.read_parquet(ev / "predictions.parquet"),
         pd.read_parquet(ev / "predictions_2026.parquet")],
        ignore_index=True,
    )  # fmt: skip
    out = cfg.reports_dir / "challenger" / "promoted"
    out.mkdir(parents=True, exist_ok=True)
    b_path = out / "walk_forward_b.parquet"
    if not b_path.exists():
        params = {**cfg.models["dixon_coles"], "promoted_pull": 0.5}
        r = run_models(matches, {"dixon_coles_promoted": params}, 2018, 2026,
                       cfg.evaluation.min_train_matches)  # fmt: skip
        r["dixon_coles_promoted"].to_parquet(b_path)
    b_preds = pd.read_parquet(b_path)
    rows = {"champion": run_part("champion", matches, champ, None, out)}
    base = maps_from(pd.read_csv(out / "champion_maps.csv"))
    for part in ("A", "B", "spread"):
        rows[part] = run_part(part, matches, champ, b_preds, out, base)
    level = 1 - 0.05 / 3
    sides = promoted_sides(matches.reset_index(drop=True))
    names = {"A": "A: promoted stretch in the calibration (ADR 0026)",
             "B": "B: weaker pull to the promoted prior (ADR 0026)",
             "spread": "Less spread, Conway-Maxwell-Poisson (ADR 0027)"}  # fmt: skip
    res = {k: compare(rows["champion"], rows[k], matches, level) for k in names}
    sec = {k: secondary(v, sides) for k, v in rows.items()}
    maps = {k: pd.read_csv(out / f"{k}_maps.csv") for k in ("A", "spread")}
    lines = ["# Promoted teams (ADR 0026) and goal spread (ADR 0027)", "",
             "Per match: match result + over/under 2.5 + both-score log loss, lower is better. "
             "Champion = the live model (Dixon-Coles + five-number calibration). Interval: 98.3% "
             "(Bonferroni for 3), blocks = league x week.", "",
             "| Challenger | Matches | Live model | Challenger | Difference | 98.3% interval "
             "| 2026/27 difference | Verdict |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, name in names.items():
        q = res[k]
        lines.append(
            f"| {name} | {q['n']:,} | {q['champ']:.5f} | {q['chall']:.5f} | {q['diff']:+.5f} | "
            f"{q['lo']:+.5f} to {q['hi']:+.5f} | {q['diff_2026']:+.5f} ({q['n_2026']}) | "
            f"{'PASS' if q['pass'] else 'FAIL'} |"
        )
    lines += ["", "Secondary (2020/21 onwards, not part of the rule):", "",
              "| Model | Promoted sides, matches 1-19: goal-difference residual | Spread home | "
              "Spread away | 0-0 predicted | 0-0 observed |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("champion", *names):
        s = sec[k]
        lines.append(f"| {k} | {s['promoted_residual']:+.3f} | {s['spread_home']:.3f} | "
                     f"{s['spread_away']:.3f} | {s['p00']:.4f} | {s['obs00']:.4f} |")  # fmt: skip
    a_p = maps["A"][["season", "a", "a_p"]].round(3).to_dict("records")
    nu = maps["spread"][["season", "nu"]].round(3).to_dict("records")
    lines += ["", f"Fitted a (others) and a_p (promoted-early) per season: {a_p}",
              f"Fitted nu per season (1 = Poisson, above 1 = less spread): {nu}"]  # fmt: skip
    passed = [names[k] for k in names if res[k]["pass"]]
    lines += ["", "**" + ("PASS: " + "; ".join(passed) if passed
                          else "FAIL: the live model stays") + ".**"]  # fmt: skip
    text = "\n".join(lines)
    (out / "verdict.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def _diagnose(args: argparse.Namespace) -> None:
    """Where the live model misses: residual checks on walk-forward predictions."""
    import pandas as pd

    from football_edge.evaluation.diagnostics import report

    cfg = load_config(args.config)
    ev = cfg.reports_dir / "evaluation" / "dixon_coles"
    champ = pd.concat(
        [pd.read_parquet(ev / "predictions.parquet"),
         pd.read_parquet(ev / "predictions_2026.parquet")],
        ignore_index=True,
    )  # fmt: skip
    text = report(load_matches(cfg), champ)
    out = cfg.reports_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "residuals.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def _challenge_all(args: argparse.Namespace) -> None:
    """ADR 0025: other model families against the locked Dixon-Coles model."""
    import numpy as np
    import pandas as pd

    from football_edge.evaluation.challenger import match_losses, run_models
    from football_edge.evaluation.ml_challenge import MARKET_COLS, build, run_all
    from football_edge.models.calibration import block_bootstrap

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    ev = cfg.reports_dir / "evaluation" / "dixon_coles"
    champ = pd.concat(
        [pd.read_parquet(ev / "predictions.parquet"),
         pd.read_parquet(ev / "predictions_2026.parquet")],
        ignore_index=True,
    )  # fmt: skip
    df = build(matches, champ)
    out = cfg.reports_dir / "challenger" / "model_families"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "features.parquet")
    level = 1 - 0.05 / 8
    m = matches.reset_index(drop=True)
    week = (m["date"] - pd.to_timedelta(m["date"].dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    block = (m["league"].astype(str) + "|" + week)

    def losses(p: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        return match_losses(p.assign(goals_home=m.loc[rows, "FTHG"].to_numpy(int),
                                     goals_away=m.loc[rows, "FTAG"].to_numpy(int)))  # fmt: skip

    results = []

    def judge(name: str, ch: pd.DataFrame, cp: pd.DataFrame) -> None:
        rows = ch.index.to_numpy()
        a, b = losses(cp.loc[rows, MARKET_COLS], rows), losses(ch[MARKET_COLS], rows)
        season = m.loc[rows, "season"].to_numpy()
        main, conf = (season >= 2020) & (season <= 2025), season == 2026
        d = (b["sum"] - a["sum"]).to_numpy()
        lo, hi = block_bootstrap(d[main], block.loc[rows].to_numpy()[main], level=level)
        results.append({
            "challenger": name,
            **{f"champ_{k}": float(a[k][main].mean()) for k in ("1x2", "ou25", "btts", "sum")},
            **{f"chall_{k}": float(b[k][main].mean()) for k in ("1x2", "ou25", "btts", "sum")},
            "diff": float(d[main].mean()), "lo": lo, "hi": hi, "diff_2026": float(d[conf].mean()),
            "pass": bool(hi < 0 and d[conf].mean() < 0),
        })  # fmt: skip

    cp = champ.set_index("row")
    for name, p in run_all(df, list(range(2020, 2027))).items():
        judge(name, p, cp)
    mt = cfg.evaluation.min_train_matches
    bv_parts: list[pd.DataFrame] = []
    bv_champ: list[pd.DataFrame] = []
    for lo_s, hi_s in ((2020, 2025), (2026, 2026)):
        pair = {"dixon_coles": cfg.models["dixon_coles"],
                "bivariate_poisson": cfg.models["dixon_coles"]}  # fmt: skip
        r = run_models(matches, pair, lo_s, hi_s, mt)
        r["bivariate_poisson"].to_parquet(out / f"bivariate_{lo_s}.parquet")
        bv_parts.append(r["bivariate_poisson"])
        bv_champ.append(r["dixon_coles"])
    judge("bivariate Poisson", pd.concat(bv_parts).set_index("row"),
          pd.concat(bv_champ).set_index("row"))  # fmt: skip
    bench = run_models(matches, {"elo": cfg.models.get("elo", {}), "glicko": {}}, 2020, 2025, mt)
    rows_b = bench["elo"]["row"].to_numpy()
    y = m.loc[rows_b, "FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    def ll(p: np.ndarray) -> float:
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))

    champ_1x2 = ll(cp.loc[rows_b, ["p_h", "p_d", "p_a"]].to_numpy())
    bench_lines = [
        f"| {n} | {ll(bench[n][['p_h', 'p_d', 'p_a']].to_numpy()):.5f} | {champ_1x2:.5f} |"
        for n in ("elo", "glicko")
    ]
    t = pd.DataFrame(results)
    t.to_csv(out / "results.csv", index=False)
    lines = ["# Other model families against Dixon-Coles (ADR 0025)", "",
             "Log loss per match, 2020/21-2025/26 (lower is better). Interval: 99.4% "
             "(Bonferroni for 8), blocks = league x week.", "",
             "| Challenger | Result | Over/under | Both score | Sum | Dixon-Coles sum | Difference "
             "| 99.4% interval | 2026/27 difference | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for row in t.sort_values("diff").to_dict("records"):
        lines.append(
            f"| {row['challenger']} | {row['chall_1x2']:.4f} | {row['chall_ou25']:.4f} | "
            f"{row['chall_btts']:.4f} | {row['chall_sum']:.4f} | {row['champ_sum']:.4f} | "
            f"{row['diff']:+.4f} | {row['lo']:+.4f} to {row['hi']:+.4f} | "
            f"{row['diff_2026']:+.4f} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines += ["", f"Dixon-Coles alone: result {t['champ_1x2'].iloc[0]:.4f}, over/under "
              f"{t['champ_ou25'].iloc[0]:.4f}, both score {t['champ_btts'].iloc[0]:.4f}.", "",
              "Benchmarks (match result only, weekly refits):", "",
              "| Model | Result log loss | Dixon-Coles |", "|---|---|---|", *bench_lines, "",
              "**" + ("PASS: " + ", ".join(t[t["pass"]]["challenger"]) if t["pass"].any()
                      else "FAIL: Dixon-Coles stays") + ".**"]  # fmt: skip
    text = "\n".join(lines)
    (out / "verdict.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def _challenge(args: argparse.Namespace) -> None:
    """Locked model against a challenger on the same weekly folds (ADR 0022)."""
    from football_edge.evaluation.challenger import compare, run_models, verdict

    cfg = load_config(args.config)
    matches = load_matches(cfg)
    models = {"dixon_coles": cfg.models["dixon_coles"], args.model: cfg.models[args.model]}
    mt = cfg.evaluation.min_train_matches
    main = run_models(matches, models, 2020, 2025, mt)
    conf = run_models(matches, models, 2026, 2026, mt)
    m_cmp = compare(main["dixon_coles"], main[args.model])
    c_cmp = compare(conf["dixon_coles"], conf[args.model])
    text, _ = verdict(m_cmp, c_cmp, "dixon_coles", args.model)
    out = cfg.reports_dir / "challenger" / args.model
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {**{f"{k}_2020_25": v for k, v in main.items()},
                     **{f"{k}_2026": v for k, v in conf.items()}}.items():  # fmt: skip
        df.to_parquet(out / f"predictions_{name}.parquet", index=False)
    (out / "verdict.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def _build_confidence(args: argparse.Namespace) -> None:
    from football_edge.predict.confidence import build_track_record
    from football_edge.trader.record import build_goal_record

    cfg = load_config(args.config)
    if not (cfg.reports_dir / "evaluation" / "dixon_coles" / "predictions.parquet").exists():
        raise SystemExit("run `football-edge evaluate --model dixon_coles` first")
    goals, counts, calibrated = _confidence_inputs(cfg)
    table = build_track_record(goals, counts, load_matches(cfg))
    out = cfg.reports_dir / "confidence"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "track_record.csv", index=False, float_format="%.4f")
    extra = build_goal_record(goals, _current_rho(cfg))  # the trader desk's extra goal bets
    extra.to_csv(out / "track_record_goals.csv", index=False, float_format="%.4f")
    print(f"extra goal bets: {extra['market'].nunique()} markets, saved alongside")
    print("labels built from " + ("CALIBRATED" if calibrated else "raw")
          + " walk-forward predictions, 2020/21-2025/26")  # fmt: skip
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
    from football_edge.predict.forecast import (
        count_forecasts,
        fit_for,
        forecast,
        load_calibration,
        market_view,
    )

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
    goals_cal, cards_cal = load_calibration(cfg)
    model.calibration = goals_cal
    fc = market_view(forecast(model, fixtures))
    fc = pd.concat([fc, count_forecasts(history, fixtures, cards_cal=cards_cal)], axis=1)
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
            picks = golden.cs_picks(fc, float(conf.get("lo", 0.12)), conf.get("leagues"))
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
    app = create_app(cfg, svc, nox=nox, track=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _snapshot_now(cfg: Any, days: int = 7) -> dict[str, int]:
    from football_edge.tracking.snapshot import take_snapshot
    from football_edge.trader.desk import Desk
    from football_edge.web.service import NoxService

    svc = NoxService(cfg, days=days)
    return take_snapshot(cfg, Desk(cfg, svc))


def _print_scorecard(cfg: Any) -> None:
    from football_edge.tracking import scorecard

    card = scorecard.build(cfg, load_matches(cfg))
    print(f"logged since {card['since'] or '-'}: {card['logged_matches']} matches, "
          f"{card['graded_matches']} graded")  # fmt: skip
    for m in card["markets"]:
        hit = f"{m['hit']:.0%}" if m["hit"] is not None else "-"
        print(f"  {m['name']:<26} {m['n']:>5} graded, came in {hit:>4}; {m['verdict']}")
    for s in card["slips"]:
        print(f"  {s['title']:<26} {s['won']}/{s['n']} came in (expected {s['expected']:.0%})")
    if not card["log_ok"]:
        print("WARNING: the prediction log was changed after it was written:")
        for p in card["log_problems"]:
            print(f"  {p['file']} {p['taken_at']}: {p['what']}")


def _daily(args: argparse.Namespace) -> None:
    """The morning run: fresh results and schedule, log every prediction before kickoff,
    grade what has been played, settle the golden-call logs."""
    cfg = load_config(args.config)
    if not args.no_update:
        try:
            _update(args)
        except (Exception, SystemExit) as e:  # keep going: logging still works on old data
            print(f"update failed, carrying on with the data we have: {e}")
    n = _snapshot_now(cfg)
    print(f"logged {n['predictions']} prediction rows and {n['slips']} desk slip rows")
    try:
        _settle(args)
    except (Exception, SystemExit) as e:
        print(f"settle skipped: {e}")
    _print_scorecard(cfg)
    if cfg.tracking.get("auto_push", False) and not getattr(args, "no_push", False):
        from football_edge.tracking.publish import publish

        print(publish(cfg.root))
    else:
        print("To make today's log public proof: commit and push records/ (see docs/tracking.md).")


def _scorecard(args: argparse.Namespace) -> None:
    _print_scorecard(load_config(args.config))


def _records_verify(args: argparse.Namespace) -> None:
    from football_edge.tracking.snapshot import ledgers

    cfg = load_config(args.config)
    bad = 0
    for name, book in ledgers(cfg).items():
        problems = book.verify()
        bad += len(problems)
        rows = len(book.read())
        print(f"{name}: {rows} rows, {'untouched' if not problems else 'CHANGED'}")
        for p in problems:
            print(f"  {p.file} {p.taken_at}: {p.what}")
    if bad:
        raise SystemExit(1)


def _schedule_daily(args: argparse.Namespace) -> None:
    """Register `football-edge daily` in Windows Task Scheduler: runs hidden (no window to
    close by accident) when the PC is on; a missed run starts as soon as the PC is back on.
    Running it again replaces the task."""
    import platform
    import re
    import subprocess
    import sys

    cfg = load_config(args.config)
    if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", args.time):
        raise SystemExit("--time must look like 10:30")
    folder = cfg.root / "data" / "interim" / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    runner = folder / "nox_daily.cmd"
    runner.write_text(
        "@echo off\r\n"
        f'cd /d "{cfg.root}"\r\n'
        f'echo ===== %date% %time% >> "{folder / "daily.log"}"\r\n'
        f'"{sys.executable}" -m football_edge.cli daily >> "{folder / "daily.log"}" 2>&1\r\n',
        encoding="utf-8",
    )
    print(f"runner written: {runner}")
    if platform.system() != "Windows":
        print("not on Windows: schedule that file with cron instead")
        return
    ps = (
        # conhost --headless: no console window, so the run cannot be closed by accident
        f"$a = New-ScheduledTaskAction -Execute 'conhost.exe' "
        f"-Argument '--headless cmd.exe /c \"{runner}\"';"
        f"$t = New-ScheduledTaskTrigger -Daily -At {args.time};"
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries "
        "-AllowStartIfOnBatteries;"
        "Register-ScheduledTask -TaskName 'Nox daily refresh' -Action $a -Trigger $t "
        "-Settings $s -Force | Out-Null"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"could not register the task: {r.stderr.strip()}")
    print(f"scheduled: 'Nox daily refresh' every day at {args.time}, hidden; "
          f"log: {folder / 'daily.log'}")  # fmt: skip


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
    # dixon_coles is tuned and locked; dixon_coles_dynamic is the ADR 0022 challenger
    p.add_argument("--model", required=True, choices=["dixon_coles", "dixon_coles_dynamic"])
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
    p.add_argument("--out", help="report folder under reports/ (default: prob_bar)")
    p.set_defaults(func=_bar_test)

    p = sub.add_parser("window-search", help="pre-registered search of every probability window")
    p.add_argument("--ou-split", action="store_true", help="over/under by side and league (0013)")
    p.add_argument("--by-league", action="store_true", help="each bet type and league (0014)")
    p.add_argument("--both", action="store_true", help="every window in both periods, best 2")
    p.add_argument("--bars", action="store_true", help="open-ended bars (X%% or more), best 2")
    p.add_argument("--above", type=float, help="with --by-league: windows from this up (0015: 0.5)")
    p.add_argument("--shuffles", type=int, default=1,
                   help="with --both: repeat the luck control this many times (e.g. 200)")
    p.add_argument("--calibrated", action="store_true",
                   help="search the calibrated chances the site shows (from 2019/20)")
    p.add_argument("--out", help="report folder under reports/ (default: by search type)")
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
    p.add_argument("--seasons", type=int, nargs=2, default=[2018, 2025])
    p.add_argument("--out-name", default="predictions",
                   help="file name under reports/evaluation/counts (no .parquet)")  # fmt: skip
    p.set_defaults(func=_evaluate_counts)

    p = sub.add_parser("update", help="download the latest results and odds, then ingest")
    p.add_argument("--src", type=Path, default=Path("raw data"), help="raw source folder")
    p.set_defaults(func=_update)

    p = sub.add_parser("schedule", help="full-season schedule from football-data.org")
    p.set_defaults(func=_schedule)

    p = sub.add_parser("build-matches", help="rebuild the processed table from data/raw only")
    p.set_defaults(func=_build_matches)

    p = sub.add_parser("tune-per-league", help="tune each league's own settings (ADR 0023)")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=_tune_per_league)

    p = sub.add_parser("per-league-calibration", help="calibration map per league (ADR 0023)")
    p.set_defaults(func=_per_league_calibration)

    p = sub.add_parser("promoted-spread", help="promoted teams and goal spread (0026, 0027)")
    p.set_defaults(func=_promoted_spread)

    p = sub.add_parser("diagnose", help="where the live model misses (residual checks)")
    p.set_defaults(func=_diagnose)

    p = sub.add_parser("challenge-all", help="ML and other model families vs Dixon-Coles (0025)")
    p.set_defaults(func=_challenge_all)

    p = sub.add_parser("league-check", help="may new leagues join the model? (ADR 0024)")
    p.add_argument("--new", nargs="+", default=["D1", "F1"])
    p.set_defaults(func=_league_check)

    p = sub.add_parser("challenge", help="locked goals model against a challenger (ADR 0022)")
    p.add_argument("--model", default="dixon_coles_dynamic")
    p.set_defaults(func=_challenge)

    p = sub.add_parser("referee-test", help="pre-registered referee test for cards (ADR 0021)")
    p.set_defaults(func=_referee_test)

    p = sub.add_parser("replay-desk", help="replay the trader desk with no future data")
    p.add_argument("--seasons", type=int, nargs=2, default=[2020, 2025])
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=_replay_desk)

    p = sub.add_parser("calibrate", help="pre-registered calibration test and live maps (ADR 0020)")
    p.set_defaults(func=_calibrate)

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
    p.add_argument("--league", help="E0, I1, SP1, D1 or F1 (only needed if names are ambiguous)")
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

    p = sub.add_parser("daily", help="morning run: update, log predictions, grade, scorecard")
    p.add_argument("--src", type=Path, default=Path("raw data"), help="raw source folder")
    p.add_argument("--no-update", action="store_true", help="skip the download step")
    p.add_argument("--no-push", action="store_true", help="do not commit and push records/")
    p.set_defaults(func=_daily)

    p = sub.add_parser("schedule-daily", help="run `daily` every morning (Windows Task Scheduler)")
    p.add_argument("--time", default="10:30", help="local time, e.g. 10:30")
    p.set_defaults(func=_schedule_daily)

    p = sub.add_parser("scorecard", help="live accuracy of every logged prediction")
    p.set_defaults(func=_scorecard)

    p = sub.add_parser("records-verify", help="check the prediction log was never edited")
    p.set_defaults(func=_records_verify)

    args = parser.parse_args(argv)
    configure_logging()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
