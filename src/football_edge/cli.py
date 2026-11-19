"""Command line entry point: `football-edge <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_edge.config import load_config
from football_edge.data.loader import ingest, load_matches
from football_edge.logging_setup import configure_logging


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
    matches = load_matches(cfg)
    folds = make_folds(matches, ev.first_test_season, ev.last_test_season, ev.min_train_matches)
    params = cfg.models.get(args.model, {})
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
    print(f"\nsaved to {out_dir}")


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

    from football_edge.markets.margin import remove_margin
    from football_edge.markets.stack import render_report, stack_test

    cfg = load_config(args.config)
    preds_path = cfg.reports_dir / "evaluation" / args.model / "predictions.parquet"
    if not preds_path.exists():
        raise SystemExit(f"run `football-edge evaluate --model {args.model}` first")
    preds = pd.read_parquet(preds_path)
    matches = load_matches(cfg)
    odds = matches.iloc[preds["row"].to_numpy()][["close_h", "close_d", "close_a"]].to_numpy()
    ok = ~pd.isna(odds).any(axis=1)
    if not ok.all():
        print(f"note: {int((~ok).sum())} matches without closing odds were left out")
    preds, odds = preds[ok].reset_index(drop=True), odds[ok].astype(float)
    p_market = remove_margin(odds)
    p_model = preds[["p_h", "p_d", "p_a"]].to_numpy()
    res = stack_test(preds, p_market, p_model, eval_seasons=args.eval_seasons)
    out = cfg.reports_dir / "stack_test" / args.model
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(res, args.model)
    (out / "verdict.md").write_text(report, encoding="utf-8")
    summary = {k: v for k, v in res.__dict__.items() if k != "criteria"}
    summary["criteria"] = res.criteria
    summary["passed"] = res.passed
    (out / "verdict.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(report)
    print(f"saved to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="football-edge")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="validate raw CSVs and build data/processed/matches.parquet")
    p.add_argument("--src", required=True, type=Path, help="folder with downloaded CSVs")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("evaluate", help="walk-forward evaluation of one model")
    p.add_argument("--model", required=True, choices=["team_blind", "elo", "dixon_coles"])
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("tune", help="grid-search settings on the tuning seasons only")
    p.add_argument("--model", required=True, choices=["dixon_coles"])
    p.set_defaults(func=_tune)

    p = sub.add_parser("stack-test", help="pre-registered test: does the model beat the market?")
    p.add_argument("--model", required=True, choices=["team_blind", "elo", "dixon_coles"])
    p.add_argument(
        "--eval-seasons", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025]
    )
    p.set_defaults(func=_stack_test)

    args = parser.parse_args(argv)
    configure_logging()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
