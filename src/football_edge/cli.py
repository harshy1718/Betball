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

    args = parser.parse_args(argv)
    configure_logging()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
