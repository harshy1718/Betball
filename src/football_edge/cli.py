"""Command line entry point: `football-edge <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_edge.config import load_config
from football_edge.data.loader import ingest
from football_edge.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="football-edge")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="validate raw CSVs and build data/processed/matches.parquet")
    p.add_argument("--src", required=True, type=Path, help="folder with downloaded CSVs")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    if args.command == "ingest":
        df = ingest(args.src, cfg)
        print(f"ok: {len(df)} matches, seasons {df.season.min()}-{df.season.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
