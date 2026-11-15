"""Central logging configuration. Call once from entry points, never from library code."""

import logging
import os


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("FOOTBALL_EDGE_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
