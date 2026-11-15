"""Save an evaluation run so it can be traced to exact code, config and data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def save_run(
    out_dir: Path,
    preds: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    model: str,
    params: dict[str, object],
    root: Path,
    data_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv", float_format="%.5f")
    meta = {
        "model": model,
        "params": params,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(root),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "n_predictions": len(preds),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
