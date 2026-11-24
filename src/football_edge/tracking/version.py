"""What produced a prediction: code commit, model settings and data, as short hashes.

Logged with every prediction (ADR 0020 amendment to 0019) so any logged number can be tied to
the exact model that made it and reproduced from the repository:
  * model_commit: the git commit of the code (+dirty when src/ or configs/ have uncommitted
    changes), or "unknown" without git;
  * params_sha: SHA-256 (first 16) of the locked goals-model settings plus the calibration file;
  * data_sha: SHA-256 (first 16) of data/processed/matches.parquet, the results the model saw.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from football_edge.config import Config

_data_cache: dict[tuple[str, float, int], str] = {}


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20, check=True
    )
    return out.stdout.strip()


def model_commit(root: Path) -> str:
    try:
        commit = _git(root, "rev-parse", "--short=12", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--", "src", "configs")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


def params_sha(cfg: Config) -> str:
    model = str(cfg.predict.get("model", "dixon_coles"))
    calib = None
    if cfg.calibration.get("enabled", False):
        path = cfg.root / str(cfg.calibration.get("path", "reports/calibration/calibration.json"))
        calib = path.read_text(encoding="utf-8") if path.exists() else None
    blob = json.dumps(
        {"model": model, "params": cfg.models.get(model, {}), "calibration": calib},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def data_sha(path: Path) -> str:
    if not path.exists():
        return "none"
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    if key not in _data_cache:
        _data_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return _data_cache[key]


def stamps(cfg: Config) -> dict[str, str]:
    return {
        "model_commit": model_commit(cfg.root),
        "params_sha": params_sha(cfg),
        "data_sha": data_sha(cfg.matches_path),
    }
