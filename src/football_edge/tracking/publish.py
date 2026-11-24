"""Commit and push the public log after the daily run, so GitHub's push time dates every block.

Only `records/` is committed (`git commit -- records`), whatever else is staged or changed, so
Harsh's own work in progress is never swept into these commits. A failed push is reported, not
fatal: the commit stays local and the next successful push carries it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def _git(root: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def publish(
    root: Path, folder: str = "records", push: bool = True, now: pd.Timestamp | None = None
) -> str:
    """Commit `folder` if it changed, then push. Returns one line saying what happened."""
    now = now or pd.Timestamp.now(tz="UTC")
    try:
        if _git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
            return "publish skipped: not a git repository"
        _git(root, "add", "--", folder)
        if _git(root, "diff", "--cached", "--quiet", "--", folder).returncode == 0:
            msg = "publish: no new log rows to commit"
        else:
            stamp = now.strftime("%Y-%m-%d %H:%M UTC")
            c = _git(root, "commit", "-m", f"records: prediction log {stamp}", "--", folder)
            if c.returncode != 0:
                return f"publish failed at commit: {(c.stderr or c.stdout).strip()[:300]}"
            msg = f"publish: committed {folder}/ ({stamp})"
        if not push:
            return msg
        ahead = _git(root, "rev-list", "--count", "@{u}..HEAD")
        if ahead.returncode == 0 and ahead.stdout.strip() == "0":
            return msg + "; nothing to push"
        p = _git(root, "push", timeout=180)
        if p.returncode != 0:
            return (
                msg + f"; push FAILED (will retry next run): {(p.stderr or p.stdout).strip()[:300]}"
            )
        return msg + "; pushed to GitHub"
    except (OSError, subprocess.SubprocessError) as e:
        return f"publish failed: {e}"
