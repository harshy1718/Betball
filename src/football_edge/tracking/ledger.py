"""An append-only, hash-chained CSV log.

Rows are only ever added, in blocks (one block = one snapshot, all rows sharing the same
`taken_at`, which is always the first column). After each block a line goes into
`chain.csv`: the block's SHA-256 and a chain hash that also covers the previous line. Editing
or deleting any old row changes its block's hash and breaks every chain hash after it, which
`verify()` reports. The log proves nothing on its own (anyone could rebuild the whole chain),
but once it is pushed to GitHub, GitHub's push record fixes when each block existed.

Hashes are taken over the text of each row without its line ending, so Windows/Unix line
endings and git's conversions do not matter.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CHAIN_COLS = ["taken_at", "file", "rows", "block_sha256", "chain_sha256"]
GENESIS = "0" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rows_text(rows: list[list[str]]) -> list[str]:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerows(rows)
    return buf.getvalue().splitlines()


@dataclass(frozen=True)
class Problem:
    taken_at: str
    file: str
    what: str


class Ledger:
    """One append-only log: monthly files `<dir>/<YYYY-MM>.csv` plus `<dir>/chain.csv` (a month
    can also have `<YYYY-MM>_v2.csv` if the columns changed during it)."""

    def __init__(self, folder: Path, columns: list[str]) -> None:
        if columns[0] != "taken_at":
            raise ValueError("the first column must be taken_at")
        self.folder = folder
        self.columns = columns

    @property
    def chain_path(self) -> Path:
        return self.folder / "chain.csv"

    def _last_chain(self) -> str:
        if not self.chain_path.exists():
            return GENESIS
        lines = [x for x in self.chain_path.read_text(encoding="utf-8").splitlines() if x]
        if len(lines) < 2:
            return GENESIS
        return lines[-1].rsplit(",", 1)[1]

    def append(self, df: pd.DataFrame, taken_at: str) -> int:
        """Add one block of rows (all stamped `taken_at`). Returns the number of rows."""
        if df.empty:
            return 0
        df = df.assign(taken_at=taken_at)
        df = df.assign(**{c: "" for c in self.columns if c not in df.columns})[self.columns]
        rows = [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
        lines = _rows_text(rows)
        block = _sha("\n".join(lines))
        prev = self._last_chain()
        chain = _sha(prev + block)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self._file_for(taken_at[:7])
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fh:
            if new:
                fh.write(",".join(self.columns) + "\n")
            fh.write("\n".join(lines) + "\n")
        cnew = not self.chain_path.exists()
        with self.chain_path.open("a", encoding="utf-8", newline="") as fh:
            if cnew:
                fh.write(",".join(CHAIN_COLS) + "\n")
            fh.write(",".join([taken_at, path.name, str(len(lines)), block, chain]) + "\n")
        return len(lines)

    def _file_for(self, month: str) -> Path:
        """`<month>.csv`, or `<month>_v2.csv`, `_v3`... when an older file of that month has
        different columns (the log gained columns): old files are never rewritten."""
        header = ",".join(self.columns)
        n = 1
        while True:
            path = self.folder / (f"{month}.csv" if n == 1 else f"{month}_v{n}.csv")
            if not path.exists():
                return path
            with path.open(encoding="utf-8") as fh:
                if fh.readline().rstrip("\r\n") == header:
                    return path
            n += 1

    def files(self) -> list[Path]:
        if not self.folder.exists():
            return []
        return sorted(p for p in self.folder.glob("*.csv") if p.name != "chain.csv")

    def read(self) -> pd.DataFrame:
        parts = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in self.files()]
        if not parts:
            return pd.DataFrame(columns=self.columns)
        out = pd.concat(parts, ignore_index=True)  # older files lack newer columns: blank
        for c in self.columns:
            if c not in out.columns:
                out[c] = ""
        return out.fillna("")

    def verify(self) -> list[Problem]:
        """Every block in the files must match its chain line, in order, and every chain hash
        must follow from the one before. Empty list = untouched."""
        problems: list[Problem] = []
        if not self.chain_path.exists():
            return problems if not self.files() else [Problem("", "", "rows but no chain.csv")]
        chain = pd.read_csv(self.chain_path, dtype=str)
        blocks: dict[tuple[str, str], list[str]] = {}
        for p in self.files():
            lines = [x.rstrip("\r") for x in p.read_text(encoding="utf-8").splitlines()][1:]
            for line in lines:
                if line:
                    blocks.setdefault((p.name, line.split(",", 1)[0]), []).append(line)
        prev = GENESIS
        seen = set()
        for r in chain.to_dict("records"):
            at, file, n = str(r["taken_at"]), str(r["file"]), int(str(r["rows"]))
            block, link = str(r["block_sha256"]), str(r["chain_sha256"])
            seen.add((file, at))
            lines = blocks.get((file, at), [])
            if len(lines) != n:
                problems.append(Problem(at, file, f"{len(lines)} rows, chain says {n}"))
            elif _sha("\n".join(lines)) != block:
                problems.append(Problem(at, file, "rows were changed"))
            if _sha(prev + block) != link:
                problems.append(Problem(at, file, "chain broken"))
            prev = link
        for key in blocks.keys() - seen:
            problems.append(Problem(key[1], key[0], "rows not in the chain"))
        return problems
