"""End to end on made-up data, no manual step: schedule -> forecast (calibrated) -> desk slips ->
prediction log with version stamps -> saved slip with the per-match cap -> results arrive ->
scorecard grades and the calibration check -> saved slip settles -> log committed and pushed."""

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from football_edge.config import load_config
from football_edge.predict.confidence import BAND
from football_edge.trader.record import goal_markets

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
TEAMS = [f"Team{i:02d}" for i in range(20)]


def _season(year: int, rng: np.random.Generator, att: np.ndarray, dfn: np.ndarray) -> list[dict]:
    """A double round robin: 38 weekends from mid-August."""
    n = len(TEAMS)
    rounds, order = [], list(range(n))
    for _ in range(n - 1):  # circle method
        rounds.append([(order[i], order[n - 1 - i]) for i in range(n // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    rounds += [[(b, a) for a, b in r] for r in rounds]
    rows = []
    start = pd.Timestamp(f"{year}-08-14")
    for k, r in enumerate(rounds):
        day = start + pd.Timedelta(days=7 * k)
        for h, a in r:
            lam, mu = np.exp(0.3 + att[h] - dfn[a]), np.exp(0.05 + att[a] - dfn[h])
            hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
            p = np.array([0.45, 0.27, 0.28])
            rows.append(
                {
                    "league": "E0",
                    "season": year,
                    "date": day,
                    "home": TEAMS[h],
                    "away": TEAMS[a],
                    "FTHG": hg,
                    "FTAG": ag,
                    "FTR": "H" if hg > ag else ("D" if hg == ag else "A"),
                    "HTHG": min(hg, 1),
                    "HTAG": min(ag, 1),
                    "HS": 12,
                    "AS": 10,
                    "HST": int(rng.poisson(4 * lam / 1.4)),
                    "AST": int(rng.poisson(4 * mu / 1.1)),
                    "HC": int(rng.poisson(5.5)),
                    "AC": int(rng.poisson(4.5)),
                    "HY": int(rng.poisson(1.8)),
                    "AY": int(rng.poisson(2.0)),
                    "HR": int(rng.random() < 0.05),
                    "AR": int(rng.random() < 0.06),
                    "close_h": 1 / p[0],
                    "close_d": 1 / p[1],
                    "close_a": 1 / p[2],
                    "close_src": "PS",
                    "pre_h": 1 / p[0],
                    "pre_d": 1 / p[1],
                    "pre_a": 1 / p[2],
                    "pre_src": "PS",
                    "referee": f"Ref{rng.integers(0, 15)}",
                }
            )
    return rows


def _track_record() -> pd.DataFrame:
    """Every market, every band: well calibrated with plenty of history (all High)."""
    markets = ["1x2", "ou25", "btts", "cards_3.5", "cards_4.5", "cards_5.5", *goal_markets()]
    rows = []
    for m in markets:
        for k in range(20):
            b = round(k * BAND, 2)
            rows.append(
                {
                    "source": "model",
                    "market": m,
                    "band": b,
                    "n": 1000,
                    "predicted": b + 0.025,
                    "hit_rate": b + 0.025,
                    "skill": True,
                    "label": "High",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(7)
    att, dfn = rng.normal(0, 0.25, 20), rng.normal(0, 0.2, 20)
    matches = pd.DataFrame([r for y in (2022, 2023, 2024) for r in _season(y, rng, att, dfn)])
    # move the calendar so the next round is tomorrow: the site and desk work from the clock
    nxt = matches.loc[matches["date"] >= "2025-03-01", "date"].min()
    shift = pd.Timestamp.now().normalize() + pd.Timedelta(days=1) - nxt
    matches["date"] = matches["date"] + shift
    played = matches[matches["date"] < nxt + shift].copy()
    upcoming = matches[matches["date"] == nxt + shift]

    cfg_raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg_raw), encoding="utf-8")
    shutil.copy(ROOT / "configs" / "team_aliases.yaml", tmp_path / "configs" / "team_aliases.yaml")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    played.to_parquet(proc / "matches.parquet", index=False)
    sched = upcoming[["league", "season", "date", "home", "away"]].copy()
    sched["kickoff_utc"] = (sched["date"] + pd.Timedelta(hours=15)).dt.tz_localize("UTC")
    sched["status"] = "TIMED"
    sched.to_parquet(proc / "schedule.parquet", index=False)
    conf = tmp_path / "reports" / "confidence"
    conf.mkdir(parents=True)
    rec = _track_record()
    rec[rec["market"].isin(["1x2", "ou25", "btts"]) | rec["market"].str.startswith("cards")].to_csv(
        conf / "track_record.csv", index=False
    )
    rec[
        ~rec["market"].isin(["1x2", "ou25", "btts"]) & ~rec["market"].str.startswith("cards")
    ].to_csv(conf / "track_record_goals.csv", index=False)
    shutil.copy(ROOT / "reports" / "confidence" / "market_skill.csv", conf / "market_skill.csv")
    cal = tmp_path / "reports" / "calibration"
    cal.mkdir(parents=True)
    (cal / "calibration.json").write_text(
        json.dumps(
            {
                "goals": {"a": 1.1, "b": 0.0, "d": 0.65, "c": 0.05, "delta": 0.1, "t_ref": 0.4},
                "cards": {"d": 0.66, "f": 0.7},
                "adopted": {"goals": True, "cards": True},
            }
        )
    )
    return tmp_path, matches, upcoming


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def test_fixtures_to_settled_slips_and_pushed_log(world):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    from football_edge.tracking import scorecard
    from football_edge.tracking.publish import publish
    from football_edge.tracking.snapshot import ledgers, take_snapshot
    from football_edge.trader.desk import Desk
    from football_edge.web.service import NoxService

    root, _matches, upcoming = world
    cfg = load_config(root / "configs" / "default.yaml")
    now = pd.Timestamp(upcoming["date"].min()).tz_localize("UTC") - pd.Timedelta(days=1)

    # 1. forecast and desk, calibrated
    svc = NoxService(cfg)
    desk = Desk(cfg, svc)
    sl = svc.slate(now)
    assert len(sl.fc) == len(upcoming) == 10
    assert sl.model.calibration is not None
    assert "corners" not in " ".join(sl.fc.columns)  # corners are no longer forecast
    board = desk.board(now)
    assert board["matches"] and any(m["singles"] for m in board["matches"])

    # 2. the log: every leg, raw chance and version stamps
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "configs")
    _git(root, "commit", "-q", "-m", "init")
    n = take_snapshot(cfg, desk, now)
    assert n["predictions"] > 0 and n["slips"] > 0
    log = ledgers(cfg)["predictions"].read()
    assert (log["model_commit"].str.len() >= 12).all() and (log["data_sha"].str.len() == 16).all()
    goal_rows = log[log["market"].isin(["1x2", "ou25", "btts"])]
    assert (goal_rows["p_raw"] != "").all() and (goal_rows["p_raw"] != goal_rows["p"]).any()
    assert take_snapshot(cfg, desk, now)["predictions"] == 0  # nothing changed: not re-logged

    # 3. save two slips on one match: the second is cut to the 4% per-match cap
    m0 = board["matches"][0]
    keys = [x["key"] for x in m0["singles"][0]["legs"]]
    first = desk.save(keys, price=None, stake=300.0)
    second = desk.check(keys)
    assert first["stake"] == 300.0 and second["stake"] <= 100.0  # 4% of 10,000 = 400
    assert "cut to stay within" in second["stake_note"]

    # 4. results arrive; grade, check calibration, settle
    later = pd.concat([pd.read_parquet(cfg.matches_path), upcoming], ignore_index=True)
    later.to_parquet(cfg.matches_path, index=False)
    card = scorecard.build(cfg, later, now=now + pd.Timedelta(days=3))
    assert card["graded_matches"] == 10 and card["log_ok"]
    assert card["calibration_check"]["matches"] == 10
    assert card["calibration_check"]["status"].startswith("Collecting")
    assert [x["league"] for x in card["by_league"]] == ["E0"]
    assert card["by_league"][0]["overall"]["n"] > 0
    hist = desk.history()
    assert hist["slips"][0]["result"] is not None

    # 5. publish: only records/ is committed, then pushed to the remote
    remote = root.parent / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "-u", "origin", "main")
    (root / "scratch.txt").write_text("not part of the log")
    msg = publish(root, now=now)
    assert "committed records/" in msg and "pushed" in msg
    files = subprocess.run(
        ["git", "-C", str(root), "show", "--name-only", "--format=", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert files and all(f.startswith("records/") for f in files)
    again = publish(root, now=now)
    assert "no new log rows" in again and "nothing to push" in again
