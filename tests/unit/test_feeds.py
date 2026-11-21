from pathlib import Path

import pandas as pd
import pytest

from football_edge.config import load_config
from football_edge.data.loader import ingest
from football_edge.data.schedule import (
    ScheduleError,
    build_name_map,
    fetch_schedule,
    norm,
    upcoming,
)
from football_edge.data.sources import download_results, season_code
from football_edge.predict.fixtures import from_schedule

ROOT = Path(__file__).resolve().parents[2]
HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"


def _csv(league, n, start="22/08/2026"):
    d0 = pd.Timestamp(pd.to_datetime(start, dayfirst=True))
    rows = [f"{league},{(d0 + pd.Timedelta(days=7 * i)):%d/%m/%Y},T{i}a,T{i}b,1,0,H"
            for i in range(n)]  # fmt: skip
    return (HEADER + "\n".join(rows) + "\n").encode()


def test_ingest_keeps_the_biggest_file_per_season(tmp_path):
    src = tmp_path / "raw data"
    src.mkdir()
    (src / "E0 (11).csv").write_bytes(_csv("E0", 3))
    (src / "E0_2026_20261123.csv").write_bytes(_csv("E0", 5))
    cfg = load_config(ROOT / "configs" / "default.yaml", root=tmp_path)
    matches = ingest(src, cfg)
    assert len(matches) == 5  # not 8: the older, smaller copy is ignored


def test_download_results_validates(tmp_path):
    def fetch(url):
        if url.endswith("/E0.csv"):
            return _csv("E0", 6)
        if url.endswith("/I1.csv"):
            return _csv("SP1", 6)  # wrong league in the file
        raise OSError("timeout")

    out = {
        d.league: d
        for d in download_results(
            tmp_path, 2026, ("E0", "I1", "SP1"), {"E0": 4}, fetch=fetch, stamp="20261123"
        )
    }
    assert out["E0"].path is not None and out["E0"].note == "6 matches (+2 new)"
    assert out["I1"].path is None and "wrong league" in out["I1"].note
    assert out["SP1"].path is None and "download failed" in out["SP1"].note
    shrunk = download_results(tmp_path, 2026, ("E0",), {"E0": 10}, fetch=fetch, stamp="x")[0]
    assert shrunk.path is None and "fewer" in shrunk.note
    assert season_code(2026) == "2627" and season_code(2099) == "9900"


KNOWN = {
    "E0": {"Man United", "Brighton", "Bournemouth", "Tottenham"},
    "SP1": {"Ath Madrid", "Ath Bilbao", "Betis", "Vallecano"},
}


def test_name_mapping(tmp_path):
    aliases = tmp_path / "a.yaml"
    aliases.write_text(
        "E0:\n  Manchester United: Man United\n"
        "SP1:\n  Atletico Madrid: Ath Madrid\n  Athletic Club: Ath Bilbao\n"
        "  Real Betis: Betis\n  Rayo Vallecano de Madrid: Vallecano\n"
    )
    m, missing = build_name_map(
        {"Manchester United FC", "Brighton & Hove Albion FC", "AFC Bournemouth",
         "Tottenham Hotspur FC", "Mystery FC"}, KNOWN, "E0", aliases)  # fmt: skip
    assert m["Manchester United FC"] == "Man United" and m["AFC Bournemouth"] == "Bournemouth"
    assert m["Brighton & Hove Albion FC"] == "Brighton" and m["Tottenham Hotspur FC"] == "Tottenham"
    assert missing == ["Mystery FC"]
    m, missing = build_name_map(
        {"Club Atlético de Madrid", "Athletic Club", "Real Betis Balompié",
         "Rayo Vallecano de Madrid"}, KNOWN, "SP1", aliases)  # fmt: skip
    assert not missing and m["Club Atlético de Madrid"] == "Ath Madrid"
    assert norm("Club Atlético de Madrid") == "atletico madrid"


PAYLOAD = {
    "matches": [
        {"id": 1, "utcDate": "2026-11-21T15:00:00Z", "matchday": 11, "status": "TIMED",
         "homeTeam": {"name": "Manchester United FC"}, "awayTeam": {"name": "AFC Bournemouth"},
         "score": {"fullTime": {"home": None, "away": None}},
         "referees": [{"name": "A Ref", "type": "REFEREE"}]},
        {"id": 2, "utcDate": "2026-11-08T14:00:00Z", "matchday": 10, "status": "FINISHED",
         "homeTeam": {"name": "Tottenham Hotspur FC"},
         "awayTeam": {"name": "Brighton & Hove Albion FC"},
         "score": {"fullTime": {"home": 2, "away": 1}}, "referees": []},
    ]
}  # fmt: skip


def test_fetch_schedule_and_upcoming(tmp_path):
    aliases = tmp_path / "a.yaml"
    aliases.write_text("E0:\n  Manchester United: Man United\n")
    sched, problems = fetch_schedule(("E0",), KNOWN, aliases, get=lambda u, t: PAYLOAD,
                                     token="t")  # fmt: skip
    assert not problems and len(sched) == 2
    assert sched.loc[sched.match_id == 1, "referee"].iat[0] == "A Ref"
    nxt = upcoming(sched, pd.Timestamp("2026-11-20"), days=4)
    assert nxt[["home", "away"]].values.tolist() == [["Man United", "Bournemouth"]]
    assert nxt["season"].iat[0] == 2026
    priced = pd.DataFrame(
        {
            "league": ["E0"],
            "date": [pd.Timestamp("2026-11-21")],
            "home": ["Man United"],
            "away": ["Bournemouth"],
            "odds_h": [1.9],
        }
    )
    fx = from_schedule(nxt, priced)
    assert fx["odds_h"].iat[0] == 1.9 and pd.isna(fx["ref_h"].iat[0])
    with pytest.raises(ScheduleError, match="token"):
        fetch_schedule(("E0",), KNOWN, aliases, get=lambda u, t: PAYLOAD, token="")
