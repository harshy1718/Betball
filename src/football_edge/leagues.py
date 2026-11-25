"""The leagues Nox knows, in one place: names, hashtags and each data source's code.

Which of them are actually ingested and predicted is `leagues:` in configs/default.yaml.
"""

from __future__ import annotations

NAMES = {
    "E0": "Premier League",
    "I1": "Serie A",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
TAGS = {  # Instagram hashtags (posts page)
    "E0": "PremierLeague",
    "I1": "SerieA",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "F1": "Ligue1",
}
FOOTBALL_DATA_ORG = {"E0": "PL", "I1": "SA", "SP1": "PD", "D1": "BL1", "F1": "FL1"}
FILE_PATTERN = r"^(" + "|".join(sorted(NAMES, key=len, reverse=True)) + ")"


def name(code: str) -> str:
    return NAMES.get(code, code)
