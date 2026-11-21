import json
import math

import pytest

from football_edge.news.evaluate import evaluate
from football_edge.news.extract import extract
from football_edge.news.impact import match_player, team_impact
from football_edge.news.schema import ArticleFacts, PlayerNews, SchemaError, parse_answer

# Fictional clubs and players only.
ARTICLE = (
    "Northbridge will be without striker Tom Avery, who tore his hamstring in training. "
    "Goalkeeper Luis Ortega faces a late fitness test. "
    "Defender Sam Kale is available again after serving his suspension. "
    "Eastvale boss Jon Price was sacked on Monday."
)


def _answer(players, manager=(), rotation=()):
    return json.dumps(
        {"players": players, "manager_change": list(manager), "rotation_expected": list(rotation)}
    )


AVERY = {
    "team": "Northbridge",
    "player": "Tom Avery",
    "status": "out",
    "reason": "hamstring",
    "evidence": ARTICLE.split(". ")[0] + ".",
}


def test_parse_accepts_fenced_json_and_drops_unsupported_facts():
    made_up = {**AVERY, "player": "Ray Moss", "evidence": "Ray Moss is injured."}
    text = "```json\n" + _answer([AVERY, made_up], manager=["Eastvale"]) + "\n```"
    facts = parse_answer(text, ARTICLE)
    assert [p.player for p in facts.players] == ["Tom Avery"]  # no quote, no fact
    assert facts.manager_change == ("Eastvale",)


def test_parse_rejects_bad_status_and_non_json():
    with pytest.raises(SchemaError, match="status"):
        parse_answer(_answer([{**AVERY, "status": "maybe"}]), ARTICLE)
    with pytest.raises(SchemaError):
        parse_answer("Avery is out.", ARTICLE)


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.answers.pop(0), {"total_tokens": 100}


def test_extract_retries_once_after_invalid_answer():
    client = FakeClient(["not json", _answer([AVERY])])
    res = extract(client, ARTICLE)
    assert res.facts is not None and len(res.facts.players) == 1
    assert client.calls == 2 and res.tokens == 200
    failed = extract(FakeClient(["no", "still no"]), ARTICLE)
    assert failed.facts is None and failed.error


def _facts(*players):
    return ArticleFacts(tuple(PlayerNews(*p, "", "") for p in players), (), ())


USUAL = {
    "Tom Avery": 40e6,
    "Luis Ortega": 20e6,
    "Sam Kale": 10e6,
    **{f"P{i}": 10e6 for i in range(8)},
}


def test_impact_weights_by_value_and_status():
    imp = team_impact(
        "Northbridge",
        [_facts(("Northbridge", "Avery", "out"), ("Northbridge", "L. Ortega", "doubtful"))],
        USUAL,
        squad=[],
    )
    usual = sum(USUAL.values())  # 150m
    assert imp.lineup_estimate == pytest.approx(math.log((usual - 40e6 - 0.5 * 20e6) / usual))
    assert imp.missing == ("Luis Ortega (50%)", "Tom Avery (100%)")


def test_impact_ignores_non_starters_other_teams_and_cancelled_absences():
    imp = team_impact(
        "Northbridge",
        [
            _facts(("Northbridge", "Youth Player", "out"), ("Eastvale", "Tom Avery", "out")),
            _facts(("Northbridge", "Sam Kale", "suspended")),
            _facts(("Northbridge", "Sam Kale", "returning")),
            _facts(("Northbridge", "Nobody Known", "out")),
        ],
        USUAL,
        squad=["Youth Player"],
    )
    assert imp.lineup_estimate == 0.0 and imp.missing == ()
    assert imp.unmatched == ("Nobody Known",)


def test_match_player_handles_accents_and_surnames():
    squad = ["José Núñez", "Tom Avery", "Ben Avery-Hill"]
    assert match_player("Jose Nunez", squad) == "José Núñez"
    assert match_player("Avery", squad) == "Tom Avery"
    assert match_player("Someone Else", squad) is None


def test_evaluate_precision_recall():
    labels = [
        {
            "id": "a1",
            "article": ARTICLE,
            "expected": {
                "players": [
                    {"team": "Northbridge", "player": "Tom Avery", "status": "out"},
                    {"team": "Northbridge", "player": "Luis Ortega", "status": "doubtful"},
                ]
            },
        },
        {"id": "a2", "article": ARTICLE, "expected": {"players": []}},
    ]
    wrong = {
        **AVERY,
        "player": "Sam Kale",
        "status": "out",
        "evidence": "Defender Sam Kale is available again after serving his suspension.",
    }
    client = FakeClient([_answer([AVERY]), _answer([wrong]), "bad", "bad"])
    r = evaluate(client, labels)
    assert (r.true_pos, r.extracted, r.expected) == (1, 2, 2)
    assert r.precision == 0.5 and r.recall == 0.5 and r.valid_rate == 1.0
