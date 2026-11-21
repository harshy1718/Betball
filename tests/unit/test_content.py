import pandas as pd

from football_edge.content.facts import FactSheet, normalise_number
from football_edge.content.picker import pick_matches
from football_edge.content.script import check_script, write_script


def _sheet(home="Athletic Bilbao", away="Espanyol", conf="High", p=0.72, goals_p=0.56):
    return FactSheet(
        league="SP1",
        date=pd.Timestamp("2026-11-20"),
        home=home,
        away=away,
        sections={
            "Prediction": [
                f"Chance of {home} win: 63%",
                "Chance of a draw: 21%",
                "Most likely scores: 1-0 (14%), 1-1 (12%), 2-0 (11%)",
                "Expected goals: 1.5 and 0.9",
            ],
            "Recent form": [f"{home} last 5: 2 wins, 2 draws, 1 loss; scored 8, conceded 5"],
        },
        data={"pick": {"result": f"{home} win", "p": p, "confidence": conf,
                       "goals": "over 2.5 goals", "goals_p": goals_p,
                       "goals_confidence": "Medium"}},
    )  # fmt: skip


GOOD = (
    "Athletic Bilbao head into this one in decent shape, with 2 wins and 2 draws from "
    "their last 5, scoring 8 along the way. Espanyol have looked shakier. We give "
    "Athletic Bilbao a 63% chance of taking the points, with the draw at 21%. The most "
    "likely score is 1-0, at 14%, though 1-1 is close behind on 12%. Expected goals "
    "point the same way: 1.5 for the home side against 0.9 for Espanyol. That gap is "
    "the story here. Athletic create more, Espanyol rely on keeping it tight, and in "
    "recent weeks they have not managed that often enough. It is not a certainty, "
    "because a draw happens more often than people think, but the home side are the "
    "clear pick on the numbers we trust. Follow the channel for the next prediction "
    "and see how this call ends up."
)


def test_good_script_passes():
    assert check_script(GOOD, _sheet()) == []


def test_invented_number_betting_words_and_cliches_are_caught():
    bad = GOOD.replace("63%", "68%") + " Great odds, a real showdown, place your bet."
    problems = " | ".join(check_script(bad, _sheet()))
    assert "68" in problems and "betting language" in problems and "showdown" in problems


def test_length_formatting_and_team_names_are_checked():
    problems = " | ".join(check_script("## Athletic Bilbao win 63%", _sheet()))
    assert "words" in problems and "team not named: Espanyol" in problems
    assert "formatting" in problems


def test_number_normalising():
    assert normalise_number("1.50") == "1.5" and normalise_number("64.0") == "64"
    assert "1.5" in _sheet().numbers() and "14" in _sheet().numbers()


class FakeLLM:
    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.drafts.pop(0), {}


def test_writer_retries_with_the_problems_listed():
    llm = FakeLLM([GOOD.replace("63%", "70%"), GOOD])
    res = write_script(llm, _sheet(), max_retries=2)
    assert res.text == GOOD and res.attempts == 2
    assert "70" in llm.prompts[1] and "numbers not in the fact sheet" in llm.prompts[1]


def test_writer_gives_up_instead_of_publishing_errors():
    res = write_script(FakeLLM(["too short"] * 3), _sheet(), max_retries=2)
    assert res.text is None and res.attempts == 3 and res.problems


REAL_MADRID = (
    "Real Madrid are on a massive tear, coming into this match with 5 wins from their last 5 "
    "games while conceding only 2. Confidence in the Real Madrid win is low because of the "
    "0-2 result in 2025. Celta Vigo have to cope with that."
)


def test_invented_confidence_reason_is_caught():
    problems = " | ".join(check_script(REAL_MADRID, _sheet()))
    assert "confidence linked to results or form" in problems


def test_too_many_numbers_is_caught():
    extra = " ".join(f"Then {n}%." for n in [31, 32, 33, 34, 35])
    assert any("at most 12" in p for p in check_script(GOOD + " " + extra, _sheet()))


def test_repeats_and_the_confidence_sentence_do_not_count():
    conf = (" Confidence is high: calls like 60 to 65% came true 66% of the time across "
            "590 matches.")  # fmt: skip
    text = GOOD + " Remember, 63%." + conf
    assert not any("numbers;" in p or "different numbers" in p
                   for p in check_script(text, _sheet()))  # fmt: skip


def test_picker_rule():
    big = {"SP1": ["Ath Bilbao"]}
    sheets = [
        (_sheet(conf="High", p=0.80), "Ath Bilbao", "Espanol", 4.8),  # qualifies
        (_sheet(conf="Medium", p=0.72), "Ath Bilbao", "Espanol", 4.8),  # not trusted
        (_sheet(conf="High", p=0.55), "Ath Bilbao", "Espanol", 4.8),  # no story
        (_sheet(conf="High", p=0.80), "Getafe", "Elche", 4.8),  # not a big club
        (_sheet(conf="High", p=0.45), "Ath Bilbao", "Espanol", 6.5),  # card-heavy story
    ]
    picks = pick_matches(sheets, big, n=3)
    assert len(picks) == 2 and picks[0].story.startswith("strong favourite")
    assert picks[1].story.startswith("card-heavy")


def test_story_must_come_from_a_trusted_call():
    # The Real Madrid case: 81% favourite rated Low, goals call rated High.
    fs = _sheet(conf="Low", p=0.81, goals_p=0.69)
    fs.data["pick"]["goals_confidence"] = "High"
    picks = pick_matches([(fs, "Ath Bilbao", "Espanol", 4.2)], {"SP1": ["Ath Bilbao"]}, n=3)
    assert len(picks) == 1 and picks[0].story.startswith("goals")


class EmptyLLM:
    def complete(self, system, user):
        return "", {"done_reason": "length", "thinking_chars": 9000}


def test_empty_answers_say_why():
    seen = []
    res = write_script(EmptyLLM(), _sheet(), max_retries=1, on_attempt=lambda *a: seen.append(a))
    assert res.text is None and len(seen) == 2
    assert "empty answer" in res.problems[0] and "length" in res.problems[0]


TONIGHT = (
    "Real Madrid are looking to maintain their perfect run against Celta Vigo. Our model "
    "gives Real Madrid an 81% chance of winning. The main focus here is the goal count, "
    "where there is a 69% chance of over 2.5 goals."
)


def _madrid():
    fs = _sheet(home="Real Madrid", away="Celta Vigo", conf="Low", p=0.81, goals_p=0.69)
    fs.data["pick"]["goals_confidence"] = "High"
    return fs


def test_story_must_open_and_low_calls_must_be_flagged():
    problems = " | ".join(check_script(TONIGHT, _madrid(), story="goals: over 2.5 goals 69%"))
    assert "first two sentences" in problems
    assert "81% is a call we rate Low" in problems and "the main focus here" in problems
    fixed = ("Over 2.5 goals has a 69% chance here. Real Madrid are big favourites at 81%, "
             "though our confidence in that win call is low.")  # fmt: skip
    problems = " | ".join(check_script(fixed, _madrid(), story="goals: over 2.5 goals 69%"))
    assert "first two sentences" not in problems and "rate Low" not in problems
