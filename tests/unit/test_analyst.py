import json

from football_edge.analyst.agent import Analyst


class StubTools:
    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "predict_match":
            lines = ["Chance of Liverpool win: 53%", "Expected goals: Liverpool 2.0, United 1.3"]
            return json.dumps({"Prediction": lines})
        return json.dumps({"error": f"unknown team '{args.get('team')}'"})


class ScriptedLLM:
    """Replays a fixed list of assistant messages; records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, tools):
        self.sent.append((list(messages), bool(tools)))
        return self.replies.pop(0), {}


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}  # fmt: skip


def _say(text):
    return {"role": "assistant", "content": text}


def test_tool_loop_and_grounded_answer():
    tools = StubTools()
    reply = _say("Liverpool are favourites at 53%, expected goals 2.0 to 1.3.")
    llm = ScriptedLLM([_call("predict_match", home="Liverpool", away="Man Utd"), reply])
    a = Analyst(llm, tools).ask("Who wins Liverpool v United?")
    assert tools.calls == [("predict_match", {"home": "Liverpool", "away": "Man Utd"})]
    assert a.text.startswith("Liverpool") and a.unverified == []
    tool_msg = llm.sent[1][0][-1]
    assert tool_msg["role"] == "tool" and "53%" in tool_msg["content"]


def test_invented_number_gets_one_rewrite():
    llm = ScriptedLLM([_call("predict_match", home="Liverpool", away="Man Utd"),
                       _say("Liverpool 61% to win."),  # 61 is invented
                       _say("Liverpool 53% to win.")])  # fmt: skip
    a = Analyst(llm, StubTools()).ask("Liverpool v United?")
    assert a.text == "Liverpool 53% to win." and a.unverified == []
    assert "61" in llm.sent[-1][0][-1]["content"] and llm.sent[-1][1] is False


def test_still_wrong_after_rewrite_is_flagged_not_hidden():
    llm = ScriptedLLM([_call("predict_match", home="Liverpool", away="Man Utd"),
                       _say("Liverpool 61%."), _say("Liverpool 61% really.")])  # fmt: skip
    a = Analyst(llm, StubTools()).ask("Liverpool v United?")
    assert a.unverified == ["61"]


def test_numbers_from_the_question_and_earlier_turns_are_allowed():
    tools = StubTools()
    llm = ScriptedLLM([_call("predict_match", home="Liverpool", away="Man Utd"), _say("53%."),
                       _say("At 8.4 and 53% it is the same match as before.")])  # fmt: skip
    an = Analyst(llm, tools)
    an.ask("Liverpool v United?")
    assert an.ask("is 8.4 good?").unverified == []


def test_step_limit_forces_an_answer():
    llm = ScriptedLLM([_call("team_form", team="X")] * 2 + [_say("I could not find team X.")])
    a = Analyst(llm, StubTools(), max_steps=2).ask("form of X?")
    assert a.text == "I could not find team X." and llm.sent[-1][1] is False


def test_price_check_looks_up_the_probability_itself():
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from football_edge.analyst.tools import AnalystTools

    class Model:  # 1-1 gets 10%, 1-0 gets 14%
        def score_matrix(self, lam, mu):
            m = np.full((11, 11), 0.006)
            m[1, 1], m[1, 0] = 0.10, 0.14
            return m

    f = pd.Series({"league": "E0", "home": "Liverpool", "away": "Man United", "p_h": 0.53,
                   "p_d": 0.23, "p_a": 0.24, "p_over25": 0.63, "p_under25": 0.37,
                   "p_btts_yes": 0.63, "p_btts_no": 0.37, "exp_home": 2.0, "exp_away": 1.3,
                   "top_score": "1-0", "top_score_p": 0.14})  # fmt: skip
    t = AnalystTools.__new__(AnalystTools)  # no data needed
    t.cfg = SimpleNamespace(staking={"golden_cs": {"lo": 0.12}})
    t.names = {}
    t._forecast = lambda home, away: (f, None, Model())
    r = t.price_check("Liverpool", "Man United", "1-1", 8.4)  # the LLM gives no probability
    assert r["model_probability"] == "10.0%" and r["fair_price"] == "10.00"
    assert r["verdict"].startswith("not worth")
    g = t.price_check("Liverpool", "Man United", "1-0", 7.5)  # the golden score
    assert g["golden"] == "correct-score golden pick" and g["fair_price"] == "6.80"
    assert t.price_check("Liverpool", "Man United", "home", 1.80)["bet"] == "home win"


def test_market_caveat_is_added_in_code_when_missing():
    class PriceTools:
        def call(self, name, args):
            return json.dumps({"verdict": "worth taking: above the fair price",
                               "fair_price": "1.37"})  # fmt: skip

    llm = ScriptedLLM([_call("price_check", home="Barcelona", away="Villarreal",
                             bet="home win", price=1.4),
                       _say("Worth taking: above the fair price of 1.37.")])  # fmt: skip
    a = Analyst(llm, PriceTools()).ask("Barcelona at 1.40?")
    assert "market" in a.text.lower() and a.unverified == []
