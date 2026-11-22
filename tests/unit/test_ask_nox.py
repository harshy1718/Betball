import json

import pytest

from football_edge.web.ask import _PRICE, REFUSAL, WEB_SCHEMAS, AskNox, BusyError


class StubTools:
    def call(self, name, args):
        return json.dumps({"Prediction": ["Chance of Liverpool win: 53%"]})


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, tools):
        self.sent.append(list(messages))
        return self.replies.pop(0), {}


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}  # fmt: skip


def _say(text):
    return {"role": "assistant", "content": text}


def _nox(cfg, tmp_path, replies):
    nox = AskNox(cfg, client=ScriptedLLM(replies), tools=StubTools())  # type: ignore[arg-type]
    nox.log_path = tmp_path / "log.jsonl"
    return nox


def test_no_price_tools_on_the_site():
    names = {s["name"] for s in WEB_SCHEMAS}
    assert "price_check" not in names and "golden_scoreboard" not in names
    assert "predict_match" in names


def test_golden_lines_lose_their_minimum_price():
    line = "Golden correct score pick: 1-1 12.9% (take at 6.80+) (unproven, being tracked)"
    assert _PRICE.sub("", line) == "Golden correct score pick: 1-1 12.9% (unproven, being tracked)"


def test_answer_with_progress_events(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [_call("predict_match", home="Liverpool", away="Man Utd"),
                               _say("I'm going Liverpool, 53%.")])  # fmt: skip
    events = []
    out = nox.ask("tab-1", "Who wins Liverpool v United?", on_event=events.append)
    assert out["answer"] == "I'm going Liverpool, 53%." and out["checked"] and not out["refused"]
    assert events == [{"type": "tool", "name": "predict_match",
                       "args": {"home": "Liverpool", "away": "Man Utd"}}]  # fmt: skip
    assert out["tools"] == ["predict_match"]
    assert json.loads((tmp_path / "log.jsonl").read_text())["question"].startswith("Who wins")


def test_betting_words_get_one_rewrite(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [_call("predict_match", home="Liverpool", away="Man Utd"),
                               _say("Liverpool 53%, a good bet."),
                               _say("Liverpool 53%, my favourite call.")])  # fmt: skip
    out = nox.ask("t", "Liverpool v United?")
    assert out["answer"] == "Liverpool 53%, my favourite call." and not out["refused"]
    assert "bet" in nox._client.sent[-1][-1]["content"]


def test_betting_questions_are_refused_without_the_llm(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [])
    out = nox.ask("t", "What odds should I take on Getafe?")
    assert out["answer"] == REFUSAL and out["refused"] and out["tools"] == []
    with pytest.raises(IndexError):  # "Betis" is a team, not a bet: this one reaches the LLM
        nox.ask("t", "Who wins Real Betis v Sevilla?")


def test_betting_words_twice_become_a_refusal(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [_say("Take the gamble."), _say("Still a gamble.")])
    out = nox.ask("t", "Who should I back tonight?")
    assert out["answer"] == REFUSAL and out["refused"]
    assert nox._sessions["t"].history[-1]["content"] == REFUSAL  # the follow-up sees the refusal


def test_one_question_at_a_time(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [_say("Hello!")])
    nox._busy.acquire()
    with pytest.raises(BusyError):
        nox.ask("t", "hi")
    nox._busy.release()
    with pytest.raises(ValueError):
        nox.ask("t", "   ")


def test_chats_are_kept_per_tab(cfg, tmp_path):
    nox = _nox(cfg, tmp_path, [_say("One."), _say("Two."), _say("Three.")])
    nox.ask("a", "first")
    nox.ask("b", "other tab")
    nox.ask("a", "follow-up")
    assert [m["content"] for m in nox._sessions["a"].history] == ["first", "One.", "follow-up",
                                                                  "Three."]  # fmt: skip


def test_ask_streams_json_lines(cfg, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from football_edge.web.app import create_app

    nox = _nox(cfg, tmp_path, [_call("team_form", team="Inter"), _say("Inter won 3 of 5."),
                               _say("Inter won 3 of 5.")])  # fmt: skip
    app = create_app(cfg, dist=tmp_path, nox=nox)
    r = TestClient(app).post("/api/ask", json={"session": "s", "question": "Inter form?"})
    lines = [json.loads(x) for x in r.text.strip().splitlines()]
    assert [x["type"] for x in lines] == ["tool", "answer"]
    assert lines[0]["name"] == "team_form" and lines[1]["answer"] == "Inter won 3 of 5."
    assert lines[1]["checked"] is False  # "3" and "5" are not in the stub tool's result
    assert TestClient(app).get("/api/ask/status").json()["ready"] is True
