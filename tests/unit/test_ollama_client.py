import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from football_edge.news.extract import LLMError, NewsConfig, OllamaClient, make_client


class _Handler(BaseHTTPRequestHandler):
    bodies: ClassVar[list[dict]] = []
    reject_think = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        if self.path != "/api/chat" or body["model"] != "gemma4:12b":
            self.send_response(404)
            self.end_headers()
            return
        if type(self).reject_think and "think" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"model does not support thinking"}')
            return
        out = {"message": {"content": "A script.", "thinking": ""}, "done_reason": "stop",
               "prompt_eval_count": 900, "eval_count": 120, "total_duration": 4e9}  # fmt: skip
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.bodies, _Handler.reject_think = [], False
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _cfg(url, model="gemma4:12b"):
    return NewsConfig(base_url=url + "/v1", model=model, api="ollama", temperature=0.8)


def test_native_api_with_thinking_off(server):
    text, usage = make_client(_cfg(server)).complete("sys", "user")
    body = _Handler.bodies[0]
    assert text == "A script." and usage["completion_tokens"] == 120
    assert body["think"] is False and body["options"]["num_ctx"] == 8192
    assert body["stream"] is False


def test_model_without_thinking_switch_is_retried_plainly(server):
    _Handler.reject_think = True
    text, _ = OllamaClient(_cfg(server)).complete("sys", "user")
    assert text == "A script." and "think" not in _Handler.bodies[-1]


def test_wrong_model_name_is_explained(server):
    with pytest.raises(LLMError, match="ollama list"):
        OllamaClient(_cfg(server, "gemma9")).complete("sys", "user")
