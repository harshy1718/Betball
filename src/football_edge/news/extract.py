"""Call an open-source LLM to fill the form for one article.

Any server that speaks the OpenAI-compatible chat API works, so the model is a config
choice, not a code change:
  * Ollama on your own PC (free, private):  base_url http://localhost:11434/v1,
    model qwen2.5:7b-instruct
  * Hugging Face Inference / TGI / vLLM, or any hosted provider with that API.
The API key, if one is needed, is read from an environment variable, never from a file.

Temperature is 0 so the same article gives the same answer, and every call records its
latency and token use so cost can be measured in the evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from football_edge.news.schema import JSON_SCHEMA_TEXT, ArticleFacts, SchemaError, parse_answer

SYSTEM_PROMPT = f"""You extract football team news from one article.
Return ONLY a JSON object of this form, nothing else:
{JSON_SCHEMA_TEXT}
Rules:
- Only facts the article states about the coming match. Do not guess or use outside knowledge.
- "evidence" must be copied word for word from the article.
- "doubtful" = fitness test, 50-50, "a doubt", "touch and go".
- "returning" = available again after an absence.
- Leave lists empty when the article says nothing about them."""


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]: ...


@dataclass(frozen=True)
class NewsConfig:
    base_url: str
    model: str
    api_key_env: str = ""
    timeout_s: float = 60.0
    max_retries: int = 1
    temperature: float = 0.0  # 0 for extraction (repeatable); higher for natural prose
    api: str = "openai"  # "openai" (/v1/chat/completions) or "ollama" (native /api/chat)
    think: bool = False  # ollama only: let "thinking" models reason first (slow on a CPU)
    num_ctx: int = 8192  # ollama only: context window (prompt + answer), in tokens
    num_predict: int = 800  # ollama only: most tokens the answer may use

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NewsConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class LLMError(RuntimeError):
    """The local LLM server could not be used (not running, wrong model name...)."""


class OpenAICompatibleClient:
    """Minimal client for /chat/completions, using only the standard library."""

    def __init__(self, cfg: NewsConfig) -> None:
        self.cfg = cfg

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        body = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise LLMError(
                    f"the LLM server has no model called '{self.cfg.model}'. Run `ollama list` "
                    "and put the exact name in configs/default.yaml (writer: model:)"
                ) from None
            raise LLMError(f"the LLM server returned HTTP {e.code}: {e.read()[:300]!r}") from None
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach the LLM server at {self.cfg.base_url} ({e.reason}). "
                "Is Ollama running? Open the Ollama app or run `ollama serve`"
            ) from None
        return str(data["choices"][0]["message"]["content"]), dict(data.get("usage") or {})


class OllamaClient:
    """Ollama's own /api/chat. Unlike the OpenAI-style route it can switch off the hidden
    "thinking" of models like Gemma 4 and set the context size; on a CPU both decide whether
    an answer comes back at all."""

    def __init__(self, cfg: NewsConfig) -> None:
        self.cfg = cfg

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                return dict(json.loads(resp.read()))
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            if e.code == 404:
                raise LLMError(
                    f"Ollama has no model called '{self.cfg.model}'. Run `ollama list` "
                    "and put the exact name in configs/default.yaml"
                ) from None
            raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach Ollama at {self.cfg.base_url} ({e.reason}). "
                "Is it running? Open the Ollama app or run `ollama serve`"
            ) from None
        except TimeoutError:
            raise LLMError(
                f"no answer within {self.cfg.timeout_s:.0f}s; raise writer.timeout_s "
                "or use a smaller model"
            ) from None

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "think": self.cfg.think,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": self.cfg.temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
            },
        }
        try:
            data = self._post(body)
        except LLMError as e:
            if "think" not in str(e).lower():
                raise
            body.pop("think")  # a model with no thinking switch: just ask normally
            data = self._post(body)
        msg = data.get("message") or {}
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "seconds": round(data.get("total_duration", 0) / 1e9, 1),
            "done_reason": data.get("done_reason", ""),
            "thinking_chars": len(msg.get("thinking") or ""),
        }
        return str(msg.get("content") or ""), usage

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """One chat step with tools (Ollama tool calling). Returns (message, usage)."""
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "think": self.cfg.think,
            "messages": messages,
            "options": {
                "temperature": self.cfg.temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
            },
        }
        if tools:
            body["tools"] = tools
        try:
            data = self._post(body)
        except LLMError as e:
            low = str(e).lower()
            if "does not support tools" in low:
                raise LLMError(
                    f"'{self.cfg.model}' cannot call tools in Ollama. Pull a model that can, e.g. "
                    "`ollama pull qwen3:8b`, and set analyst: model: in configs/default.yaml"
                ) from None
            if "think" not in low:
                raise
            body.pop("think")
            data = self._post(body)
        msg = dict(data.get("message") or {})
        usage = {"seconds": round(data.get("total_duration", 0) / 1e9, 1),
                 "completion_tokens": data.get("eval_count", 0)}  # fmt: skip
        return msg, usage


def make_client(cfg: NewsConfig) -> ChatClient:
    return OllamaClient(cfg) if cfg.api == "ollama" else OpenAICompatibleClient(cfg)


@dataclass(frozen=True)
class Extraction:
    facts: ArticleFacts | None  # None when the model never produced a valid answer
    latency_s: float
    tokens: int
    error: str = ""


def extract(client: ChatClient, article: str, max_retries: int = 1) -> Extraction:
    """Fill the form for one article; one retry with the validation error if it fails."""
    start = time.perf_counter()
    tokens = 0
    user = f"Article:\n{article}"
    error = ""
    for _ in range(max_retries + 1):
        text, usage = client.complete(SYSTEM_PROMPT, user)
        tokens += int(usage.get("total_tokens", 0))
        try:
            facts = parse_answer(text, article)
            return Extraction(facts, time.perf_counter() - start, tokens)
        except SchemaError as e:
            error = str(e)
            user = f"Article:\n{article}\n\nYour previous answer was invalid ({error}). Try again."
    return Extraction(None, time.perf_counter() - start, tokens, error)
