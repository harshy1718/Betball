"""The Nox website's API (FastAPI) and, when the React app is built, the site itself.

Run with `football-edge serve`. Routes:
  GET /api/meta              model date, data warning
  GET /api/next?n=5          the next n matches
  GET /api/matches?league=   every match in the next days (league: E0, I1, SP1)
  GET /api/matches/{id}      one match in full: grid, form, meetings, golden calls
  GET /api/golden            tonight's golden calls, the bars, the tracker
  GET /api/ask/status        is the local LLM ready for Ask Nox
  POST /api/ask              Ask Nox a question; streams JSON lines: progress, then the answer
  /api/desk/...              the private trader desk: answers requests from this PC only
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from football_edge.config import Config
from football_edge.web.ask import AskNox, BusyError, LLMError
from football_edge.web.service import MatchNotFoundError, NoxService

LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}


class SlipRequest(BaseModel):
    keys: list[str]
    price: float | None = None
    stake: float | None = None


class Question(BaseModel):
    session: str = "default"
    question: str


def create_app(
    cfg: Config,
    service: NoxService | None = None,
    dist: Path | None = None,
    nox: AskNox | None = None,
    desk: Any = None,
) -> FastAPI:
    svc = service or NoxService(cfg)
    asker = nox or AskNox(cfg)
    trader = desk
    review_lock = threading.Lock()
    app = FastAPI(title="Nox", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def guard(fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from None
        except MatchNotFoundError:
            raise HTTPException(404, "no upcoming match with that id") from None

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return dict(guard(svc.meta))

    @app.get("/api/next")
    def next_matches(n: int = 5) -> list[dict[str, Any]]:
        return list(guard(svc.next_matches, max(1, min(n, 20))))

    @app.get("/api/matches")
    def matches(league: str | None = None) -> list[dict[str, Any]]:
        return list(guard(svc.matches, league=league))

    @app.get("/api/matches/{mid}")
    def match(mid: str) -> dict[str, Any]:
        return dict(guard(svc.match, mid))

    @app.get("/api/golden")
    def golden_calls() -> dict[str, Any]:
        return dict(guard(svc.golden_calls))

    @app.get("/api/ask/status")
    def ask_status() -> dict[str, Any]:
        return asker.status()

    @app.post("/api/ask")
    def ask(q: Question) -> StreamingResponse:
        """Streams one JSON object per line: {"type": "tool", ...} while Nox works, then
        {"type": "answer", ...} or {"type": "error", "message": ...}."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def work() -> None:
            try:
                out = asker.ask(q.session, q.question, on_event=events.put)
                events.put({"type": "answer", **out})
            except BusyError as e:
                events.put({"type": "error", "message": str(e), "busy": True})
            except LLMError as e:
                events.put({"type": "error", "message": f"Nox's brain isn't answering: {e}"})
            except (ValueError, FileNotFoundError) as e:
                events.put({"type": "error", "message": str(e)})
            except Exception as e:  # keep the page alive; the server log has the trace
                events.put({"type": "error", "message": f"Something went wrong: {e}"})
                raise
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            while (ev := events.get()) is not None:
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    # -- the private trader desk ----------------------------------------------------------
    def local_only(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host not in LOCAL:
            raise HTTPException(403, "the trader desk only answers on this PC")

    def get_desk() -> Any:
        nonlocal trader
        if trader is None:
            from football_edge.trader.desk import Desk

            trader = Desk(cfg, svc)
        return trader

    def desk_guard(fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except KeyError as e:
            raise HTTPException(404, f"not found: {e}") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from None

    local = [Depends(local_only)]

    @app.get("/api/desk/board", dependencies=local)
    def desk_board() -> dict[str, Any]:
        return dict(desk_guard(get_desk().board))

    @app.get("/api/desk/options/{mid}", dependencies=local)
    def desk_options(mid: str) -> dict[str, Any]:
        return dict(desk_guard(get_desk().options, mid))

    @app.post("/api/desk/check", dependencies=local)
    def desk_check(r: SlipRequest) -> dict[str, Any]:
        return dict(desk_guard(get_desk().check, r.keys, r.price))

    @app.post("/api/desk/save", dependencies=local)
    def desk_save(r: SlipRequest) -> dict[str, Any]:
        return dict(desk_guard(get_desk().save, r.keys, r.price, r.stake))

    @app.get("/api/desk/history", dependencies=local)
    def desk_history() -> dict[str, Any]:
        return dict(desk_guard(get_desk().history))

    @app.post("/api/desk/review", dependencies=local)
    def desk_review(r: SlipRequest) -> dict[str, Any]:
        from football_edge.trader.brain import review

        c = desk_guard(get_desk().check, r.keys, r.price)
        if not review_lock.acquire(blocking=False):
            raise HTTPException(429, "the trader is reviewing another slip")
        try:
            client = getattr(asker, "_client", None) or asker.client()
            return {**review(client, c), "desk_view": c["desk_view"]}
        except LLMError as e:
            raise HTTPException(503, f"the trader's brain isn't answering: {e}") from None
        finally:
            review_lock.release()

    site = dist or cfg.root / "web" / "dist"
    if (site / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=site / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            """Any non-API path is a page of the React app (it routes on the client)."""
            if path.startswith("api/"):
                raise HTTPException(404)
            file = (site / path).resolve()
            if path and file.is_file() and site.resolve() in file.parents:
                return FileResponse(file)
            return FileResponse(site / "index.html")

    return app
