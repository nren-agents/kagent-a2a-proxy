"""
kagent-a2a-proxy — FastAPI application.

Exposes:
  POST /v1/chat/completions   OpenAI streaming chat completions
  GET  /v1/models             OpenAI model list (one entry per agent)
  GET  /healthz/ready         Liveness / readiness probe
  /mcp                        MCP server (Streamable HTTP) — one tool per agent
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import hitl, kagent_client
from .agent_runner import collect_response, translate_resume, translate_stream
from .config import settings
from .mcp_server import mcp
from .models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    DeltaContent,
    ModelList,
    ModelObject,
    StreamChoice,
)
from .translator import render_ask_user

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# Build the MCP ASGI app with path="/" so that mounting it at "/mcp" yields
# the canonical /mcp endpoint (avoiding /mcp/mcp).
_mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with _mcp_app.lifespan(app):
        try:
            yield
        finally:
            await kagent_client.aclose()


app = FastAPI(
    title="kagent-a2a-proxy",
    description=(
        "OpenAI-compatible streaming chat completions and MCP server"
        " backed by kagent A2A"
    ),
    version="0.0.8",
    lifespan=lifespan,
)

app.mount("/mcp", _mcp_app)


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------


@app.get("/healthz/ready", include_in_schema=False)
async def healthz() -> dict[str, Any]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/models — return one model entry per configured agent
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelObject(id=name) for name in settings.agent_map])


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    body = await request.json()

    try:
        req = ChatCompletionRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    messages = [m.model_dump() for m in req.messages]
    session_id = str(uuid.uuid4())
    chunks = _select_chunks(req.model, messages, session_id)

    if req.stream:
        return StreamingResponse(
            _with_heartbeat(
                _stream_response(req.model, chunks), settings.sse_heartbeat_interval
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )
    try:
        return await _blocking_response(req.model, chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    return next(
        (
            str(m.get("content") or "")
            for m in reversed(messages)
            if m.get("role") == "user"
        ),
        "",
    )


def _messages_after_last_assistant(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The newest turn: messages following the last assistant reply (usually the
    single new user message). Prior turns live in kagent's session on a
    continuation, so only this is sent."""
    last = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        default=-1,
    )
    return messages[last + 1 :]


def _select_chunks(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Pick the chunk source. With a pending HITL prompt, resume it: an ``ask_user``
    prompt maps the reply to positional answers; a tool approval keys off a clear
    approve/deny. An unmappable reply re-prompts. Otherwise, if a prior assistant
    marker carries a kagent contextId, continue that conversation (echo the
    contextId, send only the newest turn); with no marker it's a fresh call."""
    pending = hitl.extract_pending(messages, settings.hitl_secret)
    if pending is None:
        context_id = hitl.extract_context(messages, settings.hitl_secret)
        if context_id is not None:
            return translate_stream(
                model, _messages_after_last_assistant(messages), context_id=context_id
            )
        return translate_stream(model, messages, session_id)
    user_text = _last_user_text(messages)
    questions = pending.get("questions")
    if questions:
        answers = hitl.parse_ask_user_reply(user_text, questions)
        if answers is None:
            return _clarify_ask_user_chunks(model, pending)
        return translate_resume(
            model,
            pending["task_id"],
            pending["context_id"],
            "approve",
            ask_user_answers=answers,
        )
    decision = hitl.classify_decision(user_text)
    if decision is None:
        return _clarify_chunks(model, pending)
    return translate_resume(model, pending["task_id"], pending["context_id"], decision)


def _stop_chunks(model: str, text: str) -> AsyncIterator[ChatCompletionChunk]:
    """A two-chunk stream: one content delta, then finish_reason=stop."""

    async def gen() -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            model=model, choices=[StreamChoice(delta=DeltaContent(content=text))]
        )
        yield ChatCompletionChunk(
            model=model,
            choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
        )

    return gen()


def _clarify_chunks(
    model: str,
    pending: dict[str, Any],
) -> AsyncIterator[ChatCompletionChunk]:
    """Re-prompt for an ambiguous reply, re-embedding the marker so the pending
    approval survives to the next turn."""
    marker = hitl.encode_marker(
        pending["task_id"], pending["context_id"], settings.hitl_secret
    )
    text = (
        "\n⚠️ There's a pending approval. Please reply **approve** or **deny** "
        f"to continue.\n{marker}"
    )
    return _stop_chunks(model, text)


def _clarify_ask_user_chunks(
    model: str,
    pending: dict[str, Any],
) -> AsyncIterator[ChatCompletionChunk]:
    """Re-render the ask_user questions when the reply couldn't be mapped,
    re-embedding the question-carrying marker so the prompt survives the turn."""
    questions = pending["questions"]
    marker = hitl.encode_marker(
        pending["task_id"], pending["context_id"], settings.hitl_secret, questions
    )
    note = "\n⚠️ I couldn't match your reply to the question(s). Please try again:\n"
    return _stop_chunks(model, note + render_ask_user(questions, marker))


_SSE_HEARTBEAT = ": ping\n\n"


async def _with_heartbeat(
    parts: AsyncIterator[str],
    interval: float,
) -> AsyncIterator[str]:
    """Yield ``parts``, inserting an SSE comment during idle gaps.

    Agent streams can be silent for minutes (long tool runs; deemphasize mode
    suppresses partials), and idle-read timeouts on the client path (LibreChat's
    undici bodyTimeout ~300s, LB idle timers) sever quiet streams. SSE comment
    lines are spec-ignored by parsers and reset every such timer.

    The pending ``anext`` is kept alive across heartbeats (never cancelled on
    the timeout path) so no part is ever lost mid-await.
    """
    if not interval:
        async for part in parts:
            yield part
        return
    it = aiter(parts)
    upcoming = asyncio.ensure_future(anext(it, None))
    try:
        while True:
            done, _ = await asyncio.wait({upcoming}, timeout=interval)
            if not done:
                yield _SSE_HEARTBEAT
            elif (item := upcoming.result()) is None:
                return
            else:
                yield item
                upcoming = asyncio.ensure_future(anext(it, None))
    finally:
        upcoming.cancel()


async def _stream_response(
    model: str,
    chunks: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[str]:
    """Yield SSE chunks for a streaming chat completions response."""

    # Opening chunk with role
    opening = ChatCompletionChunk(
        model=model,
        choices=[StreamChoice(delta=DeltaContent(role="assistant"))],
    )
    yield opening.to_sse()

    try:
        async for chunk in chunks:
            yield chunk.to_sse()
    except httpx.HTTPStatusError as exc:
        logger.error("kagent error: %s", exc)
        yield _error_chunk(
            model, f"kagent returned {exc.response.status_code}"
        ).to_sse()
    except Exception as exc:
        logger.exception("Unexpected error streaming from kagent")
        yield _error_chunk(model, str(exc)).to_sse()

    yield "data: [DONE]\n\n"


def _error_chunk(model: str, message: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        model=model,
        choices=[
            StreamChoice(
                delta=DeltaContent(content=f"\n\n[Error: {message}]"),
                finish_reason="stop",
            )
        ],
    )


async def _blocking_response(
    model: str,
    chunks: AsyncIterator[ChatCompletionChunk],
) -> JSONResponse:
    """Accumulate the full stream and return a non-streaming response."""
    full_content = await collect_response(chunks)

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
