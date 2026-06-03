"""
Thin async client for the kagent-controller A2A endpoint.

kagent A2A URL pattern:
  POST /api/a2a/{namespace}/{agent-name}
  method: message/stream  → SSE response
  method: message/send    → JSON response (non-streaming)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# Module-level client: reused across all requests for connection pooling.
# Closed via aclose() from main.py's FastAPI lifespan.
_client = httpx.AsyncClient(
    timeout=settings.request_timeout,
    follow_redirects=True,
)


async def aclose() -> None:
    """Close the shared httpx client. Call from app shutdown."""
    await _client.aclose()


def _a2a_url(agent_name: str) -> str:
    base = str(settings.kagent_base_url).rstrip("/")
    return f"{base}/api/a2a/{settings.kagent_namespace}/{agent_name}"


def _resolve_agent(model: str) -> str:
    """Map an OpenAI model name to a kagent agent name.

    Falls back to ``settings.default_agent`` for unknown models. Raises
    ``ValueError`` if the model is not in ``agent_map`` and no default agent
    is configured — callers should surface this as a 4xx to the client.
    """
    if (agent := settings.agent_map.get(model)) is not None:
        return agent
    if settings.default_agent is not None:
        return settings.default_agent
    raise ValueError(
        f"Model {model!r} is not in PROXY_AGENT_MAP and PROXY_DEFAULT_AGENT is not set"
    )


def _format_message(message: dict[str, Any]) -> str:
    role = message.get("role", "user")
    content = message.get("content") or ""
    prefix = f"[{role}] " if role != "user" else ""
    return f"{prefix}{content}"


def _build_payload(messages: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    """
    Build an A2A message/stream JSON-RPC payload from OpenAI messages.

    We concatenate all messages into a single text prompt so the agent
    has full context. The system message is prepended if present.
    """
    parts = [
        {"kind": "text", "text": _format_message(m)}
        for m in messages
        if m.get("content")
    ]

    return {
        "jsonrpc": "2.0",
        "method": "message/stream",
        "id": str(uuid.uuid4()),
        "params": {
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": parts,
            },
        },
    }


def _build_decision_payload(
    task_id: str,
    context_id: str,
    decision: str,
    rejection_reason: str = "",
    ask_user_answers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the A2A message/stream payload that resumes a paused (input-required)
    task with a human-in-the-loop decision.

    Mirrors kagent's own client (`_remote_a2a_tool` / `_hitl_utils`): the decision
    rides a DataPart, and the paused task is referenced by taskId + contextId on
    the message itself. For the built-in ``ask_user`` tool, kagent expects an
    ``approve`` decision carrying the positional ``ask_user_answers`` list. NOTE:
    the exact wire shape is inferred from kagent source and should be validated
    against a live kagent before relying on it.
    """
    data: dict[str, Any] = {"decision_type": decision}
    if decision == "reject" and rejection_reason:
        data["rejection_reason"] = rejection_reason
    if ask_user_answers is not None:
        data["ask_user_answers"] = ask_user_answers
    message: dict[str, Any] = {
        "role": "user",
        "taskId": task_id,
        "contextId": context_id,
        "parts": [{"kind": "data", "data": data}],
    }
    return {
        "jsonrpc": "2.0",
        "method": "message/stream",
        "id": str(uuid.uuid4()),
        "params": {"message": message},
    }


async def _stream(url: str, payload: dict[str, Any]) -> AsyncIterator[str]:
    """POST a JSON-RPC payload and yield raw SSE lines (incl. the 'data:' prefix)."""
    async with _client.stream(
        "POST",
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        # read=None: kagent may be silent between SSE events for a long time
        # (long-running tools, human-in-the-loop approval). The read timeout is
        # the gap *between* reads, not a total budget, so any value would kill a
        # healthy idle stream. connect/write/pool stay bounded.
        timeout=httpx.Timeout(settings.request_timeout, read=None),
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            logger.error(
                "kagent returned %s (location=%s): %s",
                response.status_code,
                response.headers.get("location"),
                body.decode(errors="replace"),
            )
            raise httpx.HTTPStatusError(
                f"kagent {response.status_code}",
                request=response.request,
                response=response,
            )

        async for line in response.aiter_lines():
            logger.debug("kagent SSE << %s", line)
            yield line


async def stream_agent(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """
    Call kagent A2A message/stream and yield raw SSE lines.

    Yields each line as a string (including the 'data: ' prefix).
    Caller is responsible for parsing.
    """
    agent_name = _resolve_agent(model)
    url = _a2a_url(agent_name)
    sid = session_id or str(uuid.uuid4())
    payload = _build_payload(messages, sid)

    logger.info("Streaming agent=%s session=%s url=%s", agent_name, sid, url)

    async for line in _stream(url, payload):
        yield line


async def resume_stream(
    model: str,
    task_id: str,
    context_id: str,
    decision: str,
    rejection_reason: str = "",
    ask_user_answers: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Resume a paused (input-required) task with a decision (approve/reject) or,
    for ``ask_user``, an ``approve`` carrying the positional answers."""
    agent_name = _resolve_agent(model)
    url = _a2a_url(agent_name)
    payload = _build_decision_payload(
        task_id, context_id, decision, rejection_reason, ask_user_answers
    )

    logger.info(
        "Resuming agent=%s task=%s decision=%s url=%s",
        agent_name,
        task_id,
        decision,
        url,
    )

    async for line in _stream(url, payload):
        yield line
