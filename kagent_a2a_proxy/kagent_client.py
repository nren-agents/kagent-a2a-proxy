"""
Thin async client for the kagent-controller A2A endpoint.

kagent A2A URL pattern:
  POST /api/a2a/{namespace}/{agent-name}
  method: message/stream  → SSE response
  method: message/send    → JSON response (non-streaming)
"""
from __future__ import annotations

import uuid
import logging
from collections.abc import AsyncIterator

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
    """Map an OpenAI model name to a kagent agent name."""
    return settings.agent_map.get(model, settings.default_agent)


def _format_message(message: dict) -> str:
    role = message.get("role", "user")
    content = message.get("content") or ""
    prefix = f"[{role}] " if role != "user" else ""
    return f"{prefix}{content}"


def _build_payload(messages: list[dict], session_id: str) -> dict:
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


async def stream_agent(
    model: str,
    messages: list[dict],
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

    logger.info(
        "Streaming agent=%s session=%s url=%s",
        agent_name, sid, url,
    )

    async with _client.stream(
        "POST",
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
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
            yield line
