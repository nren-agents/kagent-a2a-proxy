"""
Shared test fixtures and event builders.

Sets PROXY_AGENT_MAP before any module imports so that ``Settings()`` —
constructed at import of ``kagent_a2a_proxy.config`` — sees a populated
agent_map even though the production default is empty.

kagent's A2A stream uses the pre-v1.0 protocol shape and wraps each event in
a JSON-RPC 2.0 envelope. The helpers below produce realistic event dicts and
SSE bytes so test files don't each reinvent them.
"""

from __future__ import annotations

import json
import os

import pytest

# Must run before any ``from kagent_a2a_proxy ...`` import.
os.environ.setdefault(
    "PROXY_AGENT_MAP",
    json.dumps(
        {
            "agent-one": "agent-one",
            "agent-two": "agent-two",
        }
    ),
)
os.environ.setdefault("PROXY_DEFAULT_AGENT", "agent-one")


def sse_response(events: list[dict]) -> bytes:
    """Build a fake SSE response body, wrapping each event in a JSON-RPC envelope."""
    lines = [
        f"data: {json.dumps({'jsonrpc': '2.0', 'id': str(i), 'result': e})}\n\n"
        for i, e in enumerate(events)
    ]
    return "".join(lines).encode()


def working_event(
    text: str, partial: bool | None = None, thought: bool = False
) -> dict:
    """A working-state status-update carrying agent text.

    `partial` mirrors kagent's `kagent_adk_partial` (True = streaming fragment,
    False = aggregated full copy). `thought` flags the part as ADK reasoning.
    """
    part: dict = {"kind": "text", "text": text}
    if thought:
        part["metadata"] = {"kagent_thought": True}
    metadata: dict = {}
    if partial is not None:
        metadata["kagent_adk_partial"] = partial
    return {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {"role": "agent", "parts": [part]},
        },
        "metadata": metadata,
    }


def failed_event(detail: str = "") -> dict:
    status: dict = {"state": "failed"}
    if detail:
        status["message"] = {
            "role": "agent",
            "parts": [{"kind": "text", "text": detail}],
        }
    return {"kind": "status-update", "final": True, "status": status, "metadata": {}}


def tool_call_event(
    name: str, args: dict | None = None, long_running: bool = False
) -> dict:
    """A working-state function-call event in kagent's real shape — `kagent_type`
    on the data *part* metadata, not the event metadata."""
    part_meta: dict = {"kagent_type": "function_call"}
    if long_running:
        part_meta["kagent_is_long_running"] = True
    part = {
        "kind": "data",
        "data": {"name": name, "args": args or {}},
        "metadata": part_meta,
    }
    return {
        "kind": "status-update",
        "status": {"state": "working", "message": {"role": "agent", "parts": [part]}},
        "metadata": {},
    }


def tool_response_event(name: str) -> dict:
    """A working-state function-response event (part-level `kagent_type`)."""
    part = {
        "kind": "data",
        "data": {"name": name, "response": {}},
        "metadata": {"kagent_type": "function_response"},
    }
    return {
        "kind": "status-update",
        "status": {"state": "working", "message": {"role": "agent", "parts": [part]}},
        "metadata": {},
    }


def artifact_event(text: str) -> dict:
    return {
        "kind": "artifact-update",
        "artifact": {
            "artifactId": "a-1",
            "parts": [{"kind": "text", "text": text}],
        },
        "lastChunk": True,
        "taskId": "t-1",
        "contextId": "c-1",
    }


def completed_event(context_id: str = "") -> dict:
    """A completed status-update. Real kagent events carry the conversation's
    ``contextId``; pass one to exercise the session-continuity marker."""
    event: dict = {
        "kind": "status-update",
        "status": {"state": "completed"},
        "metadata": {},
    }
    if context_id:
        event["contextId"] = context_id
    return event


def narration_aggregate(text: str, tool: str, args: dict | None = None) -> dict:
    """kagent's real shape for a narration burst that ends in a tool call: a
    non-partial aggregate (`kagent_adk_partial=false`) whose message carries the
    full burst text *and* the function-call data part."""
    return {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "agent",
                "parts": [
                    {"kind": "text", "text": text},
                    {
                        "kind": "data",
                        "data": {"name": tool, "args": args or {}},
                        "metadata": {"kagent_type": "function_call"},
                    },
                ],
            },
        },
        "metadata": {"kagent_adk_partial": False},
    }


@pytest.fixture
def stream_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the original token-streaming narration behavior for a test."""
    from kagent_a2a_proxy.config import settings

    monkeypatch.setattr(settings, "narration_mode", "stream")


@pytest.fixture
def deemphasize_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the blockquote-narration behavior for a test (also the default)."""
    from kagent_a2a_proxy.config import settings

    monkeypatch.setattr(settings, "narration_mode", "deemphasize")
