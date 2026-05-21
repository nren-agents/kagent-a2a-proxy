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


def working_event(text: str) -> dict:
    return {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": text}],
            },
        },
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


def completed_event() -> dict:
    return {
        "kind": "status-update",
        "status": {"state": "completed"},
        "metadata": {},
    }
