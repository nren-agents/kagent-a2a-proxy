"""Tests for the A2A JSON-RPC payload builders in ``kagent_client``."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from kagent_a2a_proxy.kagent_client import (
    _build_continue_payload,
    _build_decision_payload,
    _build_payload,
)


def _fresh() -> dict:
    return _build_payload([{"role": "user", "content": "hi"}], "sess-1")


def _resume() -> dict:
    return _build_decision_payload("task-1", "ctx-1", "approve")


def _continue() -> dict:
    return _build_continue_payload([{"role": "user", "content": "more"}], "ctx-1")


# kagent's a2a-go server dedups task history on ``message.messageId`` (treating a
# repeated id as a retry). Every outbound message must therefore carry a unique,
# non-empty id — mirroring kagent's own client, which always sets one.
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_fresh, id="fresh"),
        pytest.param(_resume, id="resume"),
        pytest.param(_continue, id="continue"),
    ],
)
def test_message_carries_unique_message_id(build: Callable[[], dict]) -> None:
    message_id = build()["params"]["message"]["messageId"]
    assert message_id
    uuid.UUID(message_id)  # a well-formed UUID
    assert build()["params"]["message"]["messageId"] != message_id


def test_continue_payload_carries_context_and_omits_session() -> None:
    # A follow-up turn rides the same conversation: the contextId lives on the
    # message (A2A's continuity key), and — like the resume payload — there is no
    # params.sessionId, so kagent keys the existing session off the contextId.
    payload = _build_continue_payload([{"role": "user", "content": "more"}], "ctx-42")
    assert payload["method"] == "message/stream"
    assert "sessionId" not in payload["params"]
    message = payload["params"]["message"]
    assert message["contextId"] == "ctx-42"
    assert message["role"] == "user"
    assert message["parts"] == [{"kind": "text", "text": "more"}]
    assert "taskId" not in message  # a new task within the existing context


def test_continue_payload_sends_only_given_messages() -> None:
    # main passes just the newest turn; the builder must not re-expand history.
    payload = _build_continue_payload(
        [{"role": "user", "content": "follow-up question"}], "ctx-1"
    )
    parts = payload["params"]["message"]["parts"]
    assert [p["text"] for p in parts] == ["follow-up question"]
