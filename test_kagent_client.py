"""Tests for the A2A JSON-RPC payload builders in ``kagent_client``."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from kagent_a2a_proxy.kagent_client import _build_decision_payload, _build_payload


def _fresh() -> dict:
    return _build_payload([{"role": "user", "content": "hi"}], "sess-1")


def _resume() -> dict:
    return _build_decision_payload("task-1", "ctx-1", "approve")


# kagent's a2a-go server dedups task history on ``message.messageId`` (treating a
# repeated id as a retry). Every outbound message must therefore carry a unique,
# non-empty id — mirroring kagent's own client, which always sets one.
@pytest.mark.parametrize(
    "build",
    [pytest.param(_fresh, id="fresh"), pytest.param(_resume, id="resume")],
)
def test_message_carries_unique_message_id(build: Callable[[], dict]) -> None:
    message_id = build()["params"]["message"]["messageId"]
    assert message_id
    uuid.UUID(message_id)  # a well-formed UUID
    assert build()["params"]["message"]["messageId"] != message_id
