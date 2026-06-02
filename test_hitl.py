"""
Tests for the human-in-the-loop marker + decision helpers.

The marker is an HMAC-signed, render-invisible HTML comment embedded in the
assistant reply; it round-trips through LibreChat's conversation history so the
proxy can route the user's next approve/deny back to the paused kagent task.
"""

import pytest

from kagent_a2a_proxy import hitl

SECRET = "s3cr3t"


def _messages(*turns: tuple[str, str]) -> list[dict[str, str]]:
    """Build a chat history from (role, content) turns."""
    return [{"role": role, "content": content} for role, content in turns]


def test_marker_round_trip():
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET)
    assert "<!--kagent-hitl" in marker
    messages = _messages(
        ("user", "do x"),
        ("assistant", "⚠️ Approval required" + marker),
    )
    assert hitl.extract_pending(messages, SECRET) == {
        "task_id": "task-1",
        "context_id": "ctx-1",
    }


def test_disabled_without_secret():
    assert hitl.encode_marker("t", "c", None) == ""
    assert hitl.extract_pending(_messages(("assistant", "hello")), None) is None


def test_wrong_secret_rejected():
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET)
    messages = _messages(("assistant", "x" + marker))
    assert hitl.extract_pending(messages, "different-secret") is None


def test_tampered_payload_rejected():
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET)
    # Swap the signed payload for a forged one while keeping the signature.
    head, _, sig = marker.rpartition(":")
    forged = f"{head[: head.rindex(':') + 1]}Zm9yZ2Vk:{sig}"
    messages = _messages(("assistant", "x" + forged))
    assert hitl.extract_pending(messages, SECRET) is None


def test_marker_taken_from_latest_assistant_message():
    old = hitl.encode_marker("old-task", "c", SECRET)
    new = hitl.encode_marker("new-task", "c", SECRET)
    messages = _messages(
        ("assistant", "first" + old),
        ("user", "approve"),
        ("assistant", "second" + new),
    )
    pending = hitl.extract_pending(messages, SECRET)
    assert pending is not None and pending["task_id"] == "new-task"


def test_no_marker_returns_none():
    messages = _messages(("assistant", "just a normal reply"))
    assert hitl.extract_pending(messages, SECRET) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param("approve", "approve", id="approve"),
        pytest.param("Approve, please", "approve", id="approve-caps-trailing"),
        pytest.param("yes", "approve", id="yes"),
        pytest.param("ok", "approve", id="ok"),
        pytest.param("deny", "reject", id="deny"),
        pytest.param("No.", "reject", id="no-punctuation"),
        pytest.param("cancel that", "reject", id="cancel"),
        pytest.param("what will it do?", None, id="ambiguous-question"),
        pytest.param("", None, id="empty"),
    ],
)
def test_classify_decision(text: str, expected: str | None):
    assert hitl.classify_decision(text) == expected
