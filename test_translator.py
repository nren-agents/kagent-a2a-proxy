"""
Tests for the A2A → OpenAI delta translator.

kagent's A2A stream uses the pre-v1.0 protocol shape:
  - Parts discriminate on `kind` ("text" / "data"), not `type`.
  - Each SSE line is a JSON-RPC envelope (handled by parse_sse_line).
  - Status-update text (working / tool-calls) maps to reasoning_content
    (LibreChat's "Thinking" pane); artifact-update text maps to content
    (the visible assistant reply); completed status-updates emit finish_reason
    only — the artifact carries the actual answer.
"""

import pytest

from kagent_a2a_proxy.translator import event_to_chunks, parse_sse_line

# ---------------------------------------------------------------------------
# parse_sse_line — JSON-RPC envelope unwrapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        pytest.param(
            'data: {"jsonrpc":"2.0","id":"x","result":{"kind":"status-update","status":{"state":"working"}}}',
            {"kind": "status-update", "status": {"state": "working"}},
            id="jsonrpc-result-unwrap",
        ),
        pytest.param("data: [DONE]", None, id="done-marker"),
        pytest.param("event: message", None, id="non-data-line"),
        pytest.param("", None, id="empty-line"),
        pytest.param(
            'data: {"jsonrpc":"2.0","id":"x","error":{"code":-1,"message":"nope"}}',
            None,
            id="jsonrpc-error",
        ),
    ],
)
def test_parse_sse_line(line: str, expected):
    assert parse_sse_line(line) == expected


# ---------------------------------------------------------------------------
# event_to_chunks — status-update (thinking channel)
# ---------------------------------------------------------------------------


def test_working_text_goes_to_reasoning_content():
    event = {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "Checking telemetry..."}],
            },
        },
        "metadata": {},
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    delta = chunks[0].choices[0].delta
    assert delta.reasoning_content == "Checking telemetry..."
    assert delta.content is None
    assert chunks[0].choices[0].finish_reason is None


def test_tool_call_annotation_goes_to_reasoning_content():
    event = {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "assistant",
                "parts": [
                    {"kind": "data", "data": {"name": "influxdb_query", "args": {}}}
                ],
            },
        },
        "metadata": {"kagent_type": "function_call", "kagent_is_long_running": False},
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    reasoning = chunks[0].choices[0].delta.reasoning_content
    assert "influxdb_query" in reasoning
    assert "🔧" in reasoning


def test_hitl_approval_request_goes_to_reasoning_content():
    event = {
        "kind": "status-update",
        "status": {
            "state": "input-required",
            "message": {
                "role": "assistant",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "name": "adk_request_confirmation",
                            "id": "conf-1",
                            "args": {
                                "originalFunctionCall": {"name": "restart_router"},
                                "toolConfirmation": {
                                    "hint": "Restart router spine-01?",
                                    "confirmed": False,
                                },
                            },
                        },
                    }
                ],
            },
        },
        "metadata": {"kagent_type": "function_call", "kagent_is_long_running": True},
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    reasoning = chunks[0].choices[0].delta.reasoning_content
    assert "⚠️" in reasoning
    assert "Approval required" in reasoning


# ---------------------------------------------------------------------------
# event_to_chunks — completed status-update emits only finish_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {"kind": "status-update", "status": {"state": "completed"}, "metadata": {}},
            id="no-message",
        ),
        pytest.param(
            {
                "kind": "status-update",
                "status": {
                    "state": "completed",
                    "message": {
                        "role": "assistant",
                        "parts": [{"kind": "text", "text": "All done."}],
                    },
                },
                "metadata": {},
            },
            id="with-message-text-ignored",
        ),
    ],
)
def test_completed_event_emits_only_finish_chunk(event: dict):
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    assert chunks[0].choices[0].finish_reason == "stop"
    # The artifact carries the answer; completed status-updates do not.
    assert not chunks[0].choices[0].delta.content


# ---------------------------------------------------------------------------
# event_to_chunks — artifact-update carries the actual answer text
# ---------------------------------------------------------------------------


def test_artifact_update_text_goes_to_content():
    event = {
        "kind": "artifact-update",
        "artifact": {
            "artifactId": "a-1",
            "parts": [{"kind": "text", "text": "Here is the final answer."}],
        },
        "lastChunk": True,
        "taskId": "t-1",
        "contextId": "c-1",
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    delta = chunks[0].choices[0].delta
    assert delta.content == "Here is the final answer."
    assert delta.reasoning_content is None


# ---------------------------------------------------------------------------
# event_to_chunks — malformed events produce no chunks
# ---------------------------------------------------------------------------


def test_malformed_event_produces_no_chunks():
    chunks = list(event_to_chunks({"garbage": True}, "agent-one"))
    assert chunks == []
