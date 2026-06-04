"""
Tests for the A2A → OpenAI delta translator.

kagent's A2A stream uses the pre-v1.0 protocol shape:
  - Parts discriminate on `kind` ("text" / "data"), not `type`.
  - Each SSE line is a JSON-RPC envelope (handled by parse_sse_line).
  - working text routes by role: ADK thought parts (`kagent_thought`) and tool
    calls → reasoning_content (LibreChat's "Thinking" pane); the agent's prose
    answer → content (the visible reply). The non-partial aggregate copy
    (`kagent_adk_partial=false`) is skipped.
  - input-required (free-text questions and tool-approval requests) maps to
    content, so the prompt is visible instead of buried in the Thinking pane.
  - failed / canceled / rejected / auth-required map to a visible content notice
    plus finish; completed emits finish_reason only.
"""

import pytest

from conftest import (
    narration_aggregate,
    tool_call_event,
    tool_response_event,
    working_event,
)
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
# event_to_chunks — working text routing (answer → content, thought → pane)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,channel,expected",
    [
        pytest.param(
            working_event("Pong!", partial=True),
            "content",
            "Pong!",
            id="answer-partial-to-content",
        ),
        pytest.param(
            working_event("Checking telemetry...", thought=True, partial=True),
            "reasoning",
            "Checking telemetry...",
            id="thought-to-reasoning",
        ),
        pytest.param(
            working_event("answer with no partial flag"),
            "content",
            "answer with no partial flag",
            id="answer-no-flag-to-content",
        ),
    ],
)
def test_working_text_routing(
    stream_mode: None, event: dict, channel: str, expected: str
):
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    delta = chunks[0].choices[0].delta
    if channel == "content":
        assert delta.content == expected
        assert delta.reasoning_content is None
    else:
        assert delta.reasoning_content is not None
        assert delta.reasoning_content.strip() == expected
        assert delta.content is None


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            working_event("aggregate answer copy", partial=False),
            id="non-partial-answer-skipped",
        ),
        pytest.param(
            working_event("aggregate thought copy", thought=True, partial=False),
            id="non-partial-thought-skipped",
        ),
        pytest.param(
            {
                "kind": "status-update",
                "status": {
                    "state": "working",
                    "message": {
                        "role": "agent",
                        "parts": [{"kind": "data", "data": {"result": "ok"}}],
                    },
                },
                "metadata": {"kagent_type": "function_response"},
            },
            id="function-response-dropped-event-level",
        ),
        pytest.param(
            tool_response_event("list_agents"),
            id="function-response-dropped-part-level",
        ),
        pytest.param(
            tool_call_event("adk_request_confirmation", long_running=True),
            id="confirmation-request-not-rendered",
        ),
    ],
)
def test_working_events_producing_no_chunks(stream_mode: None, event: dict):
    assert list(event_to_chunks(event, "agent-one")) == []


# ---------------------------------------------------------------------------
# event_to_chunks — deemphasize mode (the default): aggregates drive output,
# narration is blockquoted, raw partials are skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,channel,expected",
    [
        pytest.param(
            working_event("aggregate report", partial=False),
            "content",
            "aggregate report",
            id="aggregate-answer-to-content",
        ),
        pytest.param(
            working_event("Thinking...", thought=True, partial=True),
            "reasoning",
            "Thinking...",
            id="thought-partial-to-reasoning",
        ),
        pytest.param(
            working_event("no-flag answer"),
            "content",
            "no-flag answer",
            id="answer-no-flag-to-content",
        ),
    ],
)
def test_deemphasize_text_routing(
    deemphasize_mode: None, event: dict, channel: str, expected: str
):
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    delta = chunks[0].choices[0].delta
    if channel == "content":
        assert delta.content == expected
        assert delta.reasoning_content is None
    else:
        assert delta.reasoning_content is not None
        assert delta.reasoning_content.strip() == expected
        assert delta.content is None


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            working_event("streamed fragment", partial=True),
            id="answer-partial-skipped",
        ),
        pytest.param(
            working_event("aggregate thought", thought=True, partial=False),
            id="aggregate-thought-skipped",
        ),
        pytest.param(
            tool_response_event("list_agents"),
            id="function-response-dropped",
        ),
    ],
)
def test_deemphasize_events_producing_no_chunks(deemphasize_mode: None, event: dict):
    assert list(event_to_chunks(event, "agent-one")) == []


def test_deemphasize_narration_aggregate_splits_blockquote_and_tool(
    deemphasize_mode: None,
):
    # A narration aggregate carries the burst text + the tool call it triggered:
    # text → blockquote in the reply, tool → Thinking pane.
    event = narration_aggregate("Let me query the topology.", "ana_topology_agent")
    chunks = list(event_to_chunks(event, "agent-one"))
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = "".join(c.choices[0].delta.reasoning_content or "" for c in chunks)
    assert "> Let me query the topology." in content
    assert "🔧 **ana_topology_agent**" in reasoning


@pytest.mark.parametrize(
    "args,expected",
    [
        pytest.param({}, ["> 🔧 **influxdb_query**"], id="no-args"),
        pytest.param(
            {"range": "1h", "limit": 5},
            ["> 🔧 **influxdb_query**", '> `range="1h", limit=5`'],
            id="with-args",
        ),
    ],
)
def test_tool_call_renders_as_structured_block(args: dict, expected: list[str]):
    event = {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "assistant",
                "parts": [
                    {"kind": "data", "data": {"name": "influxdb_query", "args": args}}
                ],
            },
        },
        "metadata": {"kagent_type": "function_call", "kagent_is_long_running": False},
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    reasoning = chunks[0].choices[0].delta.reasoning_content
    assert reasoning is not None
    assert all(fragment in reasoning for fragment in expected)
    assert chunks[0].choices[0].delta.content is None


def test_tool_call_detected_from_part_metadata():
    # Real kagent streams carry kagent_type on the data part, not the event.
    event = tool_call_event("list_agents", args={"limit": 5})
    reasoning = (
        next(iter(event_to_chunks(event, "agent-one")))
        .choices[0]
        .delta.reasoning_content
    )
    assert reasoning is not None
    assert "> 🔧 **list_agents**" in reasoning
    assert "limit=5" in reasoning


# ---------------------------------------------------------------------------
# event_to_chunks — input-required prompts go to the visible content channel
# ---------------------------------------------------------------------------


def test_hitl_approval_request_goes_to_content():
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
    delta = chunks[0].choices[0].delta
    # The prompt must be visible (content), not hidden in the Thinking pane.
    assert delta.reasoning_content is None
    assert "⚠️" in delta.content
    assert "Approval required" in delta.content
    assert "restart_router" in delta.content
    assert "Restart router spine-01?" in delta.content


def test_hitl_parallel_approvals_list_every_pending_tool():
    """Parallel approval-required calls produce ONE prompt naming all of them,
    not just the first part (a uniform approve/deny still covers them all)."""

    def _confirmation(tool: str, hint: str) -> dict:
        return {
            "kind": "data",
            "data": {
                "name": "adk_request_confirmation",
                "args": {
                    "originalFunctionCall": {"name": tool},
                    "toolConfirmation": {"hint": hint, "confirmed": False},
                },
            },
        }

    event = {
        "kind": "status-update",
        "status": {
            "state": "input-required",
            "message": {
                "role": "assistant",
                "parts": [
                    _confirmation("restart_router", "Restart spine-01?"),
                    _confirmation("delete_file", "Delete /etc/hosts?"),
                ],
            },
        },
        "metadata": {"kagent_type": "function_call", "kagent_is_long_running": True},
    }
    content = next(iter(event_to_chunks(event, "agent-one"))).choices[0].delta.content
    assert content.count("Approval required") == 1  # a single combined prompt
    assert "restart_router" in content and "delete_file" in content
    assert "Restart spine-01?" in content and "Delete /etc/hosts?" in content


def test_free_text_input_required_goes_to_content():
    event = {
        "kind": "status-update",
        "status": {
            "state": "input-required",
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "Which interface should I check?"}],
            },
        },
        "metadata": {},
    }
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 1
    delta = chunks[0].choices[0].delta
    assert delta.reasoning_content is None
    assert "Which interface should I check?" in delta.content
    assert "❓" in delta.content


def test_approval_embeds_signed_marker_when_secret_set(monkeypatch):
    from kagent_a2a_proxy import hitl
    from kagent_a2a_proxy.config import settings

    monkeypatch.setattr(settings, "hitl_secret", "s3cr3t")
    event = {
        "kind": "status-update",
        "taskId": "t-9",
        "contextId": "c-9",
        "status": {
            "state": "input-required",
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "name": "adk_request_confirmation",
                            "args": {
                                "originalFunctionCall": {"name": "restart_router"},
                                "toolConfirmation": {"hint": "Restart spine-01?"},
                            },
                        },
                    }
                ],
            },
        },
        "metadata": {"kagent_type": "function_call", "kagent_is_long_running": True},
    }
    chunk = next(iter(event_to_chunks(event, "agent-one")))
    content = chunk.choices[0].delta.content
    assert "approve" in content  # the visible reply instruction
    # The embedded (invisible) marker round-trips back to the paused task ids.
    assert hitl.extract_pending(
        [{"role": "assistant", "content": content}], "s3cr3t"
    ) == {"task_id": "t-9", "context_id": "c-9"}


# ---------------------------------------------------------------------------
# event_to_chunks — ask_user prompts render selectable choices, not approve/deny
# ---------------------------------------------------------------------------


def _ask_user_event(questions: list[dict]) -> dict:
    """An input-required event whose confirmation wraps an ask_user call."""
    return {
        "kind": "status-update",
        "taskId": "t-au",
        "contextId": "c-au",
        "status": {
            "state": "input-required",
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "name": "adk_request_confirmation",
                            "args": {
                                "originalFunctionCall": {
                                    "name": "ask_user",
                                    "args": {"questions": questions},
                                },
                                "toolConfirmation": {
                                    "hint": "; ".join(q["question"] for q in questions),
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


def test_ask_user_single_question_renders_numbered_choices():
    event = _ask_user_event(
        [
            {
                "question": "Which database should I use?",
                "choices": ["PostgreSQL", "MySQL", "SQLite"],
                "multiple": False,
            }
        ]
    )
    content = next(iter(event_to_chunks(event, "agent-one"))).choices[0].delta.content
    assert "Which database should I use?" in content
    assert "1. PostgreSQL" in content
    assert "2. MySQL" in content
    assert "3. SQLite" in content
    # An ask_user prompt is NOT a yes/no approval gate.
    assert "Approval required" not in content
    assert "approve" not in content.lower()


def test_ask_user_multiselect_hint_and_marker(monkeypatch):
    from kagent_a2a_proxy import hitl
    from kagent_a2a_proxy.config import settings

    monkeypatch.setattr(settings, "hitl_secret", "s3cr3t")
    questions = [
        {
            "question": "Which database?",
            "choices": ["PostgreSQL", "MySQL"],
            "multiple": False,
        },
        {
            "question": "Which features?",
            "choices": ["Auth", "Logging", "Caching"],
            "multiple": True,
        },
    ]
    content = (
        next(iter(event_to_chunks(_ask_user_event(questions), "agent-one")))
        .choices[0]
        .delta.content
    )
    # Both questions render with their own numbered choices.
    assert "Which database?" in content
    assert "Which features?" in content
    assert "select all" in content.lower()  # multi-select affordance
    # The marker round-trips the question structure for the resume turn.
    pending = hitl.extract_pending(
        [{"role": "assistant", "content": content}], "s3cr3t"
    )
    assert pending is not None
    assert pending["task_id"] == "t-au"
    assert pending["questions"] == questions


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
# event_to_chunks — terminal / auth states surface a visible notice + stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,detail,expected",
    [
        pytest.param("failed", "tool timed out", "Agent run failed", id="failed"),
        pytest.param("canceled", "", "canceled", id="canceled"),
        pytest.param("rejected", "", "rejected", id="rejected"),
        pytest.param(
            "auth-required",
            "login at example.com",
            "Authentication required",
            id="auth",
        ),
    ],
)
def test_terminal_state_surfaces_notice_and_finish(
    state: str, detail: str, expected: str
):
    status: dict = {"state": state}
    if detail:
        status["message"] = {
            "role": "agent",
            "parts": [{"kind": "text", "text": detail}],
        }
    event = {"kind": "status-update", "status": status, "metadata": {}}
    chunks = list(event_to_chunks(event, "agent-one"))
    assert len(chunks) == 2
    assert expected in chunks[0].choices[0].delta.content
    if detail:
        assert detail in chunks[0].choices[0].delta.content
    assert chunks[1].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# event_to_chunks — malformed events produce no chunks
# ---------------------------------------------------------------------------


def test_malformed_event_produces_no_chunks():
    chunks = list(event_to_chunks({"garbage": True}, "agent-one"))
    assert chunks == []
