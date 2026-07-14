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


def test_marker_round_trip_and_is_invisible():
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET)
    assert marker  # non-empty
    # Invisible: nothing but zero-width characters (no renderable text).
    assert all(ch in ("\u200b", "\u200c") for ch in marker)
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
    # A payload whose signature was computed over different bytes must not verify.
    body = f"v1:tampered-payload:{hitl._sign('other-payload', SECRET)}"
    forged = hitl._encode_zw(body)
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


# ---------------------------------------------------------------------------
# Session continuity — a context-only marker (no taskId) carries the kagent
# conversation's contextId across turns so follow-ups resume the same session.
# ---------------------------------------------------------------------------


def test_context_only_marker_round_trips_via_extract_context():
    marker = hitl.encode_marker("", "ctx-9", SECRET)
    assert marker
    assert all(ch in ("\u200b", "\u200c") for ch in marker)  # still invisible
    messages = _messages(("user", "hi"), ("assistant", "the answer" + marker))
    assert hitl.extract_context(messages, SECRET) == "ctx-9"


def test_context_only_marker_is_not_a_pending_approval():
    # No taskId → extract_pending must not treat it as a paused HITL task.
    marker = hitl.encode_marker("", "ctx-9", SECRET)
    assert hitl.extract_pending(_messages(("assistant", "x" + marker)), SECRET) is None


def test_encode_marker_empty_when_nothing_to_carry():
    assert hitl.encode_marker("", "", SECRET) == ""


def test_extract_context_returns_latest_assistant_marker():
    old = hitl.encode_marker("", "ctx-old", SECRET)
    new = hitl.encode_marker("", "ctx-new", SECRET)
    messages = _messages(
        ("assistant", "first" + old),
        ("user", "more"),
        ("assistant", "second" + new),
    )
    assert hitl.extract_context(messages, SECRET) == "ctx-new"


def test_extract_context_scans_back_past_unmarked_assistant():
    # A later assistant turn without a marker (e.g. an error) must not break the
    # chain — we fall back to the most recent valid marker.
    marker = hitl.encode_marker("", "ctx-1", SECRET)
    messages = _messages(
        ("assistant", "earlier" + marker),
        ("user", "x"),
        ("assistant", "no marker here"),
    )
    assert hitl.extract_context(messages, SECRET) == "ctx-1"


def test_extract_context_also_reads_an_approval_marker():
    # A HITL approval marker carries the contextId too, so continuity survives
    # a turn that paused for approval.
    marker = hitl.encode_marker("task-1", "ctx-hitl", SECRET)
    messages = _messages(("assistant", "Approval required" + marker))
    assert hitl.extract_context(messages, SECRET) == "ctx-hitl"


def test_extract_context_disabled_without_secret():
    marker = hitl.encode_marker("", "ctx-1", SECRET)
    assert hitl.extract_context(_messages(("assistant", "a" + marker)), None) is None


def test_extract_context_rejects_wrong_secret():
    marker = hitl.encode_marker("", "ctx-1", SECRET)
    messages = _messages(("assistant", "a" + marker))
    assert hitl.extract_context(messages, "different-secret") is None


def test_extract_context_none_when_no_marker():
    assert hitl.extract_context(_messages(("assistant", "plain reply")), SECRET) is None


def test_tombstone_stops_context_scan():
    # After a terminal failure the kagent session may hold a dangling tool call;
    # the tombstone on the error turn must stop the scan so the next turn starts
    # fresh instead of reopening the poisoned context.
    tombstone = hitl.encode_tombstone(SECRET)
    assert tombstone  # non-empty
    assert all(ch in ("\u200b", "\u200c") for ch in tombstone)  # invisible
    marker = hitl.encode_marker("", "ctx-1", SECRET)
    messages = _messages(
        ("assistant", "fine answer" + marker),
        ("user", "do the risky thing"),
        ("assistant", "⚠️ Agent run failed" + tombstone),
    )
    assert hitl.extract_context(messages, SECRET) is None


def test_newer_context_marker_overrides_older_tombstone():
    # A completed turn after the failure re-establishes continuity.
    tombstone = hitl.encode_tombstone(SECRET)
    marker = hitl.encode_marker("", "ctx-2", SECRET)
    messages = _messages(
        ("assistant", "⚠️ Agent run failed" + tombstone),
        ("user", "try again"),
        ("assistant", "worked this time" + marker),
    )
    assert hitl.extract_context(messages, SECRET) == "ctx-2"


def test_tombstone_is_not_a_pending_approval():
    tombstone = hitl.encode_tombstone(SECRET)
    messages = _messages(("assistant", "⚠️ Agent run failed" + tombstone))
    assert hitl.extract_pending(messages, SECRET) is None


def test_encode_tombstone_empty_without_secret():
    assert hitl.encode_tombstone(None) == ""


def test_strip_marker_removes_zero_width_chars():
    marker = hitl.encode_marker("", "ctx-1", SECRET)
    assert hitl.strip_marker("Answer." + marker) == "Answer."
    assert hitl.strip_marker("no marker") == "no marker"


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


# ---------------------------------------------------------------------------
# ask_user — the marker carries the question structure, and the user's reply
# is parsed into the positional ask_user_answers list kagent expects.
# ---------------------------------------------------------------------------

_DB_Q = {
    "question": "Which database?",
    "choices": ["PostgreSQL", "MySQL", "SQLite"],
    "multiple": False,
}
_FEATURES_Q = {
    "question": "Which features?",
    "choices": ["Auth", "Logging", "Caching"],
    "multiple": True,
}
_FREETEXT_Q = {"question": "Anything else?", "choices": [], "multiple": False}


def test_marker_round_trip_carries_ask_user_questions():
    questions = [_DB_Q]
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET, questions)
    assert all(ch in ("\u200b", "\u200c") for ch in marker)  # still invisible
    messages = _messages(("assistant", "❓ Which database?" + marker))
    assert hitl.extract_pending(messages, SECRET) == {
        "task_id": "task-1",
        "context_id": "ctx-1",
        "questions": questions,
    }


def test_marker_without_questions_omits_questions_key():
    marker = hitl.encode_marker("task-1", "ctx-1", SECRET)
    pending = hitl.extract_pending(_messages(("assistant", "x" + marker)), SECRET)
    assert pending == {"task_id": "task-1", "context_id": "ctx-1"}


def test_parse_single_select_by_number():
    assert hitl.parse_ask_user_reply("2", [_DB_Q]) == [{"answer": ["MySQL"]}]


def test_parse_single_select_by_label_case_insensitive():
    assert hitl.parse_ask_user_reply("postgresql", [_DB_Q]) == [
        {"answer": ["PostgreSQL"]}
    ]


def test_parse_single_select_free_text_passthrough():
    assert hitl.parse_ask_user_reply("CockroachDB", [_DB_Q]) == [
        {"answer": ["CockroachDB"]}
    ]


def test_parse_free_text_question_returns_whole_reply():
    assert hitl.parse_ask_user_reply("add rate limiting", [_FREETEXT_Q]) == [
        {"answer": ["add rate limiting"]}
    ]


def test_parse_multi_select_numbers():
    assert hitl.parse_ask_user_reply("1,3", [_FEATURES_Q]) == [
        {"answer": ["Auth", "Caching"]}
    ]


def test_parse_multi_select_mixes_number_and_label():
    assert hitl.parse_ask_user_reply("auth, 3", [_FEATURES_Q]) == [
        {"answer": ["Auth", "Caching"]}
    ]


def test_parse_multi_question_one_answer_per_line():
    assert hitl.parse_ask_user_reply("1\n1,3", [_DB_Q, _FEATURES_Q]) == [
        {"answer": ["PostgreSQL"]},
        {"answer": ["Auth", "Caching"]},
    ]


def test_parse_multi_question_count_mismatch_returns_none():
    assert hitl.parse_ask_user_reply("just one line", [_DB_Q, _FEATURES_Q]) is None


def test_parse_empty_reply_returns_none():
    assert hitl.parse_ask_user_reply("   ", [_DB_Q]) is None
