"""
Tests for the agent_runner pipeline's whitespace normalization seam.

The translator emits per-fragment text (thought fragments raw, tool blocks
wrapped in blank lines). `_translate_lines` normalizes each channel
independently so the concatenated reasoning / content panes never stack more
than one blank line, never lead with whitespace, and never lose the spaces
that join streamed thought fragments.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from conftest import tool_call_event, working_event
from kagent_a2a_proxy.agent_runner import _normalize, _translate_lines
from kagent_a2a_proxy.models import ChatCompletionChunk

COMPLETED = {"kind": "status-update", "status": {"state": "completed"}}


async def _lines(events: list[dict]) -> AsyncIterator[str]:
    for i, event in enumerate(events):
        yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': str(i), 'result': event})}"


async def _run(events: list[dict]) -> tuple[str, str]:
    """Drive events through the pipeline; return (content, reasoning) panes."""
    chunks: list[ChatCompletionChunk] = [
        c async for c in _translate_lines(_lines(events), "agent-one")
    ]
    content = "".join(c.choices[0].delta.content or "" for c in chunks)
    reasoning = "".join(c.choices[0].delta.reasoning_content or "" for c in chunks)
    return content, reasoning


# ---------------------------------------------------------------------------
# _normalize — pure per-channel boundary normalizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,tail,expected_text,expected_tail",
    [
        pytest.param("\n\nHello", None, "Hello", 0, id="start-strips-leading"),
        pytest.param("\n\n", None, "", None, id="start-all-newlines-emits-nothing"),
        pytest.param("\n> tool\n\n", 2, "> tool\n\n", 2, id="boundary-caps-to-two"),
        pytest.param(
            "\n\n> tool\n\n", 0, "\n\n> tool\n\n", 2, id="boundary-keeps-one-blank"
        ),
        pytest.param("\n\nmore", 1, "\nmore", 0, id="boundary-partial-trim"),
        pytest.param("a\n\n\n\nb", 0, "a\n\nb", 0, id="collapse-internal-run"),
        pytest.param("\n\n", 1, "\n", 2, id="mid-all-newlines-capped"),
        pytest.param(
            " both segments.", 0, " both segments.", 0, id="plain-join-preserved"
        ),
    ],
)
def test_normalize(
    text: str, tail: int | None, expected_text: str, expected_tail: int | None
) -> None:
    assert _normalize(text, tail) == (expected_text, expected_tail)


# ---------------------------------------------------------------------------
# _translate_lines — reasoning pane formatting
# ---------------------------------------------------------------------------


async def test_thought_fragments_join_without_injected_blank_lines() -> None:
    content, reasoning = await _run(
        [
            working_event("I now have", thought=True, partial=True),
            working_event(" both segments.", thought=True, partial=True),
        ]
    )
    assert reasoning == "I now have both segments."
    assert content == ""


async def test_tool_block_separated_by_exactly_one_blank_line() -> None:
    _, reasoning = await _run(
        [
            working_event("First thought.\n\n", thought=True, partial=True),
            tool_call_event("influxdb_query", args={"range": "1h"}),
            working_event("Second thought.", thought=True, partial=True),
        ]
    )
    assert "\n\n\n" not in reasoning
    assert "First thought.\n\n> 🔧 **influxdb_query**" in reasoning
    assert "Second thought." in reasoning


async def test_reasoning_never_leads_with_whitespace() -> None:
    _, reasoning = await _run([tool_call_event("list_agents", args={"limit": 5})])
    assert not reasoning.startswith("\n")
    assert reasoning.startswith("> 🔧 **list_agents**")


# ---------------------------------------------------------------------------
# _translate_lines — visible content stays clean
# ---------------------------------------------------------------------------


async def test_answer_fragments_pass_through_clean() -> None:
    content, reasoning = await _run(
        [
            working_event("Hello", partial=True),
            working_event(" world", partial=True),
            COMPLETED,
        ]
    )
    assert content == "Hello world"
    assert reasoning == ""


async def test_realistic_stream_has_no_double_blank_lines() -> None:
    content, reasoning = await _run(
        [
            working_event("Let me check.", thought=True, partial=True),
            tool_call_event("influxdb_query", args={"q": "x"}),
            working_event("Got results.", thought=True, partial=True),
            working_event("The answer is 42.", partial=True),
            COMPLETED,
        ]
    )
    assert content == "The answer is 42."
    assert "\n\n\n" not in reasoning
    assert "\n\n\n" not in content
