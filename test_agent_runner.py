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

from conftest import (
    artifact_event,
    narration_aggregate,
    tool_call_event,
    tool_response_event,
    working_event,
)
from kagent_a2a_proxy.agent_runner import _normalize, _translate_lines
from kagent_a2a_proxy.models import ChatCompletionChunk

COMPLETED = {"kind": "status-update", "status": {"state": "completed"}}

# A trace faithful to a real kagent ADK run: each narration burst streams as
# token partials, then a non-partial aggregate bundles the full burst text with
# the tool call; the final burst (the report) has no tool call and is echoed as
# an artifact. See the captured surf-a2a-proxy stream that motivated this.
TRACE = [
    working_event("I'll start", partial=True),
    working_event(" by querying ANA.", partial=True),
    narration_aggregate("I'll start by querying ANA.", "ana_topology_agent"),
    tool_response_event("ana_topology_agent"),
    working_event("Now I", partial=True),
    working_event(" pull telemetry.", partial=True),
    narration_aggregate("Now I pull telemetry.", "telemetry_agent"),
    tool_response_event("telemetry_agent"),
    working_event("I now have", partial=True),
    working_event(" both sides.\n\n---\n\n# Report\n\nAll good.", partial=True),
    working_event(
        "I now have both sides.\n\n---\n\n# Report\n\nAll good.", partial=False
    ),
    artifact_event("I now have both sides.\n\n---\n\n# Report\n\nAll good."),
    COMPLETED,
]


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


async def test_answer_fragments_stream_to_thinking_pane(stream_mode: None) -> None:
    # The live working stream goes to the Thinking pane and joins cleanly there;
    # the main pane's answer arrives via the trailing artifact.
    content, reasoning = await _run(
        [
            working_event("Hello", partial=True),
            working_event(" world", partial=True),
            artifact_event("Hello world"),
            COMPLETED,
        ]
    )
    assert reasoning == "Hello world"
    assert content == "Hello world"


async def test_realistic_stream_has_no_double_blank_lines(stream_mode: None) -> None:
    content, reasoning = await _run(
        [
            working_event("Let me check.", thought=True, partial=True),
            tool_call_event("influxdb_query", args={"q": "x"}),
            working_event("Got results.", thought=True, partial=True),
            working_event("The answer is 42.", partial=True),
            artifact_event("The answer is 42."),
            COMPLETED,
        ]
    )
    # The answer streams live into Thinking and lands cleanly in the main pane.
    assert content == "The answer is 42."
    assert "The answer is 42." in reasoning
    assert "\n\n\n" not in reasoning
    assert "\n\n\n" not in content


# ---------------------------------------------------------------------------
# narration_mode — deemphasize (default) vs stream
# ---------------------------------------------------------------------------


async def test_deemphasize_narration_blockquoted_report_plain(
    deemphasize_mode: None,
) -> None:
    content, reasoning = await _run(TRACE)
    # Each narration burst → a de-emphasized blockquote in the visible reply.
    assert "> I'll start by querying ANA." in content
    assert "> Now I pull telemetry." in content
    # The final report → plain (not blockquoted), front-and-center, exactly once
    # (the duplicate artifact is dropped).
    assert "# Report\n\nAll good." in content
    assert "> # Report" not in content
    assert content.count("All good.") == 1
    # Tool calls render in the Thinking pane, never in the reply.
    assert "🔧 **ana_topology_agent**" in reasoning
    assert "🔧 **telemetry_agent**" in reasoning
    assert "ana_topology_agent" not in content
    # Narration is not duplicated into the Thinking pane.
    assert "I'll start by querying ANA." not in reasoning


async def test_deemphasize_skips_token_partials(deemphasize_mode: None) -> None:
    # Only the aggregate is emitted; the streamed fragments must not also leak.
    content, _ = await _run(TRACE)
    assert content.count("I'll start by querying ANA.") == 1
    assert "I'll start\n" not in content
    assert "I'll start by" not in content.replace("> I'll start by querying ANA.", "")


async def test_stream_mode_routes_live_stream_to_thinking(stream_mode: None) -> None:
    content, reasoning = await _run(TRACE)
    # The live working stream — narration, answer-in-progress, and tool calls —
    # goes to the Thinking pane verbatim (no blockquote de-emphasis).
    assert "I'll start by querying ANA." in reasoning
    assert "Now I pull telemetry." in reasoning
    assert "🔧 **ana_topology_agent**" in reasoning
    assert "🔧 **telemetry_agent**" in reasoning
    # The main pane shows only the final answer, delivered via the artifact;
    # the noisy narration never lands there.
    assert content == "I now have both sides.\n\n---\n\n# Report\n\nAll good."
    assert "I'll start by querying ANA." not in content
