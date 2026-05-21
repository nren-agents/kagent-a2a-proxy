"""
Tests for the MCP server surface exposed at /mcp.

Uses FastMCP's in-memory Client transport (no HTTP roundtrip) and respx to
mock the underlying kagent A2A calls.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastmcp import Client

from conftest import artifact_event, completed_event, sse_response, working_event
from kagent_a2a_proxy.config import settings
from kagent_a2a_proxy.mcp_server import _sanitise_tool_name, mcp


def _kagent_url(agent_name: str) -> str:
    base = str(settings.kagent_base_url).rstrip("/")
    return f"{base}/api/a2a/{settings.kagent_namespace}/{agent_name}"


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


async def test_tools_list_matches_agent_map():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    discovered = {t.name for t in tools}
    expected = {_sanitise_tool_name(k) for k in settings.agent_map}
    assert discovered == expected


@pytest.mark.parametrize(
    "agent_key,expected_tool_name",
    [
        pytest.param("agent-one", "agent_one", id="hyphenated"),
        pytest.param("agent-two", "agent_two", id="multi-word"),
    ],
)
def test_tool_name_sanitisation(agent_key: str, expected_tool_name: str):
    assert _sanitise_tool_name(agent_key) == expected_tool_name


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------


@respx.mock
async def test_tool_invocation_returns_artifact_text():
    events = [
        working_event("Looking at telemetry..."),
        artifact_event("All systems nominal."),
        completed_event(),
    ]
    route = respx.post(_kagent_url("agent-one")).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with Client(mcp) as client:
        result = await client.call_tool(
            "agent_one",
            {"prompt": "How are things?"},
        )

    assert route.called
    assert result.data == "All systems nominal."


@respx.mock
async def test_tool_invocation_with_explicit_session_id_forwards_it():
    events = [artifact_event("done"), completed_event()]
    route = respx.post(_kagent_url("agent-two")).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with Client(mcp) as client:
        await client.call_tool(
            "agent_two",
            {"prompt": "ping", "session_id": "fixed-session-42"},
        )

    body = json.loads(route.calls.last.request.content)
    assert body["params"]["sessionId"] == "fixed-session-42"


@respx.mock
async def test_tool_invocation_with_default_session_id_is_fresh_uuid():
    events = [artifact_event("ok"), completed_event()]
    route = respx.post(_kagent_url("agent-two")).mock(
        return_value=httpx.Response(
            200,
            content=sse_response(events),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with Client(mcp) as client:
        await client.call_tool("agent_two", {"prompt": "ping"})

    body = json.loads(route.calls.last.request.content)
    # A UUID is 36 chars (8-4-4-4-12 with hyphens).
    assert len(body["params"]["sessionId"]) == 36


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@respx.mock
async def test_kagent_error_surfaces_as_mcp_tool_error():
    respx.post(_kagent_url("agent-one")).mock(
        return_value=httpx.Response(503, content=b"unavailable"),
    )

    async with Client(mcp) as client:
        with pytest.raises(Exception):  # ToolError / httpx.HTTPStatusError
            await client.call_tool(
                "agent_one",
                {"prompt": "hi"},
            )
