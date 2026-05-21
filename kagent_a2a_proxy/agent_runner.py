"""
Shared high-level orchestration for invoking a kagent agent and collecting
its final text response.

Used by:
  - the blocking branch of /v1/chat/completions (single JSON response)
  - the MCP tools exposed under /mcp (each tool returns one final string)

Streams from kagent_client + translates via translator, accumulating the
artifact content. Optional `on_progress` callback fires for each reasoning
(thinking/working) delta so MCP tools can forward them as progress
notifications.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from itertools import chain
from typing import Any

from .kagent_client import stream_agent
from .models import DeltaContent
from .translator import event_to_chunks, parse_sse_line

ProgressCallback = Callable[[str], Awaitable[None]]


async def _events(
    line_stream: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed A2A event dicts from a raw SSE line stream."""
    async for raw in line_stream:
        if (event := parse_sse_line(raw)) is not None:
            yield event


def _iter_deltas(event: dict[str, Any], model: str) -> Iterator[DeltaContent]:
    """Flatten one A2A event into the delta objects across all chunks/choices."""
    return chain.from_iterable(
        (choice.delta for choice in chunk.choices)
        for chunk in event_to_chunks(event, model)
    )


async def collect_agent_response(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str,
    on_progress: ProgressCallback | None = None,
) -> str:
    """
    Drain a kagent A2A stream and return the artifact's final text.

    If `on_progress` is provided, await it once per reasoning/working delta
    (the same text that goes into LibreChat's "Thinking" pane). The returned
    string is the concatenation of all `delta.content` pieces — i.e. the
    artifact answer.
    """
    parts: list[str] = []
    async for event in _events(stream_agent(model, messages, session_id)):
        for delta in _iter_deltas(event, model):
            if delta.content:
                parts.append(delta.content)
            elif delta.reasoning_content and on_progress is not None:
                await on_progress(delta.reasoning_content)
    return "".join(parts)
