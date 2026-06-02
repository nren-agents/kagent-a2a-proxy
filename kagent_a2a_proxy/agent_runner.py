"""
Shared high-level orchestration for invoking a kagent agent.

Used by:
  - the streaming branch of /v1/chat/completions (SSE chunks)
  - the blocking branch of /v1/chat/completions (single JSON response)
  - the MCP tools exposed under /mcp (each tool returns one final string)

`translate_stream` is the single pipeline: it streams from kagent_client,
parses + translates via translator, and drops consecutive duplicate reasoning
blocks. `collect_agent_response` drains that pipeline into a final string,
forwarding reasoning deltas to an optional `on_progress` callback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .kagent_client import stream_agent
from .models import ChatCompletionChunk
from .translator import event_to_chunks, parse_sse_line

ProgressCallback = Callable[[str], Awaitable[None]]


async def _events(
    line_stream: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed A2A event dicts from a raw SSE line stream."""
    async for raw in line_stream:
        if (event := parse_sse_line(raw)) is not None:
            yield event


def _chunk_reasoning(chunk: ChatCompletionChunk) -> str | None:
    """Return the chunk's reasoning_content text, if it carries any."""
    return next(
        (
            choice.delta.reasoning_content
            for choice in chunk.choices
            if choice.delta.reasoning_content is not None
        ),
        None,
    )


async def translate_stream(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Stream a kagent A2A call as OpenAI chunks, dropping consecutive duplicate
    reasoning blocks.

    kagent re-emits each working-state narration twice (a streaming copy and a
    turn-final copy); left alone they double up in LibreChat's "Thinking" pane.
    We suppress a reasoning chunk whose text matches the one immediately before
    it. Content chunks always pass through and reset the streak.
    """
    last_reasoning: str | None = None
    async for event in _events(stream_agent(model, messages, session_id)):
        for chunk in event_to_chunks(event, model):
            reasoning = _chunk_reasoning(chunk)
            is_duplicate = reasoning is not None and reasoning == last_reasoning
            last_reasoning = reasoning
            if not is_duplicate:
                yield chunk


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
    artifact answer plus any user-facing prompt.
    """
    parts: list[str] = []
    async for chunk in translate_stream(model, messages, session_id):
        for delta in (choice.delta for choice in chunk.choices):
            if delta.content:
                parts.append(delta.content)
            elif delta.reasoning_content and on_progress is not None:
                await on_progress(delta.reasoning_content)
    return "".join(parts)
