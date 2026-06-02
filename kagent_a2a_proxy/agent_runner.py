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

from .kagent_client import resume_stream, stream_agent
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


def _chunk_content(chunk: ChatCompletionChunk) -> str | None:
    """Return the chunk's content text, if it carries any."""
    return next(
        (
            choice.delta.content
            for choice in chunk.choices
            if choice.delta.content is not None
        ),
        None,
    )


async def _translate_lines(
    line_stream: AsyncIterator[str],
    model: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Translate raw SSE lines to OpenAI chunks, suppressing kagent's duplicate
    answer copies.

    kagent streams the answer as `working` text partials, then re-sends it as a
    non-partial aggregate (dropped in the translator) and again as the final
    `artifact-update`. Once any answer content has streamed, we drop the
    artifact copy. We also drop a reasoning chunk identical to the one
    immediately before it (a cheap safety net for repeated thoughts).
    """
    last_reasoning: str | None = None
    content_emitted = False
    async for event in _events(line_stream):
        is_artifact = event.get("kind") == "artifact-update"
        for chunk in event_to_chunks(event, model):
            reasoning = _chunk_reasoning(chunk)
            content = _chunk_content(chunk)
            drop_artifact = is_artifact and content is not None and content_emitted
            dup_reasoning = reasoning is not None and reasoning == last_reasoning
            last_reasoning = reasoning
            if not (drop_artifact or dup_reasoning):
                content_emitted = content_emitted or bool(content)
                yield chunk


async def translate_stream(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Stream a fresh kagent A2A call as de-duplicated OpenAI chunks."""
    async for chunk in _translate_lines(
        stream_agent(model, messages, session_id), model
    ):
        yield chunk


async def translate_resume(
    model: str,
    task_id: str,
    context_id: str,
    decision: str,
    rejection_reason: str = "",
) -> AsyncIterator[ChatCompletionChunk]:
    """Resume a paused task with an approve/reject decision, as OpenAI chunks."""
    async for chunk in _translate_lines(
        resume_stream(model, task_id, context_id, decision, rejection_reason), model
    ):
        yield chunk


async def collect_response(
    chunks: AsyncIterator[ChatCompletionChunk],
    on_progress: ProgressCallback | None = None,
) -> str:
    """Drain a chunk stream into the final content string.

    If `on_progress` is provided, await it once per reasoning delta (the text
    that goes into LibreChat's "Thinking" pane).
    """
    parts: list[str] = []
    async for chunk in chunks:
        for delta in (choice.delta for choice in chunk.choices):
            if delta.content:
                parts.append(delta.content)
            elif delta.reasoning_content and on_progress is not None:
                await on_progress(delta.reasoning_content)
    return "".join(parts)


async def collect_agent_response(
    model: str,
    messages: list[dict[str, Any]],
    session_id: str,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Drain a fresh kagent A2A call into the final content string."""
    return await collect_response(
        translate_stream(model, messages, session_id), on_progress
    )
