"""
Translates kagent A2A TaskStatusUpdateEvents into OpenAI streaming delta chunks.

kagent event stream looks like:
  data: {"id":"...","status":{"state":"working","message":{"role":"assistant","parts":[{"kind":"text","text":"..."}]}}}
  data: {"id":"...","status":{"state":"working"},"metadata":{"kagent_type":"function_call",...}}
  data: {"id":"...","status":{"state":"completed","message":{"role":"assistant","parts":[{"kind":"text","text":"final answer"}]}}}
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

from .models import (
    A2ATaskArtifactUpdateEvent,
    A2ATaskStatusUpdateEvent,
    A2ATextPart,
    A2ADataPart,
    ChatCompletionChunk,
    DeltaContent,
    StreamChoice,
)

logger = logging.getLogger(__name__)


def parse_sse_line(line: str) -> dict | None:
    """
    Parse a single SSE data line into an A2A event dict.

    kagent wraps events in a JSON-RPC 2.0 envelope:
      {"jsonrpc":"2.0","id":"...","result":{...event...}}
      {"jsonrpc":"2.0","id":"...","error":{"code":...,"message":...}}

    We return just the event payload (the `result` field), or None for
    non-event lines, parse failures, and JSON-RPC errors.
    """
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Failed to parse SSE line: %s", line)
        return None

    match parsed:
        case {"result": dict() as result}:
            return result
        case {"error": error}:
            logger.warning("kagent JSON-RPC error: %s", error)
            return None
        case _:
            return parsed


def event_to_chunks(
    raw: dict,
    model: str,
) -> Iterator[ChatCompletionChunk]:
    """
    Convert one kagent A2A event dict to zero or more OpenAI delta chunks.

    Dispatches on the event's `kind` discriminator. Status-update events
    drive working/completed states and tool-call annotations;
    artifact-update events carry the actual assistant response text.
    """
    match raw.get("kind"):
        case "artifact-update":
            yield from _artifact_update_chunks(raw, model)
            return
        case "status-update" | None:
            # `None` for backward compat with events lacking the discriminator
            pass
        case other:
            logger.debug("Ignoring A2A event kind=%r", other)
            return

    try:
        event = A2ATaskStatusUpdateEvent.model_validate(raw)
    except Exception as exc:
        logger.warning("Could not parse status-update event (%s): %s", exc, raw)
        return

    state = event.status.state
    message = event.status.message
    is_tool_call = event.is_tool_call()
    tool_name = event.tool_name()

    # ------------------------------------------------------------------
    # Tool call in progress — surfaced in the "thinking" channel
    # ------------------------------------------------------------------
    if is_tool_call and tool_name:
        if event.is_long_running():
            hint = _approval_hint(message)
            suffix = f": {hint}" if hint else ""
            text = f"\n⚠️ Approval required for `{tool_name}`{suffix}\n"
        else:
            text = f"\n🔧 `{tool_name}`…\n"
        yield _thinking_chunk(text, model)
        return

    # ------------------------------------------------------------------
    # Regular assistant message (working state) — also "thinking"
    # ------------------------------------------------------------------
    if message and state == "working":
        text = message.text()
        if text:
            yield _thinking_chunk(text, model)
        return

    # ------------------------------------------------------------------
    # Completed — just signal stop; final answer arrives via artifact
    # ------------------------------------------------------------------
    if state == "completed":
        yield _finish_chunk(model)
        return

    # ------------------------------------------------------------------
    # input-required — agent is waiting for human input (user-facing)
    # ------------------------------------------------------------------
    if state == "input-required":
        if message:
            text = message.text()
            if text:
                yield _text_chunk(f"\n> ❓ {text}\n", model)
        return

    # All other states (submitted, cancelled, failed) — no output


def _approval_hint(message) -> str:
    """Extract the toolConfirmation hint string from a long-running tool message."""
    if not message:
        return ""
    return next(
        (
            part.data.get("args", {}).get("toolConfirmation", {}).get("hint", "")
            for part in message.parts
            if isinstance(part, A2ADataPart) and isinstance(part.data, dict)
        ),
        "",
    )


def _artifact_update_chunks(
    raw: dict,
    model: str,
) -> Iterator[ChatCompletionChunk]:
    """Emit the artifact text content as a delta chunk."""
    try:
        event = A2ATaskArtifactUpdateEvent.model_validate(raw)
    except Exception as exc:
        logger.warning("Could not parse artifact-update event (%s): %s", exc, raw)
        return

    text = event.artifact.text()
    if text:
        yield _text_chunk(text, model)


def _text_chunk(text: str, model: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        model=model,
        choices=[StreamChoice(delta=DeltaContent(content=text))],
    )


def _thinking_chunk(text: str, model: str) -> ChatCompletionChunk:
    """Emit a delta on the reasoning_content channel (LibreChat 'Thinking' pane)."""
    return ChatCompletionChunk(
        model=model,
        choices=[StreamChoice(delta=DeltaContent(reasoning_content=text))],
    )


def _finish_chunk(model: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        model=model,
        choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
    )
