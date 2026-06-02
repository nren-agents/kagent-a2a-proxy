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
from collections.abc import Iterator
from typing import Any

from .models import (
    A2ADataPart,
    A2AMessage,
    A2ATaskArtifactUpdateEvent,
    A2ATaskStatusUpdateEvent,
    ChatCompletionChunk,
    DeltaContent,
    StreamChoice,
)

logger = logging.getLogger(__name__)

# Upper bound on the rendered tool-args string in the "thinking" pane, so a
# large argument payload can't re-clutter what we're trying to declutter.
_MAX_ARGS_LEN = 160


def parse_sse_line(line: str) -> dict[str, Any] | None:
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
            return parsed  # type: ignore[no-any-return]  # json.loads returns Any


def event_to_chunks(
    raw: dict[str, Any],
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

    # ------------------------------------------------------------------
    # input-required — the agent is blocked waiting on the user. This is
    # the only user-facing status, so it goes to the visible `content`
    # channel; otherwise the question hides in LibreChat's "Thinking" pane.
    # ------------------------------------------------------------------
    if state == "input-required":
        text = _input_required_text(event)
        if text:
            yield _text_chunk(text, model)
        return

    # ------------------------------------------------------------------
    # Tool call in progress (working) — surfaced in the "thinking" channel
    # as a structured blockquote with the call's compact args.
    # ------------------------------------------------------------------
    call = _function_call(message)
    if event.is_tool_call() and call and call.get("name"):
        args = _format_tool_args(call.get("args"))
        yield _thinking_chunk(_tool_call_text(str(call["name"]), args), model)
        return

    # ------------------------------------------------------------------
    # Regular assistant narration (working state) — also "thinking"
    # ------------------------------------------------------------------
    if message and state == "working":
        text = message.text()
        if text:
            yield _thinking_chunk(f"{text.strip()}\n\n", model)
        return

    # ------------------------------------------------------------------
    # Completed — just signal stop; final answer arrives via artifact
    # ------------------------------------------------------------------
    if state == "completed":
        yield _finish_chunk(model)
        return

    # All other states (submitted, cancelled, failed) — no output


def _input_required_text(event: A2ATaskStatusUpdateEvent) -> str:
    """Render the user-facing prompt for an input-required event.

    A long-running tool confirmation becomes an approval prompt naming the
    underlying tool; any other input-required message falls back to its text.
    """
    message = event.status.message
    tool, hint = _approval_request(message)
    if tool or hint:
        target = f" for `{tool}`" if tool else ""
        suffix = f": {hint}" if hint else ""
        return f"\n⚠️ Approval required{target}{suffix}\n"
    text = message.text() if message else ""
    return f"\n> ❓ {text}\n" if text else ""


def _approval_request(message: A2AMessage | None) -> tuple[str, str]:
    """Pull (underlying tool name, confirmation hint) out of a tool-confirmation
    message, or ("", "") when this is not an approval request."""
    call = _function_call(message)
    args = call.get("args") if call else None
    if not isinstance(args, dict):
        return "", ""
    original = args.get("originalFunctionCall")
    confirmation = args.get("toolConfirmation")
    name = original.get("name", "") if isinstance(original, dict) else ""
    hint = confirmation.get("hint", "") if isinstance(confirmation, dict) else ""
    return name, hint


def _function_call(message: A2AMessage | None) -> dict[str, Any] | None:
    """Return the first data part's dict payload (a kagent function call), if any."""
    if message is None:
        return None
    return next(
        (
            part.data
            for part in message.parts
            if isinstance(part, A2ADataPart) and isinstance(part.data, dict)
        ),
        None,
    )


def _tool_call_text(name: str, args: str) -> str:
    """Format an in-progress tool call as a Markdown blockquote block."""
    args_line = f"\n> `{args}`" if args else ""
    return f"\n> 🔧 **{name}**{args_line}\n\n"


def _format_tool_args(args: Any) -> str:
    """Render tool-call args as a compact, truncated `key=value` list."""
    if not isinstance(args, dict) or not args:
        return ""
    rendered = ", ".join(f"{k}={_arg_value(v)}" for k, v in args.items())
    if len(rendered) <= _MAX_ARGS_LEN:
        return rendered
    return rendered[: _MAX_ARGS_LEN - 1] + "…"


def _arg_value(value: Any) -> str:
    """Compactly stringify a single arg value (strings quoted, rest as JSON)."""
    match value:
        case str():
            return f'"{value}"'
        case _:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _artifact_update_chunks(
    raw: dict[str, Any],
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
