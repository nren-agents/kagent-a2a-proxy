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

from .config import settings
from .hitl import encode_marker
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

    # ------------------------------------------------------------------
    # input-required — the agent is blocked waiting on the user → visible
    # `content` so the prompt isn't hidden in LibreChat's "Thinking" pane.
    # ------------------------------------------------------------------
    if state == "input-required":
        text = _input_required_text(event)
        if text:
            yield _text_chunk(text, model)
        return

    # ------------------------------------------------------------------
    # Terminal failure / auth states — surface a visible notice + stop,
    # otherwise the stream just ends silently with no answer.
    # ------------------------------------------------------------------
    if state in ("failed", "canceled", "rejected", "auth-required"):
        yield _text_chunk(_terminal_notice(state, event.status.message), model)
        yield _finish_chunk(model)
        return

    # ------------------------------------------------------------------
    # Completed — emit the final message text, then signal stop.
    #
    # Standard a2a-sdk agents (e.g. wfo-search) put their answer/summary in the
    # completed status message. kagent agents instead carry the answer in an
    # artifact-update and leave this message empty, so emitting it here is a
    # no-op for them.
    # ------------------------------------------------------------------
    if state == "completed":
        message = event.status.message
        if message:
            text = message.text()
            if text:
                yield _text_chunk(text, model)
        yield _finish_chunk(model)
        return

    # ------------------------------------------------------------------
    # working — route each part: tool calls + thoughts → "thinking",
    # the agent's prose answer → visible `content`.
    # ------------------------------------------------------------------
    if state == "working":
        yield from _working_chunks(event, model)

    # submitted and any other states — no output


def _working_chunks(
    event: A2ATaskStatusUpdateEvent,
    model: str,
) -> Iterator[ChatCompletionChunk]:
    """Route a working-state event's parts per the configured narration mode."""
    match settings.narration_mode:
        case "deemphasize":
            yield from _working_deemphasize(event, model)
        case _:
            yield from _working_stream(event, model)


def _tool_call_chunks(
    message: A2AMessage | None, model: str
) -> Iterator[ChatCompletionChunk]:
    """Render an in-progress tool call into the Thinking pane (results dropped)."""
    call = _function_call(message)
    name = call.get("name") if call else None
    # adk_request_confirmation is the approval mechanism itself — it surfaces
    # via the input-required branch (⚠️), not as a 🔧 tool line.
    if call and name and name != "adk_request_confirmation":
        args = _format_tool_args(call.get("args"))
        yield _thinking_chunk(_tool_call_text(str(name), args), model)


def _working_stream(
    event: A2ATaskStatusUpdateEvent, model: str
) -> Iterator[ChatCompletionChunk]:
    """Original behavior: token-stream every working text part into the reply.

    Thought fragments and tool calls go to the Thinking pane; the agent's prose
    (narration and answer alike) streams verbatim into `content`. The non-partial
    aggregate copy (is_partial() is False) is skipped to avoid duplication.
    """
    message = event.status.message
    if event.is_tool_call():
        yield from _tool_call_chunks(message, model)
        return
    if event.is_function_response():
        return
    if message is None or event.is_partial() is False:
        return
    # Emit thought fragments verbatim: the model's own newlines carry the
    # paragraph structure, and agent_runner normalizes the channel as a whole.
    if thought := message.thought_text():
        yield _thinking_chunk(thought, model)
    if answer := message.answer_text():
        yield _text_chunk(answer, model)


def _working_deemphasize(
    event: A2ATaskStatusUpdateEvent, model: str
) -> Iterator[ChatCompletionChunk]:
    """De-emphasize between-tool narration so the final answer stays prominent.

    kagent emits each narration burst as token partials followed by a non-partial
    aggregate that bundles the full burst text with the tool call it triggered;
    the final burst (the answer) has no tool call. So the aggregate is the
    authoritative per-burst copy: skip the partials, render an aggregate that
    carries a tool call as a blockquoted narration block plus the tool call, and
    an aggregate with no tool call as the plain answer. Non-ADK executors (no
    aggregate; is_partial() is None) emit their text directly.
    """
    message = event.status.message
    partial = event.is_partial()

    if event.is_tool_call():
        if (
            message is not None
            and partial is not True
            and (narration := message.answer_text().strip())
        ):
            yield _text_chunk(_narration_block(narration), model)
        yield from _tool_call_chunks(message, model)
        return
    if event.is_function_response():
        return
    if message is None:
        return
    # Thoughts still stream live (from partial fragments) to the Thinking pane.
    if partial is not False and (thought := message.thought_text()):
        yield _thinking_chunk(thought, model)
    # Answer text is taken from the aggregate (is_partial() is False) or, for
    # non-ADK executors, directly (is_partial() is None). Raw partials are
    # skipped — the aggregate carries the full burst text.
    if partial is True:
        return
    if answer := message.answer_text():
        yield _text_chunk(answer, model)


def _narration_block(text: str) -> str:
    """Render between-tool progress narration as a de-emphasized blockquote."""
    quoted = "\n".join(f"> {line}".rstrip() for line in text.splitlines())
    return f"\n\n{quoted}\n\n"


def _terminal_notice(state: str, message: A2AMessage | None) -> str:
    """Visible one-line notice for a non-completed terminal / auth state."""
    detail = message.text().strip() if message else ""
    match state:
        case "auth-required":
            label = "🔐 Authentication required to continue"
        case "failed":
            label = "⚠️ Agent run failed"
        case "canceled":
            label = "⚠️ Agent run was canceled"
        case _:
            label = "⚠️ Agent request was rejected"
    suffix = f": {detail}" if detail else ""
    return f"\n{label}{suffix}\n"


def _input_required_text(event: A2ATaskStatusUpdateEvent) -> str:
    """Render the user-facing prompt for an input-required event.

    The built-in ``ask_user`` tool becomes a question prompt listing its
    predefined choices; any other long-running confirmation becomes an approval
    prompt naming the underlying tool; anything else falls back to its text.
    """
    message = event.status.message
    original = _original_function_call(message)
    if original is not None and original.get("name") == "ask_user":
        questions = _ask_user_questions(original)
        if questions:
            marker = encode_marker(
                event.taskId, event.contextId, settings.hitl_secret, questions
            )
            return render_ask_user(questions, marker)
    approvals = _approval_requests(message)
    if approvals:
        # When HITL is enabled, embed a signed marker so the user's next
        # "approve"/"deny" reply can be routed back to this paused task.
        marker = encode_marker(event.taskId, event.contextId, settings.hitl_secret)
        return _render_approval_prompt(approvals, marker)
    text = message.text() if message else ""
    return f"\n> ❓ {text}\n" if text else ""


def _approval_requests(message: A2AMessage | None) -> list[tuple[str, str]]:
    """(underlying tool name, confirmation hint) for every tool-confirmation part.

    The agent can fire several approval-required tools in one turn (parallel
    function calls), so each pending confirmation rides its own data part — not
    just the first."""
    return [
        approval
        for call in _function_calls(message)
        if (approval := _approval_from_call(call)) is not None
    ]


def _approval_from_call(call: dict[str, Any]) -> tuple[str, str] | None:
    """Pull (tool name, hint) from one confirmation call, or None when it isn't one."""
    args = call.get("args")
    if not isinstance(args, dict):
        return None
    original = args.get("originalFunctionCall")
    confirmation = args.get("toolConfirmation")
    name = original.get("name", "") if isinstance(original, dict) else ""
    hint = confirmation.get("hint", "") if isinstance(confirmation, dict) else ""
    if not name and not hint:
        return None
    return name, hint


def _render_approval_prompt(approvals: list[tuple[str, str]], marker: str) -> str:
    """Render one or more pending tool approvals as a single visible prompt.

    A uniform approve/deny resume covers every pending call, so all of them are
    surfaced under one instruction rather than only the first."""
    instruction = " — reply **approve** or **deny** to continue" if marker else ""
    if len(approvals) == 1:
        name, hint = approvals[0]
        target = f" for `{name}`" if name else ""
        suffix = f": {hint}" if hint else ""
        return f"\n⚠️ Approval required{target}{suffix}{instruction}\n{marker}"
    header = f"\n⚠️ Approval required for {len(approvals)} tool calls{instruction}:"
    items = "\n".join(_approval_line(name, hint) for name, hint in approvals)
    return f"{header}\n{items}\n{marker}"


def _approval_line(name: str, hint: str) -> str:
    label = f"`{name}`" if name else "tool"
    suffix = f": {hint}" if hint else ""
    return f"- {label}{suffix}"


def _original_function_call(message: A2AMessage | None) -> dict[str, Any] | None:
    """Return the ``originalFunctionCall`` wrapped inside an
    ``adk_request_confirmation`` data part, if present."""
    call = _function_call(message)
    args = call.get("args") if call else None
    if not isinstance(args, dict):
        return None
    original = args.get("originalFunctionCall")
    return original if isinstance(original, dict) else None


def _ask_user_questions(original: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the ``ask_user`` questions list to ``{question, choices,
    multiple}`` dicts, dropping any entry without question text."""
    args = original.get("args")
    raw = args.get("questions") if isinstance(args, dict) else None
    items = raw if isinstance(raw, list) else []
    return [q for q in map(_normalize_question, items) if q is not None]


def _normalize_question(item: Any) -> dict[str, Any] | None:
    """Coerce one raw ask_user question into the normalized shape, or None."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("question", "")).strip()
    if not text:
        return None
    raw_choices = item.get("choices")
    choices = [str(c) for c in raw_choices] if isinstance(raw_choices, list) else []
    return {
        "question": text,
        "choices": choices,
        "multiple": bool(item.get("multiple", False)),
    }


def render_ask_user(questions: list[dict[str, Any]], marker: str) -> str:
    """Render ask_user questions as visible, numbered Markdown + the resume marker.

    Shared by the initial input-required prompt and main's re-prompt on an
    unparseable reply, so both render identically.

    A single question gets a compact numbered list; a batch numbers each question
    and asks for one answer per line. The marker (empty when HITL is disabled)
    carries the question structure so the next reply can be mapped to answers.
    """
    if len(questions) == 1:
        body = _render_one_question(questions[0])
    else:
        body = _render_question_batch(questions)
    return f"{body}\n{marker}"


def _render_one_question(question: dict[str, Any]) -> str:
    lines = [f"\n❓ {question['question']}"]
    lines += [f"{i}. {choice}" for i, choice in enumerate(question["choices"], 1)]
    if question["choices"]:
        lines += ["", _select_hint(question["multiple"])]
    return "\n".join(lines)


def _render_question_batch(questions: list[dict[str, Any]]) -> str:
    lines = ["\nThe agent needs a few answers to continue:\n"]
    for qi, question in enumerate(questions, 1):
        suffix = " _(select all that apply)_" if question["multiple"] else ""
        lines.append(f"**{qi}.** {question['question']}{suffix}")
        lines += [f"{i}. {choice}" for i, choice in enumerate(question["choices"], 1)]
        lines.append("")
    lines.append("_Answer each question on its own line, in order._")
    return "\n".join(lines)


def _select_hint(multiple: bool) -> str:
    if multiple:
        return "_Reply with the numbers (e.g. `1,3`), the option text, or your own answer._"
    return "_Reply with the number, the option text, or your own answer._"


def _function_calls(message: A2AMessage | None) -> list[dict[str, Any]]:
    """Every data part's dict payload (kagent function calls), in order."""
    if message is None:
        return []
    return [
        part.data
        for part in message.parts
        if isinstance(part, A2ADataPart) and isinstance(part.data, dict)
    ]


def _function_call(message: A2AMessage | None) -> dict[str, Any] | None:
    """Return the first data part's dict payload (a kagent function call), if any."""
    return next(iter(_function_calls(message)), None)


def _tool_call_text(name: str, args: str) -> str:
    """Format an in-progress tool call as a Markdown blockquote block."""
    args_line = f"\n> `{args}`" if args else ""
    return f"\n> 🔧 **{name}**{args_line}\n\n"


def _format_tool_args(args: Any) -> str:
    """Render tool-call args as a compact `key=value` list."""
    if not isinstance(args, dict) or not args:
        return ""
    return ", ".join(f"{k}={_arg_value(v)}" for k, v in args.items())


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
