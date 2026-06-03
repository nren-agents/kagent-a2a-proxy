"""
Human-in-the-loop (HITL) approval state, carried statelessly inside the reply.

kagent pauses a long-running tool at TaskState `input-required` and resumes when
it receives a decision DataPart on the same ``(taskId, contextId)``. LibreChat
gives us no task handle, but it resends prior assistant ``content`` verbatim, so
we embed a zero-width, render-invisible, HMAC-signed marker in the approval reply
and recover it from the conversation history on the user's next turn.

The marker is only ever placed on the *immediately preceding* assistant turn
(the approval prompt), which LibreChat never summarises away — so it is a robust
correlation key without any server-side store.
"""

from __future__ import annotations

import base64
import hmac
import json
import re
from hashlib import sha256
from typing import Any

_MARKER_VERSION = "v1"
_SIG_LEN = 16

# The marker is encoded as zero-width characters: invisible in every renderer
# (an HTML comment is shown as literal text by LibreChat's markdown), yet kept
# verbatim in LibreChat's stored message content so it survives the round-trip.
# ZWSP / ZWNJ are not stripped by JS `String.prototype.trim()`.
_BIT0 = "\u200b"  # ZERO WIDTH SPACE
_BIT1 = "\u200c"  # ZERO WIDTH NON-JOINER

_APPROVE_WORDS = frozenset(
    {"approve", "approved", "yes", "y", "ok", "okay", "confirm", "confirmed", "accept"}
)
_REJECT_WORDS = frozenset(
    {"reject", "rejected", "deny", "denied", "no", "n", "cancel", "decline", "abort"}
)


def _sign(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload_b64.encode(), sha256).hexdigest()
    return digest[:_SIG_LEN]


def _encode_zw(text: str) -> str:
    """Encode text as a run of zero-width characters (8 bits per byte)."""
    bits = "".join(format(byte, "08b") for byte in text.encode())
    return "".join(_BIT1 if bit == "1" else _BIT0 for bit in bits)


def _decode_zw(content: str) -> str:
    """Decode the zero-width characters embedded anywhere in ``content``."""
    bits = "".join(
        "1" if ch == _BIT1 else "0" for ch in content if ch in (_BIT0, _BIT1)
    )
    if not bits or len(bits) % 8:
        return ""
    decoded = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))
    return decoded.decode("utf-8", errors="ignore")


def encode_marker(
    task_id: str,
    context_id: str,
    secret: str | None,
    questions: list[dict[str, Any]] | None = None,
) -> str:
    """Return a signed, invisible (zero-width) marker, or '' when disabled.

    Disabled (returns '') when no secret is configured or there's no task to
    resume — callers then emit an informational-only prompt.

    When ``questions`` is given (an ``ask_user`` prompt), the normalized
    question structure is embedded too, so the stateless next turn can map the
    user's reply (numbers / labels / free text) back to ``ask_user_answers``.
    """
    if not secret or not task_id:
        return ""
    body_obj: dict[str, Any] = {"t": task_id, "c": context_id}
    if questions:
        body_obj["q"] = questions
    payload = (
        base64.urlsafe_b64encode(json.dumps(body_obj).encode()).decode().rstrip("=")
    )
    body = f"{_MARKER_VERSION}:{payload}:{_sign(payload, secret)}"
    return _encode_zw(body)


def extract_pending(
    messages: list[dict[str, Any]], secret: str | None
) -> dict[str, Any] | None:
    """Recover the paused task from the latest assistant marker.

    Returns ``{task_id, context_id}`` (plus ``questions`` for an ``ask_user``
    prompt), or None when HITL is disabled, no assistant marker is present, or
    the signature doesn't verify (tampered / forged).
    """
    if not secret:
        return None
    assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    content = assistant.get("content") if assistant else None
    if not isinstance(content, str):
        return None
    version, _, rest = _decode_zw(content).partition(":")
    if version != _MARKER_VERSION:
        return None
    payload_b64, _, sig = rest.partition(":")
    if not sig or not hmac.compare_digest(sig, _sign(payload_b64, secret)):
        return None
    data = _decode_payload(payload_b64)
    task_id = data.get("t")
    if not task_id:
        return None
    pending: dict[str, Any] = {
        "task_id": str(task_id),
        "context_id": str(data.get("c", "")),
    }
    questions = data.get("q")
    if isinstance(questions, list):
        pending["questions"] = questions
    return pending


def _decode_payload(payload_b64: str) -> dict[str, Any]:
    padding = "=" * (-len(payload_b64) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except (ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def classify_decision(text: str) -> str | None:
    """Map a user reply to 'approve' / 'reject', or None when ambiguous."""
    words = text.strip().lower().split()
    first = words[0].strip(".!?,") if words else ""
    if first in _APPROVE_WORDS:
        return "approve"
    if first in _REJECT_WORDS:
        return "reject"
    return None


def parse_ask_user_reply(
    text: str, questions: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Parse a free-text reply into kagent's positional ``ask_user_answers``.

    Returns a list of ``{"answer": [labels...]}`` aligned 1:1 with ``questions``,
    or None when the reply can't be mapped (empty, or — for a multi-question
    batch — the number of answer lines doesn't match the number of questions),
    so the caller can re-prompt.

    A single question consumes the whole reply. A multi-question batch expects
    one answer per line, in order.
    """
    text = text.strip()
    if not text or not questions:
        return None
    if len(questions) == 1:
        return [{"answer": _resolve_answer(text, questions[0])}]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(questions):
        return None
    return [
        {"answer": _resolve_answer(line, q)}
        for line, q in zip(lines, questions, strict=True)
    ]


def _resolve_answer(text: str, question: dict[str, Any]) -> list[str]:
    """Resolve one reply segment into the chosen label(s) for one question."""
    choices = question.get("choices") or []
    text = text.strip()
    if not choices:
        # Free-text question: the whole segment is the answer.
        return [text] if text else []
    if question.get("multiple"):
        tokens = [t.strip() for t in re.split(r"[,\n]", text) if t.strip()]
        return [_resolve_token(t, choices) for t in tokens]
    return [_resolve_token(text, choices)]


def _resolve_token(token: str, choices: list[str]) -> str:
    """Map one token to a choice by 1-based index or case-insensitive label,
    falling back to the raw token as free text."""
    if token.isdigit():
        index = int(token)
        if 1 <= index <= len(choices):
            return choices[index - 1]
    for choice in choices:
        if choice.lower() == token.lower():
            return choice
    return token
