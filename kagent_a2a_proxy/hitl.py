"""
Human-in-the-loop (HITL) approval state, carried statelessly inside the reply.

kagent pauses a long-running tool at TaskState `input-required` and resumes when
it receives a decision DataPart on the same ``(taskId, contextId)``. LibreChat
gives us no task handle, but it resends prior assistant ``content`` verbatim, so
we embed an HMAC-signed, render-invisible marker in the approval reply and
recover it from the conversation history on the user's next turn.

The marker is only ever placed on the *immediately preceding* assistant turn
(the approval prompt), which LibreChat never summarises away — so it is a robust
correlation key without any server-side store.
"""

from __future__ import annotations

import base64
import hmac
import json
from hashlib import sha256
from typing import Any

_MARKER_PREFIX = "<!--kagent-hitl:v1:"
_MARKER_SUFFIX = "-->"
_SIG_LEN = 16

_APPROVE_WORDS = frozenset(
    {"approve", "approved", "yes", "y", "ok", "okay", "confirm", "confirmed", "accept"}
)
_REJECT_WORDS = frozenset(
    {"reject", "rejected", "deny", "denied", "no", "n", "cancel", "decline", "abort"}
)


def _sign(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload_b64.encode(), sha256).hexdigest()
    return digest[:_SIG_LEN]


def encode_marker(task_id: str, context_id: str, secret: str | None) -> str:
    """Return a signed, invisible HTML-comment marker, or '' when disabled.

    Disabled (returns '') when no secret is configured or there's no task to
    resume — callers then emit an informational-only approval prompt.
    """
    if not secret or not task_id:
        return ""
    payload = (
        base64.urlsafe_b64encode(json.dumps({"t": task_id, "c": context_id}).encode())
        .decode()
        .rstrip("=")
    )
    return f"\n\n{_MARKER_PREFIX}{payload}:{_sign(payload, secret)}{_MARKER_SUFFIX}"


def extract_pending(
    messages: list[dict[str, Any]], secret: str | None
) -> dict[str, str] | None:
    """Recover ``{task_id, context_id}`` from the latest assistant marker.

    Returns None when HITL is disabled, no assistant marker is present, or the
    signature doesn't verify (tampered / forged).
    """
    if not secret:
        return None
    assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    content = assistant.get("content") if assistant else None
    if not isinstance(content, str) or _MARKER_PREFIX not in content:
        return None
    payload_b64, sig = _split_marker(content)
    if not payload_b64 or not hmac.compare_digest(sig, _sign(payload_b64, secret)):
        return None
    data = _decode_payload(payload_b64)
    task_id = data.get("t")
    if not task_id:
        return None
    return {"task_id": str(task_id), "context_id": str(data.get("c", ""))}


def _split_marker(content: str) -> tuple[str, str]:
    """Return ``(payload_b64, signature)`` from the marker in ``content``."""
    start = content.rfind(_MARKER_PREFIX) + len(_MARKER_PREFIX)
    end = content.find(_MARKER_SUFFIX, start)
    if end == -1:
        return "", ""
    body = content[start:end]
    payload_b64, _, sig = body.rpartition(":")
    return payload_b64, sig


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
