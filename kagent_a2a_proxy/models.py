"""
Pydantic models for:
  - OpenAI chat completions request / streaming response
  - kagent A2A event stream
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# OpenAI request types
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # Ignore any other OpenAI fields silently
    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# OpenAI streaming delta types
# ---------------------------------------------------------------------------


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None
    # DeepSeek-style reasoning channel; LibreChat renders this in a
    # collapsible "Thinking" pane separate from the main response.
    reasoning_content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


# ---------------------------------------------------------------------------
# OpenAI models list response
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kagent"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# kagent A2A event types (subset we care about)
# ---------------------------------------------------------------------------


class A2ATextPart(BaseModel):
    kind: Literal["text"]
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_thought(self) -> bool:
        """True if kagent flagged this part as ADK reasoning (part.thought)."""
        return bool(self.metadata.get("kagent_thought"))


class A2ADataPart(BaseModel):
    kind: Literal["data"]
    data: Any = None


A2APart = A2ATextPart | A2ADataPart


class A2AMessage(BaseModel):
    role: str
    parts: list[A2ATextPart | A2ADataPart] = []

    def text(self) -> str:
        return " ".join(p.text for p in self.parts if isinstance(p, A2ATextPart))

    def answer_text(self) -> str:
        """Concatenated non-thought text — the user-facing answer."""
        return "".join(
            p.text
            for p in self.parts
            if isinstance(p, A2ATextPart) and not p.is_thought()
        )

    def thought_text(self) -> str:
        """Concatenated thought-flagged text — the agent's reasoning."""
        return "".join(
            p.text for p in self.parts if isinstance(p, A2ATextPart) and p.is_thought()
        )


class A2ATaskStatus(BaseModel):
    state: str
    message: A2AMessage | None = None


class A2ATaskStatusUpdateEvent(BaseModel):
    """Wraps a TaskStatusUpdateEvent from the A2A stream."""

    id: str = ""
    final: bool = False
    status: A2ATaskStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_tool_call(self) -> bool:
        return self.metadata.get("kagent_type") == "function_call"

    def is_function_response(self) -> bool:
        return self.metadata.get("kagent_type") == "function_response"

    def is_long_running(self) -> bool:
        return bool(self.metadata.get("kagent_is_long_running"))

    def is_partial(self) -> bool | None:
        """ADK streaming flag: True = streaming fragment, False = aggregated
        full copy (to be skipped), None = not signalled (non-ADK executor)."""
        value = self.metadata.get("kagent_adk_partial")
        return value if isinstance(value, bool) else None


class A2AArtifact(BaseModel):
    artifactId: str = ""
    parts: list[A2ATextPart | A2ADataPart] = []

    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, A2ATextPart))


class A2ATaskArtifactUpdateEvent(BaseModel):
    """Wraps a TaskArtifactUpdateEvent from the A2A stream."""

    kind: Literal["artifact-update"]
    artifact: A2AArtifact
    lastChunk: bool = False
    taskId: str = ""
    contextId: str = ""
