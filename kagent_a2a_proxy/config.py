"""
Runtime configuration loaded from PROXY_* environment variables (and from a
local .env file when present). All fields have validators so misconfiguration
fails fast at startup rather than at first request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kagent_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://kagent-controller.kagent.svc:8083"),
        description="Base URL of the kagent-controller A2A server.",
    )
    kagent_namespace: str = Field(
        default="default",
        description="Kubernetes namespace where kagent agents are deployed.",
    )
    agent_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "JSON map of OpenAI model name → kagent agent name. "
            "Set via PROXY_AGENT_MAP as a JSON string."
        ),
    )
    default_agent: str | None = Field(
        default=None,
        description=(
            "Fallback kagent agent name used when the requested model is not "
            "present in agent_map. Must appear as a value in agent_map."
        ),
    )
    request_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Per-request timeout (seconds) for kagent A2A calls.",
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Log level for the proxy's own logger.",
    )
    narration_mode: Literal["stream", "deemphasize"] = Field(
        default="deemphasize",
        description=(
            "How an agent's between-tool progress narration is rendered. "
            "'deemphasize' (default) collapses each narration burst into a "
            "Markdown blockquote above the final answer, keeping the answer "
            "front-and-center; the answer is emitted per-burst (from kagent's "
            "non-partial aggregate) rather than token-streamed. 'stream' "
            "token-streams all working text into the reply verbatim (the "
            "original behavior)."
        ),
    )
    hitl_secret: str | None = Field(
        default=None,
        description=(
            "Secret for HMAC-signing the human-in-the-loop approval marker "
            "embedded in assistant replies. When unset, tool-approval requests "
            "are informational only (no stateful approve/deny resume)."
        ),
    )

    @model_validator(mode="after")
    def _default_agent_in_map(self) -> Settings:
        if self.default_agent and self.default_agent not in self.agent_map.values():
            raise ValueError(
                f"default_agent {self.default_agent!r} must appear as a value "
                f"in agent_map (got values: {sorted(self.agent_map.values())!r})"
            )
        return self


settings = Settings()
