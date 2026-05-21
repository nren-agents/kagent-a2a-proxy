from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROXY_", env_file=".env")

    kagent_base_url: str = Field(
        default="http://kagent-controller.kagent.svc:8083",
        description="Base URL of kagent-controller A2A server",
    )
    kagent_namespace: str = Field(
        default="troubleshooting",
        description="Kubernetes namespace where agents are deployed",
    )
    # Mapping of OpenAI model name → kagent agent name.
    # Populated from PROXY_AGENT_MAP env var as JSON, e.g.:
    # '{"troubleshoot-planner":"troubleshoot-planner","telemetry":"telemetry-agent"}'
    agent_map: dict[str, str] = Field(
        default={
            "troubleshoot-planner": "troubleshoot-planner",
            "telemetry-agent": "telemetry-agent",
            "wfo-search-agent": "wfo-search-agent",
            "alarming-agent": "alarming-agent",
        },
        description="Map of model name to kagent agent name",
    )
    default_agent: str = Field(
        default="troubleshoot-planner",
        description="Fallback agent when model is not in agent_map",
    )
    request_timeout: float = Field(
        default=300.0,
        description="Timeout in seconds for kagent A2A requests",
    )
    log_level: str = Field(default="info")


settings = Settings()
