"""
Tests for ``Settings`` — the pydantic-settings model that parses env vars.

We construct ``Settings`` explicitly with kwargs (not via env) so each test
exercises the field validators in isolation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kagent_a2a_proxy.config import Settings


def test_defaults_are_empty_and_safe(monkeypatch):
    monkeypatch.delenv("PROXY_AGENT_MAP", raising=False)
    monkeypatch.delenv("PROXY_DEFAULT_AGENT", raising=False)
    s = Settings()
    assert s.agent_map == {}
    assert s.default_agent is None
    assert s.log_level == "info"
    assert s.request_timeout == 300.0


def test_kagent_base_url_must_be_a_url():
    with pytest.raises(ValidationError):
        Settings(kagent_base_url="not a url")


def test_log_level_must_be_in_literal_set():
    with pytest.raises(ValidationError):
        Settings(log_level="trace")


def test_request_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(request_timeout=0)


def test_default_agent_must_be_in_agent_map_values():
    with pytest.raises(ValidationError) as exc:
        Settings(
            agent_map={"alpha": "alpha-agent"},
            default_agent="other-agent",
        )
    assert "default_agent" in str(exc.value)


def test_default_agent_present_in_map_values_is_accepted():
    s = Settings(
        agent_map={"alpha": "alpha-agent"},
        default_agent="alpha-agent",
    )
    assert s.default_agent == "alpha-agent"
