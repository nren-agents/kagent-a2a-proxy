"""
Tests for the model → agent resolution layer.

Conftest sets PROXY_AGENT_MAP={"agent-one":"agent-one","agent-two":"agent-two"}
and PROXY_DEFAULT_AGENT="agent-one", so:
  - known model → its mapped agent
  - unknown model → the configured default
  - unknown model with default cleared → raises ValueError
"""

from __future__ import annotations

import pytest

from kagent_a2a_proxy import kagent_client
from kagent_a2a_proxy.config import settings


def test_known_model_resolves_to_mapped_agent():
    assert kagent_client._resolve_agent("agent-one") == "agent-one"


def test_unknown_model_resolves_to_default(monkeypatch):
    assert kagent_client._resolve_agent("not-in-map") == settings.default_agent


def test_unknown_model_with_no_default_raises(monkeypatch):
    monkeypatch.setattr(settings, "default_agent", None)
    with pytest.raises(ValueError) as exc:
        kagent_client._resolve_agent("not-in-map")
    assert "not-in-map" in str(exc.value)
