# SURF-agnostic Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the SURF-flavored proxy into a generic MIT-licensed open-source repo: rename the package, drop hardcoded defaults, harden config validation, ship generic example manifests, add README/LICENSE/.env.example, and add a CI workflow that tests on Python 3.12/3.13/3.14 and publishes to GHCR.

**Architecture:** No functional changes to translation, MCP, or A2A logic. Renames `surf_a2a_proxy` → `kagent_a2a_proxy`, tightens `Settings` (AnyHttpUrl, Literal log_level, validator), drops SURF defaults, makes unresolvable models raise instead of falling back to a string-`None` URL, and adds GHCR publishing via a single workflow.

**Tech Stack:** FastAPI, FastMCP, httpx, pydantic + pydantic-settings, uv, Docker, GitHub Actions.

**Spec reference:** `docs/superpowers/specs/2026-05-21-surf-agnostic-design.md`

---

## File Structure

**Created:**
- `kagent_a2a_proxy/` (rename of `surf_a2a_proxy/`)
- `LICENSE`
- `README.md`
- `.env.example`
- `examples/deploy.yaml`
- `examples/librechat-config.yaml`
- `.github/workflows/ci.yml`
- `deploy.yaml.local` (rename of `deploy.yaml`, gitignored)
- `librechat-config.yaml.local` (rename of `10-librechat-config.yaml`, gitignored)

**Modified:**
- `Dockerfile` — package path
- `pyproject.toml` — name, description, packages, drop legacy optional-deps block
- `.gitignore` — append `.local` manifest names
- `conftest.py` — set `PROXY_AGENT_MAP` env var before imports
- `test_api.py`, `test_mcp_server.py`, `test_translator.py` — update imports + agent names
- All modules in the package — package path updates only (mechanical)

**Renamed/moved internally:**
- `kagent_a2a_proxy/config.py` — full rewrite (validation, empty defaults)
- `kagent_a2a_proxy/kagent_client.py` — `_resolve_agent` raises when no agent resolvable
- `kagent_a2a_proxy/main.py` — FastAPI title/description string
- `kagent_a2a_proxy/mcp_server.py` — FastMCP server name string
- `kagent_a2a_proxy/models.py` — `owned_by` default string

---

## Task 1: Test isolation — set PROXY_AGENT_MAP in conftest

The existing tests rely on the agent_map containing SURF-named agents because `config.py` ships those as defaults. Once we drop those defaults, tests must populate their own map. The cleanest way is to set env vars in `conftest.py` BEFORE any module under test gets imported — pydantic-settings reads env at `Settings()` construction time, which happens at import of `config.py`.

We do this *before* the rename so the existing test suite continues to pass after the agent_map default goes empty.

**Files:**
- Modify: `conftest.py`

- [ ] **Step 1: Set test env vars at top of conftest**

Replace the entire contents of `conftest.py` with:

```python
"""
Shared test fixtures and event builders.

Sets PROXY_AGENT_MAP before any module imports so that ``Settings()`` —
constructed at import of ``kagent_a2a_proxy.config`` — sees a populated
agent_map even though the production default is empty.

kagent's A2A stream uses the pre-v1.0 protocol shape and wraps each event in
a JSON-RPC 2.0 envelope. The helpers below produce realistic event dicts and
SSE bytes so test files don't each reinvent them.
"""
from __future__ import annotations

import json
import os

# Must run before any ``from kagent_a2a_proxy ...`` import.
os.environ.setdefault(
    "PROXY_AGENT_MAP",
    json.dumps(
        {
            "agent-one": "agent-one",
            "agent-two": "agent-two",
        }
    ),
)
os.environ.setdefault("PROXY_DEFAULT_AGENT", "agent-one")


def sse_response(events: list[dict]) -> bytes:
    """Build a fake SSE response body, wrapping each event in a JSON-RPC envelope."""
    lines = [
        f"data: {json.dumps({'jsonrpc': '2.0', 'id': str(i), 'result': e})}\n\n"
        for i, e in enumerate(events)
    ]
    return "".join(lines).encode()


def working_event(text: str) -> dict:
    return {
        "kind": "status-update",
        "status": {
            "state": "working",
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": text}],
            },
        },
        "metadata": {},
    }


def artifact_event(text: str) -> dict:
    return {
        "kind": "artifact-update",
        "artifact": {
            "artifactId": "a-1",
            "parts": [{"kind": "text", "text": text}],
        },
        "lastChunk": True,
        "taskId": "t-1",
        "contextId": "c-1",
    }


def completed_event() -> dict:
    return {
        "kind": "status-update",
        "status": {"state": "completed"},
        "metadata": {},
    }
```

- [ ] **Step 2: Verify tests still pass with the pre-rename codebase**

Run: `uv sync --all-groups && uv run pytest -q`
Expected: all tests pass (conftest still imports nothing under test yet — env vars take effect for later tasks).

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "test: set PROXY_AGENT_MAP in conftest before module imports"
```

---

## Task 2: Rename package directory + update all imports

Pure mechanical: `surf_a2a_proxy/` → `kagent_a2a_proxy/`. Updates every import string across tests, the Dockerfile, and `pyproject.toml`. No behavior change yet.

**Files:**
- Rename: `surf_a2a_proxy/` → `kagent_a2a_proxy/`
- Modify: `Dockerfile`
- Modify: `pyproject.toml`
- Modify: `test_api.py`
- Modify: `test_mcp_server.py`
- Modify: `test_translator.py`

- [ ] **Step 1: Rename the package directory**

```bash
git mv surf_a2a_proxy kagent_a2a_proxy
```

(`git mv` works even on an unstaged directory because nothing is tracked yet — equivalent to a plain `mv` here. Use `mv surf_a2a_proxy kagent_a2a_proxy` if `git mv` complains about the untracked state.)

- [ ] **Step 2: Confirm no leftover references**

Run: `grep -rln 'surf_a2a_proxy' --include='*.py' --include='Dockerfile' --include='*.toml' --include='*.yaml' .`
Expected: lists `test_api.py`, `test_mcp_server.py`, `test_translator.py`, `Dockerfile`, `pyproject.toml`.

- [ ] **Step 3: Update Dockerfile**

Edit `Dockerfile`. Replace:

```dockerfile
# Copy application code (full package directory)
COPY surf_a2a_proxy ./surf_a2a_proxy/


# Sync again to install the project itself
RUN uv sync --no-dev
```

with:

```dockerfile
# Copy application code (full package directory)
COPY kagent_a2a_proxy ./kagent_a2a_proxy/


# Sync again to install the project itself
RUN uv sync --no-dev
```

And replace:

```dockerfile
CMD ["uv", "run", "uvicorn", "surf_a2a_proxy.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
```

with:

```dockerfile
CMD ["uv", "run", "uvicorn", "kagent_a2a_proxy.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
```

- [ ] **Step 4: Update pyproject.toml**

Edit `pyproject.toml`. Replace the `[project]` block top with:

```toml
[project]
name = "kagent-a2a-proxy"
version = "0.1.0"
description = "OpenAI-compatible streaming chat completions and MCP server for kagent agents"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "fastmcp>=2.0",
]
```

Delete the entire `[project.optional-dependencies]` block (lines 15-21 in the original) — its dev deps are duplicated in `[dependency-groups].dev`, which is what `uv sync` honors.

Replace the wheel target:

```toml
[tool.hatch.build.targets.wheel]
packages = ["surf_a2a_proxy"]
```

with:

```toml
[tool.hatch.build.targets.wheel]
packages = ["kagent_a2a_proxy"]
```

The resulting file is:

```toml
[project]
name = "kagent-a2a-proxy"
version = "0.1.0"
description = "OpenAI-compatible streaming chat completions and MCP server for kagent agents"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "fastmcp>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["kagent_a2a_proxy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[dependency-groups]
dev = [
    "httpx>=0.28.1",
    "pytest>=9.0.3",
    "pytest-asyncio>=1.3.0",
    "respx>=0.23.1",
]
```

- [ ] **Step 5: Update test_api.py imports**

Edit `test_api.py`. Replace the two `from surf_a2a_proxy...` lines:

```python
from surf_a2a_proxy.main import app
from surf_a2a_proxy.config import settings
```

with:

```python
from kagent_a2a_proxy.main import app
from kagent_a2a_proxy.config import settings
```

- [ ] **Step 6: Update test_mcp_server.py imports**

Edit `test_mcp_server.py`. Replace:

```python
from surf_a2a_proxy.config import settings
from surf_a2a_proxy.mcp_server import _sanitise_tool_name, mcp
```

with:

```python
from kagent_a2a_proxy.config import settings
from kagent_a2a_proxy.mcp_server import _sanitise_tool_name, mcp
```

- [ ] **Step 7: Update test_translator.py imports**

Edit `test_translator.py`. Replace:

```python
from surf_a2a_proxy.translator import event_to_chunks, parse_sse_line
```

with:

```python
from kagent_a2a_proxy.translator import event_to_chunks, parse_sse_line
```

- [ ] **Step 8: Re-sync uv (package name changed)**

Run: `uv sync --all-groups`
Expected: succeeds. The `uv.lock` may be rewritten — that's fine.

- [ ] **Step 9: Run tests — they'll still pass because the agent_map default still contains the SURF names**

Run: `uv run pytest -q`
Expected: all green. Tests still reference `troubleshoot-planner` and `telemetry-agent` — that's resolved in Task 5.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: rename package surf_a2a_proxy → kagent_a2a_proxy"
```

---

## Task 3: Rewrite Settings — validation, empty defaults, optional default_agent

Adds `AnyHttpUrl` URL validation, `Literal` log level, positive-only timeout, empty `agent_map`, `Optional[str]` `default_agent`, and a `model_validator` that verifies `default_agent` (when set) is actually one of the kagent agent names in the map. After this task `Settings()` will not contain any SURF defaults.

Test first: the validator behavior is new code, so it gets unit tests.

**Files:**
- Modify: `kagent_a2a_proxy/config.py`
- Create: `test_config.py`

- [ ] **Step 1: Write the failing tests for Settings validation**

Create `test_config.py`:

```python
"""
Tests for ``Settings`` — the pydantic-settings model that parses env vars.

We construct ``Settings`` explicitly with kwargs (not via env) so each test
exercises the field validators in isolation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kagent_a2a_proxy.config import Settings


def test_defaults_are_empty_and_safe():
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest test_config.py -v`
Expected: All tests FAIL (`AnyHttpUrl` validation absent, `default_agent` validator absent, etc.).

- [ ] **Step 3: Rewrite config.py**

Replace the entire contents of `kagent_a2a_proxy/config.py` with:

```python
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
        default="http://kagent-controller.kagent.svc:8083",
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

    @model_validator(mode="after")
    def _default_agent_in_map(self) -> "Settings":
        if self.default_agent and self.default_agent not in self.agent_map.values():
            raise ValueError(
                f"default_agent {self.default_agent!r} must appear as a value "
                f"in agent_map (got values: {sorted(self.agent_map.values())!r})"
            )
        return self


settings = Settings()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test_config.py -v`
Expected: all six tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kagent_a2a_proxy/config.py test_config.py
git commit -m "feat(config): tighten Settings validation, drop SURF defaults"
```

---

## Task 4: Make `_resolve_agent` raise when no agent is configured

With `default_agent` now possibly `None`, the current `_resolve_agent` would return `None` for unknown models, and `_a2a_url(None)` would produce a URL containing the literal string `"None"`. Replace the silent fallback with an explicit `ValueError`. Adjust `main.py` so this surfaces as an HTTP 400 on `/v1/chat/completions`.

**Files:**
- Modify: `kagent_a2a_proxy/kagent_client.py:41-43`
- Modify: `kagent_a2a_proxy/main.py:88-114` (chat_completions handler)
- Create: `test_resolve_agent.py`

- [ ] **Step 1: Write the failing test for `_resolve_agent`**

Create `test_resolve_agent.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify the no-default case fails**

Run: `uv run pytest test_resolve_agent.py -v`
Expected: `test_unknown_model_with_no_default_raises` FAILS (current code returns `None`).

- [ ] **Step 3: Update `_resolve_agent` in kagent_client.py**

Edit `kagent_a2a_proxy/kagent_client.py`. Replace:

```python
def _resolve_agent(model: str) -> str:
    """Map an OpenAI model name to a kagent agent name."""
    return settings.agent_map.get(model, settings.default_agent)
```

with:

```python
def _resolve_agent(model: str) -> str:
    """Map an OpenAI model name to a kagent agent name.

    Falls back to ``settings.default_agent`` for unknown models. Raises
    ``ValueError`` if the model is not in ``agent_map`` and no default agent
    is configured — callers should surface this as a 4xx to the client.
    """
    if (agent := settings.agent_map.get(model)) is not None:
        return agent
    if settings.default_agent is not None:
        return settings.default_agent
    raise ValueError(
        f"Model {model!r} is not in PROXY_AGENT_MAP and PROXY_DEFAULT_AGENT "
        f"is not set"
    )
```

- [ ] **Step 4: Run test_resolve_agent.py to verify all three pass**

Run: `uv run pytest test_resolve_agent.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Surface the resolve error as HTTP 400 in main.py**

The streaming branch already wraps `stream_agent` in a try/except that turns errors into a stream-error chunk; that path will catch `ValueError` raised by `_resolve_agent` (called inside `stream_agent`) and emit it as an `[Error: ...]` chunk. We want the non-streaming branch to return a clean 400 instead of crashing with a 500.

Edit `kagent_a2a_proxy/main.py`. Replace:

```python
    if req.stream:
        return StreamingResponse(
            _stream_response(req.model, messages, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            },
        )
    else:
        # Non-streaming: accumulate all chunks into one response
        return await _blocking_response(req.model, messages, session_id)
```

with:

```python
    if req.stream:
        return StreamingResponse(
            _stream_response(req.model, messages, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            },
        )
    try:
        return await _blocking_response(req.model, messages, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

- [ ] **Step 6: Run the full test suite to confirm nothing regressed**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add kagent_a2a_proxy/kagent_client.py kagent_a2a_proxy/main.py test_resolve_agent.py
git commit -m "feat: raise ValueError when no agent can be resolved for a model"
```

---

## Task 5: Generalize string identifiers in package modules

Replaces remaining SURF-flavored strings inside the package — none of these are functional behavior, just identifiers shown to clients (FastAPI doc title, MCP server name, `/v1/models` owned_by).

**Files:**
- Modify: `kagent_a2a_proxy/main.py:56-60`
- Modify: `kagent_a2a_proxy/mcp_server.py:52`
- Modify: `kagent_a2a_proxy/models.py:70`

- [ ] **Step 1: Update FastAPI app title/description**

Edit `kagent_a2a_proxy/main.py`. Replace:

```python
app = FastAPI(
    title="surf-a2a-proxy",
    description="Translates OpenAI streaming chat completions and MCP tool calls to kagent A2A",
    version="0.1.0",
    lifespan=lifespan,
)
```

with:

```python
app = FastAPI(
    title="kagent-a2a-proxy",
    description="OpenAI-compatible streaming chat completions and MCP server backed by kagent A2A",
    version="0.1.0",
    lifespan=lifespan,
)
```

- [ ] **Step 2: Update FastMCP server name**

Edit `kagent_a2a_proxy/mcp_server.py`. Replace:

```python
    mcp = FastMCP("surf-a2a-proxy")
```

with:

```python
    mcp = FastMCP("kagent-a2a-proxy")
```

- [ ] **Step 3: Update `ModelObject.owned_by` default**

Edit `kagent_a2a_proxy/models.py`. Replace:

```python
class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "surf"
```

with:

```python
class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kagent"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add kagent_a2a_proxy/main.py kagent_a2a_proxy/mcp_server.py kagent_a2a_proxy/models.py
git commit -m "refactor: replace SURF identifiers with generic kagent-a2a-proxy names"
```

---

## Task 6: Update tests to use generic agent names

Tests currently hardcode `troubleshoot-planner`, `telemetry-agent`, `alarming-agent`. With the new test agent_map (set in conftest as `agent-one` / `agent-two`), update the test agent references to match.

**Files:**
- Modify: `test_api.py`
- Modify: `test_mcp_server.py`
- Modify: `test_translator.py`

- [ ] **Step 1: Update test_api.py**

Edit `test_api.py`. Replace the `KAGENT_URL` line and every test reference to SURF agent names:

```python
KAGENT_URL = (
    f"{settings.kagent_base_url}/api/a2a"
    f"/{settings.kagent_namespace}/troubleshoot-planner"
)
```

becomes:

```python
KAGENT_URL = (
    f"{settings.kagent_base_url}/api/a2a"
    f"/{settings.kagent_namespace}/agent-one"
)
```

In `test_list_models`, replace:

```python
    assert "troubleshoot-planner" in model_ids
```

with:

```python
    assert "agent-one" in model_ids
```

In `test_non_streaming_completion`, replace the JSON body model field:

```python
        "model": "troubleshoot-planner",
```

with:

```python
        "model": "agent-one",
```

Apply the same `troubleshoot-planner` → `agent-one` substitution in `test_streaming_completion` and `test_kagent_error_surfaced_in_stream`.

- [ ] **Step 2: Update test_mcp_server.py**

Edit `test_mcp_server.py`.

Replace the parametrize block:

```python
@pytest.mark.parametrize(
    "agent_key,expected_tool_name",
    [
        pytest.param("troubleshoot-planner", "troubleshoot_planner", id="hyphenated"),
        pytest.param("alarming-agent", "alarming_agent", id="multi-word"),
    ],
)
```

with:

```python
@pytest.mark.parametrize(
    "agent_key,expected_tool_name",
    [
        pytest.param("agent-one", "agent_one", id="hyphenated"),
        pytest.param("agent-two", "agent_two", id="multi-word"),
    ],
)
```

In `test_tool_invocation_returns_artifact_text`, replace the URL agent and the tool name:

```python
    route = respx.post(_kagent_url("troubleshoot-planner")).mock(
```

with:

```python
    route = respx.post(_kagent_url("agent-one")).mock(
```

and:

```python
        result = await client.call_tool(
            "troubleshoot_planner",
            {"prompt": "How are things?"},
        )
```

with:

```python
        result = await client.call_tool(
            "agent_one",
            {"prompt": "How are things?"},
        )
```

In both `test_tool_invocation_with_explicit_session_id_forwards_it` and `test_tool_invocation_with_default_session_id_is_fresh_uuid`, replace:

```python
    route = respx.post(_kagent_url("telemetry-agent")).mock(
```

with:

```python
    route = respx.post(_kagent_url("agent-two")).mock(
```

and:

```python
        await client.call_tool(
            "telemetry_agent",
```

with:

```python
        await client.call_tool(
            "agent_two",
```

In `test_kagent_error_surfaces_as_mcp_tool_error`, replace:

```python
    respx.post(_kagent_url("troubleshoot-planner")).mock(
        return_value=httpx.Response(503, content=b"unavailable"),
    )

    async with Client(mcp) as client:
        with pytest.raises(Exception):  # ToolError / httpx.HTTPStatusError
            await client.call_tool(
                "troubleshoot_planner",
                {"prompt": "hi"},
            )
```

with:

```python
    respx.post(_kagent_url("agent-one")).mock(
        return_value=httpx.Response(503, content=b"unavailable"),
    )

    async with Client(mcp) as client:
        with pytest.raises(Exception):  # ToolError / httpx.HTTPStatusError
            await client.call_tool(
                "agent_one",
                {"prompt": "hi"},
            )
```

- [ ] **Step 3: Update test_translator.py**

Edit `test_translator.py`. The SURF names appear only as the `model` argument to `event_to_chunks` — the value is opaque to the test assertions. Make these exact substitutions globally:

- `"troubleshoot-planner"` → `"agent-one"`
- `"telemetry-agent"` → `"agent-one"`

Verify there are exactly six replacements:

```bash
grep -c 'agent-one' test_translator.py
```

Expected: `6`.

- [ ] **Step 4: Verify no SURF agent names remain in test files**

Run: `grep -n 'troubleshoot-planner\|telemetry-agent\|alarming-agent\|wfo-search-agent' test_api.py test_mcp_server.py test_translator.py conftest.py`
Expected: no matches.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add test_api.py test_mcp_server.py test_translator.py
git commit -m "test: use generic agent-one/agent-two instead of SURF agent names"
```

---

## Task 7: Move root manifests to .local and update .gitignore

The current `deploy.yaml` and `10-librechat-config.yaml` are SURF-flavored. We rename them to `.local` and gitignore those names. The generalized examples land in `examples/` in Task 8.

**Files:**
- Rename: `deploy.yaml` → `deploy.yaml.local`
- Rename: `10-librechat-config.yaml` → `librechat-config.yaml.local`
- Modify: `.gitignore`

- [ ] **Step 1: Rename the root manifests**

```bash
mv deploy.yaml deploy.yaml.local
mv 10-librechat-config.yaml librechat-config.yaml.local
```

- [ ] **Step 2: Append local manifest names to .gitignore**

Edit `.gitignore`. Append at the end of the file (after the existing `.idea/` entry):

```
# Local k8s manifest copies (operator-specific, not for git)
deploy.yaml.local
librechat-config.yaml.local
```

- [ ] **Step 3: Verify the .local files are ignored**

Run: `git status --short | grep -E 'deploy.yaml.local|librechat-config.yaml.local'`
Expected: no output (the files are present on disk but gitignored).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore deploy.yaml.local and librechat-config.yaml.local"
```

---

## Task 8: Write generalized example manifests

Strips SURF-specific elements (surfconext JWT, `agents.surf.net`, agentgateway-specific `HTTPRoute`/`AgentgatewayPolicy`, hardcoded agent names). Keeps the Deployment, Service, probes, resource limits, and a generic commented-out `RemoteMCPServer` block.

**Files:**
- Create: `examples/deploy.yaml`
- Create: `examples/librechat-config.yaml`

- [ ] **Step 1: Create examples directory and deploy.yaml**

```bash
mkdir -p examples
```

Create `examples/deploy.yaml` with:

```yaml
# examples/deploy.yaml
#
# kagent-a2a-proxy — minimal Kubernetes deployment example.
#
# The proxy exposes:
#   - /v1/chat/completions  OpenAI-compatible streaming chat completions
#   - /v1/models            OpenAI model list (one entry per agent in agent_map)
#   - /mcp                  MCP Streamable HTTP (one tool per agent)
#   - /healthz/ready        Readiness/liveness probe
#
# Edit the placeholders marked <…> before applying:
#   <your-namespace>        namespace where you'll run the proxy
#   <agent-1>, <agent-2>    kagent agent names in your cluster
#
# See README.md for the full PROXY_* environment variable reference.

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kagent-a2a-proxy
  namespace: <your-namespace>
  labels:
    app: kagent-a2a-proxy
spec:
  replicas: 2
  selector:
    matchLabels:
      app: kagent-a2a-proxy
  template:
    metadata:
      labels:
        app: kagent-a2a-proxy
    spec:
      containers:
      - name: proxy
        image: ghcr.io/nren-agents/kagent-a2a-proxy:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: PROXY_KAGENT_BASE_URL
          value: "http://kagent-controller.kagent.svc:8083"
        - name: PROXY_KAGENT_NAMESPACE
          value: "<your-namespace>"
        - name: PROXY_REQUEST_TIMEOUT
          value: "300"
        # JSON map of OpenAI model name → kagent agent name.
        - name: PROXY_AGENT_MAP
          value: |
            {
              "<agent-1>": "<agent-1>",
              "<agent-2>": "<agent-2>"
            }
        # Optional: fallback agent for unknown model names. Must appear as a
        # value in PROXY_AGENT_MAP. Leave commented out to return HTTP 400
        # for unknown models instead.
        # - name: PROXY_DEFAULT_AGENT
        #   value: "<agent-1>"
        resources:
          limits:
            cpu: 500m
            memory: 256Mi
          requests:
            cpu: 50m
            memory: 64Mi
        livenessProbe:
          httpGet:
            path: /healthz/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /healthz/ready
            port: 8080
          initialDelaySeconds: 3
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: kagent-a2a-proxy
  namespace: <your-namespace>
spec:
  selector:
    app: kagent-a2a-proxy
  ports:
  - port: 8080
    targetPort: 8080
    name: http

---
# Optional: kagent RemoteMCPServer pointing at this proxy's in-cluster /mcp
# endpoint, exposing each configured agent as an MCP tool to other kagent
# agents in the same cluster. Apply (or adapt) in the namespace where you
# want consumers to discover these tools.
#
# apiVersion: kagent.dev/v1alpha2
# kind: RemoteMCPServer
# metadata:
#   name: kagent-a2a-proxy-agents
#   namespace: <your-namespace>
# spec:
#   description: "kagent agents exposed as MCP tools"
#   protocol: STREAMABLE_HTTP
#   url: http://kagent-a2a-proxy.<your-namespace>.svc:8080/mcp
#   timeout: 300s
#   sseReadTimeout: 300s
```

- [ ] **Step 2: Create examples/librechat-config.yaml**

Create `examples/librechat-config.yaml` with:

```yaml
# examples/librechat-config.yaml
#
# Example LibreChat configuration that consumes kagent-a2a-proxy as:
#   (a) MCP server  — surfaces each kagent agent as an MCP tool, and
#   (b) Custom OpenAI endpoint — surfaces each kagent agent as a "model".
#
# Replace the placeholders below before applying:
#   <your-proxy-url>   URL where kagent-a2a-proxy is reachable from LibreChat
#                      (e.g. http://kagent-a2a-proxy.<ns>.svc:8080)
#   <your-mcp-token>   bearer token your proxy/gateway expects on /mcp
#                      (omit the Authorization header if auth is not enforced)
#   <agent-1>, <agent-2>   kagent agent names exposed via PROXY_AGENT_MAP
#
# Secrets to provide via env or LibreChat-credentials Secret:
#   MCP_TOKEN            — bearer token for the proxy's /mcp surface
#   PROXY_API_KEY        — optional; only if you've put an auth proxy in front

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: librechat-config
  namespace: librechat
data:
  librechat.yaml: |
    version: 1.3.5
    cache: true

    # mcpSettings must be top-level alongside mcpServers.
    # allowedDomains whitelists in-cluster addresses — LibreChat blocks
    # internal addresses by default as SSRF protection.
    mcpSettings:
      allowedDomains:
        - "<your-proxy-url>"

    # mcpServers is top-level — NOT nested under endpoints.
    mcpServers:
      kagent-agents:
        type: streamable-http
        url: "<your-proxy-url>/mcp"
        timeout: 300000      # 5 minutes per tool call (ms)
        initTimeout: 30000   # 30s for initialize handshake
        headers:
          Authorization: "Bearer ${MCP_TOKEN}"

    endpoints:
      # Custom OpenAI endpoint exposing kagent agents as "models".
      custom:
        - name: "kagent Agents"
          apiKey: "${PROXY_API_KEY}"
          baseURL: "<your-proxy-url>/v1"
          models:
            default:
              - "<agent-1>"
              - "<agent-2>"
            fetch: false
          titleConvo: true
          titleModel: "<agent-1>"
          modelDisplayLabel: "kagent Agents"
          streamRate: 0
```

- [ ] **Step 3: Commit**

```bash
git add examples/deploy.yaml examples/librechat-config.yaml
git commit -m "docs: add generic example k8s manifests in examples/"
```

---

## Task 9: Write README, LICENSE, and .env.example

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `.env.example`

- [ ] **Step 1: Create LICENSE (MIT, SURF copyright)**

Create `LICENSE` with:

```
MIT License

Copyright (c) 2026 SURF

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Create .env.example**

Create `.env.example` at the repo root with:

```dotenv
# kagent-a2a-proxy — example .env
#
# Copy to .env (gitignored) and edit. The proxy's Settings class reads .env
# from the current working directory at startup. All vars are optional —
# defaults shown below.

# Base URL of the kagent-controller A2A server.
# PROXY_KAGENT_BASE_URL=http://kagent-controller.kagent.svc:8083

# Kubernetes namespace where kagent agents are deployed.
# PROXY_KAGENT_NAMESPACE=default

# JSON map of OpenAI model name → kagent agent name. With an empty map the
# proxy starts but exposes no models or MCP tools.
# PROXY_AGENT_MAP={"agent-one":"agent-one","agent-two":"agent-two"}

# Optional fallback agent for unknown model names. Must appear as a value
# in PROXY_AGENT_MAP. If unset, unknown models return HTTP 400.
# PROXY_DEFAULT_AGENT=agent-one

# Per-request timeout in seconds for kagent A2A calls.
# PROXY_REQUEST_TIMEOUT=300

# Log level: debug | info | warning | error | critical.
# PROXY_LOG_LEVEL=info
```

- [ ] **Step 3: Create README.md**

Create `README.md` with:

````markdown
# kagent-a2a-proxy

OpenAI-compatible streaming chat completions and MCP Streamable HTTP server,
backed by [kagent](https://kagent.dev) agents over the A2A protocol.

Drop it in front of a kagent controller and your agents look like:

- **OpenAI models** — every agent in `PROXY_AGENT_MAP` becomes a model on
  `/v1/chat/completions` and `/v1/models`. Streaming SSE works; reasoning
  output is routed to the `reasoning_content` channel so LibreChat renders
  it in the "Thinking" pane.
- **MCP tools** — every agent in `PROXY_AGENT_MAP` becomes an MCP tool on
  `/mcp` (Streamable HTTP). Working/thinking deltas are forwarded as MCP
  progress notifications.

## How it works

```
client ──► /v1/chat/completions ─┐
                                 │
client ──► /mcp tool call ───────┤  proxy  ──► POST /api/a2a/<ns>/<agent>
                                 │     │
                                 │     └── translator: A2A status/artifact
                                 │         events → OpenAI delta chunks
                                 │
                                 ▼
                          kagent-controller
```

- An OpenAI request is translated into a kagent A2A `message/stream` call.
  The agent's `working` events become `reasoning_content` deltas; the final
  artifact text becomes regular `content`.
- MCP tool calls go through the same A2A path. The tool returns the artifact
  text; `working` text fires `Context.report_progress` notifications.

## Quickstart

### Docker

```bash
docker run --rm -p 8080:8080 \
  -e PROXY_KAGENT_BASE_URL=http://your-kagent-controller:8083 \
  -e PROXY_KAGENT_NAMESPACE=default \
  -e PROXY_AGENT_MAP='{"agent-one":"agent-one"}' \
  ghcr.io/nren-agents/kagent-a2a-proxy:latest
```

### Local with uv

```bash
uv sync --all-groups
cp .env.example .env       # then edit
uv run uvicorn kagent_a2a_proxy.main:app --host 0.0.0.0 --port 8080
```

### Send a request

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "agent-one",
    "messages": [{"role":"user","content":"hello"}],
    "stream": true
  }'
```

## Configuration

All settings are read from `PROXY_*` environment variables (or a local
`.env` file). See `.env.example` for a copy-pasteable template.

| Variable | Type | Default | Description |
|---|---|---|---|
| `PROXY_KAGENT_BASE_URL` | URL | `http://kagent-controller.kagent.svc:8083` | Base URL of the kagent-controller A2A server. |
| `PROXY_KAGENT_NAMESPACE` | string | `default` | Kubernetes namespace where kagent agents are deployed. |
| `PROXY_AGENT_MAP` | JSON object | `{}` | Map of OpenAI model name → kagent agent name. |
| `PROXY_DEFAULT_AGENT` | string | _unset_ | Optional fallback for unknown models. Must appear as a value in `PROXY_AGENT_MAP`. |
| `PROXY_REQUEST_TIMEOUT` | float (seconds) | `300` | Per-request timeout for kagent A2A calls. |
| `PROXY_LOG_LEVEL` | `debug`/`info`/`warning`/`error`/`critical` | `info` | Log level for the proxy's logger. |

`PROXY_AGENT_MAP` is parsed as JSON. Example:

```bash
PROXY_AGENT_MAP='{"weather":"weather-agent","alerts":"alerting-agent"}'
```

Misconfiguration fails fast at startup: invalid URLs, unknown log levels,
non-positive timeouts, and `PROXY_DEFAULT_AGENT` values that don't appear in
the map all raise a `ValidationError`.

## Deployment

See [`examples/deploy.yaml`](examples/deploy.yaml) for a minimal Kubernetes
manifest (Deployment + Service + commented `RemoteMCPServer`).

LibreChat users: [`examples/librechat-config.yaml`](examples/librechat-config.yaml)
shows how to wire the proxy as both an MCP server and a custom OpenAI endpoint.

## Development

```bash
uv sync --all-groups
uv run pytest -q
```

Tests cover the translator, the FastAPI surface (`/v1/chat/completions`,
`/v1/models`, `/healthz/ready`), the MCP tool surface, the Settings
validators, and the model→agent resolver.

## License

MIT — see [LICENSE](LICENSE).
````

- [ ] **Step 4: Commit**

```bash
git add README.md LICENSE .env.example
git commit -m "docs: add README, MIT LICENSE, and .env.example"
```

---

## Task 10: Add GitHub Actions workflow

Single workflow, two jobs: tests on Python 3.12/3.13/3.14, then a publish job that runs only on push-to-main and `v*` tags.

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow file**

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/ci.yml` with:

```yaml
name: ci

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ['3.12', '3.13', '3.14']
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python }}

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Run tests
        run: uv run pytest -q

  publish:
    needs: test
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract image metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=sha-,enable={{is_default_branch}}
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GHCR publish workflow with multi-python test matrix"
```

---

## Task 11: Final verification

Confirm everything still passes and there are no stray SURF references in tracked files.

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run pytest -q`
Expected: all green across `test_api.py`, `test_mcp_server.py`, `test_translator.py`, `test_config.py`, `test_resolve_agent.py`.

- [ ] **Step 2: Build the Docker image**

Run: `docker build -t kagent-a2a-proxy:check .`
Expected: build succeeds.

- [ ] **Step 3: Sanity-run the image with an empty agent_map**

Run:

```bash
docker run --rm -d --name kagent-a2a-proxy-check -p 18080:8080 \
  -e PROXY_KAGENT_BASE_URL=http://example.invalid \
  kagent-a2a-proxy:check
sleep 2
curl -sf http://localhost:18080/healthz/ready
docker stop kagent-a2a-proxy-check
```

Expected: container starts, `curl` returns `{"status":"ok"}`.

- [ ] **Step 4: Audit remaining SURF references in tracked files**

Run:

```bash
git ls-files | xargs grep -nE 'surf|SURF' 2>/dev/null
```

Expected: only intentional hits — the spec/plan docs under `docs/superpowers/`, the LICENSE copyright line (`Copyright (c) 2026 SURF`), and any acknowledgments in the README. No source/test/manifest files should match.

If anything unexpected appears, fix it before continuing.

- [ ] **Step 5: Confirm git status is clean apart from the .local files**

Run: `git status --short`
Expected: empty (the `.local` files are gitignored).

---

## Task 12: Push to origin

The remote is `git@github.com:nren-agents/kagent-a2a-proxy.git`. The branch is `main` and there are no upstream commits yet, so the first push needs `-u`.

- [ ] **Step 1: Confirm the remote**

Run: `git remote -v`
Expected: `origin git@github.com:nren-agents/kagent-a2a-proxy.git` (fetch and push).

- [ ] **Step 2: Push**

Run: `git push -u origin main`
Expected: push succeeds. The CI workflow should trigger; check it on GitHub (Actions tab) — both the test matrix and the publish job should run, and GHCR should receive `:latest` + `:sha-<short>`.
