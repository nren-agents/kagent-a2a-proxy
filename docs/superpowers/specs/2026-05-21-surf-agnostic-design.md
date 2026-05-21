# Make the repo SURF-agnostic and publishable

**Date:** 2026-05-21
**Status:** Approved (design phase)

## Goal

Take the existing `surf-a2a-proxy` codebase — currently shaped around SURF-specific
infrastructure (surf.net hostnames, surfconext JWT auth, hardcoded agent names) — and
turn it into a generic, MIT-licensed open-source repo publishable on GitHub under the
SURF org. The proxy itself remains useful to anyone running kagent; only the publishing
identity is SURF.

## In scope

- Rename package and project: `surf_a2a_proxy` → `kagent_a2a_proxy`,
  `surf-a2a-proxy` → `kagent-a2a-proxy`.
- Tighten `Settings` validation (pydantic-settings) and drop SURF-specific defaults.
- Move and generalize `deploy.yaml` and `10-librechat-config.yaml` into `examples/`.
  Keep the SURF-flavored versions as gitignored local copies.
- Add `README.md`, `LICENSE` (MIT), `.env.example` (repo root).
- Add a single GitHub Actions workflow that runs tests on Python 3.12 / 3.13 / 3.14
  and publishes a Docker image to GHCR on push-to-main and on `v*` tags.
- Update `Dockerfile` and `pyproject.toml` to match the renamed package.

## Out of scope

- Functional changes to translation, MCP server, or kagent client logic.
- CONTRIBUTING.md, code of conduct, issue/PR templates.
- Multi-arch image builds (can be added later via QEMU + buildx if needed).
- Alternative config sources (YAML/TOML files). Env vars + `.env` only.
- LibreChat-specific docs beyond the example file.

## Final repo layout

```
.
├── .github/workflows/ci.yml          NEW
├── .env.example                       NEW
├── .gitignore                         + deploy.yaml.local, librechat-config.yaml.local
├── Dockerfile                         updated COPY/CMD paths
├── LICENSE                            NEW (MIT, copyright SURF)
├── README.md                          NEW
├── conftest.py
├── pyproject.toml                     name/desc/package path updated
├── kagent_a2a_proxy/               RENAMED from surf_a2a_proxy/
│   ├── __init__.py
│   ├── agent_runner.py
│   ├── config.py                      tightened, empty defaults
│   ├── kagent_client.py
│   ├── main.py                        title/desc generalized
│   ├── mcp_server.py
│   ├── models.py
│   └── translator.py
├── examples/
│   ├── deploy.yaml                    generalized
│   └── librechat-config.yaml          generalized
├── test_api.py
├── test_mcp_server.py
└── test_translator.py
```

The current root-level `deploy.yaml` is renamed to `deploy.yaml.local` and
`10-librechat-config.yaml` is renamed to `librechat-config.yaml.local`. Both names
are added to `.gitignore`, preserving the SURF-flavored manifests locally for the
operator who needs them while keeping them out of git.

## Components

### `kagent_a2a_proxy/config.py`

```python
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
        description="Base URL of the kagent-controller A2A server",
    )
    kagent_namespace: str = Field(
        default="default",
        description="Kubernetes namespace where kagent agents are deployed",
    )
    agent_map: dict[str, str] = Field(
        default_factory=dict,
        description="JSON map of OpenAI model name → kagent agent name",
    )
    default_agent: str | None = Field(
        default=None,
        description="Fallback agent when the requested model is not in agent_map",
    )
    request_timeout: float = Field(default=300.0, gt=0)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    @model_validator(mode="after")
    def _check_default_agent(self) -> "Settings":
        if self.default_agent and self.default_agent not in self.agent_map.values():
            raise ValueError(
                f"default_agent {self.default_agent!r} is not present in agent_map values"
            )
        return self


settings = Settings()
```

Key shifts from the existing `Settings`:

- `kagent_base_url` validated via `AnyHttpUrl`.
- `kagent_namespace` default is `"default"` instead of `"troubleshooting"`.
- `agent_map` default is empty (was four SURF agents).
- `default_agent` defaults to `None` and is cross-checked against `agent_map`.
- `log_level` restricted to a `Literal`.
- `request_timeout` must be positive (`gt=0`).
- `extra="ignore"` so unrelated `PROXY_*` vars don't break startup.

### Caller updates for `default_agent` becoming `Optional`

Any code in `kagent_client.py` / `agent_runner.py` that currently treats
`settings.default_agent` as a guaranteed string must be updated to handle `None`.
The expected behavior when an unknown model is requested and no default is set:
return a 4xx with a clear error message rather than silently picking an agent.
Exact call sites confirmed during implementation.

### `examples/deploy.yaml`

Strips every SURF-specific element from the current manifest:

- Namespace: `troubleshooting` → `<your-namespace>` (placeholder, comment).
- Image: `automationbeheer/a2a-proxy:latest` → `ghcr.io/nren-agents/kagent-a2a-proxy:latest`.
- `PROXY_AGENT_MAP`: example with `{"agent-one": "agent-one"}` and a comment.
- `PROXY_DEFAULT_AGENT`: commented out.
- `HTTPRoute`: removed — it depends on `agentgateway-proxy` and `agents.surf.net`,
  neither of which a generic user has.
- `AgentgatewayPolicy` (JWT/surfconext): removed entirely.
- Kept: `Deployment`, `Service`, readiness/liveness probes, resource limits, and
  the commented `RemoteMCPServer` example block — but its names are rewritten:
  `surf-agents` → `kagent-a2a-proxy-agents`, `surf-a2a-proxy.troubleshooting.svc`
  → `kagent-a2a-proxy.<your-namespace>.svc`, namespace → placeholder.

A header comment explains the file is an example and points at `README.md` for
the env var reference.

### `examples/librechat-config.yaml`

Generic version of the current `10-librechat-config.yaml`:

- `surfconext`-flow credential instructions → "obtain a bearer token for your MCP
  endpoint" (generic note).
- `${AGENTGATEWAY_API_KEY}` and `${MCP_TOKEN}` references kept — they are
  LibreChat env interpolation, not SURF-specific.
- Hostnames: `agentgateway-proxy.agentgateway-system.svc` → kept as illustrative
  with a comment that the user substitutes their gateway address.
- Model names: SURF agent names → placeholders.

### `.env.example` (repo root)

A reference `.env` documenting every `PROXY_*` var with a sensible commented-out
default. Lives at repo root so `cp .env.example .env` works for local dev — the
`Settings` class reads `.env` from CWD.

### `README.md`

```markdown
# kagent-a2a-proxy

One-paragraph description: OpenAI-compatible streaming chat completions
and MCP Streamable HTTP server backed by kagent A2A.

## How it works
- Brief architecture: OpenAI request → translator → kagent A2A SSE → OpenAI SSE
- /mcp surface: one MCP tool per agent in agent_map
- Small ascii diagram (client → proxy → kagent)

## Quickstart
- Docker: `docker run` example with PROXY_KAGENT_BASE_URL + PROXY_AGENT_MAP
- Local: `uv sync && uv run uvicorn kagent_a2a_proxy.main:app`
- curl against /v1/chat/completions

## Configuration
- Table: env var | type | default | description (every PROXY_* var)
- Note on .env support

## Deployment
- Pointer to examples/deploy.yaml
- Pointer to examples/librechat-config.yaml for LibreChat users

## Development
- uv sync, pytest

## License
- MIT — see LICENSE
```

### `LICENSE`

Standard MIT text, copyright line: `Copyright (c) 2026 SURF`.

### `.github/workflows/ci.yml`

```yaml
name: ci

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ['3.12', '3.13', '3.14']
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --all-groups
      - run: uv run pytest -q

  publish:
    needs: test
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=sha-,enable={{is_default_branch}}
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

Behavior:

- PRs run tests on every matrix cell, no publish job.
- Push to `main` runs tests, then publishes `latest` and `sha-<short>`.
- Git tag `v1.2.3` runs tests, then publishes `1.2.3`, `1.2`, `1`.
- Default `GITHUB_TOKEN` has `packages: write` — no extra secret setup needed.
- `pytest -q`: matches current quiet output convention.

### `Dockerfile`

Two edits only:

- `COPY surf_a2a_proxy ./surf_a2a_proxy/` → `COPY kagent_a2a_proxy ./kagent_a2a_proxy/`
- `CMD ["uv", "run", "uvicorn", "surf_a2a_proxy.main:app", ...]` →
  `CMD ["uv", "run", "uvicorn", "kagent_a2a_proxy.main:app", ...]`

### `pyproject.toml`

- `name = "kagent-a2a-proxy"`
- `description = "OpenAI-compatible streaming chat completions and MCP server for kagent agents"`
- `[tool.hatch.build.targets.wheel] packages = ["kagent_a2a_proxy"]`
- Drop the legacy `[project.optional-dependencies].dev` block — `[dependency-groups].dev`
  is what `uv sync --all-extras` uses.

### `.gitignore`

Append:

```
# Local k8s manifest copies (not for git)
deploy.yaml.local
librechat-config.yaml.local
```

## Migration order (informational; full plan comes from writing-plans skill)

1. Rename package directory and update all imports + tooling references.
2. Update `Settings` and verify callers handle `default_agent=None`.
3. Move root manifests to `.local` names, write generalized `examples/`.
4. Write `README.md`, `LICENSE`, `examples/.env.example`.
5. Add `.github/workflows/ci.yml`.
6. Run tests on every Python version locally to confirm nothing regressed.

## Risks

- **Package rename ripples.** Imports, tests, Dockerfile, and `pyproject.toml`
  all reference `surf_a2a_proxy`. Missing one breaks the wheel build or pytest
  collection. Mitigation: grep before claiming done.
- **`default_agent=None` callers.** If any code path assumes a string, it now
  blows up at runtime instead of falling back. Need to audit `kagent_client.py`
  and `agent_runner.py` and add explicit error handling.
- **MCP tools off an empty `agent_map`.** With the new empty default, a freshly
  started proxy with no env config will expose zero tools. Acceptable for a
  generic image — users must configure — but verify the proxy starts cleanly
  in that state (no crashes, healthz returns ok).
- **Python 3.14 in CI.** 3.14 is now stable, but `pydantic-settings` /
  `fastmcp` wheel availability for 3.14 should be sanity-checked the first time
  CI runs; if a transitive dep blocks 3.14, fall back to 3.12 + 3.13 and open a
  follow-up.

## Verification

Done means:

- `uv sync --all-groups && uv run pytest -q` passes locally on the renamed package.
- `docker build .` succeeds, image runs and `/healthz/ready` returns 200 with an
  empty `agent_map`.
- A grep for `surf` and `SURF` in tracked files returns only intentional
  occurrences (the LICENSE copyright, README acknowledgments if any, this spec).
- CI runs green on a PR and produces the right image tags on a push to main.
