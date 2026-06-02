# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-groups                                   # install (incl. dev deps)
uv run uvicorn kagent_a2a_proxy.main:app --port 8080   # run locally (needs .env)
uv run pytest -q                                        # all tests
uv run pytest test_translator.py -q                     # one test file
uv run pytest test_translator.py::test_artifact_update_text_goes_to_content  # one test
uv run ruff check . && uv run ruff format --check .     # lint + format check
uv run mypy kagent_a2a_proxy                            # typecheck
```

Tests are flat `test_*.py` files at the repo root (no `tests/` dir). `uv run ruff format .` auto-formats. CI (`.github/workflows/ci.yml`) runs tests on Python 3.12/3.13/3.14, lint on 3.12, then publishes a Docker image to GHCR; the `publish` job is gated on both `test` and `lint`.

## Architecture

A stateless proxy that makes [kagent](https://kagent.dev) agents look like OpenAI models and MCP tools. Two client-facing surfaces converge on a single kagent A2A path:

```
/v1/chat/completions ─┐
/mcp tool call ───────┼─► kagent_client.stream_agent ─► POST /api/a2a/{ns}/{agent}
                      │      (SSE) ─► translator.event_to_chunks ─► OpenAI delta chunks
```

**Request flow.** An OpenAI/MCP request maps a model name → kagent agent name (`kagent_client._resolve_agent`, via `settings.agent_map` with `default_agent` fallback), then issues a kagent A2A `message/stream` JSON-RPC call. The streamed A2A events are translated into OpenAI streaming delta chunks.

**The central translation contract** lives in `translator.py` and is the most important thing to understand. kagent streams the agent's prose answer as `working` text and routing keys off *what kind of part it is*, not the task state. The visible reply (`content`) gets the answer + user-facing prompts; the Thinking pane (`reasoning_content`) gets reasoning + tool activity.
- **`working` text part with `kagent_thought` metadata** (ADK `part.thought`) → **`reasoning_content`** (Thinking pane). Plain answer text parts → **`content`** (visible reply).
- kagent re-sends the streamed answer: as `working` **partial** deltas (`kagent_adk_partial=true`), then once as a **non-partial** aggregate copy (`kagent_adk_partial=false`), then again as an **`artifact-update`**. `translator` skips the non-partial copy (`is_partial() is False`); `agent_runner.translate_stream` drops the artifact once any answer content has streamed. The artifact is only emitted as a fallback when nothing streamed (non-ADK executors).
- In-progress tool calls (`kagent_type=function_call`) → Thinking pane as a blockquote `> 🔧 **{name}**` + optional `> \`{compact args}\`` (`_format_tool_args`, truncated to `_MAX_ARGS_LEN`). Tool results (`kagent_type=function_response`) are **dropped**.
- **`input-required`** → **`content`** (visible), so the prompt isn't buried: tool-approval requests render `⚠️ Approval required for \`{originalFunctionCall.name}\`: {hint}`; free-text questions render `> ❓ {text}`.
- **`failed` / `canceled` / `rejected` / `auth-required`** → a visible `content` notice (`_terminal_notice`) + `finish_reason: "stop"` (otherwise the stream ends silently). **`completed`** emits `finish_reason: "stop"` only.

**Streaming vs. blocking.** Both paths share `agent_runner.translate_stream` (stream → `parse_sse_line` → `event_to_chunks` → drop the duplicate artifact + a consecutive-identical-reasoning safety net). `main.py` yields chunks straight to SSE for `stream=true`. For blocking responses and all MCP tool calls, `agent_runner.collect_agent_response` drains the same pipeline and concatenates `delta.content` into the final string; `reasoning_content` deltas instead fire an optional `on_progress` callback (MCP tools forward these as progress notifications).

**`agent_map` drives everything dynamically.** Each entry becomes both a `/v1/models` model and an `@mcp.tool` (registered at import time in `mcp_server.py` via `build_mcp_server`). An empty map starts the server with no models/tools. MCP tool names are sanitized kebab→snake.

### kagent A2A protocol quirks

kagent uses the **pre-v1.0** A2A shape, which the models in `models.py` and `translator.parse_sse_line` are written against — do not "modernize" these to current A2A spec:
- Each SSE line is wrapped in a **JSON-RPC 2.0 envelope** (`{"jsonrpc","id","result"|"error"}`); `parse_sse_line` unwraps `result` and drops errors/non-events.
- Message/artifact **parts discriminate on `kind`** (`"text"`/`"data"`), not `type`.
- Tool-call metadata lives on the **data part's** `metadata` (`kagent_type` = `function_call`/`function_response`, `kagent_is_long_running`), *not* the event metadata — verified against live streams. `A2ATaskStatusUpdateEvent.kagent_type()` checks the part first and falls back to the event metadata (the shape some synthetic fixtures use). The `adk_request_confirmation` pseudo-tool is suppressed in the working/tool branch — it surfaces only via the `input-required` approval prompt.

### Configuration

All runtime config is `PROXY_*` env vars (or a `.env` file), loaded into a **module-level `settings` singleton** (`config.py`) constructed at import time. Validators make misconfiguration fail fast at startup (bad URL, unknown log level, non-positive timeout, `default_agent` not present in `agent_map`).

Because `settings` is built at import, **`conftest.py` sets `PROXY_AGENT_MAP`/`PROXY_DEFAULT_AGENT` before any package import** — preserve that ordering when adding tests that touch config.

**Human-in-the-loop (`hitl.py`).** When `PROXY_HITL_SECRET` is set, an `input-required` tool-approval prompt carries an HMAC-signed, render-invisible marker (HTML comment) encoding the paused task's `taskId`/`contextId`. On the next request, `main._select_chunks` recovers it from the assistant history (`hitl.extract_pending`), classifies the user reply (`hitl.classify_decision`), and resumes via `kagent_client.resume_stream` (a decision `DataPart`) instead of a fresh prompt. Stateless — no store, works across replicas (LibreChat resends assistant content verbatim). NOTE: the resume wire shape is inferred from kagent source and should be validated against a live kagent.

### Conventions

- mypy runs **strict** on `kagent_a2a_proxy.*`; tests/conftest are exempt. Keep new package code fully typed.
- The shared `httpx.AsyncClient` in `kagent_client.py` is a module singleton (connection pooling), closed via `aclose()` from the FastAPI `lifespan` in `main.py`.
- Pydantic models in `models.py` use `extra="ignore"` so unknown OpenAI request fields are silently dropped.
