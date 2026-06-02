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

- An OpenAI request is translated into a kagent A2A `message/stream` call. The
  agent's answer goes to `content` (the visible reply); reasoning and tool
  activity go to `reasoning_content` (LibreChat's "Thinking" pane). Between-tool
  narration is de-emphasized by default — see [Narration mode](#narration-mode).
- MCP tool calls go through the same A2A path. The tool returns the answer text;
  reasoning/tool activity fires `Context.report_progress` notifications.

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
| `PROXY_NARRATION_MODE` | `deemphasize`/`stream` | `deemphasize` | How an agent's between-tool progress narration is rendered. See [Narration mode](#narration-mode). |
| `PROXY_HITL_SECRET` | string | _unset_ | Secret for HMAC-signing the human-in-the-loop approval marker. When set, tool-approval prompts become actionable (reply `approve`/`deny`); when unset, they are informational only. Use the same value on every replica. |

`PROXY_AGENT_MAP` is parsed as JSON. Example:

```bash
PROXY_AGENT_MAP='{"weather":"weather-agent","alerts":"alerting-agent"}'
```

Misconfiguration fails fast at startup: invalid URLs, unknown log levels,
non-positive timeouts, and `PROXY_DEFAULT_AGENT` values that don't appear in
the map all raise a `ValidationError`.

### Narration mode

Agents that use tools tend to "think out loud" between tool calls ("I'll start
by querying the topology…", "Now I'll pull telemetry…"). kagent streams that
narration as plain answer text, so without help it piles up in the reply and
buries the final answer.

- **`deemphasize`** (default) collapses each narration burst into a Markdown
  blockquote above the answer, keeping the answer front-and-center. kagent
  re-sends each burst as a non-partial *aggregate* bundled with the tool call it
  triggered; the proxy uses that aggregate as the authoritative copy, so answer
  text is emitted **per-burst** rather than token-streamed (which also renders
  Markdown tables more cleanly). Tool calls still go to the "Thinking" pane.
- **`stream`** token-streams all working text into the reply verbatim — the
  original behavior. Narration and answer run together as they arrive, with the
  live typewriter effect, but the reply is busier.

Set via `PROXY_NARRATION_MODE`. It's a per-deployment, restart-free flip, handy
for validating the default against your agents before committing to it.

### Human-in-the-loop approvals

When a kagent agent calls a long-running tool that needs confirmation, the proxy
surfaces a `⚠️ Approval required …` line in the reply. Set `PROXY_HITL_SECRET`
(any random string, e.g. `openssl rand -hex 32`) to make these **actionable**:
the proxy embeds an HMAC-signed, render-invisible marker in the reply, and when
the user answers `approve`/`deny` (or `yes`/`no`), it resumes the paused kagent
task. The marker rides the conversation history, so this needs **no extra
LibreChat configuration** and works across multiple replicas — use the same
secret on each. When `PROXY_HITL_SECRET` is unset, approval prompts are
informational only.

## Deployment

See [`examples/deploy.yaml`](examples/deploy.yaml) for a minimal Kubernetes
manifest (Deployment + Service + commented `RemoteMCPServer`).

LibreChat users: [`examples/librechat-config.yaml`](examples/librechat-config.yaml)
shows how to wire the proxy as both an MCP server and a custom OpenAI endpoint.

## Development

```bash
uv sync --all-groups
uv run pytest -q                                       # tests
uv run ruff check . && uv run ruff format --check .    # lint
uv run mypy kagent_a2a_proxy                           # typecheck
```

Tests cover the translator, the FastAPI surface (`/v1/chat/completions`,
`/v1/models`, `/healthz/ready`), the MCP tool surface, the Settings
validators, and the model→agent resolver.

## License

MIT — see [LICENSE](LICENSE).
