"""
Expose configured kagent agents as MCP tools over Streamable HTTP.

One @mcp.tool per entry in settings.agent_map. Each tool invokes the
corresponding kagent agent via A2A (reusing agent_runner) and returns the
agent's final answer text. Working/thinking deltas are forwarded to the MCP
client as progress notifications.
"""
from __future__ import annotations

import itertools
import uuid

from fastmcp import Context, FastMCP

from .agent_runner import collect_agent_response
from .config import settings


def _sanitise_tool_name(name: str) -> str:
    """Convert kebab-case agent identifiers to MCP-tool-safe snake_case."""
    return name.replace("-", "_")


def _register_tool(mcp: FastMCP, model_key: str, agent_name: str) -> None:
    """Register one @mcp.tool bound to a specific kagent agent."""
    tool_name = _sanitise_tool_name(model_key)
    description = (
        f"Invoke the kagent '{agent_name}' agent and return its final "
        f"answer text. Working/thinking deltas are streamed as MCP "
        f"progress notifications."
    )

    @mcp.tool(name=tool_name, description=description)
    async def _tool(
        prompt: str,
        ctx: Context,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        messages = [{"role": "user", "content": prompt}]
        counter = itertools.count(1)

        async def on_progress(text: str) -> None:
            await ctx.report_progress(progress=next(counter), message=text)

        return await collect_agent_response(model_key, messages, sid, on_progress)


def build_mcp_server() -> FastMCP:
    """Construct a FastMCP server with one tool per entry in agent_map."""
    mcp = FastMCP("kagent-a2a-proxy")
    for model_key, agent_name in settings.agent_map.items():
        _register_tool(mcp, model_key, agent_name)
    return mcp


mcp = build_mcp_server()
