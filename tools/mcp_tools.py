"""
MCP Tool Layer — complaint routing agent
Uses the official `mcp` package (pip install mcp) with async stdio transport.
Falls back gracefully to stub logging if MCP servers are not configured.

To activate real MCP servers, set env vars and ensure npx is installed:
    export SLACK_BOT_TOKEN=xoxb-...
    export JIRA_URL=https://yourco.atlassian.net
    export JIRA_EMAIL=you@company.com
    export JIRA_API_TOKEN=...
    export PAGERDUTY_API_KEY=...
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from schemas.complaint import Priority, RoutingDecision, Team

logger = logging.getLogger(__name__)


# ── MCP server definitions ─────────────────────────────────────────────────────
# Each entry maps a server name to its stdio launch config.
# Add / remove entries freely — the agent logic never changes.

MCP_SERVER_CONFIGS: dict[str, dict] = {
    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env_keys": ["SLACK_BOT_TOKEN"],
    },
    "jira": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-jira"],
        "env_keys": ["JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"],
    },
    "pagerduty": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-pagerduty"],
        "env_keys": ["PAGERDUTY_API_KEY"],
    },
}

# Team → which MCP servers to notify
TEAM_CHANNEL_MAP: dict[Team, list[str]] = {
    Team.BILLING:      ["slack", "jira"],
    Team.TECH_SUPPORT: ["slack", "jira"],
    Team.ACCOUNT_MGMT: ["slack"],
    Team.ESCALATION:   ["slack", "pagerduty"],
    Team.REVIEW_QUEUE: ["slack"],
    Team.PRODUCT:      ["jira"],
    Team.GENERAL:      ["slack"],
}


# ── Toolkit (simple wrapper, no external adapter library needed) ───────────────

class MCPToolkit:
    """
    Thin wrapper around the `mcp` stdio client.
    Lazily connects to each server on first use.
    """

    def __init__(self, active_servers: dict[str, dict]):
        self.active_servers = active_servers  # name → config
        self._available = bool(active_servers)

    @property
    def available(self) -> bool:
        return self._available

    async def call_tool(self, server: str, tool_name: str, arguments: dict) -> Any:
        """Call a named tool on a given MCP server via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            cfg = self.active_servers[server]
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg["args"],
                env={k: os.environ[k] for k in cfg["env_keys"] if k in os.environ},
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    return result

        except ImportError:
            raise ImportError(
                "Run: pip install mcp\n"
                "MCP package is required for real tool calls."
            )
        except Exception as exc:
            logger.error("MCP tool call failed [%s/%s]: %s", server, tool_name, exc)
            raise


def build_mcp_toolkit() -> MCPToolkit:
    """
    Build a toolkit with only servers that have all env vars set.
    Returns a toolkit that stubs gracefully if nothing is configured.
    """
    active = {}
    for name, cfg in MCP_SERVER_CONFIGS.items():
        missing = [k for k in cfg["env_keys"] if not os.getenv(k)]
        if missing:
            logger.debug("MCP server '%s' skipped — missing env: %s", name, missing)
        else:
            active[name] = cfg
            logger.info("MCP server '%s' ready", name)

    if not active:
        logger.warning(
            "No MCP servers configured — using stub notifications. "
            "Set SLACK_BOT_TOKEN / JIRA_API_TOKEN / PAGERDUTY_API_KEY to activate."
        )

    return MCPToolkit(active)


# ── Notification logic ─────────────────────────────────────────────────────────

def notify_via_mcp(state: dict, toolkit: MCPToolkit) -> dict:
    """
    LangGraph node (sync wrapper around async logic).
    Calls MCP tool servers based on team + priority, falls back to stub.
    """
    routing: RoutingDecision | None = state.get("routing")
    classification = state.get("classification")

    if not routing or state.get("error"):
        return state

    if not toolkit.available:
        _stub_notify(routing, classification)
        return state

    # Run async MCP calls from the sync LangGraph node
    try:
        asyncio.run(_dispatch_all(routing, classification, toolkit))
    except RuntimeError:
        # Already inside an event loop (e.g. Jupyter / some Gradio setups)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_dispatch_all(routing, classification, toolkit))

    return state


async def _dispatch_all(
    routing: RoutingDecision,
    classification: Any,
    toolkit: MCPToolkit,
) -> None:
    servers = set(TEAM_CHANNEL_MAP.get(routing.team, ["slack"]))

    # P1 always adds PagerDuty regardless of team
    if classification and classification.priority == Priority.P1:
        servers.add("pagerduty")

    # Only dispatch to servers that are actually configured
    servers &= set(toolkit.active_servers.keys())

    tasks = [_dispatch_one(server, routing, classification, toolkit) for server in servers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for server, result in zip(servers, results):
        if isinstance(result, Exception):
            # Non-fatal — other channels still notified
            logger.error("MCP dispatch to '%s' failed: %s", server, result)


async def _dispatch_one(
    server: str,
    routing: RoutingDecision,
    classification: Any,
    toolkit: MCPToolkit,
) -> None:
    if server == "slack":
        await toolkit.call_tool(
            "slack", "slack_post_message",
            {
                "channel": f"#complaints-{routing.team.value.lower().replace(' ', '-')}",
                "text": _slack_message(routing, classification),
            },
        )

    elif server == "jira":
        await toolkit.call_tool(
            "jira", "jira_create_issue",
            {
                "project": "COMPLAINTS",
                "summary": classification.summary if classification else routing.complaint_id,
                "description": classification.raw_text if classification else "",
                "priority": classification.priority.value if classification else "P3",
                "labels": [classification.category if classification else "general"],
            },
        )

    elif server == "pagerduty":
        await toolkit.call_tool(
            "pagerduty", "pagerduty_trigger_incident",
            {
                "title": f"P1 Complaint — {routing.complaint_id}",
                "body": classification.summary if classification else routing.complaint_id,
                "severity": "critical",
                "service_id": os.getenv("PAGERDUTY_SERVICE_ID", ""),
            },
        )

    logger.info("MCP '%s' notified for %s", server, routing.complaint_id)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slack_message(routing: RoutingDecision, classification: Any) -> str:
    prio = classification.priority.value if classification else "?"
    summary = classification.summary if classification else "No summary"
    sla = routing.sla_deadline.strftime("%H:%M UTC") if routing.sla_deadline else "?"
    return (
        f":rotating_light: *{prio} complaint — {routing.complaint_id}*\n"
        f">{summary}\n"
        f"*Team:* {routing.team.value}   *SLA:* {sla}\n"
        f"{'_Auto-routed_' if routing.auto_routed else '_Needs review_'}"
    )


def _stub_notify(routing: RoutingDecision, classification: Any) -> None:
    logger.info(
        "[STUB NOTIFY] %s → %s | %s | SLA %s",
        routing.complaint_id,
        routing.team.value,
        classification.priority.value if classification else "?",
        routing.sla_deadline,
    )
