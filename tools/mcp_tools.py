"""
MCP Tool Layer — complaint routing agent
Replaces hard-coded HTTP calls with swappable MCP tool servers.

Usage in pipeline.py — replace notify_and_track with:
    from tools.mcp_tools import build_mcp_toolkit, notify_via_mcp
    toolkit = build_mcp_toolkit()
    graph.add_node("notify_and_track", lambda s: notify_via_mcp(s, toolkit))
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from schemas.complaint import Priority, RoutingDecision, Team

logger = logging.getLogger(__name__)


# ── MCP server config ──────────────────────────────────────────────────────────
# Each entry is one MCP server. Add / remove entries freely —
# the agent auto-discovers the tools each server exposes.

MCP_SERVERS: dict[str, dict] = {
    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env": {"SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN", "")},
        "transport": "stdio",
    },
    "jira": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-jira"],
        "env": {
            "JIRA_URL": os.getenv("JIRA_URL", ""),
            "JIRA_EMAIL": os.getenv("JIRA_EMAIL", ""),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", ""),
        },
        "transport": "stdio",
    },
    "pagerduty": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-pagerduty"],
        "env": {"PAGERDUTY_API_KEY": os.getenv("PAGERDUTY_API_KEY", "")},
        "transport": "stdio",
    },
    # Add more MCP servers here with zero changes to the agent logic:
    # "zendesk": { ... },
    # "servicenow": { ... },
    # "teams": { ... },
}

# Team → which MCP servers to call
TEAM_CHANNEL_MAP: dict[Team, list[str]] = {
    Team.BILLING:      ["slack", "jira"],
    Team.TECH_SUPPORT: ["slack", "jira"],
    Team.ACCOUNT_MGMT: ["slack"],
    Team.ESCALATION:   ["slack", "pagerduty"],
    Team.REVIEW_QUEUE: ["slack"],
    Team.PRODUCT:      ["jira"],
    Team.GENERAL:      ["slack"],
}

# Priority P1 always triggers PagerDuty regardless of team
P1_ALWAYS_ADDS = ["pagerduty"]


def build_mcp_toolkit() -> MultiServerMCPClient | None:
    """
    Initialise the MCP client with all configured servers.
    Returns None gracefully if MCP servers are unavailable (dev mode).
    """
    # Only include servers that have credentials configured
    active = {
        name: cfg
        for name, cfg in MCP_SERVERS.items()
        if _server_has_credentials(name, cfg)
    }

    if not active:
        logger.warning(
            "No MCP servers have credentials — falling back to stub notifications. "
            "Set SLACK_BOT_TOKEN / JIRA_API_TOKEN / PAGERDUTY_API_KEY to activate."
        )
        return None

    try:
        client = MultiServerMCPClient(active)
        logger.info("MCP client initialised with servers: %s", list(active.keys()))
        return client
    except Exception as exc:
        logger.error("MCP client init failed: %s", exc)
        return None


def _server_has_credentials(name: str, cfg: dict) -> bool:
    env = cfg.get("env", {})
    return all(v for v in env.values())


# ── Notification logic ─────────────────────────────────────────────────────────

async def notify_via_mcp(state: dict, toolkit: MultiServerMCPClient | None) -> dict:
    """
    LangGraph node: calls the right MCP tools based on team + priority.
    Falls back to stub logging if MCP unavailable.
    """
    routing: RoutingDecision | None = state.get("routing")
    classification = state.get("classification")

    if not routing or state.get("error"):
        return state

    if toolkit is None:
        # Dev / demo fallback
        _stub_notify(routing, classification)
        return state

    tools: list[BaseTool] = await toolkit.get_tools()
    tool_map = {t.name: t for t in tools}

    servers_to_call = set(TEAM_CHANNEL_MAP.get(routing.team, ["slack"]))
    if classification and classification.priority == Priority.P1:
        servers_to_call.update(P1_ALWAYS_ADDS)

    for server_name in servers_to_call:
        await _dispatch(server_name, routing, classification, tool_map)

    return state


async def _dispatch(
    server: str,
    routing: RoutingDecision,
    classification: Any,
    tool_map: dict[str, BaseTool],
) -> None:
    """Call the appropriate tool on a given MCP server."""
    try:
        if server == "slack":
            tool = tool_map.get("slack_post_message")
            if tool:
                await tool.ainvoke({
                    "channel": f"#complaints-{routing.team.value.lower().replace(' ', '-')}",
                    "text": _slack_message(routing, classification),
                })

        elif server == "jira":
            tool = tool_map.get("jira_create_issue")
            if tool:
                await tool.ainvoke({
                    "project": "COMPLAINTS",
                    "summary": classification.summary if classification else routing.complaint_id,
                    "description": classification.raw_text if classification else "",
                    "priority": classification.priority.value if classification else "P3",
                    "labels": [classification.category if classification else "general"],
                })

        elif server == "pagerduty":
            tool = tool_map.get("pagerduty_trigger_incident")
            if tool:
                await tool.ainvoke({
                    "title": f"P1 Complaint — {routing.complaint_id}",
                    "body": classification.summary if classification else routing.complaint_id,
                    "severity": "critical",
                    "service_id": os.getenv("PAGERDUTY_SERVICE_ID", ""),
                })

        logger.info("MCP %s notified for %s", server, routing.complaint_id)

    except Exception as exc:
        # Non-fatal — log and continue. Other channels still get notified.
        logger.error("MCP dispatch to %s failed for %s: %s", server, routing.complaint_id, exc)


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
