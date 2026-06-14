"""Utility: consistent error handling across agent nodes."""
from __future__ import annotations
import traceback
import logging

logger = logging.getLogger(__name__)


class AgentError(Exception):
    def __init__(self, node: str, message: str):
        self.node = node
        super().__init__(f"[{node}] {message}")


def handle_agent_error(node: str, exc: Exception) -> AgentError:
    logger.error("Node '%s' failed: %s\n%s", node, exc, traceback.format_exc())
    return AgentError(node=node, message=str(exc))
