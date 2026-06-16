"""
Customer Complaint Classification & Routing Engine
LangGraph state machine with confidence-gated feedback loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from schemas.complaint import (
    ClassifiedComplaint,
    Priority,
    RoutingDecision,
    Team,
)
from utils.error_handler import handle_agent_error
from utils.response_cleaner import extract_json
from utils.guardrails import screen_input
from utils.feedback_store import FeedbackStore
from utils.sla_calculator import SLACalculator

load_dotenv()
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75


# ── State ──────────────────────────────────────────────────────────────────────

class ComplaintState(TypedDict):
    complaint_id: str
    raw_input: str
    normalized: dict[str, Any] | None
    classification: ClassifiedComplaint | None
    routing: RoutingDecision | None
    confidence: float
    needs_human_review: bool
    human_correction: dict[str, Any] | None
    feedback_logged: bool
    error: str | None
    trace_id: str
    security: dict | None
    messages: Annotated[list, add_messages]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def security_screen(state: ComplaintState) -> ComplaintState:
    """
    Run guardrails before any LLM call:
    - Detect prompt injection
    - Mask PII
    If injection detected, short-circuit to error with security flag.
    """
    result = screen_input(state["raw_input"])
    if not result["safe"]:
        logger.warning("[SECURITY] Guardrail triggered: %s | Input: %.80s",
                       result["reason"], state["raw_input"])
        return {
            **state,
            "security": result,
            "error": f"SECURITY_EXCEPTION: {result['reason']} — input flagged and routed to triage",
        }
    # Replace raw_input with PII-masked version for all downstream nodes
    return {**state, "raw_input": result["masked_text"], "security": result}


def parse_and_validate(state: ComplaintState, llm: ChatOpenAI) -> ComplaintState:
    try:
        system = SystemMessage(content="""You are a data normalisation agent.
Given any input (plain text, JSON, CSV row, Excel snippet), extract a complaint object.
Return ONLY valid JSON matching this schema:
{
  "customer_id": "string or null",
  "channel": "email|chat|phone|web|unknown",
  "text": "the complaint text",
  "metadata": {}
}
No markdown, no explanation. /no_think""")

        response = llm.invoke([system, HumanMessage(content=state["raw_input"])])
        normalized = json.loads(extract_json(response.content))
        logger.info("Normalised complaint %s", state["complaint_id"])
        return {**state, "normalized": normalized, "error": None}

    except Exception as exc:
        err = handle_agent_error("parse_and_validate", exc)
        logger.error(err, exc_info=True)
        return {**state, "error": str(err)}


def classify_complaint(state: ComplaintState, llm: ChatOpenAI, feedback_store: FeedbackStore) -> ComplaintState:
    if state.get("error"):
        return state
    try:
        examples = feedback_store.get_recent_examples(n=5)
        examples_text = "\n".join(
            f"- Input: \"{e['text']}\" → Category: {e['category']}, Priority: {e['priority']}"
            for e in examples
        ) or "No examples yet."

        system = SystemMessage(content=f"""You are an expert complaint classifier.
Categorise the complaint and return ONLY valid JSON:
{{
  "category": "billing|technical|account|product|general",
  "priority": "P1|P2|P3|P4",
  "sentiment": "angry|frustrated|neutral|satisfied",
  "summary": "one sentence max",
  "confidence": 0.0-1.0
}}
Priority guide: P1=critical/financial loss, P2=service down, P3=inconvenience, P4=feedback.

Recent human-corrected examples (learn from these):
{examples_text}

Return JSON only. No markdown. /no_think""")

        text = (state["normalized"] or {}).get("text", state["raw_input"])
        response = llm.invoke([system, HumanMessage(content=text)])
        data = json.loads(extract_json(response.content))

        classification = ClassifiedComplaint(
            complaint_id=state["complaint_id"],
            category=data["category"],
            priority=Priority(data["priority"]),
            sentiment=data["sentiment"],
            summary=data["summary"],
            confidence=float(data["confidence"]),
            raw_text=text,
        )
        logger.info("Classified %s → %s (conf=%.2f)", state["complaint_id"], classification.category, classification.confidence)
        return {
            **state,
            "classification": classification,
            "confidence": classification.confidence,
            "needs_human_review": classification.confidence < CONFIDENCE_THRESHOLD,
        }

    except Exception as exc:
        err = handle_agent_error("classify_complaint", exc)
        logger.error(err, exc_info=True)
        return {**state, "error": str(err)}


def confidence_gate(state: ComplaintState) -> Literal["prioritise", "human_review", "error"]:
    if state.get("error"):
        return "error"
    if state["needs_human_review"]:
        return "human_review"
    return "prioritise"


def prioritise(state: ComplaintState, sla_calc: SLACalculator) -> ComplaintState:
    if state.get("error"):
        return state
    c = state["classification"]
    deadline = sla_calc.calculate(c.priority, c.category)
    enriched = c.model_copy(update={"sla_deadline": deadline})
    return {**state, "classification": enriched}


def route_complaint(state: ComplaintState) -> ComplaintState:
    if state.get("error"):
        return state
    c = state["classification"]
    ROUTING_MAP = {
        "billing": Team.BILLING,
        "technical": Team.TECH_SUPPORT,
        "account": Team.ACCOUNT_MGMT,
        "product": Team.PRODUCT,
        "general": Team.GENERAL,
    }
    team = Team.ESCALATION if c.priority == Priority.P1 else ROUTING_MAP.get(c.category, Team.GENERAL)
    routing = RoutingDecision(
        complaint_id=state["complaint_id"],
        team=team,
        channel="slack",
        assigned_at=datetime.utcnow(),
        sla_deadline=c.sla_deadline,
        auto_routed=True,
    )
    logger.info("Routed %s → %s", state["complaint_id"], team.value)
    return {**state, "routing": routing}


def notify_and_track(state: ComplaintState) -> ComplaintState:
    if state.get("error"):
        return state
    r = state["routing"]
    logger.info("[NOTIFY] %s → %s | SLA %s", r.complaint_id, r.team.value, r.sla_deadline)
    return state


def human_review_node(state: ComplaintState) -> ComplaintState:
    logger.info("LOW CONFIDENCE (%.2f): %s queued for review", state["confidence"], state["complaint_id"])
    return {
        **state,
        "needs_human_review": True,
        "routing": RoutingDecision(
            complaint_id=state["complaint_id"],
            team=Team.REVIEW_QUEUE,
            channel="dashboard",
            assigned_at=datetime.utcnow(),
            sla_deadline=datetime.utcnow() + timedelta(hours=2),
            auto_routed=False,
        ),
    }


def log_feedback(state: ComplaintState, feedback_store: FeedbackStore) -> ComplaintState:
    correction = state.get("human_correction")
    if correction and state.get("classification"):
        feedback_store.save(
            text=state["classification"].raw_text,
            original_category=state["classification"].category,
            corrected_category=correction.get("category", state["classification"].category),
            corrected_priority=correction.get("priority", state["classification"].priority.value),
            corrected_team=correction.get("team"),
        )
        logger.info("Feedback logged for %s", state["complaint_id"])
    return {**state, "feedback_logged": True}


def error_node(state: ComplaintState) -> ComplaintState:
    logger.error("Pipeline error for %s: %s", state["complaint_id"], state.get("error"))
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph(llm, feedback_store, sla_calc, toolkit=None):
    graph = StateGraph(ComplaintState)

    graph.add_node("security_screen", security_screen)
    graph.add_node("parse_and_validate", lambda s: parse_and_validate(s, llm))
    graph.add_node("classify_complaint", lambda s: classify_complaint(s, llm, feedback_store))
    graph.add_node("prioritise", lambda s: prioritise(s, sla_calc))
    graph.add_node("route_complaint", route_complaint)
    graph.add_node("human_review", human_review_node)
    graph.add_node("log_feedback", lambda s: log_feedback(s, feedback_store))
    graph.add_node("error_node", error_node)

    if toolkit is not None and toolkit.available:
        from tools.mcp_tools import notify_via_mcp
        graph.add_node("notify_and_track", lambda s: notify_via_mcp(s, toolkit))
    else:
        graph.add_node("notify_and_track", notify_and_track)

    graph.add_edge(START, "security_screen")
    graph.add_conditional_edges(
        "security_screen",
        lambda s: "error_node" if s.get("error") else "parse_and_validate",
        {"error_node": "error_node", "parse_and_validate": "parse_and_validate"},
    )
    graph.add_edge("parse_and_validate", "classify_complaint")
    graph.add_conditional_edges(
        "classify_complaint",
        confidence_gate,
        {"prioritise": "prioritise", "human_review": "human_review", "error": "error_node"},
    )
    graph.add_edge("prioritise", "route_complaint")
    graph.add_edge("route_complaint", "notify_and_track")
    graph.add_edge("notify_and_track", "log_feedback")
    graph.add_edge("log_feedback", END)
    graph.add_edge("human_review", "log_feedback")
    graph.add_edge("error_node", END)

    return graph.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

def create_agent():
    """
    Reads config from environment variables.
    Set in .env file or export before running.
    """
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError(
            "QWEN_API_KEY not set.\n"
            "Add it to your .env file:\n"
            "  QWEN_API_KEY=sk-xxxxxxxxxxxx"
        )

    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("QWEN_MODEL", "qwen-plus")

    logger.info("Using model: %s  base_url: %s", model, base_url)

    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,           # ← real key from env
        temperature=0.0,
    )

    feedback_store = FeedbackStore()
    sla_calc = SLACalculator()

    from tools.mcp_tools import build_mcp_toolkit
    toolkit = build_mcp_toolkit()

    return build_graph(llm, feedback_store, sla_calc, toolkit=toolkit)


def _build_initial_state(raw_input: str, human_correction: dict | None) -> ComplaintState:
    """Build a fresh state dict for one complaint."""
    return {
        "complaint_id": str(uuid.uuid4())[:8].upper(),
        "raw_input": raw_input,
        "normalized": None,
        "classification": None,
        "routing": None,
        "confidence": 0.0,
        "needs_human_review": False,
        "human_correction": human_correction,
        "feedback_logged": False,
        "error": None,
        "security": None,
        "trace_id": str(uuid.uuid4()),
        "messages": [],
    }


def run_complaint(raw_input: str, human_correction: dict | None = None, agent=None) -> dict:
    """Synchronous — single complaint. Used by simple UI calls."""
    if agent is None:
        agent = create_agent()
    return agent.invoke(_build_initial_state(raw_input, human_correction))


async def run_complaint_async(raw_input: str, human_correction: dict | None = None, agent=None) -> dict:
    """Async — single complaint via ainvoke."""
    if agent is None:
        agent = create_agent()
    try:
        return await agent.ainvoke(_build_initial_state(raw_input, human_correction))
    except Exception as exc:
        logger.error("Async complaint failed: %s", exc, exc_info=True)
        return {**_build_initial_state(raw_input, human_correction), "error": str(exc)}


async def run_batch_async(
    inputs: list[str],
    human_correction: dict | None = None,
    agent=None,
    max_concurrency: int = 5,
) -> list[dict]:
    """
    Async batch — runs all complaints concurrently (capped at max_concurrency).
    Much faster than sequential for file uploads with many rows.
    """
    if agent is None:
        agent = create_agent()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(raw: str) -> dict:
        async with semaphore:
            return await run_complaint_async(raw, human_correction, agent)

    logger.info("Running batch of %d complaints (max_concurrency=%d)", len(inputs), max_concurrency)
    results = await asyncio.gather(*[_run_one(raw) for raw in inputs], return_exceptions=False)
    logger.info("Batch complete — %d processed", len(results))
    return list(results)
