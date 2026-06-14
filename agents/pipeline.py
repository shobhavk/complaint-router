"""
Customer Complaint Classification & Routing Engine
LangGraph state machine with confidence-gated feedback loop.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from schemas.complaint import (
    ClassifiedComplaint,
    ComplaintInput,
    Priority,
    RoutingDecision,
    Team,
)
from utils.error_handler import AgentError, handle_agent_error
from utils.feedback_store import FeedbackStore
from utils.sla_calculator import SLACalculator

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75


# ── State ─────────────────────────────────────────────────────────────────────

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
    messages: Annotated[list, add_messages]


# ── Node implementations ───────────────────────────────────────────────────────

def parse_and_validate(state: ComplaintState, llm: ChatOpenAI) -> ComplaintState:
    """
    Parse raw input (any format) into a normalised complaint schema.
    Uses the LLM to handle unstructured / semi-structured input.
    """
    try:
        system = SystemMessage(content="""You are a data normalisation agent.
Given any input (plain text, JSON, CSV row, Excel snippet), extract a complaint object.
Return ONLY valid JSON matching this schema:
{
  "customer_id": "string or null",
  "channel": "email|chat|phone|web|unknown",
  "text": "the complaint text",
  "metadata": {"any extra fields": "..."}
}
No markdown, no explanation.""")

        response = llm.invoke([system, HumanMessage(content=state["raw_input"])])
        raw = response.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        normalized = json.loads(raw.strip())
        logger.info("Normalised complaint %s", state["complaint_id"])
        return {**state, "normalized": normalized, "error": None}

    except Exception as exc:
        err = handle_agent_error("parse_and_validate", exc)
        logger.error(err)
        return {**state, "error": str(err)}


def classify_complaint(state: ComplaintState, llm: ChatOpenAI, feedback_store: FeedbackStore) -> ComplaintState:
    """
    Classify the complaint into a category, priority, and sentiment.
    Injects recent human-correction feedback as few-shot examples.
    """
    if state.get("error"):
        return state

    try:
        # Pull recent corrections to ground the classifier (lightweight RAG-equivalent)
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
  "confidence": 0.0–1.0
}}
Priority guide: P1=critical/financial loss, P2=service down, P3=inconvenience, P4=feedback.

Recent human-corrected examples (learn from these):
{examples_text}

Return JSON only.""")

        text = state["normalized"].get("text", state["raw_input"])
        response = llm.invoke([system, HumanMessage(content=text)])
        raw = response.content.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(raw)

        classification = ClassifiedComplaint(
            complaint_id=state["complaint_id"],
            category=data["category"],
            priority=Priority(data["priority"]),
            sentiment=data["sentiment"],
            summary=data["summary"],
            confidence=float(data["confidence"]),
            raw_text=text,
        )
        logger.info(
            "Classified %s → %s (conf=%.2f)",
            state["complaint_id"], classification.category, classification.confidence,
        )
        return {
            **state,
            "classification": classification,
            "confidence": classification.confidence,
            "needs_human_review": classification.confidence < CONFIDENCE_THRESHOLD,
        }

    except Exception as exc:
        err = handle_agent_error("classify_complaint", exc)
        logger.error(err)
        return {**state, "error": str(err)}


def confidence_gate(state: ComplaintState) -> Literal["prioritise", "human_review", "error"]:
    """Router node: branch on confidence or error."""
    if state.get("error"):
        return "error"
    if state["needs_human_review"]:
        return "human_review"
    return "prioritise"


def prioritise(state: ComplaintState, sla_calc: SLACalculator) -> ComplaintState:
    """Enrich with SLA deadline and escalation flag."""
    if state.get("error"):
        return state
    c = state["classification"]
    deadline = sla_calc.calculate(c.priority, c.category)
    # Mutate a copy – no side effects on shared state
    enriched = c.model_copy(update={"sla_deadline": deadline})
    return {**state, "classification": enriched}


def route_complaint(state: ComplaintState) -> ComplaintState:
    """Deterministic routing rules → team assignment."""
    if state.get("error"):
        return state

    c = state["classification"]

    ROUTING_MAP: dict[str, Team] = {
        "billing": Team.BILLING,
        "technical": Team.TECH_SUPPORT,
        "account": Team.ACCOUNT_MGMT,
        "product": Team.PRODUCT,
        "general": Team.GENERAL,
    }

    # Override to escalation for P1 regardless of category
    if c.priority == Priority.P1:
        team = Team.ESCALATION
    else:
        team = ROUTING_MAP.get(c.category, Team.GENERAL)

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
    """Emit to webhook / notification channel (stubbed)."""
    if state.get("error"):
        return state

    r = state["routing"]
    logger.info(
        "[NOTIFY] %s assigned to %s, SLA deadline %s",
        r.complaint_id, r.team.value, r.sla_deadline,
    )
    # In production: call Slack / PagerDuty / ticketing system webhook here
    return state


def human_review_node(state: ComplaintState) -> ComplaintState:
    """
    Placeholder for human-in-the-loop review.
    In production: pushes to a review UI and awaits correction via interrupt().
    """
    logger.info(
        "LOW CONFIDENCE (%.2f): %s queued for review",
        state["confidence"], state["complaint_id"],
    )
    # LangGraph supports .interrupt() here for async human approval
    # For this demo we keep routing with a flag so the UI can show it
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
    """
    If a human correction exists, persist it so future classifications improve.
    This is the feedback loop node.
    """
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
    """Terminal error handler — logs and returns gracefully."""
    logger.error("Pipeline error for %s: %s", state["complaint_id"], state.get("error"))
    return state


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(
    llm: ChatOpenAI,
    feedback_store: FeedbackStore,
    sla_calc: SLACalculator,
    toolkit=None,
) -> StateGraph:
    graph = StateGraph(ComplaintState)

    # Bind dependencies via closures (no global state)
    graph.add_node("parse_and_validate", lambda s: parse_and_validate(s, llm))
    graph.add_node("classify_complaint", lambda s: classify_complaint(s, llm, feedback_store))
    graph.add_node("prioritise", lambda s: prioritise(s, sla_calc))
    graph.add_node("route_complaint", route_complaint)
    if toolkit is not None:
        from tools.mcp_tools import notify_via_mcp
        graph.add_node("notify_and_track", lambda s: notify_via_mcp(s, toolkit))
    else:
        graph.add_node("notify_and_track", notify_and_track)
    graph.add_node("human_review", human_review_node)
    graph.add_node("log_feedback", lambda s: log_feedback(s, feedback_store))
    graph.add_node("error_node", error_node)

    # Edges
    graph.add_edge(START, "parse_and_validate")
    graph.add_edge("parse_and_validate", "classify_complaint")
    graph.add_conditional_edges(
        "classify_complaint",
        confidence_gate,
        {
            "prioritise": "prioritise",
            "human_review": "human_review",
            "error": "error_node",
        },
    )
    graph.add_edge("prioritise", "route_complaint")
    graph.add_edge("route_complaint", "notify_and_track")
    graph.add_edge("notify_and_track", "log_feedback")
    graph.add_edge("log_feedback", END)
    graph.add_edge("human_review", "log_feedback")
    graph.add_edge("error_node", END)

    return graph.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

def create_agent(model_name: str = "Qwen/Qwen2.5-7B-Instruct", base_url: str = "http://localhost:8000/v1"):
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",  # vLLM does not require a real key
        temperature=0.0,
    )
    feedback_store = FeedbackStore()
    sla_calc = SLACalculator()
    from tools.mcp_tools import build_mcp_toolkit
    toolkit = build_mcp_toolkit()
    return build_graph(llm, feedback_store, sla_calc, toolkit=toolkit)


def run_complaint(raw_input: str, human_correction: dict | None = None, agent=None) -> dict:
    """Top-level callable used by the Gradio UI."""
    if agent is None:
        agent = create_agent()

    initial_state: ComplaintState = {
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
        "trace_id": str(uuid.uuid4()),
        "messages": [],
    }

    result = agent.invoke(initial_state)
    return result
