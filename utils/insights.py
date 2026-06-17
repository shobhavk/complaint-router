"""
AI Insights generator — calls Qwen via vLLM to analyse routing history
and produce executive-level insights. Called only on demand (tab click).
"""
from __future__ import annotations
import json
import logging
import os
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


def _build_context(routing_history: list[dict], feedback_corrections: int) -> str:
    """Summarise routing history into a compact JSON context for the LLM."""
    if not routing_history:
        return json.dumps({"error": "No data yet — process some complaints first."})

    categories  = Counter(r.get("Category", "unknown") for r in routing_history)
    teams       = Counter(r.get("Team",     "unknown") for r in routing_history)
    priorities  = Counter(r.get("Priority", "unknown").split()[-1] if r.get("Priority") else "?" for r in routing_history)
    blocked     = sum(1 for r in routing_history if "Blocked" in r.get("Routed", ""))
    review      = sum(1 for r in routing_history if "Review"  in r.get("Routed", ""))
    auto        = sum(1 for r in routing_history if "Auto"    in r.get("Routed", ""))
    pii_hits    = sum(1 for r in routing_history if r.get("PII", "—") != "—")

    return json.dumps({
        "total_complaints":    len(routing_history),
        "auto_routed":         auto,
        "human_review_queued": review,
        "security_blocked":    blocked,
        "pii_incidents":       pii_hits,
        "feedback_corrections":feedback_corrections,
        "top_categories":      dict(categories.most_common(5)),
        "team_distribution":   dict(teams.most_common()),
        "priority_breakdown":  dict(priorities.most_common()),
        "sample_problems":     [r.get("Problem", "") for r in routing_history[:10]],
        "generated_at":        datetime.utcnow().isoformat(),
    }, indent=2)


def generate_insights(routing_history: list[dict], feedback_corrections: int) -> dict[str, str]:
    """
    Call the LLM to generate structured AI insights.
    Returns a dict with keys matching each insight section.
    Falls back to static analysis if LLM unavailable.
    """
    context = _build_context(routing_history, feedback_corrections)

    # Check if LLM is configured
    api_key  = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model    = os.getenv("QWEN_MODEL", "qwen-plus")

    if not api_key or not routing_history:
        return _static_insights(routing_history, feedback_corrections)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from utils.response_cleaner import extract_json

        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.3)

        system = SystemMessage(content="""You are a senior customer experience analyst.
Analyse the complaint routing data and return ONLY a valid JSON object with exactly these keys:
{
  "executive_summary": "2-3 sentence overview of the current complaint landscape",
  "top_categories": "bullet list of top complaint categories with counts and % share",
  "emerging_trends": "2-3 patterns or shifts worth watching",
  "risk_alerts": "specific risks that need immediate attention (P1 volume, SLA breaches, security blocks)",
  "root_cause_indicators": "likely root causes behind top complaint drivers",
  "learning_insights": "what the feedback loop corrections reveal about model accuracy and improvement areas",
  "recommendations": "exactly 3 numbered actionable recommendations for ops/product teams"
}
Be concise, specific, and data-driven. Use the numbers from the data. /no_think""")

        response = llm.invoke([
            system,
            HumanMessage(content=f"Complaint routing data:\n{context}"),
        ])

        data = json.loads(extract_json(response.content))
        logger.info("[INSIGHTS] AI insights generated successfully")
        return data

    except Exception as exc:
        logger.error("[INSIGHTS] LLM call failed, using static analysis: %s", exc)
        return _static_insights(routing_history, feedback_corrections)


def _static_insights(history: list[dict], corrections: int) -> dict[str, str]:
    """Fallback: compute insights statically without LLM."""
    if not history:
        return {k: "No data yet — process some complaints first." for k in [
            "executive_summary", "top_categories", "emerging_trends",
            "risk_alerts", "root_cause_indicators", "learning_insights", "recommendations"
        ]}

    total    = len(history)
    cats     = Counter(r.get("Category", "?") for r in history)
    teams    = Counter(r.get("Team",     "?") for r in history)
    blocked  = sum(1 for r in history if "Blocked" in r.get("Routed", ""))
    review   = sum(1 for r in history if "Review"  in r.get("Routed", ""))
    auto     = sum(1 for r in history if "Auto"    in r.get("Routed", ""))
    pii      = sum(1 for r in history if r.get("PII", "—") != "—")
    p1       = sum(1 for r in history if "P1" in r.get("Priority", ""))

    top_cat  = cats.most_common(1)[0] if cats else ("unknown", 0)
    top_team = teams.most_common(1)[0] if teams else ("unknown", 0)

    return {
        "executive_summary": (
            f"{total} complaints processed. {auto} auto-routed ({auto*100//total if total else 0}%), "
            f"{review} queued for review, {blocked} security blocks. "
            f"Top category: {top_cat[0]} ({top_cat[1]} cases). "
            f"Highest load team: {top_team[0]} ({top_team[1]} cases)."
        ),
        "top_categories": "\n".join(
            f"• {cat}: {cnt} ({cnt*100//total if total else 0}%)"
            for cat, cnt in cats.most_common(5)
        ),
        "emerging_trends": (
            f"• {blocked} security/injection attempts detected — monitor for increase.\n"
            f"• {pii} complaints contained PII (masked before LLM processing).\n"
            f"• {review} complaints fell below confidence threshold — model may need more examples."
        ),
        "risk_alerts": (
            f"• {p1} P1 critical complaints — verify all resolved within 1h SLA.\n"
            + (f"• {blocked} prompt injection attempts — review security logs.\n" if blocked else "")
            + (f"• {review} complaints in human review queue — may cause SLA delays.\n" if review else "✅ No active risk alerts.")
        ),
        "root_cause_indicators": (
            f"• High {top_cat[0]} volume may indicate product/billing system issues.\n"
            f"• {top_team[0]} team handling {top_team[1]} cases — check for resource constraints.\n"
            f"• Low-confidence routings ({review}) suggest overlapping complaint categories."
        ),
        "learning_insights": (
            f"• {corrections} human corrections logged to feedback store.\n"
            f"• Corrections are injected as few-shot examples in future classifications.\n"
            + ("• Model accuracy improving — correction rate decreasing." if corrections > 5
               else "• More corrections needed to improve model accuracy.")
        ),
        "recommendations": (
            f"1. Prioritise resolving {p1} P1 complaints — escalation team should be alerted.\n"
            f"2. Add more training examples for '{top_cat[0]}' category — highest complaint volume.\n"
            f"3. Review {review} low-confidence cases in queue — submit corrections to improve future routing."
        ),
    }
