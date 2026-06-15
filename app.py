"""
Complaint Routing Engine — single-screen Gradio dashboard
Run: python app.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── LangSmith — must be set before any langchain imports ─────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "complaint-router")
    print(f"LangSmith tracing enabled — project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith not configured — set LANGCHAIN_API_KEY in .env to enable")

# ── Logging — writes to console AND agent.log ─────────────────────────────────
def setup_logging():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler("agent.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting Complaint Routing Engine...")

import asyncio
from agents.pipeline import create_agent, run_complaint, run_batch_async
from utils.feedback_store import FeedbackStore
from utils.input_parser import parse_input

# ── Globals ───────────────────────────────────────────────────────────────────

AGENT = None
FEEDBACK_STORE = FeedbackStore()
ROUTING_HISTORY: list[dict] = []

PRIORITY_EMOJI = {"P1": "🔴", "P2": "🟠", "P3": "🔵", "P4": "🟢"}


def get_agent():
    global AGENT
    if AGENT is None:
        AGENT = create_agent()
    return AGENT


# ── Processing ────────────────────────────────────────────────────────────────

def _get_event_loop():
    """Get or create an event loop safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def process_complaint(text: str, file, corr_team: str, corr_priority: str):
    inputs: list[str] = []

    if file is not None:
        try:
            inputs = parse_input(file.name)
        except Exception as e:
            return f"File parse error: {e}", _history_df(), _metrics()
    elif text.strip():
        inputs = [text.strip()]
    else:
        return "Please enter a complaint or upload a file.", _history_df(), _metrics()

    correction = None
    if corr_team and corr_team != "Auto-detect":
        correction = {"category": corr_team.lower(), "priority": corr_priority, "team": corr_team}

    try:
        if len(inputs) == 1:
            # Single complaint — sync path (simpler, no overhead)
            results = [run_complaint(inputs[0], human_correction=correction, agent=get_agent())]
        else:
            # Batch — run all concurrently via async (much faster for CSV/Excel uploads)
            logger.info("Batch mode: %d complaints — running async", len(inputs))
            loop = _get_event_loop()
            results = loop.run_until_complete(
                run_batch_async(inputs[:10], human_correction=correction, agent=get_agent())
            )

        for state in results:
            ROUTING_HISTORY.insert(0, _summarise(state))

        summary = _format_result(results[0] if len(results) == 1 else None, len(results))
        return summary, _history_df(), _metrics()

    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True)
        return f"Error: {e}", _history_df(), _metrics()


def submit_correction(ticket_id: str, new_team: str, new_priority: str):
    if not ticket_id.strip():
        return "⚠️ Please enter a ticket ID."
    FEEDBACK_STORE.save(
        text=f"Manual correction for {ticket_id}",
        original_category="unknown",
        corrected_category=new_team.lower(),
        corrected_priority=new_priority,
        corrected_team=new_team,
    )
    return f"Correction logged: {ticket_id} → {new_team} ({new_priority}). Will improve next classification."


def _summarise(state: dict) -> dict:
    c = state.get("classification")
    r = state.get("routing")
    return {
        "ID": state.get("complaint_id", "-"),
        "Time": datetime.utcnow().strftime("%H:%M:%S"),
        "Priority": (PRIORITY_EMOJI.get(c.priority.value, "?") + " " + c.priority.value) if c else "-",
        "Category": c.category.title() if c else "-",
        "Team": r.team.value if r else "-",
        "Conf.": f"{c.confidence:.0%}" if c else "-",
        "Routed": "✅ Auto" if (r and r.auto_routed) else "👤 Review",
        "SLA": r.sla_deadline.strftime("%H:%M UTC") if r and r.sla_deadline else "-",
    }


def _format_result(state: dict | None, total: int) -> str:
    if state is None:
        return f"Processed {total} complaints. See routing queue below."

    c = state.get("classification")
    r = state.get("routing")

    if state.get("error"):
        return f"Error: {state['error']}"

    if not c or not r:
        return "Classification failed — check agent.log for details."

    conf = c.confidence
    emoji = PRIORITY_EMOJI.get(c.priority.value, "?")
    sla = r.sla_deadline.strftime("%H:%M UTC") if r.sla_deadline else "-"
    conf_label = "High confidence" if conf >= 0.75 else "Low confidence — queued for review"
    review_note = "\nNote: Low confidence — please use the override below to correct and improve routing."         if state.get("needs_human_review") else ""

    # complaint_id can be top-level or inside classification
    ticket_id = (
        state.get("complaint_id")
        or (c.complaint_id if c else None)
        or "—"
    )

    return "\n".join([
        f"{emoji} {c.priority.value} — {c.category.title()}",
        f"Summary   : {c.summary}",
        f"Team      : {r.team.value}",
        f"SLA       : {sla}",
        f"Confidence: {conf:.0%} ({conf_label})",
        f"Sentiment : {c.sentiment.title()}",
        f"Ticket ID : {ticket_id}",
    ]) + review_note


def _history_df() -> pd.DataFrame:
    return pd.DataFrame(ROUTING_HISTORY[:15]) if ROUTING_HISTORY else pd.DataFrame(
        columns=["ID", "Time", "Priority", "Category", "Team", "Conf.", "Routed", "SLA"]
    )


def _metrics() -> str:
    total = len(ROUTING_HISTORY)
    auto = sum(1 for r in ROUTING_HISTORY if r.get("Routed") == "✅ Auto")
    corrections = FEEDBACK_STORE.get_stats()["total_corrections"]
    pct = f"{auto/total:.0%}" if total else "—"
    return f"**{total}** routed today · **{pct}** auto-routed · **{corrections}** feedback corrections"


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Complaint Routing Engine",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        footer { display: none !important; }
        .result-box { font-size: 14px; }
        """,
    ) as demo:

        # ── Header ──
        with gr.Row():
            gr.Markdown("## 🎯 Customer Complaint Routing Engine")
            langsmith_status = (
                "🟢 LangSmith connected"
                if os.getenv("LANGCHAIN_API_KEY")
                else "⚪ LangSmith not connected"
            )
            gr.Markdown(
                f"<div style='text-align:right;padding-top:8px;font-size:12px;color:gray'>"
                f"LangGraph · Qwen · vLLM (ROCm) · MCP<br>{langsmith_status}</div>"
            )

        metrics_md = gr.Markdown("Loading...")

        # ── Main two-column layout ──
        with gr.Row():

            # LEFT — input + result + override
            with gr.Column(scale=1):
                gr.Markdown("#### Input")
                complaint_text = gr.Textbox(
                    label="Complaint text",
                    placeholder="Paste complaint — or upload CSV / Excel / JSON below",
                    lines=4,
                )
                complaint_file = gr.File(
                    label="Upload file (CSV · Excel · JSON · TXT)",
                    file_types=[".csv", ".xlsx", ".json", ".txt"],
                )
                run_btn = gr.Button("▶  Run agent", variant="primary")

                result_md = gr.Markdown(
                    value="_Result appears here_",
                    elem_classes=["result-box"],
                )

                gr.Markdown("#### Human override → feedback loop")
                with gr.Row():
                    corr_team = gr.Dropdown(
                        label="Correct team",
                        choices=["Auto-detect", "Billing", "Tech Support", "Account Management", "Escalation", "Product"],
                        value="Auto-detect",
                        scale=2,
                    )
                    corr_priority = gr.Dropdown(
                        label="Priority",
                        choices=["P1", "P2", "P3", "P4"],
                        value="P2",
                        scale=1,
                    )
                with gr.Row():
                    ticket_id = gr.Textbox(label="Ticket ID", placeholder="e.g. A1B2C3D4", scale=2)
                    corr_btn = gr.Button("Submit correction", scale=1)
                corr_result = gr.Markdown()

                gr.Examples(
                    examples=[
                        ["I've been charged twice for my subscription. Invoice #8821. Fix this urgently.", None, "Auto-detect", "P1"],
                        ["App crashes every time I open account statements. iOS 17.", None, "Auto-detect", "P2"],
                        ["Please update my billing address.", None, "Auto-detect", "P4"],
                        ["I am contacting my lawyer if this isn't resolved in 24 hours.", None, "Auto-detect", "P1"],
                    ],
                    inputs=[complaint_text, complaint_file, corr_team, corr_priority],
                    label="Quick examples",
                )

            # RIGHT — routing queue + SLA
            with gr.Column(scale=1):
                gr.Markdown("#### Live routing queue")
                history_table = gr.DataFrame(
                    value=_history_df(),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("🔄 Refresh queue", size="sm")

                gr.Markdown("#### SLA compliance")
                sla_md = gr.Markdown("""
| Team | Compliance |
|------|-----------|
| Billing | 🟡 88% |
| Tech Support | 🟢 94% |
| Account Mgmt | 🟠 72% |
| Escalation | 🟢 96% |
""")

        # ── Event wiring ──
        run_btn.click(
            process_complaint,
            inputs=[complaint_text, complaint_file, corr_team, corr_priority],
            outputs=[result_md, history_table, metrics_md],
        )
        corr_btn.click(
            submit_correction,
            inputs=[ticket_id, corr_team, corr_priority],
            outputs=[corr_result],
        )
        refresh_btn.click(
            lambda: (_history_df(), _metrics()),
            outputs=[history_table, metrics_md],
        )
        demo.load(
            lambda: (_history_df(), _metrics()),
            outputs=[history_table, metrics_md],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=True,
        show_error=True,
    )
