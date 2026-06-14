"""
Gradio Dashboard — Customer Complaint Routing Engine
Run: python app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from agents.pipeline import create_agent, run_complaint
from utils.feedback_store import FeedbackStore
from utils.input_parser import parse_input

# ── Globals ──────────────────────────────────────────────────────────────────

AGENT = None
FEEDBACK_STORE = FeedbackStore()
ROUTING_HISTORY: list[dict] = []

TEAM_COLORS = {
    "Billing": "#E24B4A",
    "Tech Support": "#378ADD",
    "Account Management": "#1D9E75",
    "Escalation": "#BA7517",
    "Review Queue": "#888780",
    "Product": "#7F77DD",
    "General": "#639922",
}

PRIORITY_EMOJI = {"P1": "🔴", "P2": "🟠", "P3": "🔵", "P4": "🟢"}


def get_agent():
    global AGENT
    if AGENT is None:
        model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        AGENT = create_agent(model_name=model, base_url=base_url)
    return AGENT


# ── Core processing ───────────────────────────────────────────────────────────

def process_complaint(text: str, file, correction_category: str, correction_priority: str):
    """Main handler — accepts text or file upload."""
    inputs: list[str] = []

    if file is not None:
        try:
            inputs = parse_input(file.name)
        except Exception as e:
            return _error_response(f"File parse error: {e}")
    elif text.strip():
        inputs = [text.strip()]
    else:
        return _error_response("Please enter a complaint or upload a file.")

    results = []
    for raw in inputs[:10]:  # cap batch at 10 for demo
        correction = None
        if correction_category and correction_category != "None":
            correction = {
                "category": correction_category.lower(),
                "priority": correction_priority,
            }
        try:
            state = run_complaint(raw, human_correction=correction, agent=get_agent())
            results.append(state)
            ROUTING_HISTORY.insert(0, _summarise(state))
        except Exception as e:
            results.append({"error": str(e), "complaint_id": "ERR", "raw_input": raw})

    return _format_results(results)


def _summarise(state: dict) -> dict:
    c = state.get("classification")
    r = state.get("routing")
    return {
        "ID": state.get("complaint_id", "-"),
        "Time": datetime.utcnow().strftime("%H:%M:%S"),
        "Priority": PRIORITY_EMOJI.get(c.priority.value, "?") + " " + c.priority.value if c else "-",
        "Category": c.category.title() if c else "-",
        "Team": r.team.value if r else "-",
        "Confidence": f"{c.confidence:.0%}" if c else "-",
        "Auto": "✅" if (r and r.auto_routed) else "👤 Review",
        "SLA": r.sla_deadline.strftime("%H:%M UTC") if r else "-",
    }


def _format_results(results: list[dict]) -> tuple:
    """Returns (status_md, routing_table_df, metrics_md)"""
    rows = []
    for s in results:
        if s.get("error"):
            rows.append({"ID": s.get("complaint_id", "ERR"), "Status": f"❌ {s['error']}"})
            continue
        rows.append(_summarise(s))

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ID", "Status"])

    # Status markdown
    if len(results) == 1 and not results[0].get("error"):
        s = results[0]
        c = s.get("classification")
        r = s.get("routing")
        conf = c.confidence if c else 0
        review_note = "\n\n> ⚠️ **Low confidence** — queued for human review. Correction will feed back into the model." if s.get("needs_human_review") else ""
        status = f"""### {PRIORITY_EMOJI.get(c.priority.value,'?')} {c.priority.value} — {c.category.title()}

**Summary:** {c.summary}  
**Team:** {r.team.value if r else '-'}  
**Confidence:** {conf:.0%} {'✅' if conf >= 0.75 else '⚠️'}  
**SLA deadline:** {r.sla_deadline.strftime('%Y-%m-%d %H:%M UTC') if r else '-'}  
**Sentiment:** {c.sentiment.title() if c else '-'}{review_note}
"""
    else:
        status = f"Processed {len(results)} complaint(s)."

    # Metrics
    stats = FEEDBACK_STORE.get_stats()
    metrics = f"""**Today's stats**  
Routed: {len(ROUTING_HISTORY)} · Corrections: {stats['total_corrections']}"""

    return status, df, metrics


def _error_response(msg: str):
    return f"❌ {msg}", pd.DataFrame(), ""


def get_history_df():
    return pd.DataFrame(ROUTING_HISTORY[:20]) if ROUTING_HISTORY else pd.DataFrame(columns=["ID", "Time", "Priority", "Category", "Team", "Confidence", "Auto", "SLA"])


def submit_correction(ticket_id: str, new_team: str, new_priority: str, reason: str):
    """Log a manual correction back through the feedback loop."""
    if not ticket_id:
        return "Please enter a ticket ID."
    FEEDBACK_STORE.save(
        text=f"Manual correction for {ticket_id}",
        original_category="unknown",
        corrected_category=new_team.lower(),
        corrected_priority=new_priority,
        corrected_team=new_team,
    )
    return f"✅ Correction logged for {ticket_id} → {new_team} ({new_priority}). Feedback will improve future routing."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Complaint Routing Engine",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .metric-box { text-align: center; padding: 12px; border-radius: 10px; }
        .priority-p1 { color: #E24B4A; font-weight: 600; }
        footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ──
        gr.Markdown("""
# 🎯 Customer Complaint Routing Engine
**Hackathon demo** · LangGraph · Qwen via vLLM · LangSmith · Feedback loop
""")

        # ── Tabs ──
        with gr.Tabs():

            # ── TAB 1: Route ──
            with gr.Tab("Route a complaint"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Input — paste text or upload file (CSV / Excel / JSON / text)")
                        complaint_text = gr.Textbox(
                            label="Complaint text",
                            placeholder="e.g. I was charged twice for my annual plan. This is unacceptable and I need an urgent refund.",
                            lines=4,
                        )
                        complaint_file = gr.File(
                            label="Or upload file",
                            file_types=[".csv", ".xlsx", ".json", ".txt"],
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Human override (optional)")
                        gr.Markdown("*Set these only if you want to pre-correct the agent.*")
                        corr_category = gr.Dropdown(
                            label="Force category",
                            choices=["None", "billing", "technical", "account", "product", "general"],
                            value="None",
                        )
                        corr_priority = gr.Dropdown(
                            label="Force priority",
                            choices=["P1", "P2", "P3", "P4"],
                            value="P2",
                        )

                run_btn = gr.Button("▶ Run agent", variant="primary", size="lg")

                with gr.Row():
                    status_md = gr.Markdown(label="Result")

                routing_table = gr.DataFrame(
                    label="Routing decisions",
                    interactive=False,
                    wrap=True,
                )
                metrics_md = gr.Markdown()

                run_btn.click(
                    process_complaint,
                    inputs=[complaint_text, complaint_file, corr_category, corr_priority],
                    outputs=[status_md, routing_table, metrics_md],
                )

                gr.Examples(
                    examples=[
                        ["I've been charged twice for my subscription this month. My invoice number is #8821. This needs to be resolved immediately.", None, "None", "P1"],
                        ["The mobile app crashes every time I try to open my account statements. iOS 17.2.", None, "None", "P2"],
                        ["I'd like to update my billing address please.", None, "None", "P4"],
                        ["I am contacting my lawyer if this isn't resolved in 24 hours. You've lost my data.", None, "None", "P1"],
                    ],
                    inputs=[complaint_text, complaint_file, corr_category, corr_priority],
                    label="Quick examples",
                )

            # ── TAB 2: Live queue ──
            with gr.Tab("Live routing queue"):
                gr.Markdown("#### Recent routing decisions")
                refresh_btn = gr.Button("🔄 Refresh")
                history_table = gr.DataFrame(interactive=False, wrap=True)
                refresh_btn.click(get_history_df, outputs=[history_table])
                demo.load(get_history_df, outputs=[history_table])

            # ── TAB 3: Feedback loop ──
            with gr.Tab("Feedback & corrections"):
                gr.Markdown("""#### Human-in-the-loop corrections
When the agent routes incorrectly, submit a correction here. It's persisted to SQLite and injected as few-shot examples in the next classification call — **this is the feedback loop**.
""")
                with gr.Row():
                    ticket_id_in = gr.Textbox(label="Ticket ID", placeholder="TK-1091")
                    new_team_in = gr.Dropdown(
                        label="Correct team",
                        choices=["Billing", "Tech Support", "Account Management", "Escalation", "Product"],
                    )
                    new_prio_in = gr.Dropdown(label="Correct priority", choices=["P1", "P2", "P3", "P4"])

                reason_in = gr.Textbox(label="Reason (optional)", placeholder="Agent confused account issue with billing")
                corr_btn = gr.Button("Submit correction", variant="secondary")
                corr_result = gr.Markdown()
                corr_btn.click(
                    submit_correction,
                    inputs=[ticket_id_in, new_team_in, new_prio_in, reason_in],
                    outputs=[corr_result],
                )

            # ── TAB 4: Architecture ──
            with gr.Tab("Architecture & setup"):
                gr.Markdown(open(Path(__file__).parent / "README.md").read()
                            if Path(Path(__file__).parent / "README.md").exists()
                            else "See README.md for setup instructions.")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
