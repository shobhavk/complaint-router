# Customer Complaint Classification & Routing Engine

A production-grade AI agent system for the hackathon challenge. Built with **LangGraph**, **Qwen via vLLM**, **LangSmith**, and **Gradio**.

---

## Architecture

```
Input (any format)
    ↓
Parse & Validate  ──error──→  Error node
    ↓
Classify (Qwen/vLLM)
    ↓
Confidence gate
  ├─ ≥ 0.75 → Prioritise → Route → Notify → Log feedback
  └─ < 0.75 → Human review queue → Log feedback (loop ↑)
                                        ↑
                             Correction fed back as few-shot
```

**No RAG needed** — complaint routing is a classification problem, not a retrieval problem. Human corrections are injected directly as few-shot examples in the LLM prompt, which is faster and more reliable than embedding retrieval for this use case.

---

## Tech stack

| Component | Choice | Why |
|-----------|--------|-----|
| Agent orchestration | LangGraph | State machine with conditional edges, native human-in-the-loop |
| LLM inference | vLLM | High-throughput local serving |
| Model | Qwen2.5-7B-Instruct | Excellent instruction following, fits on a single A10G |
| Observability | LangSmith | Trace every node, log corrections, evaluate accuracy |
| UI | Gradio | Fast, professional, file upload built-in |
| Feedback store | SQLite | Simple, zero infrastructure, replaceable with Postgres |

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo>
cd complaint_router
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start vLLM (Qwen)

```bash
# GPU (recommended — A10G/A100/RTX 4090)
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096

# CPU fallback (slower, for demo/laptop)
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --device cpu \
    --dtype float32
```

### 3. Configure LangSmith (optional but recommended)

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__your_key_here
export LANGCHAIN_PROJECT=complaint-router
```

### 4. Run the dashboard

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python app.py
# Open http://localhost:7860
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Model identifier |
| `LANGCHAIN_API_KEY` | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | `complaint-router` | LangSmith project name |
| `PORT` | `7860` | Gradio port |

---

## Feedback loop

Every time a human submits a correction via the "Feedback & corrections" tab:
1. The corrected `(text, category, priority)` tuple is saved to SQLite.
2. On the next classification call, the 5 most recent corrections are fetched and injected as few-shot examples into the system prompt.
3. The model immediately improves without any retraining.

For production, replace SQLite with a proper database and consider periodic fine-tuning of the Qwen model on accumulated corrections.

---

## Input formats supported

- **Plain text** — paste directly
- **CSV** — one complaint per row; all columns concatenated
- **JSON** — single object or array of objects
- **Excel (.xlsx)** — first sheet, header row auto-detected
- **Unstructured text** — the LLM normalises it

---

## SLA rules

| Priority | Trigger | Deadline |
|----------|---------|----------|
| P1 | Critical / financial loss / threats | 1 hour |
| P2 | Service disruption / billing error | 4 hours (2h for billing) |
| P3 | Inconvenience / access issue | 8 hours |
| P4 | Feedback / feature request | 24 hours |

Complaints hitting P1 are always routed to the **Escalation** team regardless of category.

---

## Extending for production

- Replace SQLite feedback store with PostgreSQL + pgvector
- Add `.interrupt()` to `human_review` node for async approval flow
- Wire `notify_and_track` to Slack / PagerDuty webhooks
- Add LangSmith evaluator to track classification accuracy over time
- Fine-tune Qwen on accumulated corrections monthly
