# SupportGraph

**AI-Powered Multi-Agent Customer Support Automation System**
Built for **ABC Technologies Pvt. Ltd.** using LangGraph + Ollama

---

## Overview

An intelligent, multi-agent customer support automation system built using **LangGraph** and powered by locally-running **Ollama** models (`qwen2.5:3b` for language reasoning and `nomic-embed-text` for semantic embeddings). The system routes customer queries to specialised support agents, retrieves relevant context from an India-specific knowledge base using RAG, maintains persistent conversation memory via SQLite, and enforces a human-in-the-loop approval process for high-risk requests.

---

## Task Coverage

| Task | Description | Implementation |
|------|-------------|----------------|
| Task 1 | LangGraph workflow design | `build_support_graph()` in `customer_support_system.py` |
| Task 2 | State structure | `SupportState` TypedDict |
| Task 3 | Intent classification | `classify_intent_node()` |
| Task 4 | Conditional routing | `route_to_agent()` in `agent/router.py` |
| Task 5 | Specialized agents | `agent/sales.py`, `technical.py`, `billing.py`, `account.py` |
| Task 6 | RAG pipeline | `setup_rag_pipeline()` + `retrieve_context_node()` |
| Task 7 | SQLite memory | LangGraph `SqliteSaver` checkpointer (`memory.db`) |
| Task 8 | Human-in-the-loop | `human_approval_node()` + `interrupt_before` |
| Task 9 | Supervisor validation | `supervisor_validation_node()` |
| Task 10 | Demonstration | Interactive CLI — run the script and type queries |

---

## Project Structure

```
assignment2/
├── customer_support_system.py   # Main entry point & LangGraph workflow
├── agent/
│   ├── __init__.py              # Exports all agent nodes
│   ├── router.py                # Task 4 — Conditional routing logic
│   ├── sales.py                 # Task 5 — Sales Support Agent
│   ├── technical.py             # Task 5 — Technical Support Agent
│   ├── billing.py               # Task 5 — Billing Support Agent
│   └── account.py               # Task 5 — Account Support Agent
├── knowledge_base/              # Task 6 — RAG documents (India-specific)
│   ├── company_policy.txt       # Refund, cancellation, closure policies (Indian law)
│   ├── pricing_guide.txt        # Plans in INR with UPI/GST details
│   ├── technical_manual.txt     # Troubleshooting for uploads, UPI, GST invoices
│   └── faq_document.txt         # Common questions with India-context answers
├── memory.db                    # Task 7 — SQLite memory (auto-generated on run)
├── workflow_diagram.png         # Task 1 — LangGraph architecture diagram
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies
Install all required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Install & Configure Ollama

Ensure Ollama is installed and running, then pull the required models:

```bash
ollama pull qwen2.5:3b        # Main reasoning LLM
ollama pull nomic-embed-text  # Embedding model for RAG
```

### 3. Run the CLI Application
To start the interactive CLI application:
```bash
python customer_support_system.py
```

You will be prompted to enter a **Customer ID** and your name. Use the **same Customer ID** across sessions to test conversation memory recall (Task 7 / Query 5).

---

## Demonstration Queries (Task 10)

Run these queries in the interactive CLI using **the same Customer ID** (e.g., `cust_david`):

| # | Query | Expected Path |
|---|-------|---------------|
| 1 | `What are the pricing plans available for your software?` | Sales → RAG retrieval → Supervisor |
| 2 | `I forgot my account password.` | Account → RAG retrieval → Supervisor |
| 3 | `My application crashes whenever I upload a file.` | Technical Support → RAG retrieval → Supervisor |
| 4 | `I need a refund for my annual subscription.` | Billing → **Human Approval Required** → Supervisor |
| 5 | `What was my previous support issue?` | Memory Recall (no routing) → Supervisor |

> **Note for Query 4**: The system will pause and ask you (the human supervisor) to approve or reject the refund request before generating the final response.

> **Note for Query 5**: Must be asked using the **same Customer ID** (e.g. `cust_david`) as the earlier queries so that the conversation history can be loaded from the persistent `memory.db` (even across CLI sessions).

---

## SQLite Memory Schema

`memory.db` is automatically created by LangGraph's `SqliteSaver` checkpointer. Key tables:

| Table | Purpose |
|-------|---------|
| `checkpoints` | Full state snapshots per `(thread_id, checkpoint_id)` |
| `checkpoint_blobs` | Serialised channel values (messages, state fields) |
| `checkpoint_writes` | Pending writes for interrupted (HITL paused) graphs |

The `thread_id` = `customer_id`, so each customer's full conversation history is isolated and persists across multiple sessions.

---

## Human-in-the-Loop Triggers

The following request types trigger a supervisor approval pause (per company policy):

- Refund requests
- Subscription cancellation
- Account closure / deletion / deactivation
- Compensation or service credit requests
- Escalation to management / supervisor
- Threats of legal action / consumer court

---

## Knowledge Base (India-Specific)

All four documents have been customised for the Indian market:

- **Pricing in INR** (₹2,499 / ₹6,999 / ₹17,999 per month)
- **Payment methods**: UPI, Razorpay, Paytm, Net Banking, RuPay, NEFT/RTGS, EMI
- **Tax compliance**: GST-compliant invoices, TDS deduction support (Section 194J)
- **Data residency**: AWS Mumbai region (ap-south-1) — India data localisation
- **Legal jurisdiction**: Indian law, Consumer Protection Act 2019, IT Act
- **Support hours**: 9 AM–8 PM IST (Mon–Sat); 24/7 for Enterprise
