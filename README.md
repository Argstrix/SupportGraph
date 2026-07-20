# SupportGraph

**AI-Powered Multi-Agent Customer Support Automation System**
Built with LangGraph + Ollama

---

## Overview

SupportGraph is an intelligent, multi-agent customer support automation system built using **LangGraph** and powered by locally-running **Ollama** models (`qwen2.5:3b` for language reasoning and `nomic-embed-text` for semantic embeddings). It classifies incoming customer queries, routes them to specialised support agents, retrieves relevant context from a knowledge base using RAG, maintains persistent conversation memory via SQLite, and enforces a human-in-the-loop approval process for high-risk requests — all running fully locally with no external API dependency.

---

## Features

- **Intent classification** — every incoming query is classified into a support category before routing.
- **Conditional multi-agent routing** — queries are dispatched to dedicated Sales, Technical Support, Billing, or Account agents based on classified intent.
- **RAG-backed responses** — a Chroma vector store over the knowledge base grounds agent responses in real policy, pricing, and troubleshooting content instead of hallucinated answers.
- **Persistent conversation memory** — a SQLite-backed LangGraph checkpointer keeps full conversation history per customer, recallable across sessions.
- **Human-in-the-loop approval** — high-risk actions (refunds, cancellations, account closures, escalations) pause the graph and wait for explicit supervisor approval before proceeding.
- **Supervisor validation** — a final validation node reviews agent responses before they're returned to the customer.

---

## Prerequisites

- Python 3.10+ (developed on 3.13)
- [Ollama](https://ollama.com/) installed and running locally
- Git (optional, for cloning)
- No external vector DB service needed — Chroma runs embedded, persisted to `chroma_db/`

---

## Architecture

```
Customer Query
      │
      ▼
Intent Classification ──► Memory Recall (if applicable)
      │
      ▼
Conditional Router
      │
      ├──► Sales Agent
      ├──► Technical Support Agent
      ├──► Billing Agent
      └──► Account Agent
             │
             ▼
      RAG Context Retrieval
             │
             ▼
   Human Approval (high-risk only)
             │
             ▼
    Supervisor Validation
             │
             ▼
        Final Response
```

See `workflow_diagram.png` for the full LangGraph-generated architecture diagram.

---

## Project Structure

```
SupportGraph/
├── customer_support_system.py   # Main entry point & LangGraph workflow
├── agent/
│   ├── __init__.py              # Exports all agent nodes
│   ├── router.py                # Conditional routing logic
│   ├── sales.py                 # Sales support agent
│   ├── technical.py             # Technical support agent
│   ├── billing.py               # Billing support agent
│   └── account.py               # Account support agent
├── knowledge_base/              # RAG source documents (India-specific)
│   ├── company_policy.txt       # Refund, cancellation, closure policies (Indian law)
│   ├── pricing_guide.txt        # Plans in INR with UPI/GST details
│   ├── technical_manual.txt     # Troubleshooting for uploads, UPI, GST invoices
│   └── faq_document.txt         # Common questions with India-context answers
├── memory.db                    # SQLite conversation memory (auto-generated on run)
├── workflow_diagram.png         # LangGraph architecture diagram
└── README.md
```

---

## Getting Started

### 1. Create a virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install & configure Ollama

Ensure [Ollama](https://ollama.com) is installed and running, then pull the required models:

```bash
ollama pull qwen2.5:3b        # Main reasoning LLM
ollama pull nomic-embed-text  # Embedding model for RAG
```

Models are set in `customer_support_system.py` (`llm = ChatOllama(model="qwen2.5:3b", ...)` and the `OllamaEmbeddings` init) — swap the model name there if you want to use a different local model (e.g. `llama3:8b`, `mistral:7b`).

### 3. Run the CLI application
```bash
python customer_support_system.py
```

You'll be prompted for a **Customer ID** and your name. Reuse the **same Customer ID** across sessions to test conversation memory recall. Type `exit`, `quit`, or `q` to stop the console.

---

## Example Queries

Try these in the interactive CLI (using the same Customer ID, e.g. `cust_david`, across all of them):

| # | Query | Expected Path |
|---|-------|---------------|
| 1 | `What are the pricing plans available for your software?` | Sales → RAG retrieval → Supervisor |
| 2 | `I forgot my account password.` | Account → RAG retrieval → Supervisor |
| 3 | `My application crashes whenever I upload a file.` | Technical Support → RAG retrieval → Supervisor |
| 4 | `I need a refund for my annual subscription.` | Billing → **Human Approval Required** → Supervisor |
| 5 | `What was my previous support issue?` | Memory Recall (no routing) → Supervisor |

> **Query 4** pauses the graph and asks you (the human supervisor) to approve or reject the refund before a final response is generated.

> **Query 5** must use the **same Customer ID** as earlier queries so history can be loaded from `memory.db` — this works even across separate CLI sessions.

---

## Human Supervisor Approval Flow

When a high-risk request (refund, cancellation, closure, escalation, etc.) is detected, the graph interrupts before `human_approval_node` and prints the draft response for review:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      HUMAN SUPERVISOR APPROVAL REQUIRED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Customer Name : David
  Category      : Billing
  Query         : I need a refund for my annual subscription.

  Draft Response:
  ...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Approve this request? (y/n):
```

Answering `y` resumes the graph with `approved: "approved"`; answering `n` prompts for a rejection reason and resumes with `approved: "rejected"`. Either way, the decision flows into `supervisor_validation_node` before a final response is returned to the customer.

---

## Conversation Memory

`memory.db` is automatically created by LangGraph's `SqliteSaver` checkpointer. Key tables:

| Table | Purpose |
|-------|---------|
| `checkpoints` | Full state snapshots per `(thread_id, checkpoint_id)` |
| `checkpoint_blobs` | Serialised channel values (messages, state fields) |
| `checkpoint_writes` | Pending writes for interrupted (human-in-the-loop paused) graphs |

`thread_id` maps to `customer_id`, so each customer's full conversation history is isolated and persists across multiple sessions.

---

## Human-in-the-Loop Triggers

The following request types pause the graph for supervisor approval:

- Refund requests
- Subscription cancellation
- Account closure / deletion / deactivation
- Compensation or service credit requests
- Escalation to management / supervisor
- Threats of legal action / consumer court

---

## Knowledge Base (India-Specific)

All four knowledge base documents are customised for the Indian market:

- **Pricing in INR** (₹2,499 / ₹6,999 / ₹17,999 per month)
- **Payment methods**: UPI, Razorpay, Paytm, Net Banking, RuPay, NEFT/RTGS, EMI
- **Tax compliance**: GST-compliant invoices, TDS deduction support (Section 194J)
- **Data residency**: AWS Mumbai region (ap-south-1) — India data localisation
- **Legal jurisdiction**: Indian law, Consumer Protection Act 2019, IT Act
- **Support hours**: 9 AM–8 PM IST (Mon–Sat); 24/7 for Enterprise

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling and Chroma integration
- [Ollama](https://ollama.com) — local LLM & embedding inference (`qwen2.5:3b`, `nomic-embed-text`)
- [Chroma](https://www.trychroma.com/) — embedded vector store for RAG retrieval, no external service required
- SQLite — persistent conversation memory

---

## Key Implementation Details

| Area | Where | Notes |
|------|-------|-------|
| RAG retrieval | `setup_rag_pipeline()`, `retrieve_context_node()` | Knowledge base is chunked, embedded, and stored in Chroma (`chroma_db/`); top-3 chunks are retrieved per query. |
| Intent classification | `classify_intent_node()` | Classifies each query into Sales / Technical Support / Billing / Account / Memory Recall before routing. |
| Conditional routing | `agent/router.py` | Maps classified category to the corresponding specialized agent node. |
| Specialized agents | `agent/sales.py`, `technical.py`, `billing.py`, `account.py` | Each agent drafts a response scoped to its domain, using retrieved RAG context. |
| Conversation memory | LangGraph `SqliteSaver` (`memory.db`) | Persists full state/message history per `customer_id`, recallable across sessions. |
| Human-in-the-loop approval | `human_approval_node`, `interrupt_before` | Pauses graph execution for high-risk categories until a supervisor approves or rejects. |
| Supervisor validation | `supervisor_validation_node` | Final pass over the drafted response before it's returned to the customer. |
| Graph visualization | `workflow_diagram.png` | Generated from the compiled LangGraph workflow for a visual overview of the architecture. |

---

## Extending the Project

- **Add a new agent** — create a module under `agent/`, add its node to the graph in `customer_support_system.py`, and extend `route_to_agent()` in `agent/router.py` to dispatch to it.
- **Expand the knowledge base** — drop new `.txt` files into `knowledge_base/`; they'll be picked up and re-embedded by `setup_rag_pipeline()`.
- **Tune prompts** — response tone and business rules for each agent live in `agent/*.py`.
- **Swap the vector store** — replace the Chroma init in `setup_rag_pipeline()` with another LangChain-compatible vector store if you need something other than an embedded store.
- **Swap the LLM/embedding model** — change the model names passed to `ChatOllama` / `OllamaEmbeddings` in `customer_support_system.py`, or point at any other LangChain-compatible provider.

---

## Troubleshooting

- **`ollama: connection refused`** — make sure the Ollama daemon is running (`ollama serve`) before starting the app.
- **Slow first response** — the first query after startup triggers model loading and knowledge base embedding; subsequent queries are faster.
- **Stale/incorrect memory recall** — delete `memory.db` (and `chroma_db/` if you've changed the knowledge base) to reset local state; both are regenerated automatically on the next run.
- **Empty or malformed responses** — verify the pulled Ollama model matches what's configured in `customer_support_system.py`; smaller models can occasionally return empty completions under load.
