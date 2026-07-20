"""
=============================================================================
SupportGraph — AI-Powered Customer Support Automation System
ABC Technologies Pvt. Ltd.
=============================================================================
Built with LangGraph + Ollama (qwen2.5:3b + nomic-embed-text)

Tasks covered:
  Task 1  — LangGraph workflow design         (see build_support_graph)
  Task 2  — SupportState TypedDict definition
  Task 3  — Intent Classification node        (classify_intent_node)
  Task 4  — Conditional routing               (route_to_agent in agent/router.py)
  Task 5  — Specialized support agents        (agent/*.py)
  Task 6  — RAG pipeline                      (setup_rag_pipeline + retrieve_context_node)
  Task 7  — SQLite-based memory               (LangGraph SqliteSaver checkpointer)
  Task 8  — Human-in-the-loop approval        (human_approval_node + interrupt_before)
  Task 9  — Supervisor validation node        (supervisor_validation_node)
  Task 10 — Demo via interactive CLI          (run this script and use the chat session)
=============================================================================
"""

import sqlite3
import os
import json
import re
from typing import Annotated, List, Dict, Any

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama


# ==========================================
# Task 2: State Structure Definition
# ==========================================
class SupportState(TypedDict):
    """
    Central state object shared across all LangGraph nodes.

    Fields:
      customer_id       — Unique customer identifier; used as thread_id for SQLite memory.
      customer_name     — Customer's name, extracted from the query if they introduce themselves.
      query             — The raw query text submitted by the customer.
      category          — Classified intent: 'Sales' | 'Technical Support' | 'Billing' |
                          'Account' | 'Memory Recall'
      context           — Top-3 RAG-retrieved knowledge base chunks relevant to the query.
      response          — Drafted or final response text.
      requires_approval — True when the request is high-risk and needs a human supervisor.
      approved          — Approval status: 'pending' | 'approved' | 'rejected' | 'none'
      supervisor_notes  — Notes entered by the human supervisor during the approval step.
      messages          — Full conversation history (managed by LangGraph's add_messages reducer).
    """
    customer_id:      str
    customer_name:    str
    query:            str
    category:         str
    context:          str
    response:         str
    requires_approval: bool
    approved:         str   # 'pending' | 'approved' | 'rejected' | 'none'
    supervisor_notes: str
    messages:         Annotated[List[BaseMessage], add_messages]


# ==========================================
# Task 6: LLM and RAG Setup
# ==========================================
print("[Setup] Initializing ChatOllama LLM (qwen2.5:3b)...")
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# Global vector store — populated by setup_rag_pipeline()
vector_store = None

# Directory where ChromaDB persists the vector index to disk
CHROMA_PERSIST_DIR = "./chroma_db"


def _get_kb_fingerprint(kb_dir: str) -> str:
    """
    Returns a fingerprint string of all .txt file modification times in kb_dir.
    Used to detect whether the knowledge base has changed since the last index.
    """
    parts = []
    for filename in sorted(os.listdir(kb_dir)):
        if filename.endswith(".txt"):
            path  = os.path.join(kb_dir, filename)
            mtime = os.path.getmtime(path)
            parts.append(f"{filename}:{mtime}")
    return "|".join(parts)


def setup_rag_pipeline(kb_dir: str = "./knowledge_base"):
    """
    Task 6 — RAG Pipeline Setup (ChromaDB persistent vector store):

    Uses ChromaDB to store document embeddings on disk at ./chroma_db/.
    This means embeddings are generated ONCE (using nomic-embed-text via Ollama)
    and then reloaded instantly on every subsequent run — no re-embedding needed.

    Smart re-index logic:
      - On first run  : reads .txt files → chunks → embeds → saves to chroma_db/
      - On later runs : detects chroma_db/ exists + KB unchanged → loads from disk
      - If KB changes : deletes old index → re-embeds → saves fresh index

    Knowledge Base Documents (India-specific):
      - company_policy.txt  : Refund, cancellation, closure, compensation policies.
      - pricing_guide.txt   : Subscription plans in INR with Indian payment methods.
      - technical_manual.txt: Troubleshooting for uploads, login, UPI, GST invoices.
      - faq_document.txt    : Common questions with India-context answers.
    """
    global vector_store
    print("[RAG] Initializing RAG pipeline (ChromaDB persistent store)...")

    embeddings    = OllamaEmbeddings(model="nomic-embed-text")
    fingerprint   = _get_kb_fingerprint(kb_dir) if os.path.exists(kb_dir) else ""
    fp_file       = os.path.join(CHROMA_PERSIST_DIR, ".kb_fingerprint")
    index_exists  = os.path.isdir(CHROMA_PERSIST_DIR)
    stored_fp     = ""

    if index_exists and os.path.exists(fp_file):
        with open(fp_file, "r") as f:
            stored_fp = f.read().strip()

    # ── Case 1: Existing index is up-to-date → load from disk ────────────
    if index_exists and stored_fp == fingerprint and fingerprint:
        print(f"[RAG] Loading existing ChromaDB index from '{CHROMA_PERSIST_DIR}/' (KB unchanged).")
        vector_store = Chroma(
            collection_name="abc_tech_kb",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vector_store._collection.count()
        print(f"[RAG] Loaded {count} chunks from persistent vector store. Ready.\n")
        return vector_store

    # ── Case 2: No index or KB changed → build fresh index ───────────────
    if index_exists and stored_fp != fingerprint:
        print("[RAG] Knowledge base has changed. Rebuilding vector index...")
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
    else:
        print("[RAG] No existing index found. Building for the first time...")

    if not os.path.exists(kb_dir):
        print(f"[RAG] Warning: Knowledge base directory '{kb_dir}' not found.")
        vector_store = Chroma(
            collection_name="abc_tech_kb",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return vector_store

    # Load and chunk documents
    documents = []
    for filename in sorted(os.listdir(kb_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({"source": filename, "text": content})
                print(f"[RAG]   Loaded: {filename}")
            except Exception as e:
                print(f"[RAG]   Error reading {filename}: {e}")

    if not documents:
        print("[RAG] No documents found to index.")
        vector_store = Chroma(
            collection_name="abc_tech_kb",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return vector_store

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc["text"]):
            chunks.append(f"Source: {doc['source']}\n{chunk}")

    print(f"[RAG] Embedding {len(chunks)} chunks with nomic-embed-text (this runs once)...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="abc_tech_kb",
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Save fingerprint so next run can skip re-embedding
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    with open(fp_file, "w") as f:
        f.write(fingerprint)

    print(f"[RAG] {len(chunks)} chunks embedded and saved to '{CHROMA_PERSIST_DIR}/'.")
    print(f"[RAG] Future runs will load from disk instantly.\n")
    return vector_store


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly parses the LLM's text output as JSON.
    Falls back to keyword-based category detection when JSON is malformed.
    """
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Keyword fallback
    lower = text.lower()
    if "sales" in lower:
        category = "Sales"
    elif "technical" in lower or "tech" in lower:
        category = "Technical Support"
    elif "billing" in lower or "refund" in lower:
        category = "Billing"
    elif "account" in lower:
        category = "Account"
    else:
        category = "Sales"

    return {"category": category, "reason": text}


def is_memory_recall_query(query: str) -> bool:
    """
    Detects whether the customer is asking about their previous interactions.
    These queries skip department routing and are handled by memory_recall_node.

    Example triggers: "What was my previous issue?", "What did I ask last time?"
    """
    patterns = [
        r"previous (issue|query|question|request|problem|ticket|support)",
        r"(what|what's) (was|were|is) my (last|previous|earlier)",
        r"(recall|remember|remind me|what did i)",
        r"(earlier|before|last time) (i|we) (said|asked|mentioned|reported)",
        r"(my|our) (history|conversation|past (issue|query|problem))",
    ]
    lower = query.lower()
    return any(re.search(p, lower) for p in patterns)


# ==========================================
# Task 3: Intent Classification Node
# ==========================================
def classify_intent_node(state: SupportState) -> Dict[str, Any]:
    """
    Node 1 — Intent Classification (Task 3):
    Uses the LLM to classify the customer's query into one of:
      Sales | Technical Support | Billing | Account

    Also extracts the customer's name if they introduce themselves.
    Conversation history is included so the classifier stays aware of context.
    """
    query    = state["query"]
    messages = state.get("messages", [])

    # --- Extract customer name ---
    name = state.get("customer_name", "")
    if not name:
        name_match = re.search(
            r"(?:my name is|i am|this is|i'm)\s+([A-Za-z]+)", query.lower()
        )
        if name_match:
            name = name_match.group(1).capitalize()
            print(f"[Node: classify_intent] Extracted customer name: {name}")

    # --- Build conversation history string ---
    history_str = ""
    for msg in messages[:-1]:  # Exclude the current query message
        role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
        history_str += f"{role}: {msg.content}\n"

    prompt = f"""You are an intent classifier for a customer support system of ABC Technologies Pvt. Ltd., an Indian SaaS company.

Classify the customer query into EXACTLY one category:
1. Sales           — Pricing plans (INR), features, subscriptions, free trial, upgrades.
2. Technical Support — Application errors, crashes, login issues, file uploads, browser problems.
3. Billing         — GST invoices, UPI/net banking payments, refund requests, payment failures.
4. Account         — Password reset, profile updates, account activation, account closure.

Output ONLY a raw JSON object (no markdown, no extra text):
{{"category": "Sales" | "Technical Support" | "Billing" | "Account", "reason": "<brief reason>"}}

Current Query: "{query}"
Conversation History:
{history_str}
"""
    print(f"\n[Node: classify_intent] Classifying: '{query}'")
    parsed   = extract_json(llm.invoke([SystemMessage(content=prompt)]).content)
    category = parsed.get("category", "Sales")
    print(f"[Node: classify_intent]  Category: {category} | Reason: {parsed.get('reason', 'N/A')}")

    return {
        "category":         category,
        "customer_name":    name,
        "context":          "",     # Initialise to avoid KeyError in later nodes
        "response":         "",
        "requires_approval": False,
        "approved":         "none",
        "supervisor_notes": "",
    }


# ==========================================
# Task 7: Memory Recall Node
# ==========================================
def memory_recall_node(state: SupportState) -> Dict[str, Any]:
    """
    Node — Memory Recall (Task 7):
    Handles queries where the customer asks about their previous interactions.
    Reads from the full message history persisted in SQLite by LangGraph's
    SqliteSaver checkpointer and generates a clear, structured recall response.

    Routes directly to END — does NOT go through supervisor_validation,
    because recall is based on stored facts, not company policy/context.

    Demonstration query:
      "What was my previous support issue?"
    """
    query         = state["query"]
    messages      = state.get("messages", [])
    customer_name = state.get("customer_name", "")

    print("[Node: memory_recall] Customer is asking about previous interactions.")

    # Extract ONLY the customer's past queries (exclude current one and agent responses)
    # We use HumanMessages only to get a clean numbered list of what the customer asked
    past_queries = []
    for msg in messages[:-1]:          # exclude the current query (last message)
        if isinstance(msg, HumanMessage):
            past_queries.append(msg.content)

    name_str = customer_name if customer_name else "there"

    if not past_queries:
        recall_response = (
            f"Hello {name_str}! This appears to be your first interaction with us — "
            "we don't have any previous support history on file for your account. "
            "How can I assist you today?"
        )
        print("[Node: memory_recall] No prior history found in SQLite memory.")
    else:
        # Build a numbered list of past queries
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(past_queries))
        prompt = f"""You are a customer support agent for ABC Technologies Pvt. Ltd.

The customer's name is {name_str}.
They are asking about their previous support interactions.

Here are the exact queries the customer raised in their previous session(s),
retrieved from the SQLite memory store:
{numbered}

Current query: "{query}"

Write a warm, brief reply that:
1. Greets the customer by name.
2. Lists the topics they previously raised (from the numbered list above — do NOT invent new ones).
3. Offers to help further.

Do NOT say you have no records. Do NOT add any topics not in the list above.
Do NOT include any agent response content — only what the customer asked.
Keep it concise (under 100 words).
"""
        recall_response = llm.invoke([SystemMessage(content=prompt)]).content
        print(f"[Node: memory_recall] Recall response generated from {len(past_queries)} prior customer queries.")

    return {
        "response":          recall_response,
        "category":          "Memory Recall",
        "context":           "",
        "requires_approval": False,
        "approved":          "none",
        "messages":          [AIMessage(content=recall_response)],
    }


# ==========================================
# Task 6: RAG Context Retrieval Node
# ==========================================
def retrieve_context_node(state: SupportState) -> Dict[str, Any]:
    """
    Node 2 — RAG Retrieval (Task 6):
    Performs a semantic similarity search against the indexed knowledge base
    and returns the top-3 most relevant chunks to provide grounded context
    to the specialized support agent.
    """
    query = state["query"]
    print(f"[Node: retrieve_context] Searching knowledge base for: '{query}'")
    docs       = vector_store.similarity_search(query, k=3)
    context_str = "\n---\n".join([doc.page_content for doc in docs])
    print(f"[Node: retrieve_context] Retrieved {len(docs)} relevant chunks.")
    return {"context": context_str}


# ==========================================
# Task 8: Risk Check Node
# ==========================================
def check_risk_node(state: SupportState) -> Dict[str, Any]:
    """
    Node 4 — Risk Assessment (Task 8):
    Scans the customer's query and the agent's draft response for high-risk
    keywords that require human supervisor approval per company policy.

    High-risk triggers:
      - Refund requests
      - Subscription cancellation
      - Account closure / deletion / deactivation
      - Compensation or service credit requests
      - Escalation to management / supervisor
      - Threats of legal action / consumer court
    """
    query    = state["query"].lower()
    response = state.get("response", "").lower()
    category = state.get("category", "")

    high_risk_keywords = [
        "refund", "cancel subscription", "subscription cancellation",
        "close account", "account closure", "delete account", "deactivate account",
        "compensation", "service credit", "escalate", "speak to manager",
        "supervisor", "legal action", "consumer court", "lawsuit",
    ]

    requires_approval = False
    trigger_reason    = ""

    # Rule 1: High-risk keywords in the query
    for kw in high_risk_keywords:
        if kw in query:
            requires_approval = True
            trigger_reason    = f"Query contains high-risk keyword: '{kw}'"
            break

    # Rule 2: Category + keyword combination
    if not requires_approval:
        if category == "Billing" and ("refund" in query or "compensation" in query):
            requires_approval = True
            trigger_reason    = "Billing query involving refund/compensation"
        elif category == "Account" and any(k in query for k in ["close", "delete", "deactivate"]):
            requires_approval = True
            trigger_reason    = "Account query involving closure/deletion"

    # Rule 3: High-risk content in the drafted agent response
    if not requires_approval:
        for kw in ["refund", "cancel", "compensation", "escalate", "close your account"]:
            if kw in response:
                requires_approval = True
                trigger_reason    = f"Draft response mentions high-risk keyword: '{kw}'"
                break

    if requires_approval:
        print(f"[Node: check_risk]  High-risk detected! Reason: {trigger_reason}")
        return {"requires_approval": True,  "approved": "pending"}
    else:
        print("[Node: check_risk]  Low-risk request. No human approval needed.")
        return {"requires_approval": False, "approved": "none"}


def route_after_risk_check(state: SupportState) -> str:
    """
    Conditional router — After Risk Check:
    Routes to 'human_approval_node' if supervisor approval is pending,
    otherwise directly to 'supervisor_validation'.
    """
    if state.get("requires_approval") and state.get("approved") == "pending":
        return "human_approval_node"
    return "supervisor_validation"


def route_after_classify(state: SupportState) -> str:
    """
    Conditional router — After Intent Classification:
    If the query is a memory recall request, route directly to 'memory_recall_node'
    (skipping RAG retrieval and department agents).
    Otherwise, proceed to 'retrieve_context' for the normal RAG + agent flow.
    """
    if is_memory_recall_query(state.get("query", "")):
        return "memory_recall_node"
    return "retrieve_context"


# ==========================================
# Task 8: Human-in-the-Loop Approval Node
# ==========================================
def human_approval_node(state: SupportState) -> Dict[str, Any]:
    """
    Node 5 — Human Approval Interrupt (Task 8):
    This node executes AFTER the graph resumes from the LangGraph interrupt.
    The human supervisor's decision ('approved' / 'rejected') is injected
    into state via app.update_state() in run_query() before resuming.

    If rejected: overwrites the response with a polite rejection message.
    If approved: returns empty dict so the existing draft is used downstream.
    """
    approved = state.get("approved", "pending")
    notes    = state.get("supervisor_notes", "")

    print(f"[Node: human_approval_node] Resuming after supervisor review. Status: {approved}")

    if approved == "rejected":
        rejection_msg = (
            "Thank you for contacting ABC Technologies Pvt. Ltd. "
            "Your request has been reviewed by our support supervisor. "
            "Unfortunately, we are unable to process this request at this time."
        )
        if notes:
            rejection_msg += f"\n\nSupervisor's note: {notes}"
        rejection_msg += (
            "\n\nIf you have further questions or wish to discuss this further, "
            "please contact us again. We're here to help."
        )
        print("[Node: human_approval_node] Request rejected by supervisor.")
        return {"response": rejection_msg}

    print("[Node: human_approval_node] Request approved by supervisor.")
    return {}  # Draft response remains unchanged; supervisor_validation will polish it


# ==========================================
# Task 9: Supervisor Validation Node
# ==========================================
def supervisor_validation_node(state: SupportState) -> Dict[str, Any]:
    """
    Node 6 — Supervisor Validation (Task 9):
    The AI Supervisor agent reviews and polishes the drafted response to ensure
    it is professional, accurate, empathetic, and fully addresses the query
    before it is delivered as the final customer-facing response.

    Also appends the final response to the message history so it is available
    for future memory recall queries (Task 7 integration).
    """
    query          = state["query"]
    draft_response = state.get("response", "")
    context        = state.get("context", "")
    approved       = state.get("approved", "none")

    # Pass rejection responses through without LLM modification
    if approved == "rejected":
        print("[Node: supervisor_validation] Rejected request — passing through rejection message.")
        return {"response": draft_response, "messages": [AIMessage(content=draft_response)]}

    print("[Node: supervisor_validation] Polishing the drafted response...")

    approval_note = (
        "Note: This request has been reviewed and approved by a human supervisor.\n"
        if approved == "approved" else ""
    )

    prompt = f"""You are the AI Support Supervisor for ABC Technologies Pvt. Ltd., an Indian SaaS company.
Your task is to validate and improve the drafted support response before it reaches the customer.

Ensure the final response:
1. Is polite, empathetic, and professionally written.
2. Directly and completely addresses the customer's query.
3. Aligns with the retrieved company policy and knowledge base context.
4. Uses Indian English conventions (e.g., "Please do the needful", currency in INR ₹).
5. Contains no placeholder text, internal jargon, or sensitive internal references.
6. Is well-structured (numbered steps if troubleshooting, bullet points if listing options).
{approval_note}
Retrieved Knowledge Base Context:
{context}

Customer Query: {query}

Drafted Response:
{draft_response}

Output ONLY the final customer-facing response text. No prefixes like "Final Response:" or "Supervisor:".
"""
    final_response = llm.invoke([SystemMessage(content=prompt)]).content
    print("[Node: supervisor_validation] Final response polished and ready.")

    return {
        "response": final_response,
        "messages": [AIMessage(content=final_response)],  # Saved to SQLite memory
    }


# ==========================================
# Task 1: LangGraph Workflow Compilation
# ==========================================
def build_support_graph(checkpointer):
    """
    Task 1 — Workflow Design:
    Builds and compiles the complete LangGraph StateGraph.

    Workflow (abbreviated):
      START → classify_intent
            ↓ (memory recall query?)
            ├── YES → memory_recall_node → supervisor_validation → END
            └── NO  → retrieve_context → [route_to_agent]
                          → sales / technical / billing / account agent
                          → check_risk
                          ↓ (high-risk?)
                          ├── YES → human_approval_node [INTERRUPT] → supervisor_validation → END
                          └── NO  → supervisor_validation → END

    Memory: SqliteSaver (SQLite) persists all state per thread_id = customer_id.
    """
    from agent import (
        sales_agent_node,
        technical_agent_node,
        billing_agent_node,
        account_agent_node,
        route_to_agent,
    )

    workflow = StateGraph(SupportState)

    # --- Register nodes ---
    workflow.add_node("classify_intent",       classify_intent_node)
    workflow.add_node("memory_recall_node",    memory_recall_node)
    workflow.add_node("retrieve_context",      retrieve_context_node)
    workflow.add_node("sales_agent",           sales_agent_node)
    workflow.add_node("technical_agent",       technical_agent_node)
    workflow.add_node("billing_agent",         billing_agent_node)
    workflow.add_node("account_agent",         account_agent_node)
    workflow.add_node("check_risk",            check_risk_node)
    workflow.add_node("human_approval_node",   human_approval_node)
    workflow.add_node("supervisor_validation", supervisor_validation_node)

    # --- Entry ---
    workflow.add_edge(START, "classify_intent")

    # --- After classification: memory recall vs normal RAG path ---
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "memory_recall_node": "memory_recall_node",
            "retrieve_context":   "retrieve_context",
        },
    )

    # Memory recall bypasses department agents
    # Memory recall goes straight to END — no supervisor polish needed
    # (recall is fact-based, not policy-based; supervisor context would confuse the LLM)
    workflow.add_edge("memory_recall_node", END)

    # --- Route to appropriate department agent ---
    workflow.add_conditional_edges(
        "retrieve_context",
        route_to_agent,
        {
            "sales_agent":     "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent":   "billing_agent",
            "account_agent":   "account_agent",
        },
    )

    # All agents feed into risk check
    workflow.add_edge("sales_agent",     "check_risk")
    workflow.add_edge("technical_agent", "check_risk")
    workflow.add_edge("billing_agent",   "check_risk")
    workflow.add_edge("account_agent",   "check_risk")

    # --- After risk check: human approval or straight to supervisor ---
    workflow.add_conditional_edges(
        "check_risk",
        route_after_risk_check,
        {
            "human_approval_node":   "human_approval_node",
            "supervisor_validation": "supervisor_validation",
        },
    )

    workflow.add_edge("human_approval_node",   "supervisor_validation")
    workflow.add_edge("supervisor_validation",  END)

    # Compile with SQLite checkpointer and HITL interrupt
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval_node"],  # Task 8 — HITL pause point
    )
    return app


# ==========================================
# Core Query Runner
# ==========================================
def run_query(app, customer_id: str, query: str, customer_name: str = ""):
    """
    Executes a single customer query through the LangGraph support workflow.

    Steps:
      1. Submit query → graph runs until END or HITL interrupt.
      2. If interrupted (high-risk): prompt human supervisor for approve/reject.
      3. Resume graph with supervisor's decision.
      4. Display the final response.
    """
    print("\n" + "=" * 65)
    print(f"  Customer [{customer_id}] › '{query}'")
    print("=" * 65)

    config = {"configurable": {"thread_id": customer_id}}

    # Check if this customer already has a saved checkpoint (returning customer)
    existing_state = app.get_state(config)
    is_returning   = bool(existing_state.values)

    if is_returning:
        # Returning customer: only update the per-query fields.
        # The 'messages' list in the checkpoint already has full history;
        # we append just the new query message via the add_messages reducer.
        msg_count = len(existing_state.values.get("messages", []))
        print(f"[Memory] Returning customer '{customer_id}' — {msg_count} prior message(s) loaded from SQLite.")
        inputs = {
            "query":            query,
            "category":         "",
            "context":          "",
            "response":         "",
            "requires_approval": False,
            "approved":         "none",
            "supervisor_notes": "",
            "messages":         [HumanMessage(content=query)],  # add_messages appends this
        }
        # Preserve the stored customer_name if not provided fresh
        if customer_name:
            inputs["customer_name"] = customer_name
    else:
        # New customer: send full initial state
        print(f"[Memory] New customer '{customer_id}' — starting fresh session.")
        inputs = {
            "customer_id":      customer_id,
            "customer_name":    customer_name,
            "query":            query,
            "category":         "",
            "context":          "",
            "response":         "",
            "requires_approval": False,
            "approved":         "none",
            "supervisor_notes": "",
            "messages":         [HumanMessage(content=query)],
        }

    print("[Graph] Running...")
    app.invoke(inputs, config)

    # --- Handle HITL interrupt ---
    state = app.get_state(config)
    if state.next and "human_approval_node" in state.next:
        print("\n" + "!" * 65)
        print("      HUMAN SUPERVISOR APPROVAL REQUIRED   ")
        print("!" * 65)
        draft  = state.values.get("response", "")
        cat    = state.values.get("category", "")
        cname  = state.values.get("customer_name", "Unknown")
        print(f"  Customer Name : {cname}")
        print(f"  Category      : {cat}")
        print(f"  Query         : {query}")
        print(f"\n  Draft Response:\n{draft}\n")
        print("!" * 65)

        choice = ""
        while choice not in ["y", "n"]:
            choice = input("Approve this request? (y/n): ").strip().lower()

        if choice == "y":
            print("\n  [Supervisor] Request APPROVED.")
            app.update_state(
                config,
                {"approved": "approved", "supervisor_notes": "Approved by human supervisor."},
                as_node="check_risk",
            )
        else:
            reason = input("  Enter rejection reason (or press Enter for default): ").strip()
            reason = reason or "Request rejected by supervisor."
            print(f"\n  [Supervisor] Request REJECTED. Reason: {reason}")
            app.update_state(
                config,
                {"approved": "rejected", "supervisor_notes": reason},
                as_node="check_risk",
            )

        print("\n[Graph] Resuming after supervisor decision...")
        app.invoke(None, config)

    # --- Display final response ---
    final_state    = app.get_state(config)
    final_response = final_state.values.get("response", "No response generated.")

    print("\n" + "=" * 65)
    print("  FINAL RESPONSE TO CUSTOMER")
    print("=" * 65)
    print(final_response)
    print("=" * 65 + "\n")

    return final_response


# ==========================================
# Database Setup (Task 7 — SQLite Memory)
# ==========================================
def setup_database(db_file: str = "memory.db"):
    """
    Task 7 — SQLite Memory Initialization:
    Opens (or creates) the SQLite database used by LangGraph's SqliteSaver
    checkpointer for persistent, cross-session conversation memory.

    IMPORTANT: The database is NEVER deleted on startup. This allows customers
    to return in a new session and have their full conversation history recalled
    (e.g., 'What was my previous support issue?' — Query 5 of the demo).

    LangGraph automatically creates and manages these tables:
      - checkpoints       : Full state snapshots per (thread_id, checkpoint_id)
      - checkpoint_blobs  : Serialised channel values
      - checkpoint_writes : Pending writes for interrupted (HITL) graphs

    The thread_id = customer_id isolates each customer's history.
    Use '--fresh' flag or delete memory.db manually to reset all history.
    """
    is_new = not os.path.exists(db_file)
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        status = "Created new" if is_new else "Loaded existing"
        print(f"[Setup] {status} SQLite memory database: '{db_file}'")
        return conn, db_file
    except Exception as e:
        fallback = "memory_fallback.db"
        print(f"[Setup] Could not open '{db_file}' ({e}). Using '{fallback}'.")
        conn = sqlite3.connect(fallback, check_same_thread=False)
        return conn, fallback


# ==========================================
# Workflow Diagram
# ==========================================
def generate_diagram(app):
    """Generates a PNG diagram of the LangGraph workflow using Mermaid."""
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("workflow_diagram.png", "wb") as f:
            f.write(png_data)
        print("[Setup]  Workflow diagram saved as 'workflow_diagram.png'")
    except Exception as e:
        print(f"[Setup] Note: Could not generate diagram ({e}). Existing diagram will be used.")


# ==========================================
# Main Entry Point
# ==========================================
def main():
    """
    Task 10 — CLI entry point for demonstrating the system.

    Usage:
      python customer_support_system.py          # Normal run (keeps memory)
      python customer_support_system.py --fresh  # Wipe memory.db and start fresh

    Enter Customer ID (use the same ID across sessions to test memory recall).
    Type queries at the prompt. High-risk requests trigger HITL approval.
    Type 'exit' or 'quit' to end the session.
    """
    import sys

    # Handle --fresh flag: delete memory.db before starting
    db_file = "memory.db"
    if "--fresh" in sys.argv:
        for suffix in ["", "-shm", "-wal"]:
            f = db_file + suffix
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        print("[Setup] --fresh flag: memory.db cleared. Starting with clean memory.")

    # Step 1: RAG pipeline
    setup_rag_pipeline()

    # Step 2: SQLite memory (persistent across runs — never deleted automatically)
    conn, db_file = setup_database(db_file)
    checkpointer  = SqliteSaver(conn)

    # Step 3: Build graph
    print("[Setup] Building LangGraph workflow...")
    app = build_support_graph(checkpointer)
    print("[Setup] LangGraph workflow compiled successfully.")

    # Step 4: Generate diagram
    generate_diagram(app)

    # Step 5: CLI banner
    print("\n" + "=" * 65)
    print("  ABC Technologies Pvt. Ltd.")
    print("  AI-Powered Customer Support Automation System")
    print("=" * 65)
    print(f"  LLM        : Ollama qwen2.5:3b")
    print(f"  Embeddings : nomic-embed-text (ChromaDB)")
    print(f"  Memory     : SQLite ({db_file})  [persistent across sessions]")
    print(f"  Knowledge  : company_policy | pricing_guide | technical_manual | faq")
    print("=" * 65)
    print("  Tip: Use --fresh flag to wipe memory and start a clean session.")
    print("=" * 65)

    # Step 6: Get customer details
    customer_id = input("\nEnter Customer ID (press Enter for 'cust_001'): ").strip()
    if not customer_id:
        customer_id = "cust_001"

    customer_name = input("Enter your name (press Enter to skip): ").strip()

    # Check if this customer has existing history in SQLite
    config = {"configurable": {"thread_id": customer_id}}
    existing = app.get_state(config)
    prior_msgs = len(existing.values.get("messages", [])) if existing.values else 0

    if prior_msgs > 0:
        print(f"\n  Welcome back! Found {prior_msgs} prior message(s) in memory for '{customer_id}'.")
        print("  You can ask 'What was my previous support issue?' to recall history.")
    else:
        print(f"\n  New session started for Customer ID: '{customer_id}'.")

    print("  Type your query below. Type 'exit' or 'quit' to end.\n")
    print("-" * 65)

    # Step 7: Interactive query loop (Task 10 demonstration)
    while True:
        try:
            user_q = input("\nCustomer > ").strip()
            if not user_q:
                continue
            if user_q.lower() in ["exit", "quit"]:
                print("\n  Thank you for contacting ABC Technologies. Goodbye!\n")
                break
            run_query(app, customer_id, user_q, customer_name=customer_name)
        except KeyboardInterrupt:
            print("\n\n  Session interrupted. Goodbye!\n")
            break

    conn.close()


if __name__ == "__main__":
    main()