"""
Task 5 — Sales Support Agent
Handles: Product information, subscription plans, pricing, features, upgrades.
"""
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from customer_support_system import SupportState, llm


def sales_agent_node(state: SupportState) -> Dict[str, Any]:
    """
    Specialized Agent Node — Sales Support:
    Answers queries related to:
      - Subscription plans and pricing (Starter, Growth, Enterprise)
      - Product features and capabilities
      - Free trial information
      - Upgrade / downgrade guidance

    Uses RAG-retrieved context from pricing_guide.txt and company_policy.txt
    to provide accurate, up-to-date information.
    """
    query         = state["query"]
    context       = state.get("context", "")
    messages      = state.get("messages", [])
    customer_name = state.get("customer_name", "")

    # Build conversation history for multi-turn context
    history_str = ""
    for msg in messages[:-1]:
        role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
        history_str += f"{role}: {msg.content}\n"

    name_clause = (
        f"The customer's name is {customer_name}. Address them by name."
        if customer_name
        else "Customer name is not known yet."
    )

    system_prompt = f"""You are the Sales Support Agent for ABC Technologies.
{name_clause}

Your role is to provide clear, accurate, and enthusiastic information about ABC Technologies'
subscription plans, features, and pricing.

Guidelines:
- Always reference the specific plan names: Starter (₹2,499/mo), Growth (₹6,999/mo), Enterprise (₹17,999/mo).
- Mention the 14-day free trial where relevant.
- Be persuasive but honest — never overpromise.
- Keep the response concise (under 200 words) unless more detail is clearly needed.

Retrieved Knowledge Base Context:
{context}

Conversation History:
{history_str}
"""
    print("[Node: sales_agent] Running Sales Support agent...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    draft = response.content
    print(f"[Node: sales_agent] Draft response: '{draft[:100].strip()}...'")
    return {"response": draft}
