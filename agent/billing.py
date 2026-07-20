"""
Task 5 — Billing Support Agent
Handles: Invoice requests, payment issues, refund requests, billing cycles, payment history.
"""
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from customer_support_system import SupportState, llm


def billing_agent_node(state: SupportState) -> Dict[str, Any]:
    """
    Specialized Agent Node — Billing Support:
    Handles billing and financial queries such as:
      - Refund requests (requires human supervisor approval — high-risk)
      - Invoice retrieval and billing cycle questions
      - Payment failures and method updates
      - Subscription changes and pro-rated billing

    Uses RAG-retrieved context from company_policy.txt and pricing_guide.txt.

    IMPORTANT: Refund requests are automatically flagged as high-risk by the
    check_risk_node and routed to human supervisor approval before the final
    response is sent to the customer.
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

    system_prompt = f"""You are the Billing Support Agent for ABC Technologies.
{name_clause}

Your role is to assist customers with billing-related queries.

Guidelines:
- For REFUND requests: Acknowledge the request warmly, explain the 30-day refund eligibility
  policy, and clearly inform the customer that their refund request will be reviewed and
  approved by a human supervisor before processing. Do NOT promise immediate refund.
- For INVOICES: Guide the customer to Settings > Billing > Invoice History in their dashboard.
- For PAYMENT ISSUES: Suggest updating payment method in Settings > Billing > Payment Methods.
- Always be empathetic when discussing financial matters.
- Reference the company policy accurately.

Retrieved Knowledge Base Context (Company Policy & Pricing Guide):
{context}

Conversation History:
{history_str}
"""
    print("[Node: billing_agent] Running Billing Support agent...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    draft = response.content
    print(f"[Node: billing_agent] Draft response: '{draft[:100].strip()}...'")
    return {"response": draft}
