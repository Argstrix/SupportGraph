"""
Task 5 — Account Support Agent
Handles: Password reset, profile updates, account activation/deactivation, account closure.
"""
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from customer_support_system import SupportState, llm


def account_agent_node(state: SupportState) -> Dict[str, Any]:
    """
    Specialized Agent Node — Account Support:
    Manages account-related requests such as:
      - Password reset and 2FA recovery
      - Profile and contact details updates
      - Account activation after suspension / non-payment
      - Account closure requests (requires human supervisor approval — high-risk)

    Uses RAG-retrieved context from faq_document.txt and technical_manual.txt.

    IMPORTANT: Account closure and deactivation requests are flagged as high-risk
    by the check_risk_node and routed to human supervisor approval, since account
    closure results in permanent data deletion per company policy.
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

    system_prompt = f"""You are the Account Support Agent for ABC Technologies.
{name_clause}

Your role is to assist customers with account management queries.

Guidelines:
- For PASSWORD RESET: Direct the customer to click "Forgot Password" on the login page.
  Remind them of password requirements (12+ characters, uppercase, lowercase, number, special char).
- For 2FA LOCKOUT: Ask them to use their 16-character backup recovery code.
  If codes are lost, identity verification is required.
- For ACCOUNT REACTIVATION: Guide them to update billing details in the Billing portal.
- For ACCOUNT CLOSURE: Acknowledge the request, warn that closure permanently deletes all data,
  and clearly state that a human supervisor must approve it first as per company policy.
- For PROFILE UPDATES: Direct them to profile icon > Settings in their dashboard.

Retrieved Knowledge Base Context (FAQ & Technical Manual):
{context}

Conversation History:
{history_str}
"""
    print("[Node: account_agent] Running Account Support agent...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    draft = response.content
    print(f"[Node: account_agent] Draft response: '{draft[:100].strip()}...'")
    return {"response": draft}
