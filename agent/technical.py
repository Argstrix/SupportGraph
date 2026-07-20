"""
Task 5 — Technical Support Agent
Handles: Application errors, crashes, installation issues, login problems, configuration issues.
"""
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from customer_support_system import SupportState, llm


def technical_agent_node(state: SupportState) -> Dict[str, Any]:
    """
    Specialized Agent Node — Technical Support:
    Troubleshoots and resolves technical issues such as:
      - Application crashes (e.g., during file uploads)
      - Login failures and 2FA lockouts
      - Browser compatibility problems
      - Installation and configuration issues

    Uses RAG-retrieved context from technical_manual.txt and faq_document.txt
    to provide step-by-step troubleshooting guidance.
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

    system_prompt = f"""You are the Technical Support Agent for ABC Technologies.
{name_clause}

Your role is to diagnose and resolve technical issues the customer is facing with the software.

Guidelines:
- Provide numbered, step-by-step troubleshooting instructions — easy to follow.
- Reference specific limits or requirements from the technical manual where applicable.
- If the issue persists after all steps, ask for browser console logs (F12 → Console tab).
- Be empathetic — acknowledge that technical issues are frustrating.
- Keep steps clear and avoid heavy technical jargon.

Retrieved Knowledge Base Context (Technical Manual & FAQ):
{context}

Conversation History:
{history_str}
"""
    print("[Node: technical_agent] Running Technical Support agent...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    draft = response.content
    print(f"[Node: technical_agent] Draft response: '{draft[:100].strip()}...'")
    return {"response": draft}
