from typing import Literal, Dict, Any


def route_to_agent(
    state: Dict[str, Any],
) -> Literal["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
    """
    Task 4 — Conditional Routing:
    Determines which specialized support agent node to route the query to,
    based on the category classified by the Intent Classification node.

    Routing Map:
      "Sales"             → sales_agent
      "Technical Support" → technical_agent
      "Billing"           → billing_agent
      "Account"           → account_agent  (default for anything else)

    Note: "Memory Recall" queries are intercepted earlier by route_after_classify
    in customer_support_system.py and never reach this router.
    """
    category = state.get("category", "")

    if category == "Sales":
        return "sales_agent"
    elif category == "Technical Support":
        return "technical_agent"
    elif category == "Billing":
        return "billing_agent"
    else:
        # Covers "Account" and any unexpected categories
        return "account_agent"
