from langchain_core.tools import tool
from vector_store import retriever


# ── Tool 1: Knowledge Base Search (RAG — retrieval only, no LLM) ──────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search store policies: shipping, returns, payments, availability, contact."""

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


# ── Tool 2: Order Lookup ───────────────────────────────────────────────────────
@tool
def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    # TODO: replace with real DB query
    FAKE_ORDERS = {
        "1001": {"status": "Shipped",    "eta": "2 days",              "carrier": "Amana"},
        "1002": {"status": "Processing", "eta": "Ships within 24 hrs", "carrier": "N/A"},
        "1003": {"status": "Delivered",  "eta": "Delivered Apr 18",    "carrier": "CTM"},
        "1004": {"status": "Cancelled",  "eta": "N/A",                 "carrier": "N/A"},
    }
    order = FAKE_ORDERS.get(str(order_id).strip())
    if not order:
        return (
            f"No order found with ID '{order_id}'. "
            "Please verify the order number and try again."
        )
    return (
        f"Order #{order_id} → Status: {order['status']} | "
        f"ETA: {order['eta']} | Carrier: {order['carrier']}"
    )


# ── Tool 3: Escalate to Human ─────────────────────────────────────────────────
@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human agent when customer requests it or issue is complex."""
    return f"__ESCALATE__:{reason}"


ALL_TOOLS = [search_knowledge_base, lookup_order, escalate_to_human]
