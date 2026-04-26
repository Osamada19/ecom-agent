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
    """Look up order status, items, carrier, and estimated delivery by order ID."""

    # TODO: replace with real DB query
    FAKE_ORDERS = {

        # ── Confirmed / Processing ──────────────────────────────────────────
        "1001": {
            "customer": "Fatima Zahra B.",
            "status": "Confirmed",
            "items": "Kaftan Nour Classique (M, blush pink) x1",
            "total": "515 MAD",
            "payment": "COD",
            "city": "Casablanca",
            "order_date": "2025-04-23",
            "carrier": "N/A",
            "eta": "Will be dispatched within 24 hrs",
            "notes": "Gift wrapping requested.",
        },
        "1002": {
            "customer": "Younes A.",
            "status": "Processing",
            "items": "Djellaba Homme Classique (L, beige) x1, Ceinture Cuir Artisanale (95cm, brown) x1",
            "total": "650 MAD",
            "payment": "COD",
            "city": "Marrakech",
            "order_date": "2025-04-22",
            "carrier": "N/A",
            "eta": "Ships within 24 hrs",
            "notes": "",
        },

        # ── Packed / Ready to Ship ──────────────────────────────────────────
        "1003": {
            "customer": "Nadia H.",
            "status": "Packed — Awaiting Pickup by Carrier",
            "items": "Abaya Moderne (M, black) x2",
            "total": "735 MAD",
            "payment": "CMI Card",
            "city": "Rabat",
            "order_date": "2025-04-21",
            "carrier": "Amana",
            "eta": "Estimated dispatch: today",
            "notes": "",
        },

        # ── Shipped / In Transit ────────────────────────────────────────────
        "1004": {
            "customer": "Omar K.",
            "status": "Shipped — In Transit",
            "items": "Polo Premium Homme (L, navy) x1, Pantalon Chino Slim (42, khaki) x1",
            "total": "415 MAD",
            "payment": "COD",
            "city": "Fès",
            "order_date": "2025-04-19",
            "carrier": "CTM",
            "eta": "2–3 business days",
            "notes": "Tracking SMS sent to customer's phone.",
        },
        "1005": {
            "customer": "Salma E.",
            "status": "Shipped — In Transit",
            "items": "Robe Casual Lina (S, terracotta) x1, Foulard Soie Marocaine x1",
            "total": "345 MAD",
            "payment": "Visa Card",
            "city": "Tanger",
            "order_date": "2025-04-20",
            "carrier": "Amana",
            "eta": "1–2 business days",
            "notes": "",
        },
        "1006": {
            "customer": "Hamza T.",
            "status": "Shipped — In Transit",
            "items": "Set Loungewear Femme (M, sage green) x1",
            "total": "255 MAD",
            "payment": "COD",
            "city": "Agadir",
            "order_date": "2025-04-18",
            "carrier": "Aramex",
            "eta": "3–4 business days",
            "notes": "Remote city — slight delay possible.",
        },

        # ── Out for Delivery ────────────────────────────────────────────────
        "1007": {
            "customer": "Zineb M.",
            "status": "Out for Delivery — Arriving Today",
            "items": "Caftan Fille Mini Nour (7–8 ans, pink) x1",
            "total": "255 MAD",
            "payment": "COD",
            "city": "Kenitra",
            "order_date": "2025-04-21",
            "carrier": "CTM",
            "eta": "Today — delivery agent will call before arriving",
            "notes": "Please have 255 MAD ready.",
        },

        # ── Delivered ───────────────────────────────────────────────────────
        "1008": {
            "customer": "Rachid L.",
            "status": "Delivered",
            "items": "Djellaba Homme Classique (XL, white) x1",
            "total": "555 MAD",
            "payment": "CMI Card",
            "city": "Meknès",
            "order_date": "2025-04-15",
            "carrier": "Amana",
            "eta": "Delivered on Apr 18",
            "notes": "",
        },
        "1009": {
            "customer": "Houda B.",
            "status": "Delivered",
            "items": "Jilbab Deux Pièces Satin (L, grey) x1, Sac à Main Cuir Artisanal (black) x1",
            "total": "835 MAD",
            "payment": "Visa Card",
            "city": "Casablanca",
            "order_date": "2025-04-14",
            "carrier": "CTM",
            "eta": "Delivered on Apr 17",
            "notes": "",
        },

        # ── Failed Delivery ─────────────────────────────────────────────────
        "1010": {
            "customer": "Karim O.",
            "status": "Failed Delivery — Customer Unreachable",
            "items": "Gandoura Légère Été (M, white) x1",
            "total": "225 MAD",
            "payment": "COD",
            "city": "Oujda",
            "order_date": "2025-04-17",
            "carrier": "Amana",
            "eta": "Redelivery can be requested — 35 MAD reshipping fee applies",
            "notes": "Carrier attempted delivery twice. Please contact support to reschedule.",
        },

        # ── Cancelled ───────────────────────────────────────────────────────
        "1011": {
            "customer": "Imane S.",
            "status": "Cancelled",
            "items": "Robe Casual Lina (XS, black) x1",
            "total": "215 MAD",
            "payment": "COD",
            "city": "Salé",
            "order_date": "2025-04-16",
            "carrier": "N/A",
            "eta": "N/A",
            "notes": "Cancelled by customer within the 1-hour window. No charge.",
        },

        # ── Return Requested ────────────────────────────────────────────────
        "1012": {
            "customer": "Meryem F.",
            "status": "Return Requested — Awaiting Pickup",
            "items": "Kaftan Nour Classique (L, ivory) x1",
            "total": "480 MAD",
            "payment": "COD",
            "city": "Tétouan",
            "order_date": "2025-04-10",
            "carrier": "CTM",
            "eta": "Return pickup scheduled — we will contact you to confirm the date",
            "notes": "Reason: wrong size ordered (customer wanted M).",
        },

        # ── Return in Transit ───────────────────────────────────────────────
        "1013": {
            "customer": "Yassine N.",
            "status": "Return In Transit — Received by Carrier",
            "items": "Abaya Moderne (S, camel) x1",
            "total": "385 MAD",
            "payment": "Visa Card",
            "city": "Marrakech",
            "order_date": "2025-04-08",
            "carrier": "Amana",
            "eta": "Refund will be processed within 5–7 business days of inspection",
            "notes": "Reason: item color different from website photo.",
        },

        # ── Refunded ────────────────────────────────────────────────────────
        "1014": {
            "customer": "Loubna A.",
            "status": "Refunded",
            "items": "Set Loungewear Femme (L, pink) x1",
            "total": "220 MAD",
            "payment": "CMI Card",
            "city": "Casablanca",
            "order_date": "2025-04-01",
            "carrier": "CTM",
            "eta": "Refund of 220 MAD issued on Apr 10",
            "notes": "Refund sent to original CMI card. May take 3–5 bank days to appear.",
        },

        # ── Wrong Item Received ─────────────────────────────────────────────
        "1015": {
            "customer": "Tariq B.",
            "status": "Delivered — Issue Reported",
            "items": "Polo Premium Homme (M, white) x1  ← ordered | Received: Polo Premium Homme (M, red)",
            "total": "185 MAD",
            "payment": "COD",
            "city": "Fès",
            "order_date": "2025-04-19",
            "carrier": "Amana",
            "eta": "Replacement dispatched — arriving in 2–3 business days",
            "notes": "Wrong color sent by warehouse. Replacement confirmed at no cost.",
        },

        # ── Large/High-Value Order ──────────────────────────────────────────
        "1016": {
            "customer": "Najat R.",
            "status": "Shipped — In Transit",
            "items": (
                "Kaftan Nour Classique (M, emerald) x2, "
                "Jilbab Deux Pièces Satin (L, beige) x1, "
                "Sac à Main Cuir Artisanal (tan) x1, "
                "Foulard Soie Marocaine x2"
            ),
            "total": "1970 MAD",
            "payment": "Visa Card",
            "city": "Casablanca",
            "order_date": "2025-04-20",
            "carrier": "DHL",
            "eta": "1–2 business days (priority shipping)",
            "notes": "COD not available for this order value — paid by card.",
        },
    }

    order = FAKE_ORDERS.get(str(order_id).strip())

    if not order:
        return (
            f"No order found with ID '{order_id}'. "
            "Please double-check the order number and try again. "
            "Order IDs are 4-digit numbers (e.g. 1004)."
        )

    return (
        f"📦 Order #{order_id}\n"
        f"  Customer : {order['customer']}\n"
        f"  Status   : {order['status']}\n"
        f"  Items    : {order['items']}\n"
        f"  Total    : {order['total']} ({order['payment']})\n"
        f"  City     : {order['city']}\n"
        f"  Ordered  : {order['order_date']}\n"
        f"  Carrier  : {order['carrier']}\n"
        f"  ETA/Info : {order['eta']}"
        + (f"\n  Note     : {order['notes']}" if order['notes'] else "")
    )


# ── Tool 3: Escalate to Human ─────────────────────────────────────────────────

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human agent when customer requests it or issue is complex."""
    return f"__ESCALATE__:{reason}"


ALL_TOOLS = [search_knowledge_base, lookup_order, escalate_to_human]