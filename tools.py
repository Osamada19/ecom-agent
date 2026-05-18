from langchain_core.tools import tool
from vector_store import retriever
import os
import logging
import requests

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(query: str) -> str:
    """Search store policies, products, shipping, returns, payments, sizing."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found."
        return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        logger.error(f"search_knowledge_base failed for query '{query}': {e}", exc_info=True)
        return "I'm having a quick technical hiccup accessing the database. Please try your question again in about 1 minute!"


@tool
def lookup_order(order_id: str) -> str:
    """Look up order status by order ID (4-digit number)."""
    FAKE_ORDERS = {
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
        }
    }
    try:
        order = FAKE_ORDERS.get(str(order_id).strip())
        if not order:
            return f"No order found with ID '{order_id}'. Please double-check the ID and try again."
        return (
            f"📦 Order #{order_id}\n"
            f"Status: {order['status']}\n"
            f"Items: {order['items']}\n"
            f"Total: {order['total']}\n"
            f"City: {order['city']}\n"
            f"ETA: {order['eta']}"
        )
    except Exception as e:
        logger.error(f"lookup_order failed for order_id '{order_id}': {e}", exc_info=True)
        return f"I couldn't retrieve order {order_id} right now. Please try again in a moment."


@tool
def escalate_to_human(reason: str, language: str ,user_phone:int) -> str:
    """
    Escalate to human agent and notify the store owner via WhatsApp.
    Use ONLY when user explicitly asks for a human agent, or is extremely angry.

    Args:
        reason: brief description of why escalation is needed (e.g. 'customer requested human', 'very angry about wrong item'). 
        user_phone: you already have access to the customer phone number as thread id 
        language: the language the customer is using. Must be exactly one of: english, french, arabic, darija
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    message = (
        f"⚠️ *Escalation Request*\n\n"
        f"💬 Reason: {reason}\n\n" 
        f"phone number :{user_phone}\n\n "
        f"_Customer requested human support — please follow up._"
    )

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "messaging_product": "whatsapp",
                "to": owner_number,
                "type": "text",
                "text": {"body": message}
            },
            timeout=10
        )
        resp.raise_for_status()
        logger.info(f"Escalation owner notification sent. Reason: {reason}, Language: {language}")
    except Exception as e:
        logger.error(f"escalate_to_human owner notify failed: {e}", exc_info=True)

    # Always return the trigger — even if the WhatsApp call failed,
    # the customer still gets the escalation message.
    return f"[ESCALATE_TRIGGERED:{language}]"


@tool
def notify_owner(order_summary: str) -> str:
    """
    Send a draft order to the store owner for confirmation.

    STRICT CONDITIONS — call this tool ONLY when ALL of these are true:
    1. The user has explicitly stated they want to place an order
       (e.g. 'I want to order', 'I want to buy', 'confirm my order').
    2. You have collected: product name, color, size, quantity,
       customer name, delivery address, payment method, and customer_phone.

    Format order_summary EXACTLY like this:

     الاسم: [name]
    الهاتف: [customer phone]
    المنتج: [product name]
    اللون: [color]
    المقاس: [size]
    الكمية: [quantity]
    العنوان: [full address + city]
    الدفع: [COD or card]

    Do NOT call this just because you know product details from
    a product question. Intent to buy must be explicit.

    BEFORE calling this tool, verify you have ALL of these in the CURRENT conversation:
    - customer_name, product, color, size, quantity, address, city, payment_method, customer_phone.
    If ANY is missing, ask for it first. Never call with incomplete data.
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    message = f"🛒 *طلب جديد*\n\n{order_summary}\n\n_تم جمع الطلب بواسطة RELIA — يرجى تأكيده مع العميل._"

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "messaging_product": "whatsapp",
                "to": owner_number,
                "type": "text",
                "text": {"body": message}
            },
            timeout=10
        )
        resp.raise_for_status()
        return "Order sent to store owner. Tell the customer: the owner will confirm their order shortly via WhatsApp."
    except requests.exceptions.Timeout:
        logger.error("notify_owner timed out sending WhatsApp message to owner")
        return "The order notification timed out. Please ask the customer to try confirming again in a moment."
    except requests.exceptions.HTTPError as e:
        logger.error(f"notify_owner HTTP error: {e}", exc_info=True)
        return "There was a problem sending the order to the owner. Please try again or contact support."
    except Exception as e:
        logger.error(f"notify_owner failed: {e}", exc_info=True)
        return "I couldn't send the order notification right now. Please try again in a moment."


ALL_TOOLS = [search_knowledge_base, lookup_order, escalate_to_human, notify_owner]




### english order summary format 

# Customer: [name]
    # Phone: [customer phone]
    # Product: [product name]
    # Color: [color]
    # Size: [size]
    # Quantity: [quantity]
    # Address: [full address + city]
    # Payment: [COD or card]
