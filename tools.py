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
            "items": "Trottinette Urban X7 (Black) x1",
            "total": "3,200 MAD",
            "payment": "COD",
            "city": "Casablanca",
            "order_date": "2025-04-23",
            "carrier": "N/A",
            "eta": "Will be dispatched within 24 hrs",
            "notes": "",
        },
        "1002": {
            "customer": "Younes A.",
            "status": "Processing",
            "items": "Trottinette Rider Pro 500 (Grey) x1, Casque de Protection (L) x1",
            "total": "5,050 MAD",
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
            "items": "E-Bike City Comfort (Sand Beige) x1",
            "total": "8,900 MAD",
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
            "items": "Trottinette Kids Mini E100 (Blue) x1",
            "total": "1,600 MAD",
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
            "items": "Chargeur de Rechange Universel x1, Sacoche de Rangement x1",
            "total": "500 MAD",
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
            "items": "Trottinette Urban X7 (White) x1",
            "total": "3,200 MAD",
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
            "items": "Trottinette Kids Mini E100 (Pink) x1",
            "total": "1,600 MAD",
            "payment": "COD",
            "city": "Kenitra",
            "order_date": "2025-04-21",
            "carrier": "CTM",
            "eta": "Today — delivery agent will call before arriving",
            "notes": "Please have 1,600 MAD ready.",
        },
        "1008": {
            "customer": "Rachid L.",
            "status": "Delivered",
            "items": "Trottinette Rider Pro 500 (Black) x1",
            "total": "4,800 MAD",
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
            "items": "E-Bike Trail Sport (Black/Orange) x1",
            "total": "11,500 MAD",
            "payment": "Visa Card + 30% deposit",
            "city": "Casablanca",
            "order_date": "2025-04-14",
            "carrier": "CTM",
            "eta": "Delivered on Apr 17",
            "notes": "",
        },
        "1010": {
            "customer": "Karim O.",
            "status": "Failed Delivery — Customer Unreachable",
            "items": "Trottinette Urban X7 (Black) x1",
            "total": "3,200 MAD",
            "payment": "COD",
            "city": "Oujda",
            "order_date": "2025-04-17",
            "carrier": "Amana",
            "eta": "Redelivery can be requested — 50 MAD reshipping fee applies",
            "notes": "Carrier attempted delivery twice. Please contact support to reschedule.",
        },
        "1011": {
            "customer": "Imane S.",
            "status": "Cancelled",
            "items": "Casque de Protection (M) x1",
            "total": "250 MAD",
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
            "items": "Trottinette Kids Mini E100 (Black) x1",
            "total": "1,600 MAD",
            "payment": "COD",
            "city": "Tétouan",
            "order_date": "2025-04-10",
            "carrier": "CTM",
            "eta": "Return pickup scheduled — we will contact you to confirm the date",
            "notes": "Reason: too advanced for a first-time young rider, wants the lower speed cap version.",
        },
        "1013": {
            "customer": "Yassine N.",
            "status": "Return In Transit — Received by Carrier",
            "items": "Trottinette Urban X7 (White) x1",
            "total": "3,200 MAD",
            "payment": "Visa Card",
            "city": "Marrakech",
            "order_date": "2025-04-08",
            "carrier": "Amana",
            "eta": "Refund will be processed within 5–7 business days of inspection",
            "notes": "Reason: unused, customer changed mind, within 7-day return window.",
        },
        "1014": {
            "customer": "Loubna A.",
            "status": "Refunded",
            "items": "Antivol U-Lock Renforcé x1",
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
            "items": "Trottinette Rider Pro 500 (Red) x1  ← ordered | Received: Rider Pro 500 (Grey)",
            "total": "4,800 MAD",
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
                "E-Bike Cargo Delivery Pro (Black) x1, "
                "spare battery pack x1, "
                "Casque de Protection (L) x1"
            ),
            "total": "15,950 MAD",
            "payment": "Bank transfer",
            "city": "Casablanca",
            "order_date": "2025-04-20",
            "carrier": "DHL",
            "eta": "1–2 business days (priority shipping)",
            "notes": "COD not available above 5,000 MAD — 30% deposit paid, rest by transfer.",
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
def escalate_to_human(reason: str, language: str, user_phone: str) -> str:
    """
    Escalate to human agent and notify the store owner via WhatsApp, using an approved template
    (works even if the owner hasn't messaged the bot in the last 24 hours).

    Use ONLY when user explicitly asks for a human agent, or is extremely angry.

    Args:
        reason: brief description of why escalation is needed (e.g. 'customer requested human', 'very angry about wrong item').
        user_phone: the customer's WhatsApp number already in conversation context — never ask for it.
        language: the language the customer is using. Must be exactly one of: english, french, arabic, darija
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    payload = {
        "messaging_product": "whatsapp",
        "to": owner_number,
        "type": "template",
        "template": {
            "name": "escalation_request",
            "language": {"code": "ar"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": user_phone},
                        {"type": "text", "text": reason},
                        {"type": "text", "text": language},
                    ]
                }
            ]
        }
    }

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
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
def notify_owner(name: str, phone: str, product: str, color: str, size: str, quantity: str, address: str, payment: str) -> str:
    """
    Send a confirmed order to the store owner via WhatsApp, using an approved template
    (works even if the owner hasn't messaged the bot in the last 24 hours).

    Call this tool ONLY after ALL three conditions are met:
    1. All order details are collected: name, product, color, size, quantity, address, city, payment method.
    2. The full order summary was shown to the customer.
    3. The customer explicitly confirmed or indicated agreement in ANY way, language, or affirmative word or phrase (e.g. ah, iyeh, oui, yes, d'accord, confirm, etc.).

    Customer phone is already in the conversation context — extract it automatically.
    NEVER ask the customer for their phone number.

    Args:
        name: customer full name
        phone: customer WhatsApp number (from context, never ask)
        product: product name
        color: product color
        size: size if applicable (accessories like helmets), empty string if not applicable
        quantity: quantity ordered
        address: full delivery address including city
        payment: "COD" or payment method used
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    payload = {
        "messaging_product": "whatsapp",
        "to": owner_number,
        "type": "template",
        "template": {
            "name": "new_order_notification",
            "language": {"code": "ar"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": "ElectroMA"},
                        {"type": "text", "text": name},
                        {"type": "text", "text": phone},
                        {"type": "text", "text": product},
                        {"type": "text", "text": color},
                        {"type": "text", "text": size or "-"},
                        {"type": "text", "text": quantity},
                        {"type": "text", "text": address},
                        {"type": "text", "text": payment},
                    ]
                }
            ]
        }
    }

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
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







 



# ## notify order tool exaclty before doing the template : 

# @tool
# def notify_owner(order_summary: str) -> str:
#     """
#      Send a confirmed order to the store owner via WhatsApp.

#     Call this tool ONLY after ALL three conditions are met:
#     1. All order details are collected: name, product, color, size, quantity, address, city, payment method.
#     2. The full order summary was shown to the customer .
#     3. The customer explicitly confirmed with yes / iyeh / oui / nam.

#     Customer phone is already in the conversation context — extract it automatically.
#     NEVER ask the customer for their phone number.

#     Format order_summary EXACTLY like this:
#     الاسم: [name]
#     الهاتف: [customer phone from context]
#     المنتج: [product name]
#     اللون: [color]
#     المقاس: [size]
#     الكمية: [quantity]
#     العنوان: [full address + city]
#     الدفع: [COD or card]
    
#     """
#     owner_number = os.getenv("OWNER_PHONE")
#     token = os.getenv("WHATSAPP_TOKEN")
#     phone_id = os.getenv("PHONE_NUMBER_ID")

#     message = f"🛒 *طلب جديد*\n\n{order_summary}\n\n_تم جمع الطلب بواسطة RELIA — يرجى تأكيده مع العميل._"

#     try:
#         resp = requests.post(
#             f"https://graph.facebook.com/v19.0/{phone_id}/messages",
#             headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
#             json={
#                 "messaging_product": "whatsapp",
#                 "to": owner_number,
#                 "type": "text",
#                 "text": {"body": message}
#             },
#             timeout=10
#         )
#         resp.raise_for_status()
#         return "Order sent to store owner. Tell the customer: the owner will confirm their order shortly via WhatsApp."
#     except requests.exceptions.Timeout:
#         logger.error("notify_owner timed out sending WhatsApp message to owner")
#         return "The order notification timed out. Please ask the customer to try confirming again in a moment."
#     except requests.exceptions.HTTPError as e:
#         logger.error(f"notify_owner HTTP error: {e}", exc_info=True)
#         return "There was a problem sending the order to the owner. Please try again or contact support."
#     except Exception as e:
#         logger.error(f"notify_owner failed: {e}", exc_info=True)
#         return "I couldn't send the order notification right now. Please try again in a moment."








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