from langchain_core.tools import tool
from vector_store import retriever
import os
import logging
import requests
from alerts import send_alert

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
    """Look up order status by its 4-digit order ID."""
    # Fictional international demo orders; totals are merchandise only, excluding shipping/taxes.
    FAKE_ORDERS = {
        "1001": {
            "customer": "Fatima Zahra B.",
            "status": "Confirmed",
            "items": "Scooter Urban X7 (Black) x1",
            "total": "320 USD",
            "payment": "Visa Card",
            "city": "New York, NY, United States",
            "order_date": "2025-04-23",
            "carrier": "N/A",
            "eta": "Will be dispatched within 24 hrs",
            "notes": "",
        },
        "1002": {
            "customer": "Younes A.",
            "status": "Processing",
            "items": "Scooter Rider Pro 500 (Grey) x1, Protective Helmet (L) x1",
            "total": "505 USD",
            "payment": "Visa Card",
            "city": "London, United Kingdom",
            "order_date": "2025-04-22",
            "carrier": "N/A",
            "eta": "Ships within 24 hrs",
            "notes": "",
        },
        "1003": {
            "customer": "Nadia H.",
            "status": "Packed — Awaiting Pickup by Carrier",
            "items": "E-Bike City Comfort (Sand Beige) x1",
            "total": "890 USD",
            "payment": "Mastercard",
            "city": "Toronto, ON, Canada",
            "order_date": "2025-04-21",
            "carrier": "Shipping partner",
            "eta": "Estimated dispatch: today",
            "notes": "",
        },
        "1004": {
            "customer": "Omar K.",
            "status": "Shipped — In Transit",
            "items": "Scooter Kids Mini E100 (Blue) x1",
            "total": "160 USD",
            "payment": "Visa Card",
            "city": "Chicago, IL, United States",
            "order_date": "2025-04-19",
            "carrier": "Shipping partner",
            "eta": "2–3 business days",
            "notes": "Tracking SMS sent to customer's phone.",
        },
        "1005": {
            "customer": "Salma E.",
            "status": "Shipped — In Transit",
            "items": "Universal Replacement Charger x1, Scooter Storage Bag x1",
            "total": "50 USD",
            "payment": "Visa Card",
            "city": "Manchester, United Kingdom",
            "order_date": "2025-04-20",
            "carrier": "Shipping partner",
            "eta": "1–2 business days",
            "notes": "",
        },
        "1006": {
            "customer": "Hamza T.",
            "status": "Shipped — In Transit",
            "items": "Scooter Urban X7 (White) x1",
            "total": "320 USD",
            "payment": "Visa Card",
            "city": "Vancouver, BC, Canada",
            "order_date": "2025-04-18",
            "carrier": "Shipping partner",
            "eta": "3–4 business days",
            "notes": "Remote city — slight delay possible.",
        },
        "1007": {
            "customer": "Zineb M.",
            "status": "Out for Delivery — Arriving Today",
            "items": "Scooter Kids Mini E100 (Pink) x1",
            "total": "160 USD",
            "payment": "Visa Card",
            "city": "Austin, TX, United States",
            "order_date": "2025-04-21",
            "carrier": "Shipping partner",
            "eta": "Today — delivery agent will call before arriving",
            "notes": "Prepaid order; no cash payment on delivery.",
        },
        "1008": {
            "customer": "Rachid L.",
            "status": "Delivered",
            "items": "Scooter Rider Pro 500 (Black) x1",
            "total": "480 USD",
            "payment": "Mastercard",
            "city": "Birmingham, United Kingdom",
            "order_date": "2025-04-15",
            "carrier": "Shipping partner",
            "eta": "Delivered on Apr 18",
            "notes": "",
        },
        "1009": {
            "customer": "Houda B.",
            "status": "Delivered",
            "items": "E-Bike Trail Sport (Black/Orange) x1",
            "total": "1,150 USD",
            "payment": "Visa Card",
            "city": "New York, NY, United States",
            "order_date": "2025-04-14",
            "carrier": "Shipping partner",
            "eta": "Delivered on Apr 17",
            "notes": "",
        },
        "1010": {
            "customer": "Karim O.",
            "status": "Failed Delivery — Customer Unreachable",
            "items": "Scooter Urban X7 (Black) x1",
            "total": "320 USD",
            "payment": "Visa Card",
            "city": "Ottawa, ON, Canada",
            "order_date": "2025-04-17",
            "carrier": "Shipping partner",
            "eta": "Redelivery can be requested — contact the store team to confirm any fee",
            "notes": "Carrier attempted delivery twice. Please contact support to reschedule.",
        },
        "1011": {
            "customer": "Imane S.",
            "status": "Cancelled",
            "items": "Protective Helmet (M) x1",
            "total": "25 USD",
            "payment": "Visa Card",
            "city": "Seattle, WA, United States",
            "order_date": "2025-04-16",
            "carrier": "N/A",
            "eta": "N/A",
            "notes": "Cancelled by customer within the 1-hour window. Payment refunded.",
        },
        "1012": {
            "customer": "Meryem F.",
            "status": "Return Requested — Awaiting Pickup",
            "items": "Scooter Kids Mini E100 (Black) x1",
            "total": "160 USD",
            "payment": "Visa Card",
            "city": "Boston, MA, United States",
            "order_date": "2025-04-10",
            "carrier": "Shipping partner",
            "eta": "Return pickup scheduled — we will contact you to confirm the date",
            "notes": "Reason: too advanced for a first-time young rider, wants the lower speed cap version.",
        },
        "1013": {
            "customer": "Yassine N.",
            "status": "Return In Transit — Received by Carrier",
            "items": "Scooter Urban X7 (White) x1",
            "total": "320 USD",
            "payment": "Visa Card",
            "city": "London, United Kingdom",
            "order_date": "2025-04-08",
            "carrier": "Shipping partner",
            "eta": "Refund will be processed within 5–7 business days of inspection",
            "notes": "Reason: unused, customer changed mind, within 7-day return window.",
        },
        "1014": {
            "customer": "Loubna A.",
            "status": "Refunded",
            "items": "Reinforced U-Lock x1",
            "total": "22 USD",
            "payment": "Mastercard",
            "city": "New York, NY, United States",
            "order_date": "2025-04-01",
            "carrier": "Shipping partner",
            "eta": "Refund of 22 USD issued on Apr 10",
            "notes": "Refund sent to original Mastercard. May take 3–5 bank days to appear.",
        },
        "1015": {
            "customer": "Tariq B.",
            "status": "Delivered — Issue Reported",
            "items": "Scooter Rider Pro 500 (Red) x1  ← ordered | Received: Rider Pro 500 (Grey)",
            "total": "480 USD",
            "payment": "Visa Card",
            "city": "Chicago, IL, United States",
            "order_date": "2025-04-19",
            "carrier": "Shipping partner",
            "eta": "Replacement dispatched — arriving in 2–3 business days",
            "notes": "Wrong color sent by warehouse. Replacement confirmed at no cost.",
        },
        "1016": {
            "customer": "Najat R.",
            "status": "Shipped — In Transit",
            "items": (
                "E-Bike Cargo Delivery Pro (Black) x1, "
                "spare battery pack x1, "
                "Protective Helmet (L) x1"
            ),
            "total": "1,595 USD",
            "payment": "Bank transfer",
            "city": "New York, NY, United States",
            "order_date": "2025-04-20",
            "carrier": "Shipping partner",
            "eta": "1–2 business days (priority shipping)",
            "notes": "Order paid in full by bank transfer; cash on delivery is not available.",
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
            f"Merchandise subtotal: {order['total']} (shipping/taxes excluded)\n"
            f"City: {order['city']}\n"
            f"ETA: {order['eta']}"
        )
    except Exception as e:
        logger.error(f"lookup_order failed for order_id '{order_id}': {e}", exc_info=True)
        return f"I couldn't retrieve order {order_id} right now. Please try again in a moment."



## escalation with the template
# @tool
# def escalate_to_human(reason: str, language: str, user_phone: str) -> str:
#     """
#     Escalate to human agent and notify the store owner via WhatsApp, using an approved template
#     (works even if the owner hasn't messaged the bot in the last 24 hours).

#     Use ONLY when user explicitly asks for a human agent, or is extremely angry.

#     Args:
#         reason: brief description of why escalation is needed (e.g. 'customer requested human', 'very angry about wrong item').
#         user_phone: the customer's WhatsApp number already in conversation context — never ask for it.
#         language: the customer's language or dialect; any language is supported.
#     """
#     owner_number = os.getenv("OWNER_PHONE")
#     token = os.getenv("WHATSAPP_TOKEN")
#     phone_id = os.getenv("PHONE_NUMBER_ID")

#     payload = {
#         "messaging_product": "whatsapp",
#         "to": owner_number,
#         "type": "template",
#         "template": {
#             "name": "escalation_request",
#             "language": {"code": "en"},
#             "components": [
#                 {
#                     "type": "body",
#                     "parameters": [
#                         {"type": "text", "text": user_phone},
#                         {"type": "text", "text": reason},
#                         {"type": "text", "text": language},
#                     ]
#                 }
#             ]
#         }
#     }

#     try:
#         resp = requests.post(
#             f"https://graph.facebook.com/v19.0/{phone_id}/messages",
#             headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
#             json=payload,
#             timeout=10
#         )
#         resp.raise_for_status()
#         logger.info(f"Escalation owner notification sent. Reason: {reason}, Language: {language}")
#     except Exception as e:
#         logger.error(f"escalate_to_human owner notify failed: {e}", exc_info=True)
#         send_alert(
#             "escalate_notify_fail",
#             "Escalation owner notification failed",
#             f"Failed to notify owner via WhatsApp for customer {user_phone}.\nReason: {reason}\nLanguage: {language}\nError: {e}"
#         )

#     return "Escalation done. Owner notified via WhatsApp."




@tool
def escalate_to_human(reason: str, language: str, user_phone: str) -> str:
    """
    Escalate to human agent and notify the store owner via WhatsApp, 


    Use ONLY when user explicitly asks for a human agent, or is extremely angry.

    Args:
        reason: brief description in English of why escalation is needed (e.g. 'customer requested human', 'very angry about wrong item').
        user_phone: the customer's WhatsApp number already in conversation context — never ask for it.
        language: the customer's language or dialect; any language is supported.

    The owner notification uses English labels with this format:
    
    reason : [reason of escalation]
    language : [language of the user]
    user phone : [phone number of the user]
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    message = f"Human support requested\nReason: {reason}\nLanguage: {language}\nCustomer phone: {user_phone}"


    payload = {
        "messaging_product": "whatsapp",
        "to": owner_number,
        "type": "text",
        "text": {"body":message}
            
        
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
        return f"Escalation owner notification sent. Reason: {reason}, Language: {language}"
    except Exception as e:
        logger.error(f"escalate_to_human owner notify failed: {e}", exc_info=True)
        
        send_alert(
            "escalate_notify_fail",
            "Escalation owner notification failed",
            f"Failed to notify owner via WhatsApp for customer {user_phone}.\nReason: {reason}\nLanguage: {language}\nError: {e}"
        )
        return f"escalate_to_human owner notify failed"

    







## notify owner with template 


# @tool
# def notify_owner(name: str, phone: str, product: str, color: str, size: str, quantity: str, address: str, payment: str) -> str:
#     """
#     Send a confirmed order to the store owner via WhatsApp, using an approved template
#     (works even if the owner hasn't messaged the bot in the last 24 hours).

#     Call this tool ONLY after ALL three conditions are met:
#     1. All order details are collected: name, product, color, size, quantity, address, city, payment method.
#     2. The full order summary was shown to the customer.
#     3. The customer explicitly confirmed or indicated agreement in ANY way, language, or affirmative word or phrase (e.g. ah, iyeh, oui, yes, d'accord, confirm, etc.).

#     Customer phone is already in the conversation context — extract it automatically.
#     NEVER ask the customer for their phone number.

#     Args:
#         name: customer full name
#         phone: customer WhatsApp number (from context, never ask)
#         product: product name
#         color: product color
#         size: size if applicable (accessories like helmets), empty string if not applicable
#         quantity: quantity ordered
#         address: full delivery address including city
#         payment: accepted payment method from the knowledge base
#     """
#     owner_number = os.getenv("OWNER_PHONE")
#     token = os.getenv("WHATSAPP_TOKEN")
#     phone_id = os.getenv("PHONE_NUMBER_ID")

#     payload = {
#         "messaging_product": "whatsapp",
#         "to": owner_number,
#         "type": "template",
#         "template": {
#             "name": "new_order_notification",
#             "language": {"code": "en"},
#             "components": [
#                 {
#                     "type": "body",
#                     "parameters": [
#                         {"type": "text", "text": "ElectroMA"},
#                         {"type": "text", "text": name},
#                         {"type": "text", "text": phone},
#                         {"type": "text", "text": product},
#                         {"type": "text", "text": color},
#                         {"type": "text", "text": size or "-"},
#                         {"type": "text", "text": quantity},
#                         {"type": "text", "text": address},
#                         {"type": "text", "text": payment},
#                     ]
#                 }
#             ]
#         }
#     }

#     try:
#         resp = requests.post(
#             f"https://graph.facebook.com/v19.0/{phone_id}/messages",
#             headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
#             json=payload,
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







 



## notify order tool exaclty before doing the template : 

@tool
def notify_owner(order_summary: str) -> str:
    """
     Send a confirmed order to the store owner via WhatsApp.

    Call this tool ONLY after ALL three conditions are met:
    1. All order details are collected: name, product, color, size, quantity, address, city, payment method.
    2. The full order summary was shown to the customer .
    3. The customer explicitly confirmed the order in their own language or dialect.

    Customer phone is already in the conversation context — extract it automatically.
    NEVER ask the customer for their phone number.

    Format order_summary with English labels like this, preserving customer details:
    Customer: [name]
    Phone: [customer phone from context, including country code]
    Product: [product name]
    Color: [color]
    Size: [size, if applicable]
    Quantity: [quantity]
    Address: [full address]
    Payment: [payment method from the knowledge base]
    Total: [confirmed total and catalog currency; do not invent fees or convert currencies]
    
    """
    owner_number = os.getenv("OWNER_PHONE")
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("PHONE_NUMBER_ID")

    message = f"🛒 *New order*\n\n{order_summary}\n\n_Order collected by RELIA — please confirm with the customer._"

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
    # Address: [full address ]
    # Payment: [accepted payment method from the knowledge base]
