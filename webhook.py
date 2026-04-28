import os
import time
import hashlib
import requests
import logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from graph import graph
from langchain_core.messages import HumanMessage
from ingest import ingest

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("error.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Idempotency: track processed message IDs ──────────────────────────────────
_processed_messages = set()

def _get_message_id(data: dict) -> str:
    """Generate unique ID for a message to prevent duplicate processing."""
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" in entry:
            msg = entry["messages"][0]
            # Use WhatsApp message ID + timestamp
            msg_id = msg.get("id", "")
            timestamp = msg.get("timestamp", "")
            return hashlib.sha256(f"{msg_id}:{timestamp}".encode()).hexdigest()[:16]
    except (KeyError, IndexError):
        pass
    return None

# ── Ingest knowledge base (only if empty) ─────────────────────────────────────
if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
    logger.info("ChromaDB not found, running ingest...")
    ingest()
else:
    logger.info("ChromaDB already exists, skipping ingest.")

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# ── Webhook verification (Meta calls this once when you register) ──────────────
@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(params.get("hub.challenge"))
    return PlainTextResponse("Invalid token", status_code=403)

# ── Incoming messages ──────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    # Deduplication check
    msg_id = _get_message_id(data)
    if msg_id and msg_id in _processed_messages:
        logger.info(f"Duplicate message {msg_id}, ignoring.")
        return {"status": "duplicate_ignored"}
    if msg_id:
        _processed_messages.add(msg_id)
        # Keep set size manageable
        if len(_processed_messages) > 10000:
            _processed_messages.clear()

    try:
        entry = data["entry"][0]["changes"][0]["value"]

        # Ignore status updates (delivered, read, failed)
        if "messages" not in entry:
            return {"status": "ignored"}

        message = entry["messages"][0]

        # Ignore non-text messages (images, reactions, etc.)
        if message.get("type") != "text":
            return {"status": "ignored"}

        from_number = message["from"]
        text = message["text"]["body"]

        logger.info(f"Message from {from_number}: {text[:50]}...")

    except (KeyError, IndexError) as e:
        logger.warning(f"Malformed webhook data: {e}")
        return {"status": "ignored"}

    # Invoke graph with error handling
    try:
        config = {"configurable": {"thread_id": from_number}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=config
        )
        reply = result["messages"][-1].content

        # Reset escalation state after successful non-escalation response
        if not result.get("escalate", False):
            logger.info(f"Normal response to {from_number}")
        else:
            logger.warning(f"Escalation triggered for {from_number}")

    except Exception as e:
        logger.error(f"Graph invocation failed: {e}", exc_info=True)
        reply = (
            "Sorry, I'm having trouble processing your message right now. "
            "Please contact our support team at +212-6XX-XXXXXX."
        )

    # Send with retry logic
    success = send_message(from_number, reply)
    if not success:
        logger.error(f"Failed to send message to {from_number}")

    return {"status": "ok" if success else "send_failed"}


def send_message(to: str, text: str, max_retries: int = 3) -> bool:
    """Send WhatsApp message with retry logic."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"Message sent to {to}")
                return True
            else:
                logger.warning(f"WhatsApp API error {response.status_code}: {response.text}")
                if response.status_code == 429:  # Rate limited
                    time.sleep(2 ** attempt)
                elif response.status_code >= 500:  # Server error, retry
                    time.sleep(1)
                else:
                    break  # Client error, don't retry
        except requests.RequestException as e:
            logger.error(f"Request failed (attempt {attempt + 1}): {e}")
            time.sleep(1)

    return False