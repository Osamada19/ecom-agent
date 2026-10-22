# Load environment variables (API keys, tokens) from the .env file.
from dotenv import load_dotenv
load_dotenv()
# Core libraries for system operations, hashing, logging, and data structures.
import os
import hashlib  
import logging
from collections import OrderedDict
# HTTP requests client for making API calls to WhatsApp.
import requests
# Web framework components for handling HTTP requests and responses.
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
# Message data types for communicating with LangChain chat models.
from langchain_core.messages import HumanMessage, AIMessage
# Pre-configured LangChain agent instance with tools and conversation memory.
from agent import agent
# Background tasks runner for executing long operations without delaying HTTP responses.
from fastapi import BackgroundTasks
# Alerting helper to send notifications (e.g. to Telegram) when errors occur.
from alerts import send_alert
# Async utilities for running non-blocking tasks and managing server lifecycles.
import asyncio
from contextlib import asynccontextmanager
# Knowledge base ingestion function to populate ChromaDB.
from ingest import ingest


#universal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Cache storing recently processed message IDs to avoid answering duplicates sent by Meta retries.
# An OrderedDict keeps insertion order so we can cleanly evict the oldest items when the cache fills up.
_processed = OrderedDict()

# replacement for ingest()

# Manages application startup and shutdown lifecycle events.
# Automatically indexes the knowledge base into the vector store when the server starts or redeploys.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs automatically on every startup / redeploy."""
    # Check whether automatic ingestion on startup is enabled via environment variables.
    if os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() in ("true", "1", "yes"):
        logger.info("Redeploy/startup detected: running knowledge base ingestion...")
        try:
            # Run the synchronous ingest function in a background thread so the async event loop is not blocked.
            await asyncio.to_thread(ingest, sync_alert=True)
            logger.info("Knowledge base ingestion finished successfully on startup.")
        except Exception as e:
            logger.error(f"Startup knowledge base ingestion failed: {e}", exc_info=True)
            # ingest() has already dispatched a telegram alert via send_alert()
    yield


# Initialize the FastAPI web application with the startup/shutdown lifespan handler.
app = FastAPI(lifespan=lifespan)




# Meta WhatsApp Cloud API credentials and verification tokens loaded from environment variables.
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# Friendly pre-written responses sent to customers when their inquiry is handed over to a human team member.
ESCALATION_MSGS = {
    "english": "I've notified our team and they'll reach out to you shortly on WhatsApp. In the meantime, feel free to ask me anything else — I'm still here! 😊",
    "french": "J'ai informé notre équipe et ils vous contacteront bientôt sur WhatsApp. En attendant, n'hésitez pas à me poser d'autres questions — je suis toujours là ! 😊",
    "arabic": "لقد أبلغت فريقنا وسيتواصلون معك قريباً على واتساب. في هذه الأثناء، لا تتردد في سؤالي عن أي شيء آخر — أنا هنا! 😊",
    "darija": "أعلمت الفريق وغادي يتواصلو معاك قريبًا على واتساب. فينما كتسنا، سول على أي حاجة أخرى — أنا هنا! 😊",
}


# Webhook verification endpoint used by Meta WhatsApp to confirm server ownership during initial setup.
# Returns Meta's challenge code if the verification token matches our secret token.
@app.get("/webhook")
async def verify(request: Request):
    # Extract query parameters from the GET request URL sent by Meta.
    params = dict(request.query_params)
    # Check if Meta's verify token matches ours; return hub.challenge on success or 403 on failure.
    return PlainTextResponse(params.get("hub.challenge")) if params.get("hub.verify_token") == VERIFY_TOKEN else PlainTextResponse("Invalid", status_code=403)




# Runs the AI agent logic and dispatches the answer to WhatsApp in a background thread.
# This prevents WhatsApp timeouts by letting the webhook reply with HTTP 200 OK right away.
def process_customer_message(phone: str, text: str):
    """Process agent logic in the background so Meta gets a 200 OK immediately."""
    reply = None
    try:
        # Use the customer's phone number as the thread ID to maintain continuous chat history for each customer.
        config = {"configurable": {"thread_id": phone}}
        # Count existing messages in the conversation history before running the agent.
        n_before = len(agent.get_state(config).values.get("messages", []))
        # Send the customer message with internal context to the agent to get a response.
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"{text}\n\n[Internal Context: Customer WhatsApp {phone}]")]},
            config=config
        )
        # Extract the content of the latest message generated by the agent.
        reply = result["messages"][-1].content
        # Handle cases where the model returns response chunks as a list by joining them into a string.
        if isinstance(reply, list):
            reply = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in reply)
        elif reply is not None:
            reply = str(reply)

        # Slice out only the new messages generated during this specific invocation.
        new_messages = result["messages"][n_before:]

        # ESCALATION INTERCEPT
        # Check if the agent called the escalation tool during this turn to hand off to human support.
        try:
            for m in new_messages:
                # Look for an AIMessage that made tool calls.
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    # Check if one of the tool calls was escalate_to_human.
                    esc_call = next(
                        (tc for tc in m.tool_calls if tc.get("name") == "escalate_to_human"),
                        None
                    )
                    # If escalation was requested, send a friendly canned message in the customer's language.
                    if esc_call:
                        lang = esc_call.get("args", {}).get("language", "english")
                        if not lang:
                            lang = "english"
                        lang = str(lang).lower()
                        reply = ESCALATION_MSGS.get(lang, ESCALATION_MSGS["english"])
                        logger.info(f"Escalation triggered for {phone} in lang={lang}")
                        break
        except Exception as esc_err:
            logger.error(f"Escalation intercept failed for {phone}: {esc_err}", exc_info=True)

    # Handle timeouts if the agent takes too long to respond.
    except TimeoutError as e:
        logger.error(f"Agent timed out for {phone}: {e}", exc_info=True)
        # Send a multilingual fallback message asking the customer to retry.
        reply = (
            "طلبك خد وقت بزاف — جرب عاود من جديد 🙏\n"
            "Ça a pris trop de temps — veuillez réessayer 🙏\n"
            "That took too long. Please try again."
        )
    # Handle general exceptions, log them, notify the admin team, and send a friendly customer notice.
    except Exception as e:
        err_str = str(e).lower()
        logger.error(f"Agent error for {phone}: {e}", exc_info=True)
        # Alert: agent crashed processing a customer message
        send_alert("agent_crash", "Agent processing error",
                    f"Error for customer {phone}: {e}")

        # Choose an appropriate user-facing message based on the error category.
        if any(kw in err_str for kw in ("tool_use_failed", "failed to call", "tool call", "toolexception")):
            reply = (
                "معلاش، كاين مشكل صغير مع واحد من الأدوات. واش تقدر تعاود تسأل بطريقة أخرى؟ 🙏\n"
                "Un petit problème technique — pouvez-vous reformuler votre question? 🙏\n"
                "I'm having trouble accessing that right now. Could you try asking differently?"
            )
        elif "timeout" in err_str or "timed out" in err_str:
            reply = (
                "طلبك خد وقت بزاف — جرب عاود من جديد 🙏\n"
                "Ça a pris trop de temps — veuillez réessayer 🙏\n"
                "That took too long. Please try again."
            )
        elif "rate limit" in err_str or "quota" in err_str or "429" in err_str:
            reply = (
                "كاين ضغط دابا — صبر شوية وجرب عاود 🙏\n"
                "Trop de demandes en ce moment — réessayez dans un instant 🙏\n"
                "We're a bit busy right now. Please try again in a moment."
            )
        else:
            reply = (
                "معلاش، كاين مشكل تقني دابا. صبر شوية وجرب عاود، ولا تواصل معانا: +212-6XX-XXXXXX\n"
                "Désolé, problème technique. Réessayez plus tard ou contactez-nous: +212-6XX-XXXXXX\n"
                "Sorry, I'm having trouble. Please try again or contact support at +212-6XX-XXXXXX."
            )

        print(config)

    # If an answer was generated, send it back to the customer on WhatsApp.
    if reply:
        _send(phone, reply)





# Receives incoming webhook events (messages, status updates) from Meta WhatsApp Cloud API.
# Dispatches incoming messages to background tasks and responds with HTTP 200 immediately.
@app.post("/webhook")
async def receive(request: Request, background_tasks: BackgroundTasks):
    # Parse the incoming JSON payload from Meta.
    data = await request.json()

    # Generate a unique hash for the message to detect and ignore duplicate webhook requests.
    msg_id = _get_msg_id(data)
    if msg_id and msg_id in _processed:
        return {"status": "duplicate"}
    if msg_id:
        _processed[msg_id] = True
        # Keep the deduplication set at 10,000 entries max to prevent memory bloat.
        if len(_processed) > 10000:
            _processed.popitem(last=False)  # smoothly evict the oldest entry

    try:
        # Extract the core payload data dictionary from the webhook payload.
        entry = data["entry"][0]["changes"][0]["value"]

        # Handle delivery status updates from Meta (sent, delivered, read, failed)
        # Check for message delivery statuses such as sent, delivered, read, or failed.
        for status in entry.get("statuses", []):
            s = status.get("status")
            recipient = status.get("recipient_id", "unknown")
            # If WhatsApp could not deliver a message, log the error and trigger an alert.
            if s == "failed":
                err = status.get("errors", [{}])[0]
                code = err.get("code", "?")
                title = err.get("title", "unknown error")
                logger.error(f"WhatsApp delivery failed to {recipient}: [{code}] {title}")
                send_alert("wa_delivery_failed", "WhatsApp delivery failed",
                            f"Recipient: {recipient}\nError: [{code}] {title}")
            else:
                logger.debug(f"WhatsApp status '{s}' for {recipient}")

        # If the payload only contains delivery receipts and no customer message, exit early.
        if "messages" not in entry:
            return {"status": "ignored"}
        # Extract the first customer message and the sender's phone number.
        msg = entry["messages"][0]
        phone = msg["from"]

        # # Ignore incoming messages from the owner's phone number to avoid bot response loops.
        # if phone == os.getenv("OWNER_PHONE"):
        #     return {"status": "ignored"}

        # Identify the type of message received (text, image, audio, location).
        msg_type = msg.get("type")

        # 1. Images
        # Let the customer know we cannot process images and ask for a text description.
        if msg_type == "image":
            _send(phone, "ما كنقدرش نشوف الصور — صيفط لينا سؤالك كتابةً وغادي نعاونك دابا 🙏\n"
                         "I can't view images — could you describe what you're looking for and I'll help you right away!\n"
                         "Je ne peux pas voir les images — décrivez ce que vous cherchez et je vous aide tout de suite 🙏\n")
            return {"status": "ok"}

        # 2. Audio Voice Notes
        # Let the customer know voice notes are unsupported and ask them to type their question.
        if msg_type in ("audio", "voice"):
            _send(phone, "ما كنقدرش نسمع الرسائل الصوتية دابا — عفاك كتب ليا سؤالك وغادي نجاوبك فالحين 🙏\n"
                         "Je ne peux pas écouter les messages vocaux pour le moment — écrivez votre message et je vous réponds tout de suite 🙏")
            return {"status": "ok"}

        # 3. Location Pins
        # Convert GPS coordinates into a Google Maps URL and hand off to the agent for delivery location handling.
        if msg_type == "location":
            loc = msg.get("location", {})
            lat = loc.get("latitude")
            lng = loc.get("longitude")
            text = f"📍 Customer sent their delivery location: https://maps.google.com/?q={lat},{lng}"
            background_tasks.add_task(process_customer_message, phone, text)
            return {"status": "ok"}

        # 4. Standard Text
        # Queue the standard text message for background processing by the AI agent.
        if msg_type == "text":
            text = msg["text"]["body"]
            background_tasks.add_task(process_customer_message, phone, text)
            return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error parsing webhook message: {e}", exc_info=True)

    return {"status": "ok"}


# Generates a 16-character SHA-256 fingerprint for a message using its ID and timestamp.
# Used by deduplication logic to ensure identical messages are not processed multiple times.
def _get_msg_id(data):
    try:
        # Extract the message ID and timestamp from Meta's payload structure and hash them.
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        return hashlib.sha256(f"{msg['id']}:{msg['timestamp']}".encode()).hexdigest()[:16]
    except:
        return None


# Helper function to send outbound text messages via the Meta WhatsApp Cloud API.
# Constructs the HTTP request, handles string conversion, and logs errors or triggers alerts on failure.
def _send(to, text):
    try:
        # Normalize the text to a clean string if it was provided as a list of chunks.
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        else:
            text = str(text) if text is not None else ""

        # Set up the Meta Cloud API URL, authorization bearer token, and JSON message body.
        url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
        payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
        # Send HTTP POST request to Meta with a 10-second timeout.
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        # Check if the API returned an error response status code (4xx or 5xx).
        if resp.status_code >= 400:
            logger.error(f"WhatsApp API error [{resp.status_code}]: {resp.text}")
            # Alert: outbound message delivery failed
            send_alert("wa_send_fail", "WhatsApp send failed",
                        f"Status {resp.status_code} sending to {to}.\nAPI response: {resp.text}")
        else:
            logger.info(f"WhatsApp message sent successfully to {to} [{resp.status_code}]")
    # Catch any unexpected network or execution exceptions during sending.
    except Exception as e:
        logger.error(f"Send failed: {e}")
        send_alert("wa_send_exception", "WhatsApp send exception",
                    f"Exception sending to {to}: {e}")