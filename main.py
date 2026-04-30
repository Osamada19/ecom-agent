import os
import hashlib
import logging
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage
from agent import agent
from ingest import ingest
from fastapi import BackgroundTasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_processed = set()

if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
    ingest()

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

ESCALATION_MSGS = {
    "english": "I'm escalating your case to a human agent right away. Please reach us on WhatsApp at +212-6XX-XXXXXX (Mon–Sat, 9am–6pm). We'll respond within 2 hours.",
    "french": "Je transmets votre demande à un agent humain immédiatement. Contactez-nous sur WhatsApp au +212-6XX-XXXXXX (Lun–Sam, 9h–18h). Nous répondrons dans les 2 heures.",
    "arabic": "سأقوم بتحويل طلبك إلى أحد الوكلاء البشريين على الفور. يرجى التواصل معنا عبر واتساب على +212-6XX-XXXXXX (من الاثنين إلى السبت، 9 صباحًا حتى 6 مساءً). سنرد خلال ساعتين.",
    "darija": "غادي نحول ليك مع واحد من الفريق دابا. تواصل معانا على واتساب: +212-6XX-XXXXXX (من الاثنين للسبت، 9 الصباح حتى 6 المغرب). غادي يردو عليك خلال ساعتين.",
}

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    return PlainTextResponse(params.get("hub.challenge")) if params.get("hub.verify_token") == VERIFY_TOKEN else PlainTextResponse("Invalid", status_code=403)

@app.post("/webhook")
async def receive(request: Request):
    data = await request.json()

    
    
    
    
    msg_id = _get_msg_id(data)
    if msg_id and msg_id in _processed:
        return {"status": "duplicate"}
    if msg_id:
        _processed.add(msg_id)
        if len(_processed) > 5000: _processed.clear()

    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" not in entry: return {"status": "ignored"}
        msg = entry["messages"][0]
        if msg.get("type") != "text": return {"status": "ignored"}
        phone, text = msg["from"], msg["text"]["body"]
    except Exception:
        return {"status": "ignored"}

    # INPUT GUARD
    text_stripped = text.strip()
    if len(text_stripped) < 2 or not any(c.isalpha() for c in text_stripped):
        _send(phone, "I didn't catch that. Could you please rephrase your question?")
        return {"status": "gibberish"}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": phone}}
        )
        reply = result["messages"][-1].content
        
        # ESCALATION INTERCEPT
        if "[ESCALATE_TRIGGERED]" in reply:
            # Detect language from the user's message for the correct escalation text
            lang = _detect_language_quick(text)
            reply = ESCALATION_MSGS.get(lang, ESCALATION_MSGS["english"])
            logger.info(f"Escalation triggered for {phone} in {lang}")
            
    except Exception as e:
        logger.error(f"Agent error: {e}")
        reply = "Sorry, I'm having trouble. Please contact support at +212-6XX-XXXXXX."

    _send(phone, reply)
    return {"status": "ok"}

def _get_msg_id(data):
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        return hashlib.sha256(f"{msg['id']}:{msg['timestamp']}".encode()).hexdigest()[:16]
    except: 
        return None

def _detect_language_quick(text: str) -> str:
    """Fast heuristic for escalation message language."""
    t = text.lower()
    # Darija check
    darija_words = ["salam", "bghit", "chof", "3afak", "kifach", "fin", "wech", "mashi", "zwin", "daba", "l3afass", "wakha", "shukran", "labas"]
    if any(w in t for w in darija_words) or any(c in t for c in "37952"):
        return "darija"
    # Arabic script check
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "arabic"
    # French check
    french_words = ["bonjour", "merci", "commande", "livraison", "bon", "svp", "s'il", "comment", "prix"]
    if any(w in t for w in french_words):
        return "french"
    return "english"

def _send(to, text):
    try:
        requests.post(
            f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Send failed: {e}")