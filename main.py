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
from langchain_core.messages import ToolMessage


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_processed = set()
_escalated = set()

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
        phone = msg["from"]
        if msg.get("type") != "text":
            if msg.get("type") == "image":
                _send(phone, "ما كنقدرش نشوف الصور — صيفط لينا سؤالك كتابةً وغادي نعاونك دابا 🙏\n"
                             "Je ne peux pas voir les images — décrivez ce que vous cherchez et je vous aide tout de suite 🙏\n"
                             "I can't view images — could you describe what you're looking for and I'll help you right away!\n")
            return {"status": "ignored"}

        text = msg["text"]["body"]
    except Exception:
        return {"status": "ignored"}

    if phone == os.getenv("OWNER_PHONE"):
        return {"status": "ignored"}

    reply = None
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=f"[Customer WhatsApp: {phone}]\n{text}")]},
            config={"configurable": {"thread_id": phone}}
        )
        reply = result["messages"][-1].content

        # ESCALATION INTERCEPT
        # Scan ALL messages for the trigger — the LLM rewrites the final message
        # in its own words, so the trigger only exists in the tool result message.
        # The tool includes the language: [ESCALATE_TRIGGERED:darija]
        # NEW
        try:
            last_tool = next(
                (m for m in reversed(result["messages"]) if isinstance(m, ToolMessage)),
                None
            )
            if last_tool and "[ESCALATE_TRIGGERED" in last_tool.content and phone not in _escalated:
                _escalated.add(phone)
                lang = "english"
                try:
                    lang = last_tool.content.split("[ESCALATE_TRIGGERED:")[1].split("]")[0].lower()
                except Exception as e:
                    logger.warning(f"Could not parse escalation language for {phone}: {e}. Defaulting to english.")
                reply = ESCALATION_MSGS.get(lang, ESCALATION_MSGS["english"])
                logger.info(f"Escalation triggered for {phone} in lang={lang}")
        except Exception as esc_err:
            logger.error(f"Escalation intercept failed for {phone}: {esc_err}", exc_info=True)

    except TimeoutError as e:
        logger.error(f"Agent timed out for {phone}: {e}", exc_info=True)
        reply = (
            "طلبك خد وقت بزاف — جرب عاود من جديد 🙏\n"
            "Ça a pris trop de temps — veuillez réessayer 🙏\n"
            "That took too long. Please try again."
        )
    except Exception as e:
        err_str = str(e).lower()
        logger.error(f"Agent error for {phone}: {e}", exc_info=True)

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

    _send(phone, reply)
    return {"status": "ok"}


def _get_msg_id(data):
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        return hashlib.sha256(f"{msg['id']}:{msg['timestamp']}".encode()).hexdigest()[:16]
    except:
        return None


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