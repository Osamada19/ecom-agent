import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from graph import graph
from langchain_core.messages import HumanMessage
from ingest import ingest

# Populate ChromaDB on every startup
ingest()

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

    try:
        entry = data["entry"][0]["changes"][0]["value"]

        # ✅ FIX 1: ignore status updates (delivered, read, failed)
        if "messages" not in entry:
            return {"status": "ignored"}

        message = entry["messages"][0]

        # ✅ FIX 2: ignore non-text messages (images, reactions, etc.)
        if message.get("type") != "text":
            return {"status": "ignored"}

        from_number = message["from"]
        text = message["text"]["body"]

    except (KeyError, IndexError):
        return {"status": "ignored"}

    config = {"configurable": {"thread_id": from_number}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=config
    )

    reply = result["messages"][-1].content
    send_message(from_number, reply)
    return {"status": "ok"}


def send_message(to: str, text: str):
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
    response = requests.post(url, json=payload, headers=headers)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)