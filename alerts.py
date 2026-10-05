"""
Centralized alerting — all notifications route through send_alert() via Telegram.
"""
import os
import re
import time
import logging
import threading
import requests

logger = logging.getLogger(__name__)

# Rate-limit: max 1 alert per key every 15 minutes
_last_sent: dict[str, float] = {}
_COOLDOWN = 900  # seconds

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STORE_NAME = os.getenv("STORE_NAME", "ElectroMA")

# True only if both required Telegram vars are present
_TELEGRAM_CONFIGURED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _mask_phones(text: str) -> str:
    """Replace phone numbers with masked versions showing only last 4 digits."""
    return re.sub(r"\b(\+?\d{5,})\b", lambda m: "***" + m.group(1)[-4:], text)


def _send_telegram(subject: str, body: str):
    """Blocking Telegram send — always called from a background thread."""
    try:
        masked_body = _mask_phones(body)
        message = f"🚨 [{STORE_NAME}] {subject}\n\n{masked_body}"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.error(f"Telegram alert failed [{resp.status_code}]: {resp.text}")
        else:
            logger.info(f"Telegram alert sent: {subject}")
    except Exception as e:
        logger.error(f"Telegram alert error: {e}", exc_info=True)


def send_alert(key: str, subject: str, body: str, sync: bool = False):
    """
    Fire-and-forget alert (or blocking if sync=True). Never raises, never blocks the caller.
    - key: dedup key for rate-limiting (e.g. "wa_send_fail", "agent_crash", "kb_ingest_fail")
    - subject: alert title
    - body: alert message (phone numbers are auto-masked)
    - sync: if True, waits for the background thread to finish before returning
    Skips silently if Telegram is not configured or cooldown hasn't elapsed.
    """
    if not _TELEGRAM_CONFIGURED:
        return

    now = time.time()
    if now - _last_sent.get(key, 0) < _COOLDOWN:
        return
    _last_sent[key] = now

    thread = threading.Thread(target=_send_telegram, args=(subject, body), daemon=True)
    thread.start()
    if sync:
        thread.join(timeout=15)


