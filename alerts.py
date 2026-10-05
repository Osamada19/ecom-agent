"""
Centralized alerting — all notifications route through send_alert().
Swap this file later for Sentry, PagerDuty, or a fleet dashboard.
"""
import os
import re
import time
import logging
import smtplib
import threading
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# Rate-limit: max 1 email per key every 15 minutes
_last_sent: dict[str, float] = {}
_COOLDOWN = 900  # seconds

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
STORE_NAME = os.getenv("STORE_NAME", "ElectroMA")

# True only if all required SMTP vars are present
_SMTP_CONFIGURED = all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO])


def _mask_phones(text: str) -> str:
    """Replace phone numbers with masked versions showing only last 4 digits."""
    return re.sub(r"\b(\+?\d{5,})\b", lambda m: "***" + m.group(1)[-4:], text)


def _send_email(subject: str, body: str):
    """Blocking email send — always called from a background thread."""
    try:
        msg = MIMEText(_mask_phones(body))
        msg["Subject"] = f"[{STORE_NAME}] {subject}"
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL_TO

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [ALERT_EMAIL_TO], msg.as_string())
        logger.info(f"Alert email sent: {subject}")
    except Exception as e:
        logger.error(f"Alert email failed: {e}", exc_info=True)


def send_alert(key: str, subject: str, body: str):
    """
    Fire-and-forget alert. Never raises, never blocks the caller.
    - key: dedup key for rate-limiting (e.g. "wa_send_fail", "agent_crash")
    - subject: email subject line
    - body: email body (phone numbers are auto-masked)
    Skips silently if SMTP is not configured or cooldown hasn't elapsed.
    """
    if not _SMTP_CONFIGURED:
        return

    now = time.time()
    if now - _last_sent.get(key, 0) < _COOLDOWN:
        return
    _last_sent[key] = now

    threading.Thread(target=_send_email, args=(subject, body), daemon=True).start()

