"""Dashboard routes. Plug this into the existing FastAPI app in main.py.

Usage in main.py (see INTEGRATION.md for the full steps):
    from dashboard import register_dashboard
    register_dashboard(app)

That adds only these routes, nothing else:
    GET  /dashboard                          -> the HTML page
    GET  /dashboard/api/threads              -> sidebar list
    GET  /dashboard/api/threads/{phone}/messages -> one chat
    POST /dashboard/api/threads/{phone}/seen -> clear the green dot

Auth, in plain words:
  A simple browser username/password check, using the password in
  the DASHBOARD_PASSWORD env var. The browser will ask for it once
  and remember it for the session. If the env var is not set, the
  dashboard refuses to open at all, so it can never be open by
  accident on a public Railway URL. Never put this password in code
  or in GitHub. One password per store, set in that store's Railway
  env vars.
"""

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from chat_log import get_messages, get_threads, mark_thread_seen

HTML_PATH = Path(__file__).parent / "dashboard.html"


def _authorized(request: Request) -> bool:
    password = os.getenv("DASHBOARD_PASSWORD", "")
    if not password:
        return False
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode()
    except Exception:
        return False
    # Format is "username:password". Username is not checked, any
    # username works, only the password matters, so the owner can
    # type anything as the username in the browser prompt.
    if ":" not in decoded:
        return False
    _, supplied = decoded.split(":", 1)
    return secrets.compare_digest(supplied, password)


def _need_auth():
    return JSONResponse(
        {"detail": "Dashboard password required. Set DASHBOARD_PASSWORD."},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Owner dashboard"'},
    )


def register_dashboard(app: FastAPI):
    @app.get("/dashboard")
    async def dashboard_page(request: Request):
        if not _authorized(request):
            return _need_auth()
        return FileResponse(HTML_PATH)

    @app.get("/dashboard/api/threads")
    async def api_threads(request: Request):
        if not _authorized(request):
            return _need_auth()
        return get_threads()

    @app.get("/dashboard/api/threads/{phone}/messages")
    async def api_messages(request: Request, phone: str):
        if not _authorized(request):
            return _need_auth()
        # Reading does NOT mark as seen on its own. The browser calls
        # the seen endpoint after it shows the chat, so a background
        # refresh of a thread you are not looking at cannot clear
        # another thread's dot.
        return get_messages(phone)

    @app.post("/dashboard/api/threads/{phone}/seen")
    async def api_seen(request: Request, phone: str):
        if not _authorized(request):
            return _need_auth()
        mark_thread_seen(phone)
        return {"ok": True}
