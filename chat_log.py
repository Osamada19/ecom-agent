"""Simple chat log for the owner dashboard.

Why this file exists, in plain words:
  LangGraph saves the agent conversation in memory.db, but it saves it
  as a whole checkpoint blob. That blob is hard to query like a table.
  So we do NOT read memory.db for the dashboard.

  Instead, main.py writes one small row here for every message:
  one row when a customer message arrives, one row when the agent
  reply is actually sent. This file is the only place the dashboard
  reads from. That is the main idea of the whole dashboard.

Database file:
  chats.db, in the app folder, unless CHATS_DB_PATH is set.
  Never commit chats.db to GitHub. It holds real customer chats.
  On Railway it must live on a persistent volume, or a redeploy
  wipes it, exactly like memory.db.
"""

import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("CHATS_DB_PATH", "chats.db")


def _connect():
    """Open a fresh connection. One connection per call, so no
    connection is shared between the webhook thread and the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL = the bot can write while the dashboard reads, without locking.
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create the messages table if it does not exist yet.
    Safe to call on every startup. It never deletes old rows."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT NOT NULL,   -- customer phone, same as LangGraph thread_id
                direction  TEXT NOT NULL,   -- 'customer', 'agent', or 'system'
                text       TEXT NOT NULL,
                created_at TEXT NOT NULL,   -- UTC, ISO format, e.g. 2026-10-28T01:30:00
                seen       INTEGER NOT NULL DEFAULT 0  -- 1 = owner already opened this thread
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_time ON messages(thread_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seen ON messages(thread_id, seen)")
        conn.commit()
    finally:
        conn.close()


def log_message(thread_id, direction, text):
    """Write one message row. Called from main.py, never from the browser.

    Customer messages start as unseen (seen = 0), so they make the
    green dot appear. Agent replies start as seen = 1, because the
    owner does not need to act on his own bot's reply. The green dot
    means 'a customer is waiting', not 'the bot talked'.
    """
    if not thread_id or not text:
        return
    from datetime import datetime, timezone

    init_db()
    conn = _connect()
    try:
        seen = 0 if direction == "customer" else 1
        conn.execute(
            "INSERT INTO messages (thread_id, direction, text, created_at, seen)"
            " VALUES (?, ?, ?, ?, ?)",
            (thread_id, direction, str(text),
             datetime.now(timezone.utc).isoformat(timespec="seconds"), seen),
        )
        conn.commit()
    finally:
        conn.close()


def get_threads():
    """One row per customer, for the sidebar.

    Sorted by last message time, newest first, like WhatsApp Web.
    An old thread that gets a new message jumps to the top, because
    its last message time just became the newest.
    """
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, thread_id, direction, text, created_at, seen"
            " FROM messages ORDER BY created_at ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()

    threads = {}
    for row in rows:
        thread = threads.setdefault(row["thread_id"], {
            "thread_id": row["thread_id"],
            "last_text": "",
            "last_direction": "",
            "last_at": "",
            "last_id": 0,
            "unread_count": 0,
        })
        # Rows arrive oldest first, so each row overwrites the last
        # message. At the end, the last message is the newest one.
        thread["last_text"] = row["text"]
        thread["last_direction"] = row["direction"]
        thread["last_at"] = row["created_at"]
        thread["last_id"] = row["id"]
        if row["direction"] == "customer" and row["seen"] == 0:
            thread["unread_count"] += 1

    # Newest thread first. last_id breaks ties when two messages
    # land in the same second, so the truly latest thread wins.
    ordered = sorted(threads.values(),
                     key=lambda t: (t["last_at"], t["last_id"]), reverse=True)
    for thread in ordered:
        del thread["last_id"]
    return ordered


def get_messages(thread_id):
    """All messages of one thread, oldest first, for the main pane."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, direction, text, created_at, seen FROM messages"
            " WHERE thread_id = ? ORDER BY created_at ASC, id ASC",
            (thread_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def mark_thread_seen(thread_id):
    """Owner opened this thread, so clear its green dot.
    Marks every message in the thread as seen, not just customer ones."""
    init_db()
    conn = _connect()
    try:
        conn.execute("UPDATE messages SET seen = 1 WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()
