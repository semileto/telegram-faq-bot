"""SQLite storage for order requests and questions the bot could not answer."""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

PHONE_RE = re.compile(r"^\+?\d{10,15}$")


def normalize_phone(raw: str) -> str | None:
    """'+38 (067) 123-45-67' -> '+380671234567'; None if it doesn't look like a phone number."""
    digits = re.sub(r"[\s\-()]", "", raw.strip())
    return digits if PHONE_RE.match(digits) else None


class Storage:
    def __init__(self, path: str = "bot.db"):
        self.path = path
        with closing(self._conn()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER, name TEXT, phone TEXT, item TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS unanswered (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER, question TEXT, best_score REAL, created_at TEXT);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_order(self, user_id: int, name: str, phone: str, item: str) -> int:
        with closing(self._conn()) as conn, conn:
            cur = conn.execute("INSERT INTO orders (user_id, name, phone, item, created_at) VALUES (?,?,?,?,?)",
                               (user_id, name, phone, item, self._now()))
            return cur.lastrowid

    def log_unanswered(self, user_id: int, question: str, best_score: float) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("INSERT INTO unanswered (user_id, question, best_score, created_at) VALUES (?,?,?,?)",
                         (user_id, question, best_score, self._now()))

    def orders(self) -> list[dict]:
        with closing(self._conn()) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM orders ORDER BY id")]

    def top_unanswered(self, limit: int = 10) -> list[dict]:
        """Most frequent questions the bot failed on: what to add to the knowledge base next."""
        with closing(self._conn()) as conn:
            rows = conn.execute("""SELECT lower(question) AS question, COUNT(*) AS times
                                   FROM unanswered GROUP BY lower(question)
                                   ORDER BY times DESC LIMIT ?""", (limit,))
            return [dict(r) for r in rows]
