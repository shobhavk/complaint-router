"""
Stateful Session Persistence — Enhancement #4.
Saves/loads routing history and incident data to SQLite on disk.
On app start, hydrates the dashboard with previous session data.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("session.db")


class SessionStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    complaint_id TEXT,
                    saved_at    TEXT,
                    data_json   TEXT
                )
            """)

    def save_incident(self, record: dict) -> None:
        """Persist a single routing history record."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO routing_history (complaint_id, saved_at, data_json) VALUES (?,?,?)",
                (record.get("ID", "-"), datetime.utcnow().isoformat(), json.dumps(record)),
            )
        logger.debug("[DB] State saved for %s", record.get("ID"))

    def load_history(self, limit: int = 200) -> list[dict]:
        """Load previous session records, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data_json FROM routing_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        records = []
        for row in rows:
            try:
                records.append(json.loads(row["data_json"]))
            except Exception:
                pass
        logger.info("[DB] Loaded %d records from previous session", len(records))
        return records

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM routing_history")
        logger.info("[DB] Session cleared")

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM routing_history").fetchone()[0]
