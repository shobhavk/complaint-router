"""
FeedbackStore — lightweight SQLite persistence for human corrections.
In production, swap for a vector store or fine-tuning pipeline.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


DB_PATH = Path("feedback.db")


class FeedbackStore:
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
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    complaint_text TEXT NOT NULL,
                    original_category TEXT,
                    corrected_category TEXT,
                    corrected_priority TEXT,
                    corrected_team TEXT,
                    created_at TEXT
                )
            """)

    def save(
        self,
        text: str,
        original_category: str,
        corrected_category: str,
        corrected_priority: str,
        corrected_team: str | None = None,
    ) -> str:
        record_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback VALUES (?,?,?,?,?,?,?)""",
                (
                    record_id, text, original_category,
                    corrected_category, corrected_priority,
                    corrected_team, datetime.utcnow().isoformat(),
                ),
            )
        return record_id

    def get_recent_examples(self, n: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [
            {
                "text": r["complaint_text"],
                "category": r["corrected_category"],
                "priority": r["corrected_priority"],
            }
            for r in rows
        ]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {"total_corrections": total}
