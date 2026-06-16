"""
Governance Audit Log — Enhancement #1c.
Persists a metadata record for every processed complaint.
"""
from __future__ import annotations
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("audit.db")


class AuditLog:
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
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          TEXT PRIMARY KEY,
                    complaint_id TEXT,
                    timestamp   TEXT,
                    model_used  TEXT,
                    token_count INTEGER,
                    confidence  REAL,
                    category    TEXT,
                    priority    TEXT,
                    team        TEXT,
                    pii_detected TEXT,
                    injection_flag INTEGER,
                    security_reason TEXT,
                    auto_routed INTEGER,
                    error       TEXT
                )
            """)

    def record(self, state: dict, screening: dict | None = None, model: str = "unknown") -> None:
        import uuid
        c = state.get("classification")
        r = state.get("routing")
        pii = json.dumps((screening or {}).get("pii_detected", []))
        injection = 1 if (screening and not screening.get("safe")) else 0
        sec_reason = (screening or {}).get("reason", "")

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                str(uuid.uuid4()),
                state.get("complaint_id", "-"),
                datetime.utcnow().isoformat(),
                model,
                0,  # token count placeholder (hook into LangSmith for real value)
                c.confidence if c else 0.0,
                c.category if c else None,
                c.priority.value if c else None,
                r.team.value if r else None,
                pii,
                injection,
                sec_reason,
                1 if (r and r.auto_routed) else 0,
                state.get("error"),
            ))
        logger.debug("[DB] Audit record saved for %s", state.get("complaint_id"))

    def get_recent(self, n: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]
