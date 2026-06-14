"""Utility: SLA deadline calculator."""
from __future__ import annotations
from datetime import datetime, timedelta
from schemas.complaint import Priority


class SLACalculator:
    SLA_HOURS: dict[Priority, int] = {
        Priority.P1: 1,
        Priority.P2: 4,
        Priority.P3: 8,
        Priority.P4: 24,
    }

    def calculate(self, priority: Priority, category: str = "") -> datetime:
        hours = self.SLA_HOURS.get(priority, 8)
        # Billing P1/P2 gets tighter SLA
        if category == "billing" and priority == Priority.P2:
            hours = 2
        return datetime.utcnow() + timedelta(hours=hours)
