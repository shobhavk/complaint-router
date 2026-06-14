"""
Pydantic schemas — single source of truth for the complaint data model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Priority(str, Enum):
    P1 = "P1"  # Critical / financial loss — SLA 1h
    P2 = "P2"  # Service disruption — SLA 4h
    P3 = "P3"  # Inconvenience — SLA 8h
    P4 = "P4"  # Feedback / general — SLA 24h


class Team(str, Enum):
    BILLING = "Billing"
    TECH_SUPPORT = "Tech Support"
    ACCOUNT_MGMT = "Account Management"
    PRODUCT = "Product"
    ESCALATION = "Escalation"
    GENERAL = "General"
    REVIEW_QUEUE = "Review Queue"


class ComplaintInput(BaseModel):
    """Raw input before normalisation."""
    raw: str = Field(..., description="Raw complaint in any format")
    source_format: str = Field(default="unknown", description="csv|json|excel|text|unknown")


class ClassifiedComplaint(BaseModel):
    complaint_id: str
    category: str
    priority: Priority
    sentiment: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    sla_deadline: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    complaint_id: str
    team: Team
    channel: str = "slack"
    assigned_at: datetime
    sla_deadline: datetime
    auto_routed: bool = True
    notes: str = ""


class FeedbackRecord(BaseModel):
    """Persisted when a human corrects an agent decision."""
    id: str
    complaint_text: str
    original_category: str
    corrected_category: str
    corrected_priority: str
    corrected_team: str | None
    created_at: datetime = Field(default_factory=datetime.utcnow)
