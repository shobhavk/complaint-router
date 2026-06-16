"""
Guardrails — Security, PII masking, prompt injection detection.
Enhancement #1.
"""
from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

# ── PII patterns ──────────────────────────────────────────────────────────────
PII_PATTERNS: list[tuple[str, str, str]] = [
    ("credit_card",   r"\b(?:\d[ -]?){13,16}\b",                          "[CARD-MASKED]"),
    ("ssn",           r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",                "[SSN-MASKED]"),
    ("api_key",       r"\b(sk-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9]+)\b","[KEY-MASKED]"),
    ("email",         r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z]{2,}\b","[EMAIL-MASKED]"),
    ("phone",         r"\b(\+?\d[\d\s\-().]{7,}\d)\b",                    "[PHONE-MASKED]"),
    ("ip_address",    r"\b\d{1,3}(?:\.\d{1,3}){3}\b",                    "[IP-MASKED]"),
]

# ── Prompt injection patterns ─────────────────────────────────────────────────
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"you\s+are\s+now\s+(a\s+)?(?!customer)",
        r"act\s+as\s+(if\s+you\s+are\s+)?(?!a\s+customer)",
        r"disregard\s+(your\s+)?(previous|prior|all)\s+(instructions?|rules?|constraints?)",
        r"new\s+instructions?:",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\[\s*system\s*\]",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
    ]
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Mask PII in text. Returns (masked_text, list_of_detected_types).
    """
    detected: list[str] = []
    result = text
    for pii_type, pattern, replacement in PII_PATTERNS:
        new_result, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
        if count > 0:
            detected.append(pii_type)
            result = new_result
    if detected:
        logger.info("[SECURITY] PII masked: %s", detected)
    return result, detected


def detect_injection(text: str) -> tuple[bool, str]:
    """
    Detect prompt injection attempts.
    Returns (is_injection, matched_pattern).
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            logger.warning("[SECURITY] Prompt injection detected: %r", match.group(0))
            return True, match.group(0)
    return False, ""


def screen_input(text: str) -> dict:
    """
    Full security screening pipeline.
    Returns a screening result dict consumed by the agent pipeline.
    """
    is_injection, injection_match = detect_injection(text)
    if is_injection:
        return {
            "safe": False,
            "reason": "prompt_injection",
            "matched": injection_match,
            "masked_text": text,
            "pii_detected": [],
        }

    masked_text, pii_detected = mask_pii(text)
    return {
        "safe": True,
        "reason": None,
        "matched": None,
        "masked_text": masked_text,
        "pii_detected": pii_detected,
    }
