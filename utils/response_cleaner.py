"""
Utility to clean LLM responses before JSON parsing.
Handles Qwen3 <think>...</think> blocks, markdown fences, and whitespace.
"""
import re
import json


def extract_json(raw: str) -> str:
    """
    Strips Qwen3 thinking blocks, markdown fences, and extracts
    the first valid JSON object or array from the response.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from model")

    # 1. Remove <think>...</think> blocks (Qwen3 chain-of-thought)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 2. Remove markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    if not text:
        raise ValueError("Response was empty after stripping think blocks")

    # 3. Try parsing as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 4. Extract first {...} or [...] block
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]!r}")
