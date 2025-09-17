"""
Prompt sanitization utilities for reducing injection surface.

This module provides utilities to safely sanitize user-provided text and code
content before embedding in prompts, helping to reduce prompt injection risks.
"""

from __future__ import annotations

import re


def sanitize_text(text: str) -> str:
    """
    Apply light sanitization to reduce prompt injection surface.
    - Normalizes potentially problematic sequences
    - Removes control sequences commonly used in jailbreak attempts
    """
    # Handle non-string inputs by converting to string first
    if not isinstance(text, str):
        text = str(text)
        
    # Remove null bytes and non-printable chars
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    # Collapse multiple backticks to at most triple to preserve formatting
    text = re.sub(r"`{4,}", "```", text)
    # Remove common injection phrases while preserving content meaning
    blocked = [
        "ignore previous instructions",
        "disregard previous instructions",
        "override system prompt",
        "act as system",
    ]
    lowered = text.lower()
    for phrase in blocked:
        lowered = lowered.replace(phrase, "")
    return lowered


def sanitize_code(code: str) -> str:
    """
    Apply minimal sanitization to code content while preserving case sensitivity.
    - Removes control sequences that could break formatting
    - Preserves all case-sensitive identifiers and keywords
    - Does NOT remove injection phrases (code legitimately contains these patterns)
    """
    # Remove null bytes and non-printable chars that could break code formatting
    code = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", code)
    # Collapse multiple backticks to at most triple to preserve code block formatting
    code = re.sub(r"`{4,}", "```", code)
    # Return original case-preserved code
    return code
