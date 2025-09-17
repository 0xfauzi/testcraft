"""
Prompt template rendering utilities for deterministic output.

This module provides safe template rendering with JSON serialization and
error-resistant formatting to ensure deterministic prompt generation.
"""

from __future__ import annotations

import json
from typing import Any


class PromptError(Exception):
    """Raised when prompt generation, customization, or validation fails."""


def to_pretty_json(value: Any) -> str:
    """
    Convert value to pretty JSON string with fallback to string representation.
    
    Args:
        value: Any value to convert to JSON
        
    Returns:
        Pretty JSON string or string representation if not JSON-serializable
    """
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        # Fallback to string representation if not JSON-serializable
        return str(value)


def render_template(template: str, values: dict[str, Any]) -> str:
    """
    Safely render a template with the given values.
    
    Uses a safe replacement strategy to avoid KeyErrors and template execution.
    Non-string values are converted to pretty JSON or string representation.
    
    Args:
        template: Template string with {key} placeholders
        values: Dictionary of values to substitute
        
    Returns:
        Rendered template string
        
    Raises:
        PromptError: If template rendering fails
    """
    # Use a simple safe replacement to avoid KeyErrors and avoid executing templates
    class SafeDict(dict):
        def __missing__(self, key: str) -> str:  # type: ignore[override]
            return "{" + key + "}"

    # Convert non-str to strings safely (pretty JSON for dict-like values)
    prepared: dict[str, str] = {}
    for k, v in values.items():
        if isinstance(v, dict | list):
            prepared[k] = to_pretty_json(v)
        else:
            prepared[k] = str(v)

    try:
        return template.format_map(SafeDict(prepared))
    except Exception as exc:
        raise PromptError(f"Failed to render template: {exc}") from exc
