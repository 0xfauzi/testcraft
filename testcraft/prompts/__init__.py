"""
Prompt templates and registry with versioning.

This package provides a versioned prompt registry that supplies system prompts,
user prompt templates, JSON schemas, and safe template rendering for TestCraft.
"""

from __future__ import annotations

# Import all components from registry for backward compatibility
from .registry import (
    PromptRegistry,
    PromptError,
    SchemaDefinition,
    sanitize_text,
    sanitize_code,
)

# Re-export everything for stable imports
__all__ = [
    "PromptRegistry",
    "PromptError", 
    "SchemaDefinition",
    "sanitize_text",
    "sanitize_code",
]