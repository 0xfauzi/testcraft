"""Compatibility shim for OpenAI adapter - preserves existing imports."""

# Re-export the OpenAI adapter from the new package structure
from .openai.adapter import OpenAIAdapter

# Re-export the error class for backward compatibility
from .openai.client import OpenAIError

__all__ = ["OpenAIAdapter", "OpenAIError"]
