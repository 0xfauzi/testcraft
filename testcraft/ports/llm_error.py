"""Unified error type for LLM provider failures.

This module defines `LLMError`, a provider-agnostic exception that all
LLM adapters should raise at their public boundary. It preserves the
original provider exception via Python's exception chaining (``from e``)
and carries normalized context useful for retries, telemetry, and UX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class LLMError(Exception):
    """Provider-agnostic LLM error with normalized context.

    Attributes:
        message: Human-friendly error summary.
        provider: Provider key (e.g., "openai", "anthropic", "azure-openai", "bedrock").
        operation: High-level operation (e.g., "generate_tests", "analyze_code",
            "refine_content", "generate_test_plan", "chat_completion", "responses").
        model: The model/deployment identifier used for the request.
        status_code: Optional HTTP/status code if available.
        metadata: Additional structured details (request ids, api_version, etc.).
    """

    message: str
    provider: str | None = None
    operation: str | None = None
    model: str | None = None
    status_code: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts: list[str] = []
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.operation:
            parts.append(f"op={self.operation}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        ctx = (" ".join(parts)) if parts else ""
        if ctx:
            return f"LLMError({ctx}): {self.message}"
        return f"LLMError: {self.message}"


