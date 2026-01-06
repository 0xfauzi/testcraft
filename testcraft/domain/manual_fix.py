"""
Domain models for Manual Fix Guidance.

Defines immutable Pydantic models to carry manual-fix requests and
recommendations, including minimal metadata for provenance and hashing.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ManualFixContextAttachment(BaseModel):
    """Attachment describing the LLM/prompt context used for guidance."""

    prompt_hash: str = Field(..., description="Hash of the effective prompt content")
    model_id: str | None = Field(None, description="Model identifier used for guidance")
    params: dict[str, Any] = Field(default_factory=dict, description="LLM parameters")
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Timestamp (UTC)"
    )

    model_config = ConfigDict(frozen=True)


class ManualFixRequest(BaseModel):
    """Request payload for obtaining manual-fix guidance."""

    project_root: Path = Field(..., description="Project root directory")
    target_file: Path = Field(
        ..., description="Source file containing target under test"
    )
    target_object: str = Field(
        ..., description="Target object (e.g., 'Class.method' or 'function')"
    )
    trace_excerpt: str = Field(default="", description="Optional traceback excerpt")
    notes: str = Field(
        default="", description="Additional notes for prompt customization"
    )

    model_config = ConfigDict(frozen=True)


class ManualFixRecommendation(BaseModel):
    """Structured manual-fix recommendation returned from LLM guidance."""

    test_code: str = Field(
        ..., description="Deliberately failing pytest module content"
    )
    bug_note_markdown: str = Field(..., description="Markdown BUG NOTE content")
    attachment: ManualFixContextAttachment | None = Field(
        default=None, description="Model/prompt metadata attachment"
    )
    recommendation_hash: str = Field(
        ..., description="Deterministic hash of the recommendation"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Timestamp (UTC)"
    )

    @staticmethod
    def compute_hash(
        test_code: str, bug_note_markdown: str, prompt_hash: str | None = None
    ) -> str:
        """Compute a short stable hash for deduplication and artifact naming."""
        hasher = sha256()
        hasher.update(test_code.encode("utf-8"))
        hasher.update(bug_note_markdown.encode("utf-8"))
        if prompt_hash:
            hasher.update(prompt_hash.encode("utf-8"))
        return hasher.hexdigest()[:16]

    model_config = ConfigDict(frozen=True)
