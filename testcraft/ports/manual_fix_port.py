"""
Ports for Manual Fix Guidance.

Defines the guidance port (use-case facing) and presenter port (UI/CLI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..domain.manual_fix import ManualFixRecommendation, ManualFixRequest


class ManualFixGuidancePort(ABC):
    """Port to request manual-fix guidance and return a structured recommendation."""

    @abstractmethod
    async def request_guidance(
        self, request: ManualFixRequest
    ) -> ManualFixRecommendation:
        """Obtain guidance from LLM/manual-fix pipeline."""
        raise NotImplementedError


class ManualFixPresenterPort(Protocol):
    """Presenter for previewing and accepting manual-fix recommendations (CLI/TUI)."""

    def present(
        self, recommendation: ManualFixRecommendation, *, dry_run: bool = False
    ) -> bool:  # noqa: D401
        """Return True if user accepts the recommendation; False otherwise."""
        ...
