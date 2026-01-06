"""
Planning port interface for test generation planning operations.

This module defines the port (interface) for planning operations, including
plan generation, persistence, retrieval, and listing with filtering capabilities.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..domain.models import ContextPack, PlanningRequest, PlanningResult, PlanOption


class PlanningPort(ABC):
    """
    Port for planning operations.

    This interface defines the contract for planning adapters, enabling
    plan generation via LLM, persistence to storage, and retrieval for
    later use in test generation workflows.
    """

    @abstractmethod
    async def generate_plan(
        self, request: PlanningRequest, context_pack: ContextPack
    ) -> PlanOption:
        """
        Generate a plan using LLM.

        Args:
            request: Planning request with target information
            context_pack: Context pack for the target

        Returns:
            Generated plan option with hash and metadata

        Raises:
            PlanningError: If plan generation fails
        """
        pass

    @abstractmethod
    async def persist_plan(self, result: PlanningResult) -> str:
        """
        Persist plan with acceptance status and edit history.

        Args:
            result: Planning result to persist

        Returns:
            Artifact ID for the persisted plan

        Raises:
            PlanningError: If persistence fails
        """
        pass

    @abstractmethod
    async def retrieve_plan(self, plan_id: str) -> PlanningResult | None:
        """
        Retrieve a persisted plan by ID.

        Args:
            plan_id: SHA-256 hash of plan content

        Returns:
            Planning result if found, None otherwise

        Raises:
            PlanningError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_plans(
        self, target_file: Path | None = None, accepted_only: bool = False
    ) -> list[PlanningResult]:
        """
        List plans with optional filtering.

        Args:
            target_file: Optional filter by target file path
            accepted_only: If True, only return accepted plans

        Returns:
            List of planning results matching filters

        Raises:
            PlanningError: If listing fails
        """
        pass


class PlanningError(Exception):
    """Exception raised when planning operations fail."""

    pass
