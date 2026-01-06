"""
Planning presenter port interface for user interaction.

This module defines the port (interface) for presenting plans to users
and collecting acceptance decisions, enabling different UI implementations
(CLI, Textual, web, etc.).
"""

from abc import ABC, abstractmethod

from ..domain.models import PlanOption


class PlanningPresenterPort(ABC):
    """
    Port for presenting plans to users.

    This interface defines the contract for plan presentation adapters,
    enabling different UI implementations while maintaining a consistent
    planning workflow.
    """

    @abstractmethod
    def present_plan(self, plan_option: PlanOption) -> None:
        """
        Display plan to user for review.

        Args:
            plan_option: Plan option to present

        Raises:
            PresentationError: If display fails
        """
        pass

    @abstractmethod
    async def prompt_acceptance(self) -> tuple[bool, str | None]:
        """
        Prompt user to accept/reject/edit plan.

        Returns:
            Tuple of (accepted, edit_text) where:
            - accepted: True if user accepts the plan
            - edit_text: Optional user modifications (included in edit history)

        Raises:
            PresentationError: If prompt fails
        """
        pass


class PresentationError(Exception):
    """Exception raised when plan presentation fails."""

    pass
