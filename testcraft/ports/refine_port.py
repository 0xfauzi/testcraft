from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class RefinePort(Protocol):
    """Port interface for refine operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
