from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class ContextPort(Protocol):
    """Port interface for context operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
