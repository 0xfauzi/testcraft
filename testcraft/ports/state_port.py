from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class StatePort(Protocol):
    """Port interface for state operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
