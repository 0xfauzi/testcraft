from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class UiPort(Protocol):
    """Port interface for ui operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
