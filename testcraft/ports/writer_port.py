from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class WriterPort(Protocol):
    """Port interface for writer operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
