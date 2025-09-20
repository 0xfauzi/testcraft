from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class ParserPort(Protocol):
    """Port interface for parser operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
