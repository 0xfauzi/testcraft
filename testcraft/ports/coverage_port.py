from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class CoveragePort(Protocol):
    """Port interface for coverage operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
