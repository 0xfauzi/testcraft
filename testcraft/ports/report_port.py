from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class ReportPort(Protocol):
    """Port interface for report operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
