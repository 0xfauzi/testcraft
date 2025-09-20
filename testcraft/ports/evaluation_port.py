from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class EvaluationPort(Protocol):
    """Port interface for evaluation operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass


__version__ = "0.1.0"
