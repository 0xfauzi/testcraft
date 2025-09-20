from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class LlmPort(Protocol):
    """Port interface for llm operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
