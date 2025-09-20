from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class PromptPort(Protocol):
    """Port interface for prompt operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
