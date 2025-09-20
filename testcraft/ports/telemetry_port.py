from abc import abstractmethod
from typing import Any, Protocol

"""Module docstring."""


class TelemetryPort(Protocol):
    """Port interface for telemetry operations."""

    @abstractmethod
    def placeholder_method(self) -> Any:
        """Placeholder method that needs implementation."""
        pass
