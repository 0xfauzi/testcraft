"""
Telemetry adapters for different observability backends.

This module contains implementations for various telemetry backends
including OpenTelemetry, Datadog, and others, plus cost management
and routing/factory functionality.
"""

from .opentelemetry_adapter import OpenTelemetryAdapter
from .noop_adapter import NoOpTelemetryAdapter
from .cost_manager import CostManager
from .router import (
    create_telemetry_adapter,
    register_telemetry_adapter,
    get_available_backends,
    create_telemetry_context,
    TelemetryContextManager
)

__all__ = [
    "OpenTelemetryAdapter",
    "NoOpTelemetryAdapter",
    "CostManager",
    "create_telemetry_adapter",
    "register_telemetry_adapter",
    "get_available_backends",
    "create_telemetry_context",
    "TelemetryContextManager",
]
