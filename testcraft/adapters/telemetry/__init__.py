"""
Telemetry adapters for different observability backends.

This module contains implementations for various telemetry backends
including OpenTelemetry, Datadog, and others, plus cost management
and routing/factory functionality.
"""

from .cost_manager import CostManager
from .noop_adapter import NoOpTelemetryAdapter
from .opentelemetry_adapter import OpenTelemetryAdapter
from .router import (
    TelemetryContextManager,
    create_telemetry_adapter,
    create_telemetry_context,
    get_available_backends,
    register_telemetry_adapter,
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
