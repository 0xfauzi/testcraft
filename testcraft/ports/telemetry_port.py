"""
Telemetry Port interface definition.

This module defines the interface for telemetry operations, including
tracing, metrics collection, and observability features.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from typing_extensions import Protocol


class SpanKind(Enum):
    """Types of telemetry spans."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """Context information for a telemetry span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, Any] | None = None


@dataclass
class MetricValue:
    """A metric value with metadata."""

    name: str
    value: int | float
    unit: str | None = None
    labels: dict[str, str] | None = None
    timestamp: datetime | None = None


class TelemetryPort(Protocol):
    """
    Interface for telemetry operations.

    This protocol defines the contract for telemetry backends, supporting
    different providers like OpenTelemetry, Datadog, New Relic, etc.
    """

    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent_context: SpanContext | None = None,
    ) -> AbstractContextManager[SpanContext]:
        """
        Create a new telemetry span.

        Args:
            name: Name of the operation being traced
            kind: Type of span being created
            attributes: Key-value attributes to attach to the span
            parent_context: Parent span context for nested spans

        Returns:
            Context manager that yields the span context

        Usage:
            with telemetry.create_span("llm_call", attributes={"model": "o4-mini"}) as span:
                # Do work here
                span.set_attribute("tokens_used", 150)
        """
        ...

    def record_metric(self, metric: MetricValue) -> None:
        """
        Record a metric value.

        Args:
            metric: The metric value to record

        Usage:
            telemetry.record_metric(MetricValue(
                name="tests_generated",
                value=5,
                labels={"framework": "pytest"}
            ))
        """
        ...

    def record_metrics(self, metrics: list[MetricValue]) -> None:
        """
        Record multiple metric values efficiently.

        Args:
            metrics: List of metric values to record
        """
        ...

    def increment_counter(
        self,
        name: str,
        value: int | float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Amount to increment (default: 1)
            labels: Optional labels for the counter
        """
        ...

    def record_histogram(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record a histogram value (for timing, sizes, etc).

        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels for the histogram
        """
        ...

    def record_gauge(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Set a gauge value (for current state metrics).

        Args:
            name: Gauge name
            value: Current value to set
            labels: Optional labels for the gauge
        """
        ...

    def set_global_attributes(self, attributes: dict[str, Any]) -> None:
        """
        Set attributes that will be attached to all spans and metrics.

        Args:
            attributes: Global attributes (e.g., service version, environment)
        """
        ...

    def flush(self, timeout_seconds: float | None = None) -> bool:
        """
        Force flush all pending telemetry data.

        Args:
            timeout_seconds: Maximum time to wait for flush completion

        Returns:
            True if flush completed successfully, False otherwise
        """
        ...

    def is_enabled(self) -> bool:
        """
        Check if telemetry is currently enabled.

        Returns:
            True if telemetry is active, False if disabled/no-op
        """
        ...

    def get_trace_context(self) -> SpanContext | None:
        """
        Get the current active span context.

        Returns:
            Current span context if available, None otherwise
        """
        ...

    def create_child_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> AbstractContextManager[SpanContext]:
        """
        Create a child span from the current active span.

        Args:
            name: Name of the operation being traced
            kind: Type of span being created
            attributes: Key-value attributes to attach to the span

        Returns:
            Context manager that yields the child span context
        """
        ...


class SpanContextManager(Protocol):
    """
    Context manager protocol for span operations.

    This provides additional methods available within a span context.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        ...

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes on the current span."""
        ...

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the current span."""
        ...

    def set_status(self, status_code: str, description: str | None = None) -> None:
        """Set the status of the current span."""
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record an exception that occurred during the span."""
        ...
