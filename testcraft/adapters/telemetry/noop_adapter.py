"""
No-op telemetry adapter for disabled telemetry.

This adapter provides a no-operation implementation of the TelemetryPort
interface, useful when telemetry is disabled or for testing.
"""

from contextlib import AbstractContextManager, contextmanager
from typing import Any

from ...ports.telemetry_port import MetricValue, SpanContext, SpanKind


class NoOpSpanContext:
    """No-op span context that does nothing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.trace_id: str = "noop-trace"
        self.span_id: str = "noop-span"
        self.parent_span_id: str | None = None
        self.baggage: dict[str, Any] | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op attribute setting."""
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """No-op attributes setting."""
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """No-op event adding."""
        pass

    def set_status(self, status_code: str, description: str | None = None) -> None:
        """No-op status setting."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op exception recording."""
        pass

    def get_trace_context(self) -> SpanContext:
        """
        Get the trace context for this span.

        Returns:
            Self since NoOpSpanContext implements SpanContext protocol
        """
        return self


class NoOpTelemetryAdapter:
    """
    No-operation telemetry adapter.

    This adapter implements the TelemetryPort interface but performs
    no actual telemetry operations. Useful for testing or when
    telemetry is disabled.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the no-op adapter.

        Args:
            config: Configuration (ignored for no-op adapter)
        """
        self.config = config or {}

    @contextmanager
    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent_context: SpanContext | None = None,
    ) -> AbstractContextManager[SpanContext]:
        """
        Create a no-op span.

        Args:
            name: Name of the operation (ignored)
            kind: Type of span (ignored)
            attributes: Span attributes (ignored)
            parent_context: Parent span context (ignored)

        Yields:
            A no-op span context
        """
        span_context = NoOpSpanContext(name)
        yield span_context

    def record_metric(self, metric: MetricValue) -> None:
        """
        Record a metric (no-op).

        Args:
            metric: The metric value (ignored)
        """
        pass

    def record_metrics(self, metrics: list[MetricValue]) -> None:
        """
        Record multiple metrics (no-op).

        Args:
            metrics: List of metric values (ignored)
        """
        pass

    def increment_counter(
        self,
        name: str,
        value: int | float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter (no-op).

        Args:
            name: Counter name (ignored)
            value: Amount to increment (ignored)
            labels: Counter labels (ignored)
        """
        pass

    def record_histogram(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record a histogram value (no-op).

        Args:
            name: Histogram name (ignored)
            value: Value to record (ignored)
            labels: Histogram labels (ignored)
        """
        pass

    def record_gauge(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Set a gauge value (no-op).

        Args:
            name: Gauge name (ignored)
            value: Current value (ignored)
            labels: Gauge labels (ignored)
        """
        pass

    def set_global_attributes(self, attributes: dict[str, Any]) -> None:
        """
        Set global attributes (no-op).

        Args:
            attributes: Global attributes (ignored)
        """
        pass

    def flush(self, timeout_seconds: float | None = None) -> bool:
        """
        Flush telemetry data (no-op).

        Args:
            timeout_seconds: Timeout (ignored)

        Returns:
            Always True for no-op adapter
        """
        return True

    def is_enabled(self) -> bool:
        """
        Check if telemetry is enabled.

        Returns:
            Always False for no-op adapter
        """
        return False

    def get_trace_context(self) -> SpanContext | None:
        """
        Get current trace context.

        Returns:
            Always None for no-op adapter
        """
        return None

    @contextmanager
    def create_child_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> AbstractContextManager[SpanContext]:
        """
        Create a child span (no-op).

        Args:
            name: Name of the operation (ignored)
            kind: Type of span (ignored)
            attributes: Span attributes (ignored)

        Yields:
            A no-op span context
        """
        span_context = NoOpSpanContext(name)
        yield span_context
