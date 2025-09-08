"""
OpenTelemetry adapter for telemetry collection.

This adapter implements the TelemetryPort interface using OpenTelemetry
for distributed tracing and metrics collection.
"""

import hashlib
import os
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Union

from ...ports.telemetry_port import MetricValue, SpanContext, SpanKind

# OpenTelemetry imports (optional - graceful degradation if not available)
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import AttributeValue

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Define stubs for type hints when OpenTelemetry is not available
    OPENTELEMETRY_AVAILABLE = False

    # Type stubs for OpenTelemetry types
    AttributeValue = Union[str, bool, int, float]

    class MockTracer:
        pass

    class MockMeter:
        pass


class OtelSpanContextManager:
    """OpenTelemetry span context manager with enhanced functionality."""

    def __init__(self, span, span_context: SpanContext, anonymize_paths: bool = True):
        self.span = span
        self.span_context = span_context
        self.anonymize_paths = anonymize_paths

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Sanitize sensitive data
        sanitized_value = self._sanitize_value(key, value)
        if sanitized_value is not None:
            self.span.set_attribute(key, sanitized_value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes on the current span."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the current span."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        sanitized_attributes = {}
        if attributes:
            for key, value in attributes.items():
                sanitized_value = self._sanitize_value(key, value)
                if sanitized_value is not None:
                    sanitized_attributes[key] = sanitized_value

        self.span.add_event(name, sanitized_attributes)

    def set_status(self, status_code: str, description: str | None = None) -> None:
        """Set the status of the current span."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Map string status codes to OpenTelemetry StatusCode enum
        status_mapping = {
            "OK": StatusCode.OK,
            "ERROR": StatusCode.ERROR,
            "CANCELLED": StatusCode.ERROR,  # Map cancelled to error
        }

        otel_status = status_mapping.get(status_code.upper(), StatusCode.UNSET)
        self.span.set_status(Status(otel_status, description))

    def record_exception(self, exception: Exception) -> None:
        """Record an exception that occurred during the span."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        self.span.record_exception(exception)
        self.set_status("ERROR", str(exception))

    def _sanitize_value(self, key: str, value: Any) -> AttributeValue | None:
        """Sanitize attribute values for privacy and OpenTelemetry compatibility."""
        # Handle file paths
        if "path" in key.lower() or "file" in key.lower():
            if self.anonymize_paths and isinstance(value, str):
                # Hash the path to preserve uniqueness while anonymizing
                return hashlib.md5(value.encode()).hexdigest()[:16]

        # Handle code content
        if "code" in key.lower() or "content" in key.lower():
            if isinstance(value, str) and len(value) > 100:
                return f"<content_hash:{hashlib.md5(value.encode()).hexdigest()[:8]}>"

        # Convert to OpenTelemetry-compatible types
        if isinstance(value, str | int | float | bool):
            return value
        elif isinstance(value, list):
            # Convert list to string representation
            return str(value)[:100]  # Limit length
        elif isinstance(value, dict):
            # Convert dict to string representation
            return str(value)[:100]  # Limit length
        else:
            return str(value)[:100]  # Convert everything else to string


class OpenTelemetryAdapter:
    """
    OpenTelemetry adapter for telemetry collection.

    This adapter implements distributed tracing and metrics collection
    using the OpenTelemetry framework.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the OpenTelemetry adapter.

        Args:
            config: Telemetry configuration dictionary
        """
        self.config = config
        self.anonymize_paths = config.get("anonymize_file_paths", True)
        self.anonymize_code = config.get("anonymize_code_content", True)
        self.service_name = config.get("service_name", "testcraft")
        self.service_version = config.get("service_version")
        self.environment = config.get("environment", "development")

        # Global attributes
        self.global_attributes = config.get("global_attributes", {})

        if OPENTELEMETRY_AVAILABLE and not config.get("opt_out_data_collection", False):
            self._initialize_providers()
        else:
            self.tracer = None
            self.meter = None

    def _initialize_providers(self) -> None:
        """Initialize OpenTelemetry providers and exporters."""
        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version or "unknown",
                "service.environment": self.environment,
                **self.global_attributes,
            }
        )

        # Initialize tracing
        self._initialize_tracing(resource)

        # Initialize metrics
        self._initialize_metrics(resource)

    def _initialize_tracing(self, resource: "Resource") -> None:
        """Initialize OpenTelemetry tracing."""
        # Create tracer provider
        tracer_provider = TracerProvider(
            resource=resource, sampler=self._create_sampler()
        )

        # Create and configure span processor
        span_processor = self._create_span_processor()
        tracer_provider.add_span_processor(span_processor)

        # Set the global tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__, instrumenting_library_version="1.0.0"
        )

    def _initialize_metrics(self, resource: "Resource") -> None:
        """Initialize OpenTelemetry metrics."""
        # Create metric reader
        metric_reader = self._create_metric_reader()

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader] if metric_reader else []
        )

        # Set the global meter provider
        metrics.set_meter_provider(meter_provider)

        # Get meter
        self.meter = metrics.get_meter(name=self.service_name, version="1.0.0")

        # Create metric instruments
        self._create_metric_instruments()

    def _create_sampler(self):
        """Create a sampler based on configuration."""
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler

        sampling_rate = self.config.get("trace_sampling_rate", 1.0)
        return TraceIdRatioBasedSampler(sampling_rate)

    def _create_span_processor(self):
        """Create span processor with appropriate exporter."""
        # Check for OTLP endpoint configuration
        otlp_endpoint = self.config.get("backends", {}).get("opentelemetry", {}).get(
            "endpoint"
        ) or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

        if otlp_endpoint:
            # Use OTLP exporter
            headers = (
                self.config.get("backends", {})
                .get("opentelemetry", {})
                .get("headers", {})
            )
            insecure = (
                self.config.get("backends", {})
                .get("opentelemetry", {})
                .get("insecure", False)
            )

            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint, headers=headers, insecure=insecure
            )
        else:
            # Fallback to console exporter for development
            exporter = ConsoleSpanExporter()

        return BatchSpanProcessor(exporter)

    def _create_metric_reader(self):
        """Create metric reader with appropriate exporter."""
        if not self.config.get("collect_metrics", True):
            return None

        # Check for OTLP endpoint configuration
        otlp_endpoint = self.config.get("backends", {}).get("opentelemetry", {}).get(
            "endpoint"
        ) or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

        interval = self.config.get("metrics_interval_seconds", 30)

        if otlp_endpoint:
            # Use OTLP exporter
            headers = (
                self.config.get("backends", {})
                .get("opentelemetry", {})
                .get("headers", {})
            )
            insecure = (
                self.config.get("backends", {})
                .get("opentelemetry", {})
                .get("insecure", False)
            )

            exporter = OTLPMetricExporter(
                endpoint=otlp_endpoint, headers=headers, insecure=insecure
            )
        else:
            # Fallback to console exporter for development
            exporter = ConsoleMetricExporter()

        return PeriodicExportingMetricReader(
            exporter=exporter, export_interval_millis=interval * 1000
        )

    def _create_metric_instruments(self) -> None:
        """Create metric instruments for common measurements."""
        if not self.meter:
            return

        # Counters
        self.test_generation_counter = self.meter.create_counter(
            name="testcraft.tests.generated",
            description="Number of tests generated",
            unit="1",
        )

        self.llm_calls_counter = self.meter.create_counter(
            name="testcraft.llm.calls", description="Number of LLM API calls", unit="1"
        )

        self.token_usage_counter = self.meter.create_counter(
            name="testcraft.llm.tokens", description="LLM token usage", unit="1"
        )

        # Histograms
        self.llm_latency_histogram = self.meter.create_histogram(
            name="testcraft.llm.duration", description="LLM call duration", unit="ms"
        )

        self.coverage_histogram = self.meter.create_histogram(
            name="testcraft.coverage.percentage",
            description="Code coverage percentage",
            unit="%",
        )

        # Gauges (using up-down counter as approximation)
        self.active_operations_gauge = self.meter.create_up_down_counter(
            name="testcraft.operations.active",
            description="Number of active operations",
            unit="1",
        )

    @contextmanager
    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent_context: SpanContext | None = None,
    ) -> AbstractContextManager[OtelSpanContextManager]:
        """
        Create a new telemetry span.

        Args:
            name: Name of the operation being traced
            kind: Type of span being created
            attributes: Key-value attributes to attach to the span
            parent_context: Parent span context for nested spans

        Yields:
            Enhanced span context manager
        """
        if not self.tracer:
            # Return no-op context if tracing is disabled
            yield OtelSpanContextManager(
                None, SpanContext("noop", "noop"), self.anonymize_paths
            )
            return

        # Map SpanKind to OpenTelemetry span kind
        kind_mapping = {
            SpanKind.INTERNAL: trace.SpanKind.INTERNAL,
            SpanKind.SERVER: trace.SpanKind.SERVER,
            SpanKind.CLIENT: trace.SpanKind.CLIENT,
            SpanKind.PRODUCER: trace.SpanKind.PRODUCER,
            SpanKind.CONSUMER: trace.SpanKind.CONSUMER,
        }
        otel_kind = kind_mapping.get(kind, trace.SpanKind.INTERNAL)

        # Create span
        with self.tracer.start_as_current_span(
            name=name, kind=otel_kind, attributes=attributes or {}
        ) as span:
            # Create span context
            otel_span_context = span.get_span_context()
            span_context = SpanContext(
                trace_id=f"{otel_span_context.trace_id:032x}",
                span_id=f"{otel_span_context.span_id:016x}",
            )

            # Create enhanced context manager
            context_manager = OtelSpanContextManager(
                span, span_context, self.anonymize_paths
            )

            try:
                yield context_manager
            except Exception as e:
                context_manager.record_exception(e)
                raise

    def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        if not self.meter:
            return

        labels = metric.labels or {}

        # Use appropriate metric instrument based on naming convention
        if "counter" in metric.name or metric.name.endswith("_total"):
            if hasattr(self, f"{metric.name}_counter"):
                getattr(self, f"{metric.name}_counter").add(metric.value, labels)
        elif "histogram" in metric.name or "duration" in metric.name:
            if hasattr(self, f"{metric.name}_histogram"):
                getattr(self, f"{metric.name}_histogram").record(metric.value, labels)
        elif "gauge" in metric.name:
            if hasattr(self, f"{metric.name}_gauge"):
                getattr(self, f"{metric.name}_gauge").add(metric.value, labels)

    def record_metrics(self, metrics: list[MetricValue]) -> None:
        """Record multiple metric values efficiently."""
        for metric in metrics:
            self.record_metric(metric)

    def increment_counter(
        self,
        name: str,
        value: int | float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""
        if name == "tests_generated" and hasattr(self, "test_generation_counter"):
            self.test_generation_counter.add(value, labels or {})
        elif name == "llm_calls" and hasattr(self, "llm_calls_counter"):
            self.llm_calls_counter.add(value, labels or {})
        elif name == "tokens_used" and hasattr(self, "token_usage_counter"):
            self.token_usage_counter.add(value, labels or {})

    def record_histogram(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram value."""
        if name == "llm_duration" and hasattr(self, "llm_latency_histogram"):
            self.llm_latency_histogram.record(value, labels or {})
        elif name == "coverage_percentage" and hasattr(self, "coverage_histogram"):
            self.coverage_histogram.record(value, labels or {})

    def record_gauge(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        if name == "active_operations" and hasattr(self, "active_operations_gauge"):
            self.active_operations_gauge.add(value, labels or {})

    def set_global_attributes(self, attributes: dict[str, Any]) -> None:
        """Set attributes that will be attached to all spans and metrics."""
        self.global_attributes.update(attributes)

    def flush(self, timeout_seconds: float | None = None) -> bool:
        """Force flush all pending telemetry data."""
        if not OPENTELEMETRY_AVAILABLE:
            return True

        try:
            # Flush traces
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "force_flush"):
                tracer_provider.force_flush(
                    timeout_millis=int((timeout_seconds or 5) * 1000)
                )

            # Flush metrics
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, "force_flush"):
                meter_provider.force_flush(
                    timeout_millis=int((timeout_seconds or 5) * 1000)
                )

            return True
        except Exception:
            return False

    def is_enabled(self) -> bool:
        """Check if telemetry is currently enabled."""
        return OPENTELEMETRY_AVAILABLE and self.tracer is not None

    def get_trace_context(self) -> SpanContext | None:
        """Get the current active span context."""
        if not self.tracer:
            return None

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            otel_context = current_span.get_span_context()
            return SpanContext(
                trace_id=f"{otel_context.trace_id:032x}",
                span_id=f"{otel_context.span_id:016x}",
            )
        return None

    @contextmanager
    def create_child_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> AbstractContextManager[OtelSpanContextManager]:
        """Create a child span from the current active span."""
        # The OpenTelemetry SDK automatically handles parent-child relationships
        # when using start_as_current_span within an existing span context
        with self.create_span(name, kind, attributes) as span_context:
            yield span_context
