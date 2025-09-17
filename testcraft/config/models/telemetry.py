"""Telemetry and observability configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TelemetryBackendConfig(BaseModel):
    """Configuration for specific telemetry backends."""

    opentelemetry: dict[str, Any] = Field(
        default_factory=lambda: {
            "endpoint": None,  # Auto-detect or use OTEL_EXPORTER_OTLP_ENDPOINT
            "headers": {},  # Additional headers for OTLP exporter
            "insecure": False,  # Use insecure gRPC connection
            "timeout": 10,  # Timeout for exports in seconds
        },
        description="OpenTelemetry-specific configuration",
    )

    datadog: dict[str, Any] = Field(
        default_factory=lambda: {
            "api_key": None,  # DD_API_KEY env var if None
            "site": "datadoghq.com",  # Datadog site
            "service": "testcraft",  # Service name
            "env": "development",  # Environment
            "version": None,  # Service version
        },
        description="Datadog-specific configuration",
    )

    jaeger: dict[str, Any] = Field(
        default_factory=lambda: {
            "endpoint": "http://localhost:14268/api/traces",
            "agent_host_name": "localhost",
            "agent_port": 6831,
        },
        description="Jaeger-specific configuration",
    )


class TelemetryConfig(BaseModel):
    """Configuration for telemetry and observability."""

    enabled: bool = Field(default=False, description="Enable telemetry collection")

    backend: Literal["opentelemetry", "datadog", "jaeger", "noop"] = Field(
        default="opentelemetry", description="Telemetry backend to use"
    )

    service_name: str = Field(
        default="testcraft", description="Service name for telemetry"
    )

    service_version: str | None = Field(
        default=None, description="Service version (auto-detected if None)"
    )

    environment: str = Field(
        default="development",
        description="Environment name (development, staging, production)",
    )

    # Tracing configuration
    trace_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate (0.0 to 1.0)"
    )

    capture_llm_calls: bool = Field(default=True, description="Trace LLM API calls")

    capture_coverage_runs: bool = Field(
        default=True, description="Trace coverage analysis operations"
    )

    capture_file_operations: bool = Field(
        default=True, description="Trace file read/write operations"
    )

    capture_test_generation: bool = Field(
        default=True, description="Trace test generation processes"
    )

    # Metrics configuration
    collect_metrics: bool = Field(default=True, description="Enable metrics collection")

    metrics_interval_seconds: int = Field(
        default=30, ge=1, description="Metrics collection interval"
    )

    track_token_usage: bool = Field(
        default=True, description="Track LLM token usage metrics"
    )

    track_coverage_delta: bool = Field(
        default=True, description="Track coverage improvement metrics"
    )

    track_test_pass_rate: bool = Field(
        default=True, description="Track test success/failure rates"
    )

    # Privacy and anonymization
    anonymize_file_paths: bool = Field(
        default=True, description="Hash file paths in telemetry data"
    )

    anonymize_code_content: bool = Field(
        default=True, description="Exclude actual code content from telemetry"
    )

    opt_out_data_collection: bool = Field(
        default=False,
        description="Completely disable data collection (overrides enabled)",
    )

    # Resource attributes
    global_attributes: dict[str, Any] = Field(
        default_factory=dict, description="Global attributes to attach to all telemetry"
    )

    # Backend-specific configurations
    backends: TelemetryBackendConfig = Field(
        default_factory=TelemetryBackendConfig,
        description="Backend-specific configuration",
    )

    @field_validator("trace_sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v):
        """Ensure sampling rate is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("trace_sampling_rate must be between 0.0 and 1.0")
        return v
