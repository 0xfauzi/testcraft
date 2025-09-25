"""
Tests for telemetry adapters including OpenTelemetry, NoOp, and Cost Management.

This module tests all telemetry functionality including tracing, metrics,
cost tracking, and budget enforcement.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from testcraft.adapters.telemetry import (
    CostManager,
    NoOpTelemetryAdapter,
    OpenTelemetryAdapter,
)
from testcraft.ports.telemetry_port import MetricValue


class TestNoOpTelemetryAdapter:
    """Tests for the no-operation telemetry adapter."""

    def test_initialization(self):
        """Test NoOp adapter initialization."""
        adapter = NoOpTelemetryAdapter()
        assert not adapter.is_enabled()
        assert adapter.get_trace_context() is None

    def test_span_creation(self):
        """Test no-op span creation."""
        adapter = NoOpTelemetryAdapter()

        with adapter.create_span("test_operation") as span:
            assert span.name == "test_operation"
            assert span.trace_id == "noop-trace"
            assert span.span_id == "noop-span"

            # Test span operations do nothing
            span.set_attribute("key", "value")
            span.set_attributes({"key1": "value1", "key2": "value2"})
            span.add_event("test_event")
            span.set_status("OK")
            span.record_exception(Exception("test"))

    def test_metric_operations(self):
        """Test no-op metric operations."""
        adapter = NoOpTelemetryAdapter()

        # All metric operations should do nothing
        adapter.record_metric(MetricValue("test_metric", 1.0))
        adapter.record_metrics(
            [MetricValue("metric1", 1.0), MetricValue("metric2", 2.0)]
        )
        adapter.increment_counter("test_counter", 1)
        adapter.record_histogram("test_histogram", 100)
        adapter.record_gauge("test_gauge", 50)

    def test_global_operations(self):
        """Test no-op global operations."""
        adapter = NoOpTelemetryAdapter()

        adapter.set_global_attributes({"service": "test"})
        assert adapter.flush() is True

    def test_child_span_creation(self):
        """Test no-op child span creation."""
        adapter = NoOpTelemetryAdapter()

        with adapter.create_child_span("child_operation") as span:
            assert span.name == "child_operation"


class TestOpenTelemetryAdapter:
    """Tests for the OpenTelemetry adapter."""

    def test_initialization_without_otel(self):
        """Test initialization when OpenTelemetry is not available."""
        config = {
            "service_name": "testcraft",
            "anonymize_file_paths": True,
            "opt_out_data_collection": False,
        }

        with patch(
            "testcraft.adapters.telemetry.opentelemetry_adapter.OPENTELEMETRY_AVAILABLE",
            False,
        ):
            adapter = OpenTelemetryAdapter(config)
            assert not adapter.is_enabled()
            assert adapter.tracer is None
            assert adapter.meter is None

    @pytest.mark.skip(reason="Skipping test for now")
    @patch(
        "testcraft.adapters.telemetry.opentelemetry_adapter.OPENTELEMETRY_AVAILABLE",
        True,
    )
    def test_initialization_with_otel(self):
        """Test initialization when OpenTelemetry is available."""
        config = {
            "service_name": "testcraft",
            "service_version": "1.0.0",
            "environment": "test",
            "anonymize_file_paths": True,
            "opt_out_data_collection": False,
            "trace_sampling_rate": 1.0,
        }

        with (
            patch(
                "testcraft.adapters.telemetry.opentelemetry_adapter.trace"
            ) as mock_trace,
            patch(
                "testcraft.adapters.telemetry.opentelemetry_adapter.metrics"
            ) as mock_metrics,
        ):
            mock_tracer = Mock()
            mock_meter = Mock()
            mock_trace.get_tracer.return_value = mock_tracer
            mock_metrics.get_meter.return_value = mock_meter

            adapter = OpenTelemetryAdapter(config)
            adapter._initialize_providers()

            assert adapter.tracer == mock_tracer
            assert adapter.meter == mock_meter
            assert adapter.service_name == "testcraft"
            assert adapter.service_version == "1.0.0"
            assert adapter.environment == "test"

    @pytest.mark.skip(reason="Skipping test for now")
    def test_span_context_manager_sanitization(self):
        """Test span context manager value sanitization."""
        from testcraft.adapters.telemetry.opentelemetry_adapter import (
            OtelSpanContextManager,
        )

        mock_span = Mock()
        from testcraft.adapters.telemetry.opentelemetry_adapter import (
            ConcreteSpanContext,
        )

        span_context = ConcreteSpanContext("trace123", "span123")

        manager = OtelSpanContextManager(mock_span, span_context, anonymize_paths=True)

        # Test file path anonymization
        manager.set_attribute("file_path", "/sensitive/path/to/file.py")
        mock_span.set_attribute.assert_called()

        # Test code content handling
        manager.set_attribute(
            "code_content", "def sensitive_function():\n" + "    pass\n" * 100
        )
        mock_span.set_attribute.assert_called()

    def test_span_creation_without_otel(self):
        """Test span creation when OpenTelemetry is disabled."""
        config = {"opt_out_data_collection": True}
        adapter = OpenTelemetryAdapter(config)

        with adapter.create_span("test_operation") as span:
            assert span is not None
            span.set_attribute("test", "value")  # Should not raise


class TestCostManager:
    """Tests for the cost management adapter."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_telemetry(self):
        """Create mock telemetry adapter."""
        telemetry = Mock()
        telemetry.create_span.return_value.__enter__ = Mock(return_value=Mock())
        telemetry.create_span.return_value.__exit__ = Mock(return_value=None)
        return telemetry

    @pytest.fixture
    def cost_config(self):
        """Create cost management configuration."""
        return {
            "cost_thresholds": {
                "daily_limit": 10.0,
                "per_request_limit": 1.0,
                "warning_threshold": 0.50,
            }
        }

    def test_initialization(self, temp_storage, mock_telemetry, cost_config):
        """Test cost manager initialization."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        assert manager.daily_limit == 10.0
        assert manager.per_request_limit == 1.0
        assert manager.warning_threshold == 0.50
        assert manager.storage_path == temp_storage
        assert len(manager.cost_limits) == 1  # Default daily limit

    def test_track_usage_basic(self, temp_storage, mock_telemetry, cost_config):
        """Test basic usage tracking."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        cost_data = {"cost": 0.25, "tokens_used": 100, "api_calls": 1, "duration": 1.5}

        result = manager.track_usage("llm", "generate_tests", cost_data)

        assert "tracking_id" in result
        assert result["total_cost"] == 0.25
        assert result["usage_metadata"]["tokens_used"] == 100
        assert result["usage_metadata"]["warnings"] == []
        assert result["usage_metadata"]["within_limits"] is True

        # Verify entry was saved
        assert len(manager.cost_entries) == 1
        entry = manager.cost_entries[0]
        assert entry.service == "llm"
        assert entry.operation == "generate_tests"
        assert entry.cost == 0.25

    def test_track_usage_with_warnings(self, temp_storage, mock_telemetry, cost_config):
        """Test usage tracking with warning thresholds."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Track usage above warning threshold
        cost_data = {"cost": 0.75}  # Above 0.50 warning threshold
        result = manager.track_usage("llm", "generate_tests", cost_data)

        assert len(result["usage_metadata"]["warnings"]) == 1
        assert "warning threshold" in result["usage_metadata"]["warnings"][0]

    def test_track_usage_exceeds_limit(self, temp_storage, mock_telemetry, cost_config):
        """Test usage tracking that exceeds per-request limit."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Track usage above per-request limit
        cost_data = {"cost": 1.50}  # Above 1.0 per-request limit
        result = manager.track_usage("llm", "generate_tests", cost_data)

        assert len(result["usage_metadata"]["warnings"]) == 1
        assert "exceeds limit" in result["usage_metadata"]["warnings"][0]

    def test_get_summary_daily(self, temp_storage, mock_telemetry, cost_config):
        """Test getting daily cost summary."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Add some cost entries
        manager.track_usage("llm", "generate_tests", {"cost": 0.25, "tokens_used": 100})
        manager.track_usage("llm", "refine_tests", {"cost": 0.15, "tokens_used": 50})
        manager.track_usage("coverage", "analyze", {"cost": 0.05, "api_calls": 1})

        summary = manager.get_summary(time_period="daily")

        assert summary["total_cost"] == 0.45
        assert summary["service_breakdown"]["llm"]["cost"] == 0.40
        assert summary["service_breakdown"]["llm"]["tokens"] == 150
        assert summary["service_breakdown"]["coverage"]["cost"] == 0.05
        assert summary["usage_stats"]["total_operations"] == 3

    def test_cost_breakdown(self, temp_storage, mock_telemetry, cost_config):
        """Test detailed cost breakdown."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Add entries with different dates (mocking by adding to entries directly)
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        import uuid

        from testcraft.adapters.telemetry.cost_manager import CostEntry

        manager.cost_entries.extend(
            [
                CostEntry(str(uuid.uuid4()), now, "llm", "generate", 0.30),
                CostEntry(str(uuid.uuid4()), yesterday, "llm", "generate", 0.20),
                CostEntry(str(uuid.uuid4()), now, "coverage", "analyze", 0.10),
            ]
        )

        breakdown = manager.get_cost_breakdown(
            start_date=yesterday, end_date=now + timedelta(hours=1)
        )

        assert len(breakdown["daily_costs"]) == 2
        assert breakdown["service_costs"]["llm"] == 0.50
        assert breakdown["service_costs"]["coverage"] == 0.10
        assert breakdown["breakdown_metadata"]["total_cost"] == 0.60

    def test_cost_limits_management(self, temp_storage, mock_telemetry, cost_config):
        """Test cost limit setting and checking."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Set a new weekly limit
        result = manager.set_cost_limit(25.0, "weekly")
        assert result["success"] is True
        assert result["limit_details"]["amount"] == 25.0

        # Check limits with no usage
        limit_check = manager.check_cost_limit()
        assert limit_check["within_limits"] is True
        assert limit_check["current_usage"] == 0

        # Add some usage
        manager.track_usage("llm", "test", {"cost": 5.0})

        # Check limits again
        limit_check = manager.check_cost_limit()
        assert limit_check["within_limits"] is True
        assert limit_check["current_usage"] == 5.0

    def test_cost_limit_warnings(self, temp_storage, mock_telemetry, cost_config):
        """Test cost limit warning generation."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Add usage approaching daily limit (10.0)
        manager.track_usage("llm", "test1", {"cost": 8.0})  # 80% of limit

        limit_check = manager.check_cost_limit()
        assert len(limit_check["warnings"]) == 1
        assert "80.0%" in limit_check["warnings"][0]

        # Add more usage to exceed 90%
        manager.track_usage("llm", "test2", {"cost": 1.5})  # Total: 9.5 (95%)

        limit_check = manager.check_cost_limit()
        assert len(limit_check["warnings"]) == 1
        assert "95.0%" in limit_check["warnings"][0]

    def test_export_cost_data_csv(self, temp_storage, mock_telemetry, cost_config):
        """Test exporting cost data to CSV."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Add some cost entries
        manager.track_usage("llm", "generate", {"cost": 0.25, "tokens_used": 100})
        manager.track_usage("coverage", "analyze", {"cost": 0.05})

        result = manager.export_cost_data(export_format="csv")

        assert result["success"] is True
        assert result["export_format"] == "csv"
        assert result["export_metadata"]["entries_exported"] == 2

        # Verify file was created
        export_path = Path(result["export_path"])
        assert export_path.exists()

        # Verify CSV content
        with open(export_path) as f:
            content = f.read()
            assert "llm,generate" in content
            assert "coverage,analyze" in content

    def test_export_cost_data_json(self, temp_storage, mock_telemetry, cost_config):
        """Test exporting cost data to JSON."""
        manager = CostManager(
            config=cost_config, telemetry=mock_telemetry, storage_path=temp_storage
        )

        # Add a cost entry
        manager.track_usage("llm", "generate", {"cost": 0.30, "tokens_used": 150})

        result = manager.export_cost_data(export_format="json")

        assert result["success"] is True
        assert result["export_format"] == "json"

        # Verify JSON content
        export_path = Path(result["export_path"])
        with open(export_path) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["service"] == "llm"
            assert data[0]["cost"] == 0.30

    def test_persistence_across_instances(self, temp_storage, cost_config):
        """Test that cost data persists across manager instances."""
        # Create first manager and add data
        manager1 = CostManager(config=cost_config, storage_path=temp_storage)
        manager1.track_usage("llm", "test", {"cost": 0.50})

        # Create second manager with same storage
        manager2 = CostManager(config=cost_config, storage_path=temp_storage)

        # Verify data was loaded
        assert len(manager2.cost_entries) == 1
        assert manager2.cost_entries[0].cost == 0.50

    def test_error_handling_invalid_export_format(self, temp_storage, cost_config):
        """Test error handling for invalid export format."""
        manager = CostManager(config=cost_config, storage_path=temp_storage)

        result = manager.export_cost_data(export_format="invalid")

        assert result["success"] is False
        assert "Unsupported export format" in result["error"]


class TestTelemetryIntegration:
    """Integration tests for telemetry and cost management."""

    def test_cost_manager_with_telemetry_integration(self):
        """Test cost manager integration with telemetry."""
        telemetry = NoOpTelemetryAdapter()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cost_thresholds": {
                    "daily_limit": 5.0,
                    "per_request_limit": 1.0,
                    "warning_threshold": 0.50,
                }
            }

            manager = CostManager(
                config=config, telemetry=telemetry, storage_path=Path(temp_dir)
            )

            # Track usage with telemetry
            result = manager.track_usage(
                "llm",
                "generate_tests",
                {"cost": 0.75, "tokens_used": 200, "duration": 2.5},
            )

            assert result["total_cost"] == 0.75
            assert len(result["usage_metadata"]["warnings"]) == 1

    def test_telemetry_adapter_factory_pattern(self):
        """Test factory pattern for creating telemetry adapters."""

        # This would typically be in a factory class, but we'll test the concept
        def create_telemetry_adapter(backend: str, config: dict):
            if backend == "opentelemetry":
                return OpenTelemetryAdapter(config)
            elif backend == "noop" or not config.get("enabled", False):
                return NoOpTelemetryAdapter(config)
            else:
                raise ValueError(f"Unsupported telemetry backend: {backend}")

        # Test factory creates correct adapters
        otel_config = {"backend": "opentelemetry", "enabled": True}
        noop_config = {"backend": "noop", "enabled": False}

        otel_adapter = create_telemetry_adapter("opentelemetry", otel_config)
        noop_adapter = create_telemetry_adapter("noop", noop_config)

        assert isinstance(otel_adapter, OpenTelemetryAdapter)
        assert isinstance(noop_adapter, NoOpTelemetryAdapter)


if __name__ == "__main__":
    pytest.main([__file__])
