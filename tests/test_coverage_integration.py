"""
Integration tests for coverage measurement with real adapter.

Tests the complete coverage workflow from adapter to use case.
"""

import pytest

from testcraft.adapters.coverage.coverage_py_adapter import (
    CoveragePyAdapter,
    NoOpCoverageAdapter,
)
from testcraft.adapters.io.file_discovery import FileDiscoveryService
from testcraft.adapters.io.state_json import StateJsonAdapter
from testcraft.adapters.telemetry.noop_adapter import NoOpTelemetryAdapter
from testcraft.application.coverage_usecase import CoverageUseCase
from testcraft.domain.models import CoverageResult


class TestCoveragePyAdapterIntegration:
    """Integration tests for CoveragePyAdapter."""

    @pytest.fixture
    def sample_python_file(self, tmp_path):
        """Create a sample Python file for coverage testing."""
        test_file = tmp_path / "sample.py"
        test_file.write_text(
            """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

if __name__ == "__main__":
    print(add(1, 2))
"""
        )
        return test_file

    def test_adapter_creation(self):
        """Test that CoveragePyAdapter can be created."""
        try:
            adapter = CoveragePyAdapter()
            assert adapter.get_measurement_method() == "coverage.py"
        except ImportError:
            pytest.skip("coverage.py not installed")

    def test_noop_adapter_creation(self):
        """Test that NoOpCoverageAdapter provides graceful fallback."""
        adapter = NoOpCoverageAdapter()
        assert adapter.get_measurement_method() == "noop"

        # Should return empty results gracefully
        results = adapter.measure_coverage(["/fake/path.py"])
        assert results == {}

    def test_measure_coverage_real_file(self, sample_python_file):
        """Test coverage measurement with a real Python file."""
        try:
            adapter = CoveragePyAdapter()
        except ImportError:
            pytest.skip("coverage.py not installed")

        results = adapter.measure_coverage([str(sample_python_file)])

        assert len(results) == 1
        assert str(sample_python_file) in results

        coverage_result = results[str(sample_python_file)]
        assert isinstance(coverage_result, CoverageResult)
        assert 0.0 <= coverage_result.line_coverage <= 1.0
        assert 0.0 <= coverage_result.branch_coverage <= 1.0
        assert isinstance(coverage_result.missing_lines, list)

    def test_get_coverage_summary(self, sample_python_file):
        """Test coverage summary generation."""
        try:
            adapter = CoveragePyAdapter()
        except ImportError:
            pytest.skip("coverage.py not installed")

        coverage_data = adapter.measure_coverage([str(sample_python_file)])
        summary = adapter.get_coverage_summary(coverage_data)

        assert "overall_line_coverage" in summary
        assert "overall_branch_coverage" in summary
        assert "files_covered" in summary
        assert summary["files_covered"] == 1

    def test_report_coverage_formats(self, sample_python_file):
        """Test different report formats."""
        try:
            adapter = CoveragePyAdapter()
        except ImportError:
            pytest.skip("coverage.py not installed")

        coverage_data = adapter.measure_coverage([str(sample_python_file)])

        # Test summary format
        summary_report = adapter.report_coverage(coverage_data, "summary")
        assert summary_report["format"] == "summary"
        assert "Coverage Summary" in summary_report["report_content"]

        # Test detailed format
        detailed_report = adapter.report_coverage(coverage_data, "detailed")
        assert detailed_report["format"] == "detailed"
        assert "Detailed Coverage Report" in detailed_report["report_content"]

        # Test JSON format
        json_report = adapter.report_coverage(coverage_data, "json")
        assert json_report["format"] == "json"
        import json

        parsed = json.loads(json_report["report_content"])
        assert "summary" in parsed
        assert "files" in parsed

    def test_identify_gaps(self, sample_python_file):
        """Test coverage gap identification."""
        try:
            adapter = CoveragePyAdapter()
        except ImportError:
            pytest.skip("coverage.py not installed")

        coverage_data = adapter.measure_coverage([str(sample_python_file)])

        # With high threshold, should identify gaps
        gaps_high = adapter.identify_gaps(coverage_data, threshold=0.95)
        # With low threshold, might not identify gaps
        gaps_low = adapter.identify_gaps(coverage_data, threshold=0.1)

        # Gaps should be dict mapping file paths to missing lines
        assert isinstance(gaps_high, dict)
        assert isinstance(gaps_low, dict)


@pytest.mark.integration
class TestCoverageUseCaseIntegration:
    """Integration tests for CoverageUseCase with real adapter."""

    @pytest.fixture
    def sample_project(self, tmp_path):
        """Create a sample project structure."""
        project_path = tmp_path / "project"
        project_path.mkdir()

        # Create source files
        (project_path / "module1.py").write_text(
            """
def function1():
    return "test"

def function2(x):
    if x > 0:
        return x
    return 0
"""
        )

        (project_path / "module2.py").write_text(
            """
class MyClass:
    def method1(self):
        return True

    def method2(self):
        return False
"""
        )

        return project_path

    @pytest.mark.asyncio
    async def test_coverage_usecase_with_real_adapter(self, sample_project):
        """Test complete coverage workflow with real adapter."""
        try:
            adapter = CoveragePyAdapter()
        except ImportError:
            pytest.skip("coverage.py not installed")

        # Create use case with real dependencies
        use_case = CoverageUseCase(
            coverage_port=adapter,
            state_port=StateJsonAdapter(),
            telemetry_port=NoOpTelemetryAdapter(),
            file_discovery_service=FileDiscoveryService(),
        )

        # Measure coverage
        result = await use_case.measure_and_report(project_path=sample_project)

        # Verify results
        assert result["success"] is True
        assert "coverage_data" in result
        assert "coverage_summary" in result
        assert "reports" in result
        assert result["files_measured"] >= 0
        assert 0.0 <= result["overall_line_coverage"] <= 1.0

    @pytest.mark.asyncio
    async def test_coverage_usecase_with_noop_adapter(self, sample_project):
        """Test coverage workflow with no-op adapter (graceful degradation)."""
        adapter = NoOpCoverageAdapter()

        use_case = CoverageUseCase(
            coverage_port=adapter,
            state_port=StateJsonAdapter(),
            telemetry_port=NoOpTelemetryAdapter(),
            file_discovery_service=FileDiscoveryService(),
        )

        # Should complete without errors even with no-op adapter
        result = await use_case.measure_and_report(project_path=sample_project)

        assert result["success"] is True
        assert result["files_measured"] == 0  # No-op returns empty
        assert result["overall_line_coverage"] == 0.0
