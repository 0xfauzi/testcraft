"""
Tests for coverage adapters.

This module contains unit tests for the coverage measurement adapters,
including pytest coverage, AST fallback, and composite adapters.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from testcraft.adapters.coverage import (
    AstFallbackAdapter,
    CompositeCoverageAdapter,
    TestcraftCoverageAdapter,
)
from testcraft.domain.models import CoverageResult


class TestAstFallbackAdapter:
    """Test cases for the AST fallback coverage adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = AstFallbackAdapter()

    @pytest.mark.skip(reason="AstFallbackAdapter methods not yet implemented")
    def test_measure_coverage_simple_file(self):
        """Test coverage measurement for a simple Python file."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, a, b):
        return a * b

    def _private_method(self):
        return "private"

if __name__ == "__main__":
    print("Hello")
"""
            )
            temp_file = Path(f.name)

        try:
            # Measure coverage
            result = self.adapter.measure_coverage([str(temp_file)])

            # Verify result structure
            assert len(result) == 1
            assert str(temp_file.resolve()) in result

            coverage_result = result[str(temp_file.resolve())]
            assert isinstance(coverage_result, CoverageResult)
            assert 0.0 <= coverage_result.line_coverage <= 1.0
            assert 0.0 <= coverage_result.branch_coverage <= 1.0
            assert isinstance(coverage_result.missing_lines, list)

        finally:
            # Clean up
            temp_file.unlink()

    @pytest.mark.skip(reason="AstFallbackAdapter methods not yet implemented")
    def test_measure_coverage_missing_file(self):
        """Test coverage measurement for non-existent files."""
        result = self.adapter.measure_coverage(["/path/to/nonexistent/file.py"])

        assert len(result) == 1
        coverage_result = list(result.values())[0]
        assert coverage_result.line_coverage == 0.0
        assert coverage_result.branch_coverage == 0.0

    @pytest.mark.skip(reason="AstFallbackAdapter methods not yet implemented")
    def test_report_coverage_summary(self):
        """Test summary report generation."""
        # Create mock coverage data
        coverage_data = {
            "/path/to/file1.py": CoverageResult(
                line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10, 15]
            ),
            "/path/to/file2.py": CoverageResult(
                line_coverage=0.9, branch_coverage=0.85, missing_lines=[20]
            ),
        }

        report = self.adapter.report_coverage(coverage_data, "summary")

        assert report["format"] == "summary"
        assert "report_content" in report
        assert "summary_stats" in report
        assert report["summary_stats"]["files_covered"] == 2

    @pytest.mark.skip(reason="AstFallbackAdapter methods not yet implemented")
    def test_get_coverage_summary(self):
        """Test coverage summary generation."""
        coverage_data = {
            "/path/to/file1.py": CoverageResult(
                line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10]
            ),
            "/path/to/file2.py": CoverageResult(
                line_coverage=0.6, branch_coverage=0.5, missing_lines=[1, 2, 3]
            ),
        }

        summary = self.adapter.get_coverage_summary(coverage_data)

        assert summary["overall_line_coverage"] == 0.7  # (0.8 + 0.6) / 2
        assert summary["overall_branch_coverage"] == 0.6  # (0.7 + 0.5) / 2
        assert summary["files_covered"] == 2

    @pytest.mark.skip(reason="AstFallbackAdapter methods not yet implemented")
    def test_identify_gaps(self):
        """Test coverage gap identification."""
        coverage_data = {
            "/path/to/file1.py": CoverageResult(
                line_coverage=0.9, branch_coverage=0.8, missing_lines=[5]
            ),
            "/path/to/file2.py": CoverageResult(
                line_coverage=0.5,  # Below threshold
                branch_coverage=0.4,
                missing_lines=[1, 2, 3, 4, 5],
            ),
        }

        gaps = self.adapter.identify_gaps(coverage_data, threshold=0.8)

        # Only file2 should be identified as having gaps
        assert len(gaps) == 1
        assert "/path/to/file2.py" in gaps
        assert gaps["/path/to/file2.py"] == [1, 2, 3, 4, 5]


class TestCompositeCoverageAdapter:
    """Test cases for the composite coverage adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = CompositeCoverageAdapter()

    @patch("testcraft.adapters.coverage.composite.PytestCoverageAdapter")
    @pytest.mark.skip(reason="CompositeCoverageAdapter methods not yet implemented")
    def test_measure_coverage_pytest_success(self, mock_pytest_adapter):
        """Test successful coverage measurement with pytest."""
        # Mock successful pytest coverage
        mock_instance = Mock()
        mock_instance.measure_coverage.return_value = {
            "/path/to/file.py": CoverageResult(
                line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10]
            )
        }
        mock_pytest_adapter.return_value = mock_instance

        # Create new adapter with mocked pytest adapter
        adapter = CompositeCoverageAdapter()
        adapter.pytest_adapter = mock_instance

        result = adapter.measure_coverage(["/path/to/file.py"])

        # Verify pytest was used
        assert adapter.get_method_used() == "pytest"
        assert adapter.get_fallback_reason() is None
        assert len(result) == 1

    @patch("testcraft.adapters.coverage.composite.PytestCoverageAdapter")
    @pytest.mark.skip(reason="CompositeCoverageAdapter methods not yet implemented")
    def test_measure_coverage_pytest_failure_ast_fallback(self, mock_pytest_adapter):
        """Test fallback to AST when pytest fails."""
        from testcraft.adapters.coverage.pytest_coverage import CoverageError

        # Mock pytest failure
        mock_instance = Mock()
        mock_instance.measure_coverage.side_effect = CoverageError("Pytest failed")
        mock_pytest_adapter.return_value = mock_instance

        # Create new adapter
        adapter = CompositeCoverageAdapter()
        adapter.pytest_adapter = mock_instance

        # Create a simple test file for AST analysis
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): return True\n")
            temp_file = Path(f.name)

        try:
            result = adapter.measure_coverage([str(temp_file)])

            # Verify AST fallback was used
            assert adapter.get_method_used() == "ast_fallback"
            assert adapter.get_fallback_reason() == "Pytest failed"
            assert len(result) == 1

        finally:
            temp_file.unlink()

    @pytest.mark.skip(reason="CompositeCoverageAdapter methods not yet implemented")
    def test_force_ast_mode(self):
        """Test forcing AST mode instead of pytest."""
        # Create a simple test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): return True\n")
            temp_file = Path(f.name)

        try:
            result = self.adapter.measure_coverage([str(temp_file)], force_ast=True)

            # Verify AST was used directly
            assert self.adapter.get_method_used() == "ast_fallback"
            assert len(result) == 1

        finally:
            temp_file.unlink()


class TestTestcraftCoverageAdapter:
    """Test cases for the main testcraft coverage adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = TestcraftCoverageAdapter()

    @pytest.mark.skip(reason="TestcraftCoverageAdapter methods not yet implemented")
    def test_validate_source_files(self):
        """Test source file validation."""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create valid Python file
            py_file = temp_dir / "test.py"
            py_file.write_text("def test(): pass\n")

            # Create invalid file
            txt_file = temp_dir / "test.txt"
            txt_file.write_text("not python\n")

            # Test validation
            validated = self.adapter._validate_source_files(
                [str(py_file), str(txt_file), "/nonexistent/file.py"]
            )

            # Only the valid Python file should remain
            assert len(validated) == 1
            assert str(py_file.resolve()) in validated

    @pytest.mark.skip(reason="TestcraftCoverageAdapter methods not yet implemented")
    def test_get_coverage_summary_empty(self):
        """Test coverage summary with no data."""
        summary = self.adapter.get_coverage_summary({})

        assert summary["overall_line_coverage"] == 0.0
        assert summary["overall_branch_coverage"] == 0.0
        assert summary["files_covered"] == 0

    @pytest.mark.skip(reason="TestcraftCoverageAdapter methods not yet implemented")
    def test_analyze_coverage_distribution(self):
        """Test coverage distribution analysis."""
        coverage_data = {
            "excellent.py": CoverageResult(
                line_coverage=0.95, branch_coverage=0.9, missing_lines=[]
            ),
            "good.py": CoverageResult(
                line_coverage=0.85, branch_coverage=0.8, missing_lines=[1]
            ),
            "fair.py": CoverageResult(
                line_coverage=0.65, branch_coverage=0.6, missing_lines=[1, 2]
            ),
            "poor.py": CoverageResult(
                line_coverage=0.45, branch_coverage=0.4, missing_lines=[1, 2, 3]
            ),
            "very_poor.py": CoverageResult(
                line_coverage=0.25, branch_coverage=0.2, missing_lines=[1, 2, 3, 4]
            ),
        }

        distribution = self.adapter._analyze_coverage_distribution(coverage_data)

        assert distribution["excellent"] == 1
        assert distribution["good"] == 1
        assert distribution["fair"] == 1
        assert distribution["poor"] == 1
        assert distribution["very_poor"] == 1

    @pytest.mark.skip(reason="TestcraftCoverageAdapter methods not yet implemented")
    def test_html_report_generation(self):
        """Test HTML report generation."""
        coverage_data = {
            "/path/to/test.py": CoverageResult(
                line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10]
            )
        }

        report = self.adapter._generate_html_report(coverage_data)

        assert report["format"] == "html"
        assert "report_content" in report
        assert "<html>" in report["report_content"]
        assert "80.0%" in report["report_content"]  # Line coverage

    @pytest.mark.skip(reason="TestcraftCoverageAdapter methods not yet implemented")
    def test_json_report_generation(self):
        """Test JSON report generation."""
        coverage_data = {
            "/path/to/test.py": CoverageResult(
                line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10]
            )
        }

        report = self.adapter._generate_json_report(coverage_data)

        assert report["format"] == "json"
        assert "report_content" in report

        # Parse JSON to verify structure
        import json

        json_data = json.loads(report["report_content"])
        assert "summary" in json_data
        assert "files" in json_data
        assert "/path/to/test.py" in json_data["files"]


@pytest.fixture
def sample_coverage_data():
    """Fixture providing sample coverage data for tests."""
    return {
        "/path/to/file1.py": CoverageResult(
            line_coverage=0.8, branch_coverage=0.7, missing_lines=[5, 10, 15]
        ),
        "/path/to/file2.py": CoverageResult(
            line_coverage=0.9, branch_coverage=0.85, missing_lines=[20]
        ),
        "/path/to/file3.py": CoverageResult(
            line_coverage=0.5, branch_coverage=0.4, missing_lines=[1, 2, 3, 4, 5]
        ),
    }


@pytest.mark.skip(reason="Coverage adapter methods not yet implemented")
def test_integration_coverage_measurement(sample_coverage_data):
    """Integration test for coverage measurement workflow."""
    adapter = TestcraftCoverageAdapter()

    # Test summary generation
    summary = adapter.get_coverage_summary(sample_coverage_data)
    assert summary["files_covered"] == 3
    assert 0.0 <= summary["overall_line_coverage"] <= 1.0

    # Test gap identification
    gaps = adapter.identify_gaps(sample_coverage_data, threshold=0.8)
    assert len(gaps) == 1  # Only file3.py should have gaps
    assert "/path/to/file3.py" in gaps

    # Test report generation
    report = adapter.report_coverage(sample_coverage_data, "detailed")
    assert "report_content" in report
    assert "summary_stats" in report
