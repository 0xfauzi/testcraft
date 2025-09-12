"""
Tests for JSON reporter adapter.

This module tests the JsonReportAdapter implementation,
verifying report generation for various data types and scenarios.
"""

import json
import tempfile
from pathlib import Path

import pytest

from testcraft.adapters.io.reporter_json import JsonReportAdapter, ReportError
from testcraft.domain.models import (AnalysisReport, CoverageResult,
                                     GenerationResult, TestElement,
                                     TestElementType, TestGenerationPlan)


class TestJsonReportAdapter:
    """Test cases for JsonReportAdapter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter = JsonReportAdapter(output_base_path=self.temp_dir)

        # Sample test data
        self.sample_coverage_result = CoverageResult(
            line_coverage=0.85, branch_coverage=0.78, missing_lines=[10, 15, 22]
        )

        self.sample_analysis_report = AnalysisReport(
            files_to_process=["src/module1.py", "src/module2.py"],
            reasons={
                "src/module1.py": "No existing tests found",
                "src/module2.py": "Low coverage detected",
            },
            existing_test_presence={"src/module1.py": False, "src/module2.py": True},
        )

        self.sample_generation_results = [
            GenerationResult(
                file_path="tests/test_module1.py",
                content="# Test content here",
                success=True,
                error_message=None,
            ),
            GenerationResult(
                file_path="tests/test_module2.py",
                content=None,
                success=False,
                error_message="Failed to generate tests",
            ),
        ]

    def test_generate_coverage_report(self) -> None:
        """Test coverage report generation."""
        coverage_data = {
            "files": {
                "src/module1.py": {
                    "line_coverage": 0.85,
                    "branch_coverage": 0.78,
                    "missing_lines": [10, 15, 22],
                }
            },
            "overall_line_coverage": 0.85,
            "overall_branch_coverage": 0.78,
            "files_analyzed": ["src/module1.py"],
        }

        result = self.adapter.generate_coverage_report(coverage_data)

        assert "report_content" in result
        assert "coverage_summary" in result
        assert "coverage_trends" in result
        assert "coverage_metadata" in result

        # Verify JSON structure
        report_data = json.loads(result["report_content"])
        assert report_data["report_type"] == "coverage"
        assert "timestamp" in report_data
        assert report_data["files_analyzed"] == ["src/module1.py"]

    def test_generate_analysis_report(self) -> None:
        """Test analysis report generation."""
        result = self.adapter.generate_analysis_report(self.sample_analysis_report)

        assert "report_content" in result
        assert "summary" in result
        assert "recommendations" in result
        assert "analysis_metadata" in result

        # Verify content structure
        report_data = json.loads(result["report_content"])
        assert report_data["report_type"] == "analysis"
        assert report_data["total_files"] == 2
        assert len(report_data["files_without_tests"]) == 1
        assert len(report_data["files_with_tests"]) == 1

        # Verify recommendations are generated
        assert len(result["recommendations"]) > 0

    def test_generate_analysis_report_without_recommendations(self) -> None:
        """Test analysis report generation without recommendations."""
        result = self.adapter.generate_analysis_report(
            self.sample_analysis_report, include_recommendations=False
        )

        report_data = json.loads(result["report_content"])
        assert "recommendations" not in report_data
        assert result["recommendations"] == []

    def test_generate_summary_report(self) -> None:
        """Test summary report generation."""
        project_data = {
            "total_files": 10,
            "files_with_tests": 7,
            "overall_coverage": 0.82,
            "tests_generated": 15,
            "generation_success_rate": 0.93,
        }

        result = self.adapter.generate_summary_report(project_data)

        assert "summary_content" in result
        assert "key_metrics" in result
        assert "trends" in result
        assert "summary_metadata" in result

        # Verify metrics extraction
        metrics = result["key_metrics"]
        assert metrics["total_files"] == 10
        assert metrics["files_with_tests"] == 7
        assert metrics["overall_coverage"] == 0.82

    def test_generate_report_generic(self) -> None:
        """Test generic report generation."""
        test_data = {
            "generation_results": self.sample_generation_results,
            "verbose": True,
            "prompts_used": ["Generate tests for module1"],
            "llm_responses": ["Generated test content..."],
        }

        result = self.adapter.generate_report("generation", test_data)

        assert "report_content" in result
        assert "generation_summary" in result
        assert "generation_metadata" in result

        # Verify verbose details are included
        report_data = json.loads(result["report_content"])
        assert report_data["verbose_details"]
        assert "prompts_used" in report_data
        assert "llm_responses" in report_data

    def test_export_report_json(self) -> None:
        """Test exporting report to JSON file."""
        test_data = {"test": "data", "timestamp": "2024-01-01T00:00:00Z"}
        report_content = json.dumps(test_data, indent=2)
        output_path = self.temp_dir / "test_report.json"

        result = self.adapter.export_report(report_content, output_path, "json")

        assert result["success"]
        assert Path(result["export_path"]).exists()
        assert result["file_size"] > 0

        # Verify file contents
        with open(output_path) as f:
            exported_data = json.load(f)

        assert exported_data == test_data

    def test_export_report_text(self) -> None:
        """Test exporting report to text file."""
        report_content = "This is a test report\nWith multiple lines"
        output_path = self.temp_dir / "test_report.txt"

        result = self.adapter.export_report(report_content, output_path, "text")

        assert result["success"]
        assert Path(result["export_path"]).exists()

        # Verify file contents
        with open(output_path) as f:
            content = f.read()

        assert content == report_content

    def test_unsupported_report_type(self) -> None:
        """Test error handling for unsupported report types."""
        with pytest.raises(ReportError) as exc_info:
            self.adapter.generate_report("unsupported_type", {})

        assert "Unsupported report type" in str(exc_info.value)

    def test_export_nonexistent_directory(self) -> None:
        """Test export to non-existent directory creates directory."""
        nested_path = self.temp_dir / "nested" / "directory" / "report.json"
        test_data = {"test": "data"}

        result = self.adapter.export_report(json.dumps(test_data), nested_path, "json")

        assert result["success"]
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_coverage_summary_calculation(self) -> None:
        """Test coverage summary calculation logic."""
        before_coverage = CoverageResult(
            line_coverage=0.70, branch_coverage=0.60, missing_lines=[1, 2, 3, 4, 5]
        )

        after_coverage = CoverageResult(
            line_coverage=0.85, branch_coverage=0.78, missing_lines=[1, 2]
        )

        summary = self.adapter._calculate_coverage_summary(
            before_coverage, after_coverage
        )

        assert "before" in summary
        assert "after" in summary
        assert "improvement" in summary

        improvement = summary["improvement"]
        assert improvement["line_coverage_delta"] == 0.15
        assert improvement["branch_coverage_delta"] == 0.18
        assert improvement["lines_covered"] == 3  # 5 - 2 = 3 lines now covered

    def test_coverage_trends_analysis(self) -> None:
        """Test coverage trends analysis."""
        before_coverage = CoverageResult(
            line_coverage=0.70, branch_coverage=0.80, missing_lines=[1, 2, 3]
        )

        after_coverage = CoverageResult(
            line_coverage=0.85,  # improved
            branch_coverage=0.75,  # declined
            missing_lines=[1],
        )

        trends = self.adapter._analyze_coverage_trends(before_coverage, after_coverage)

        assert trends["line_coverage_trend"] == "improved"
        assert trends["branch_coverage_trend"] == "declined"
        assert trends["overall_trend"] == "declined"  # Branch declined affects overall

    def test_generation_result_serialization(self) -> None:
        """Test serialization of GenerationResult objects."""
        result = self.sample_generation_results[0]
        serialized = self.adapter._serialize_generation_result(result)

        assert serialized["file_path"] == "tests/test_module1.py"
        assert serialized["success"]
        assert serialized["error_message"] is None

    def test_test_plan_serialization(self) -> None:
        """Test serialization of TestGenerationPlan objects."""
        test_elements = [
            TestElement(
                name="test_function",
                type=TestElementType.FUNCTION,
                line_range=(10, 20),
                docstring="Test function docstring",
            )
        ]

        plan = TestGenerationPlan(
            elements_to_test=test_elements,
            existing_tests=["tests/existing.py"],
            coverage_before=self.sample_coverage_result,
        )

        serialized = self.adapter._serialize_test_plan(plan)

        assert len(serialized["elements_to_test"]) == 1
        assert serialized["elements_to_test"][0]["name"] == "test_function"
        assert serialized["elements_to_test"][0]["type"] == "function"
        assert serialized["existing_tests"] == ["tests/existing.py"]
        assert serialized["coverage_before"] is not None

    def test_analysis_recommendations_generation(self) -> None:
        """Test generation of analysis recommendations."""
        recommendations = self.adapter._generate_analysis_recommendations(
            self.sample_analysis_report
        )

        assert len(recommendations) > 0
        # Should recommend generating tests for files without tests
        assert any("generate tests" in rec.lower() for rec in recommendations)
        # Should provide priority information based on reasons
        assert any("priority focus" in rec.lower() for rec in recommendations)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
