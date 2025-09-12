"""
Tests for Rich UI components.

This module tests the Rich-based UI components including
tables, panels, progress indicators, and the UIPort adapter.
"""

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from testcraft.adapters.io.rich_cli import TESTCRAFT_THEME, RichCliComponents
from testcraft.adapters.io.ui_rich import RichUIAdapter, UIError


class TestRichCliComponents:
    """Test cases for RichCliComponents."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Use StringIO to capture console output for testing
        self.output = StringIO()
        self.console = Console(file=self.output, theme=TESTCRAFT_THEME, width=120)
        self.cli = RichCliComponents(console=self.console)

        # Sample test data
        self.sample_coverage_data = {
            "files": {
                "src/module1.py": {
                    "line_coverage": 0.85,
                    "branch_coverage": 0.78,
                    "missing_lines": [10, 15, 22],
                },
                "src/module2.py": {
                    "line_coverage": 0.95,
                    "branch_coverage": 0.88,
                    "missing_lines": [5],
                },
            },
            "overall_line_coverage": 0.90,
            "overall_branch_coverage": 0.83,
        }

        self.sample_test_results = [
            {
                "source_file": "src/module1.py",
                "test_file": "tests/test_module1.py",
                "status": "success",
                "tests_generated": 5,
                "pass_rate": 0.9,
            },
            {
                "source_file": "src/module2.py",
                "test_file": "tests/test_module2.py",
                "status": "failed",
                "tests_generated": 0,
                "pass_rate": 0.0,
            },
        ]

        self.sample_project_data = {
            "total_files": 10,
            "files_with_tests": 7,
            "overall_coverage": 0.82,
            "tests_generated": 15,
            "generation_success_rate": 0.85,
        }

        self.sample_analysis_data = {
            "files_to_process": ["src/module1.py", "src/module2.py", "src/module3.py"],
            "reasons": {
                "src/module1.py": "No existing tests found",
                "src/module2.py": "Low coverage detected",
                "src/module3.py": "No existing tests found",
            },
            "existing_test_presence": {
                "src/module1.py": False,
                "src/module2.py": True,
                "src/module3.py": False,
            },
        }

    def test_create_coverage_table(self) -> None:
        """Test coverage table creation."""
        table = self.cli.create_coverage_table(self.sample_coverage_data)

        assert isinstance(table, Table)
        assert "Code Coverage Report" in str(table.title)

        # Verify table has expected columns
        column_headers = [col.header for col in table.columns]
        assert "File" in column_headers
        assert "Line Coverage" in column_headers
        assert "Branch Coverage" in column_headers

    def test_create_coverage_table_simple(self) -> None:
        """Test coverage table creation without details."""
        table = self.cli.create_coverage_table(
            self.sample_coverage_data, show_details=False
        )

        assert isinstance(table, Table)
        column_headers = [col.header for col in table.columns]
        assert "Missing Lines" not in column_headers
        assert "Status" not in column_headers

    def test_create_test_results_table(self) -> None:
        """Test test results table creation."""
        table = self.cli.create_test_results_table(self.sample_test_results)

        assert isinstance(table, Table)
        assert "Test Generation Results" in str(table.title)

        # Verify table has expected columns
        column_headers = [col.header for col in table.columns]
        expected_columns = [
            "Source File",
            "Test File",
            "Status",
            "Tests Generated",
            "Pass Rate",
        ]
        for col in expected_columns:
            assert col in column_headers

    def test_create_project_summary_panel(self) -> None:
        """Test project summary panel creation."""
        panel = self.cli.create_project_summary_panel(self.sample_project_data)

        assert isinstance(panel, Panel)
        assert "Project Summary" in str(panel.title)

        # Panel content should contain key metrics
        content = str(panel.renderable)
        assert "Files Analyzed" in content
        assert "Overall Coverage" in content
        assert "Tests Generated" in content

    def test_create_recommendations_panel_with_recommendations(self) -> None:
        """Test recommendations panel with recommendations."""
        recommendations = [
            "Add tests for uncovered functions",
            "Improve branch coverage in module2.py",
            "Consider integration tests",
        ]

        panel = self.cli.create_recommendations_panel(recommendations)

        assert isinstance(panel, Panel)
        assert "Recommendations" in str(panel.title)

        content = str(panel.renderable)
        for rec in recommendations:
            assert rec in content

    def test_create_recommendations_panel_empty(self) -> None:
        """Test recommendations panel with no recommendations."""
        panel = self.cli.create_recommendations_panel([])

        assert isinstance(panel, Panel)
        content = str(panel.renderable)
        assert "No specific recommendations" in content

    def test_create_analysis_tree(self) -> None:
        """Test analysis tree creation."""
        tree = self.cli.create_analysis_tree(self.sample_analysis_data)

        assert isinstance(tree, Tree)
        assert "Analysis Results" in str(tree.label)

    def test_format_coverage_percentage(self) -> None:
        """Test coverage percentage formatting."""
        # Test high coverage (>= 0.85)
        high_coverage = self.cli._format_coverage_percentage(0.95)
        assert "coverage_high" in high_coverage
        assert "95.0%" in high_coverage

        # Test good coverage (>= 0.70, < 0.85)
        good_coverage = self.cli._format_coverage_percentage(0.75)
        assert "coverage_good" in good_coverage
        assert "75.0%" in good_coverage

        # Test medium coverage (>= 0.50, < 0.70)
        medium_coverage = self.cli._format_coverage_percentage(0.55)
        assert "coverage_medium" in medium_coverage
        assert "55.0%" in medium_coverage

        # Test low coverage (< 0.50)
        low_coverage = self.cli._format_coverage_percentage(0.45)
        assert "coverage_low" in low_coverage
        assert "45.0%" in low_coverage

        # Test bold formatting with high coverage
        bold_coverage = self.cli._format_coverage_percentage(0.85, bold=True)
        assert "bold coverage_high" in bold_coverage

    def test_progress_tracker_creation(self) -> None:
        """Test progress tracker creation."""
        progress = self.cli.create_progress_tracker()

        # Should create a Progress instance with expected columns
        assert progress is not None
        # Verify it has spinner and progress columns by checking column count
        assert len(progress.columns) > 3  # Should have multiple columns

    def test_status_spinner_creation(self) -> None:
        """Test status spinner creation."""
        status = self.cli.create_status_spinner("Processing...")

        assert status is not None
        # Status should contain the message
        assert "Processing..." in str(status)

    def test_display_methods(self) -> None:
        """Test various display methods."""
        # Test error display
        self.cli.display_error("Test error message", "Custom Error")
        output = self.output.getvalue()
        assert "Test error message" in output
        assert "Custom Error" in output

        # Reset output
        self.output.truncate(0)
        self.output.seek(0)

        # Test warning display
        self.cli.display_warning("Test warning", "Custom Warning")
        output = self.output.getvalue()
        assert "Test warning" in output
        assert "Custom Warning" in output

        # Reset output
        self.output.truncate(0)
        self.output.seek(0)

        # Test success display
        self.cli.display_success("Operation completed", "Success")
        output = self.output.getvalue()
        assert "Operation completed" in output
        assert "Success" in output

        # Reset output
        self.output.truncate(0)
        self.output.seek(0)

        # Test info display
        self.cli.display_info("Information message", "Info")
        output = self.output.getvalue()
        assert "Information message" in output
        assert "Info" in output

    @patch("rich.prompt.Confirm.ask")
    def test_get_user_confirmation(self, mock_confirm) -> None:
        """Test user confirmation prompt."""
        mock_confirm.return_value = True

        result = self.cli.get_user_confirmation("Continue?", default=False)

        assert result
        mock_confirm.assert_called_once()

    @patch("rich.prompt.Prompt.ask")
    def test_get_user_input(self, mock_prompt) -> None:
        """Test user input prompt."""
        mock_prompt.return_value = "test input"

        result = self.cli.get_user_input("Enter value:", default="default")

        assert result == "test input"
        mock_prompt.assert_called_once()

    def test_create_comprehensive_layout(self) -> None:
        """Test comprehensive layout creation."""
        layout = self.cli.create_comprehensive_layout(
            self.sample_project_data,
            self.sample_coverage_data,
            self.sample_test_results,
            ["Test recommendation"],
        )

        assert layout is not None
        # Layout should have multiple sections
        assert "header" in layout._layout_items
        assert "body" in layout._layout_items
        assert "footer" in layout._layout_items

    def test_print_methods(self) -> None:
        """Test print methods don't raise errors."""
        # Create sample components
        table = self.cli.create_coverage_table(self.sample_coverage_data)
        panel = self.cli.create_project_summary_panel(self.sample_project_data)
        tree = self.cli.create_analysis_tree(self.sample_analysis_data)
        layout = self.cli.create_comprehensive_layout(
            self.sample_project_data,
            self.sample_coverage_data,
            self.sample_test_results,
            [],
        )

        # Test print methods (should not raise exceptions)
        self.cli.print_table(table)
        self.cli.print_panel(panel)
        self.cli.print_tree(tree)
        self.cli.print_layout(layout)
        self.cli.print_divider("Test Section")
        self.cli.print_divider()  # Without title

    def test_enhanced_format_coverage_percentage(self) -> None:
        """Test enhanced coverage percentage formatting with icons."""
        # Test excellent coverage (>= 0.95)
        excellent = self.cli._format_coverage_percentage(0.97, with_icon=True)
        assert "coverage_excellent" in excellent
        assert "🟢" in excellent
        assert "97.0%" in excellent

        # Test high coverage (>= 0.85)
        high = self.cli._format_coverage_percentage(0.88, with_icon=True)
        assert "coverage_high" in high
        assert "🟢" in high

        # Test good coverage (>= 0.70)
        good = self.cli._format_coverage_percentage(0.75, with_icon=True)
        assert "coverage_good" in good
        assert "🟡" in good

        # Test medium coverage (>= 0.50)
        medium = self.cli._format_coverage_percentage(0.55, with_icon=True)
        assert "coverage_medium" in medium
        assert "🟠" in medium

        # Test low coverage (< 0.50)
        low = self.cli._format_coverage_percentage(0.25, with_icon=True)
        assert "coverage_low" in low
        assert "🔴" in low

        # Test without icons
        no_icon = self.cli._format_coverage_percentage(0.85, with_icon=False)
        assert "🟢" not in no_icon
        assert "85.0%" in no_icon

        # Test bold formatting
        bold = self.cli._format_coverage_percentage(0.90, bold=True, with_icon=True)
        assert "bold coverage_high" in bold

    def test_display_code_snippet(self) -> None:
        """Test code snippet display with syntax highlighting."""
        code = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b
'''

        # Test with title
        self.cli.display_code_snippet(code, "python", "Sum Function", True)
        output = self.output.getvalue()
        assert len(output) > 0

        # Reset output
        self.output.truncate(0)
        self.output.seek(0)

        # Test without title
        self.cli.display_code_snippet(code, "python", None, True)
        output = self.output.getvalue()
        assert len(output) > 0

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_create_configuration_wizard(self, mock_prompt, mock_confirm) -> None:
        """Test configuration wizard creation."""
        # Mock user inputs
        mock_prompt.side_effect = ["TestApp", "5", "1"]  # string, number, choice
        mock_confirm.side_effect = [True, True]  # boolean, save confirmation

        config_steps = [
            {
                "title": "Application Settings",
                "description": "Configure your application",
                "fields": [
                    {
                        "name": "app_name",
                        "title": "Application Name",
                        "type": "string",
                        "default": "DefaultApp",
                        "required": True,
                    },
                    {
                        "name": "max_workers",
                        "title": "Max Workers",
                        "type": "number",
                        "default": 4,
                        "integer": True,
                    },
                    {
                        "name": "log_level",
                        "title": "Log Level",
                        "type": "choice",
                        "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "default": 1,
                    },
                    {
                        "name": "enable_cache",
                        "title": "Enable Caching",
                        "type": "boolean",
                        "default": True,
                    },
                ],
            }
        ]

        result = self.cli.create_configuration_wizard(config_steps)

        # Should return expected configuration
        expected = {
            "app_name": "TestApp",
            "max_workers": 5,
            "log_level": "DEBUG",
            "enable_cache": True,
        }
        assert result == expected


class TestRichUIAdapter:
    """Test cases for RichUIAdapter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.output = StringIO()
        self.console = Console(file=self.output, theme=TESTCRAFT_THEME, width=120)
        self.ui = RichUIAdapter(console=self.console)

    def test_display_progress_spinner(self) -> None:
        """Test spinner progress display."""
        progress_data = {"current": 0, "total": 100, "message": "Processing files..."}

        # Should not raise exception
        self.ui.display_progress(progress_data, progress_type="spinner")

        # Spinner should be active
        assert self.ui._active_status is not None

        # Clean up
        self.ui._stop_progress_indicators()

    def test_display_progress_bar(self) -> None:
        """Test progress bar display."""
        progress_data = {
            "current": 50,
            "total": 100,
            "message": "Generating tests...",
            "percentage": 0.5,
        }

        # Should not raise exception
        self.ui.display_progress(progress_data, progress_type="bar")

        # Progress tracker should be active
        assert self.ui._active_progress is not None

        # Clean up
        self.ui._stop_progress_indicators()

    def test_display_coverage_results(self) -> None:
        """Test coverage results display."""
        results = {
            "success": True,
            "summary": "Coverage analysis completed successfully",
            "details": {
                "files": {
                    "src/test.py": {
                        "line_coverage": 0.85,
                        "branch_coverage": 0.78,
                        "missing_lines": [10, 15],
                    }
                }
            },
        }

        # Should not raise exception
        self.ui.display_results(results, result_type="coverage")

        # Check that output contains expected content
        output = self.output.getvalue()
        assert "Coverage analysis completed" in output

    def test_display_test_generation_results(self) -> None:
        """Test test generation results display."""
        results = {
            "success": True,
            "summary": "Test generation completed",
            "details": {
                "test_results": [
                    {
                        "source_file": "src/module.py",
                        "test_file": "tests/test_module.py",
                        "status": "success",
                        "tests_generated": 3,
                        "pass_rate": 0.8,
                    }
                ]
            },
        }

        # Should not raise exception
        self.ui.display_results(results, result_type="test_generation")

        output = self.output.getvalue()
        assert "Test generation completed" in output

    def test_display_error(self) -> None:
        """Test error display."""
        self.ui.display_error("Something went wrong", "Test Error")

        output = self.output.getvalue()
        assert "Something went wrong" in output
        assert "Test Error" in output

    def test_display_error_with_details(self) -> None:
        """Test error display with details."""
        self.ui.display_error(
            "Main error message",
            "Error Type",
            details={"detail1": "value1", "detail2": "value2"},
        )

        output = self.output.getvalue()
        assert "Main error message" in output
        assert "detail1" in output
        assert "value1" in output

    def test_display_warning(self) -> None:
        """Test warning display."""
        self.ui.display_warning("This is a warning", "Warning Type")

        output = self.output.getvalue()
        assert "This is a warning" in output
        assert "Warning Type" in output

    def test_display_info(self) -> None:
        """Test info display."""
        self.ui.display_info("Information message", "Info Type")

        output = self.output.getvalue()
        assert "Information message" in output
        assert "Info Type" in output

    @patch("testcraft.adapters.io.rich_cli.RichCliComponents.get_user_confirmation")
    def test_get_user_input_boolean(self, mock_confirm) -> None:
        """Test boolean user input."""
        mock_confirm.return_value = True

        result = self.ui.get_user_input(
            "Continue?", input_type="boolean", default=False
        )

        assert result
        mock_confirm.assert_called_once_with("Continue?", False)

    @patch("testcraft.adapters.io.rich_cli.RichCliComponents.get_user_input")
    def test_get_user_input_string(self, mock_input) -> None:
        """Test string user input."""
        mock_input.return_value = "test response"

        result = self.ui.get_user_input(
            "Enter name:", input_type="string", default="default"
        )

        assert result == "test response"
        mock_input.assert_called_once_with("Enter name:", "default")

    @patch("testcraft.adapters.io.rich_cli.RichCliComponents.get_user_input")
    def test_get_user_input_number(self, mock_input) -> None:
        """Test number user input."""
        mock_input.return_value = "42"

        result = self.ui.get_user_input("Enter number:", input_type="number")

        assert result == 42.0

    @patch("testcraft.adapters.io.rich_cli.RichCliComponents.get_user_input")
    def test_get_user_input_choice(self, mock_input) -> None:
        """Test choice user input."""
        mock_input.return_value = "2"

        result = self.ui.get_user_input(
            "Select option:",
            input_type="choice",
            choices=["Option A", "Option B", "Option C"],
        )

        assert result == "Option B"

    def test_get_user_input_choice_no_choices(self) -> None:
        """Test choice input without choices raises error."""
        with pytest.raises(UIError) as exc_info:
            self.ui.get_user_input("Select:", input_type="choice")

        assert "Choices must be provided" in str(exc_info.value)

    @patch("testcraft.adapters.io.rich_cli.RichCliComponents.get_user_confirmation")
    def test_confirm_action(self, mock_confirm) -> None:
        """Test action confirmation."""
        mock_confirm.return_value = True

        result = self.ui.confirm_action("Delete file?", default=False)

        assert result
        mock_confirm.assert_called_once_with("Delete file?", False)

    def test_stop_progress_indicators(self) -> None:
        """Test stopping progress indicators."""
        # Start some indicators
        self.ui.display_progress({"message": "test"}, progress_type="spinner")
        self.ui.display_progress({"message": "test", "total": 100}, progress_type="bar")

        assert self.ui._active_status is not None
        assert self.ui._active_progress is not None

        # Stop indicators
        self.ui._stop_progress_indicators()

        assert self.ui._active_status is None
        assert self.ui._active_progress is None

    def test_finalize(self) -> None:
        """Test UI finalization."""
        # Start indicators
        self.ui.display_progress({"message": "test"}, progress_type="spinner")

        # Finalize should clean up
        self.ui.finalize()

        assert self.ui._active_status is None
        assert self.ui._active_progress is None

    def test_print_divider(self) -> None:
        """Test divider printing."""
        self.ui.print_divider("Test Section")
        self.ui.print_divider()  # Without title

        # Should not raise exceptions
        output = self.output.getvalue()
        assert len(output) > 0  # Should have produced some output

    def test_set_quiet_mode(self) -> None:
        """Test setting quiet mode."""
        # Initially not quiet
        assert not self.ui.console.quiet

        # Enable quiet mode
        self.ui.set_quiet_mode(True)
        assert self.ui.console.quiet

        # Disable quiet mode
        self.ui.set_quiet_mode(False)
        assert not self.ui.console.quiet

    def test_error_handling_in_display_methods(self) -> None:
        """Test error handling in display methods."""
        # Mock console to raise exception
        with patch.object(
            self.ui.rich_cli, "display_error", side_effect=Exception("Test error")
        ):
            with pytest.raises(UIError) as exc_info:
                self.ui.display_error("test message")

            assert "Failed to display error" in str(exc_info.value)

    def test_run_configuration_wizard(self) -> None:
        """Test configuration wizard execution."""
        config_steps = [
            {
                "title": "Basic Settings",
                "description": "Configure basic application settings",
                "fields": [
                    {
                        "name": "app_name",
                        "title": "Application Name",
                        "type": "string",
                        "default": "TestCraft",
                    },
                    {
                        "name": "enable_feature",
                        "title": "Enable Feature",
                        "type": "boolean",
                        "default": True,
                    },
                ],
            }
        ]

        # Mock the wizard to return expected results
        with patch.object(
            self.ui.rich_cli, "create_configuration_wizard"
        ) as mock_wizard:
            mock_wizard.return_value = {"app_name": "TestCraft", "enable_feature": True}

            result = self.ui.run_configuration_wizard(config_steps)

            assert result == {"app_name": "TestCraft", "enable_feature": True}
            mock_wizard.assert_called_once_with(config_steps)

    def test_display_code_snippet(self) -> None:
        """Test code snippet display with syntax highlighting."""
        code = """
def hello_world():
    print("Hello, TestCraft!")
    return True
"""

        # Should not raise exception
        self.ui.display_code_snippet(code, "python", "Example Function")

        # Check that output contains code
        output = self.output.getvalue()
        assert len(output) > 0  # Should have produced output

    def test_create_beautiful_summary_comprehensive(self) -> None:
        """Test beautiful summary with comprehensive layout."""
        data = {
            "summary_data": {"total_files": 10, "files_with_tests": 8},
            "coverage_data": {
                "files": {"test.py": {"line_coverage": 0.9, "branch_coverage": 0.8}}
            },
            "test_results": [
                {"source_file": "test.py", "status": "success", "tests_generated": 5}
            ],
            "recommendations": ["Add more tests"],
        }

        # Should not raise exception
        self.ui.create_beautiful_summary(data, layout_style="comprehensive")

        # Check that output was produced
        output = self.output.getvalue()
        assert len(output) > 0

    def test_create_beautiful_summary_simple(self) -> None:
        """Test beautiful summary with simple layout."""
        data = {
            "summary_data": {"total_files": 5, "files_with_tests": 4},
            "coverage_data": {
                "files": {"module.py": {"line_coverage": 0.85, "branch_coverage": 0.75}}
            },
            "test_results": [],
            "recommendations": [],
        }

        # Should not raise exception
        self.ui.create_beautiful_summary(data, layout_style="simple")

        # Check that output was produced
        output = self.output.getvalue()
        assert len(output) > 0

    def test_enhanced_theme_colors(self) -> None:
        """Test that enhanced theme colors are available."""
        from testcraft.adapters.io.rich_cli import TESTCRAFT_THEME

        # Check that enhanced theme has new colors
        theme_styles = TESTCRAFT_THEME.styles

        # Core colors should exist
        assert "success" in theme_styles
        assert "error" in theme_styles
        assert "warning" in theme_styles
        assert "info" in theme_styles

        # Enhanced coverage colors should exist
        assert "coverage_excellent" in theme_styles
        assert "coverage_high" in theme_styles
        assert "coverage_good" in theme_styles
        assert "coverage_medium" in theme_styles
        assert "coverage_low" in theme_styles

        # Status indicators should exist
        assert "status_pass" in theme_styles
        assert "status_fail" in theme_styles
        assert "status_skip" in theme_styles
        assert "status_partial" in theme_styles

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.ui.finalize()
