"""
Test suite for Minimal UI implementation.

Tests the new minimal UI functionality including:
- UI style detection and selection
- Minimal theme application
- Compact rendering without duplication
- Live tracking with minimal layout
"""

import os
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from testcraft.adapters.io.enhanced_logging import LoggerManager, LogMode
from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter, MinimalRenderer
from testcraft.adapters.io.rich_cli import MINIMAL_THEME, TESTCRAFT_THEME, get_theme
from testcraft.adapters.io.ui_rich import UIStyle
from testcraft.cli.main import _display_immediate_mode_results, detect_ui_style


class TestUIStyleDetection:
    """Test UI style detection logic."""

    def test_explicit_flag_overrides_environment(self):
        """Test that --ui flag takes precedence over environment variables."""
        # Test minimal flag
        assert detect_ui_style("minimal") == UIStyle.MINIMAL
        assert detect_ui_style("classic") == UIStyle.CLASSIC

    @patch.dict(os.environ, {"TESTCRAFT_UI": "minimal"})
    def test_environment_variable_detection(self):
        """Test that TESTCRAFT_UI environment variable is respected."""
        assert detect_ui_style(None) == UIStyle.MINIMAL

    @patch.dict(os.environ, {"CI": "true"})
    @patch(
        "sys.stdout.isatty", return_value=True
    )  # Even if TTY, CI should force minimal
    def test_ci_environment_forces_minimal(self, mock_isatty):
        """Test that CI=true environment forces minimal UI."""
        assert detect_ui_style(None) == UIStyle.MINIMAL

    @patch("sys.stdout.isatty", return_value=False)
    def test_non_tty_forces_minimal(self, mock_isatty):
        """Test that non-TTY output forces minimal UI."""
        assert detect_ui_style(None) == UIStyle.MINIMAL

    @patch("sys.stdout.isatty", return_value=True)
    @patch.dict(os.environ, {}, clear=True)  # Clear environment
    def test_interactive_terminal_defaults_classic(self, mock_isatty):
        """Test that interactive terminal defaults to classic UI."""
        assert detect_ui_style(None) == UIStyle.CLASSIC


class TestMinimalTheme:
    """Test minimal theme configuration."""

    def test_get_theme_returns_minimal_for_minimal_style(self):
        """Test that get_theme returns MINIMAL_THEME for minimal style."""
        theme = get_theme(UIStyle.MINIMAL)
        assert theme == MINIMAL_THEME

    def test_get_theme_returns_classic_for_classic_style(self):
        """Test that get_theme returns TESTCRAFT_THEME for classic style."""
        theme = get_theme(UIStyle.CLASSIC)
        assert theme == TESTCRAFT_THEME

    @pytest.mark.skip(reason="Theme color restriction not yet implemented")
    def test_minimal_theme_color_restriction(self):
        """Test that minimal theme uses only core colors."""
        # Core colors should be different
        core_colors = {
            MINIMAL_THEME.styles["success"],
            MINIMAL_THEME.styles["error"],
            MINIMAL_THEME.styles["status_working"],
            MINIMAL_THEME.styles["accent"],
        }

        # Should have 4 distinct core colors (green, red, yellow, cyan)
        assert len(core_colors) == 4

        # All other colors should map to these core colors or basic whites/dims
        allowed_values = core_colors | {"white", "dim white", "dim"}

        for key, value in MINIMAL_THEME.styles.items():
            assert value in allowed_values, (
                f"Color '{key}': '{value}' not in allowed minimal set"
            )


class TestEnhancedUIAdapterMinimal:
    """Test EnhancedUIAdapter with minimal UI style."""

    def test_ui_adapter_stores_ui_style(self):
        """Test that EnhancedUIAdapter stores the UI style."""
        console = Console()
        ui = EnhancedUIAdapter(
            console, enable_rich_logging=False, ui_style=UIStyle.MINIMAL
        )
        assert ui.ui_style == UIStyle.MINIMAL

        ui_classic = EnhancedUIAdapter(
            console, enable_rich_logging=False, ui_style=UIStyle.CLASSIC
        )
        assert ui_classic.ui_style == UIStyle.CLASSIC

    def test_get_renderer_returns_minimal_for_minimal_style(self):
        """Test that get_renderer returns MinimalRenderer for minimal style."""
        console = Console()
        ui = EnhancedUIAdapter(
            console, enable_rich_logging=False, ui_style=UIStyle.MINIMAL
        )
        renderer = ui.get_renderer()
        assert isinstance(renderer, MinimalRenderer)

    def test_get_renderer_returns_self_for_classic_style(self):
        """Test that get_renderer returns self for classic style."""
        console = Console()
        ui = EnhancedUIAdapter(
            console, enable_rich_logging=False, ui_style=UIStyle.CLASSIC
        )
        renderer = ui.get_renderer()
        assert renderer is ui


class TestMinimalRenderer:
    """Test MinimalRenderer output format."""

    def test_render_generation_results_single_line_summary(self):
        """Test that MinimalRenderer produces a single-line summary."""
        console = Mock()
        renderer = MinimalRenderer()

        # Mock results with essential data
        results = {
            "files_written": 5,
            "files_processed": 7,
            "tests_generated": 23,
            "total_duration": 45.3,
            "coverage_delta": {"line_coverage_delta": 0.15},
            "generation_results": [],
            "refinement_results": [],
        }

        renderer.render_generation_results(results, console)

        # Verify console.print was called for summary line
        console.print.assert_called()

        # Get the summary line (first call to print)
        summary_call = console.print.call_args_list[0]
        summary_line = summary_call[0][0]  # First positional argument

        # Verify summary format: "done {files_written}/{files_processed} • tests {tests_generated} • Δcov {coverage_delta:+.1%} • time {total_duration:.1f}s"
        assert "done 5/7" in summary_line
        assert "tests 23" in summary_line
        assert "Δcov +15.0%" in summary_line  # 0.15 formatted as +15.0%
        assert "time 45.3s" in summary_line
        assert " • " in summary_line  # Uses bullet separator

    def test_render_generation_results_no_coverage_delta_when_zero(self):
        """Test that coverage delta is omitted when zero."""
        console = Mock()
        renderer = MinimalRenderer()

        results = {
            "files_written": 3,
            "files_processed": 3,
            "tests_generated": 12,
            "total_duration": 30.0,
            "coverage_delta": {"line_coverage_delta": 0.0},  # Zero delta
            "generation_results": [],
            "refinement_results": [],
        }

        renderer.render_generation_results(results, console)

        summary_call = console.print.call_args_list[0]
        summary_line = summary_call[0][0]

        # Should not include coverage delta
        assert "Δcov" not in summary_line
        assert "done 3/3" in summary_line
        assert "tests 12" in summary_line
        assert "time 30.0s" in summary_line

    def test_render_generation_results_single_file_omits_table(self):
        """Single-file runs should print only the summary line (no table)."""
        console = Mock()
        renderer = MinimalRenderer()
        results = {
            "files_written": 1,
            "files_processed": 1,
            "tests_generated": 5,
            "total_duration": 12.0,
            "generation_results": [{"file_path": "foo.py", "success": True}],
            "refinement_results": [
                {
                    "test_file": "foo_test.py",
                    "success": True,
                    "duration": 3.2,
                    "tests_generated": 5,
                }
            ],
        }
        renderer.render_generation_results(results, console)
        # Only summary printed
        assert console.print.call_count == 1


class TestMinimalLoggingPolicy:
    """Tests for minimal logging sanitization and levels."""

    def test_minimal_prepare_message_strips_tags_and_emojis(self):
        LoggerManager.set_log_mode(LogMode.MINIMAL, verbose=False, quiet=False)
        msg = "[primary]hello[/] ✅ done"
        sanitized = LoggerManager._prepare_message(msg)
        assert "[" not in sanitized and "]" not in sanitized
        assert "✅" not in sanitized
        assert "hello" in sanitized and "done" in sanitized


class TestClassicImmediateNoDuplication:
    """Ensure classic immediate mode does not duplicate table and summary."""

    @pytest.mark.skip(
        reason="Immediate mode results functionality not yet fully implemented"
    )
    def test_immediate_mode_results_no_duplicate_summary(self):
        ui = Mock()
        # Prepare minimal set to allow call without errors
        ui.display_file_progress_table = Mock()
        ui.display_success_summary = Mock()

        results = {
            "generation_results": [{"file_path": "a.py", "success": True}],
            "refinement_results": [
                {
                    "test_file": "a.py",
                    "success": True,
                    "duration": 1.0,
                    "tests_generated": 3,
                }
            ],
            "files_written": 1,
            "files_processed": 1,
            "total_duration": 1.0,
        }

        _display_immediate_mode_results(results, ui)
        ui.display_file_progress_table.assert_called_once()
        ui.display_success_summary.assert_not_called()


class TestMinimalRendererTableFormat:
    """Test MinimalRenderer table formatting."""

    @pytest.mark.skip(reason="Compact table minimal styling not yet implemented")
    def test_compact_table_uses_minimal_styling(self):
        """Test that _render_compact_table uses proper minimal styling."""
        console = Mock()
        renderer = MinimalRenderer()

        files_data = [
            {
                "file_path": "test_file.py",
                "status": "completed",
                "progress": 1.0,
                "tests_generated": 5,
                "duration": 12.3,
            }
        ]

        # This should not raise any errors and should call console.print with a Table
        renderer._render_compact_table(files_data, console)

        # Verify console.print was called with a table
        console.print.assert_called_once()

        # The table should be passed as the argument
        table_arg = console.print.call_args[0][0]

        # Verify it's a Rich Table with the expected properties
        from rich.table import Table

        assert isinstance(table_arg, Table)

        # Check table configuration (these are set in _render_compact_table)
        assert table_arg.box is None  # No borders
        assert table_arg.padding == (0, 1)  # Minimal padding

        # Check column headers are lowercase
        column_headers = [col.header for col in table_arg.columns]
        expected_headers = ["file", "status", "progress", "tests", "time"]
        assert column_headers == expected_headers


# Integration test placeholder - would require more complex setup
class TestMinimalUIIntegration:
    """Integration tests for minimal UI workflow."""

    def test_minimal_ui_produces_compact_output(self):
        """Test that minimal UI produces significantly less output than classic."""
        # This would be a more complex integration test that actually runs
        # a command with both UI styles and compares output line counts.
        # For now, we'll keep it as a placeholder to demonstrate the concept.
        pytest.skip("Integration test placeholder - requires full CLI setup")

    def test_logging_no_duplication_in_minimal_mode(self):
        """Test that logging doesn't produce duplicates in minimal mode."""
        # This would test the logging centralization from Phase 1
        pytest.skip("Integration test placeholder - requires logging capture setup")


if __name__ == "__main__":
    pytest.main([__file__])
