"""
Tests for the python_formatters module.

This module tests Python-specific code formatting utilities
built on top of the subprocess_safe module.
"""

from unittest.mock import patch

from testcraft.adapters.io.python_formatters import (
    format_python_content,
    run_black_safe,
    run_formatter_safe,
    run_isort_safe,
)
from testcraft.adapters.io.subprocess_safe import (
    SubprocessExecutionError,
    SubprocessTimeoutError,
)


class TestRunFormatterSafe:
    """Test the formatter-specific subprocess function."""

    @patch("testcraft.adapters.io.python_formatters.run_subprocess_safe")
    def test_run_formatter_safe_success(self, mock_safe):
        """Test formatter function with successful execution."""
        # Mock successful formatting
        mock_safe.return_value.__enter__.return_value = ("", "")

        # Create a mock temporary file
        content = "import os\ndef test(): pass"

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("pathlib.Path.read_text") as mock_read,
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            # Setup temp file mock
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.py"
            mock_read.return_value = "formatted content"

            result = run_formatter_safe(["python", "-m", "black"], content)

            assert result == "formatted content"
            mock_unlink.assert_called_once_with(missing_ok=True)

    @patch("testcraft.adapters.io.python_formatters.run_subprocess_safe")
    def test_run_formatter_safe_failure(self, mock_safe):
        """Test formatter function with failed execution."""
        # Mock failed formatting
        mock_safe.side_effect = SubprocessExecutionError("Black failed")

        content = "import os\ndef test(): pass"

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.py"

            result = run_formatter_safe(["python", "-m", "black"], content)

            # Should return original content on failure
            assert result == content
            mock_unlink.assert_called_once_with(missing_ok=True)

    @patch("testcraft.adapters.io.python_formatters.run_subprocess_safe")
    def test_run_formatter_safe_timeout(self, mock_safe):
        """Test formatter function with timeout."""
        # Mock timeout
        mock_safe.side_effect = SubprocessTimeoutError("Formatter timed out")

        content = "import os\ndef test(): pass"

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.py"

            result = run_formatter_safe(["python", "-m", "black"], content)

            # Should return original content on timeout
            assert result == content
            mock_unlink.assert_called_once_with(missing_ok=True)

    @patch("testcraft.adapters.io.python_formatters.run_subprocess_safe")
    def test_run_formatter_safe_custom_timeout(self, mock_safe):
        """Test formatter function with custom timeout."""
        # Mock successful formatting
        mock_safe.return_value.__enter__.return_value = ("", "")

        content = "import os\ndef test(): pass"

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("pathlib.Path.read_text") as mock_read,
            patch("pathlib.Path.unlink"),
        ):
            # Setup temp file mock
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.py"
            mock_read.return_value = "formatted content"

            result = run_formatter_safe(["python", "-m", "black"], content, timeout=60)

            # Should pass custom timeout to run_subprocess_safe
            mock_safe.assert_called_once_with(
                ["python", "-m", "black", "/tmp/test.py"], timeout=60
            )
            assert result == "formatted content"


class TestPythonFormatters:
    """Test the Python-specific formatter functions."""

    @patch("testcraft.adapters.io.python_formatters.run_formatter_safe")
    def test_run_black_safe(self, mock_formatter):
        """Test the Black formatter wrapper."""
        mock_formatter.return_value = "formatted"

        result = run_black_safe("code", timeout=60)

        mock_formatter.assert_called_once_with(
            ["python", "-m", "black", "--quiet"], "code", temp_suffix=".py", timeout=60
        )
        assert result == "formatted"

    @patch("testcraft.adapters.io.python_formatters.run_formatter_safe")
    def test_run_isort_safe(self, mock_formatter):
        """Test the isort formatter wrapper."""
        mock_formatter.return_value = "formatted"

        result = run_isort_safe("code")

        mock_formatter.assert_called_once_with(
            ["python", "-m", "isort", "--quiet"], "code", temp_suffix=".py", timeout=30
        )
        assert result == "formatted"

    @patch("testcraft.adapters.io.python_formatters.run_isort_safe")
    @patch("testcraft.adapters.io.python_formatters.run_black_safe")
    def test_format_python_content(self, mock_black, mock_isort):
        """Test the combined Python formatter."""
        mock_isort.return_value = "isort_formatted"
        mock_black.return_value = "fully_formatted"

        result = format_python_content("code", timeout=45)

        # Should call isort first, then black
        mock_isort.assert_called_once_with("code", 45)
        mock_black.assert_called_once_with("isort_formatted", 45)
        assert result == "fully_formatted"

    def test_format_python_content_default_timeout(self):
        """Test that format_python_content uses default timeout."""
        with (
            patch(
                "testcraft.adapters.io.python_formatters.run_isort_safe"
            ) as mock_isort,
            patch(
                "testcraft.adapters.io.python_formatters.run_black_safe"
            ) as mock_black,
        ):
            mock_isort.return_value = "isort_formatted"
            mock_black.return_value = "fully_formatted"

            result = format_python_content("code")

            # Should call both with default timeout of 30
            mock_isort.assert_called_once_with("code", 30)
            mock_black.assert_called_once_with("isort_formatted", 30)
            assert result == "fully_formatted"


class TestIntegration:
    """Integration tests using actual formatter commands (mocked at subprocess level)."""

    def test_formatter_command_construction(self):
        """Test that formatter commands are constructed correctly."""
        with patch(
            "testcraft.adapters.io.python_formatters.run_subprocess_safe"
        ) as mock_safe:
            mock_safe.return_value.__enter__.return_value = ("", "")

            with (
                patch("tempfile.NamedTemporaryFile") as mock_temp,
                patch("pathlib.Path.read_text") as mock_read,
                patch("pathlib.Path.unlink"),
            ):
                # Setup temp file mock
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.py"
                mock_read.return_value = "formatted"

                # Test Black command construction
                run_black_safe("code")
                mock_safe.assert_called_with(
                    ["python", "-m", "black", "--quiet", "/tmp/test.py"], timeout=30
                )

                mock_safe.reset_mock()

                # Test isort command construction
                run_isort_safe("code")
                mock_safe.assert_called_with(
                    ["python", "-m", "isort", "--quiet", "/tmp/test.py"], timeout=30
                )
