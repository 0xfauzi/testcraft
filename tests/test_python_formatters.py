"""
Tests for the python_formatters module.

This module tests Python-specific code formatting utilities
built on top of the subprocess_safe module, including the smart
formatter selection system with Ruff support.
"""

import subprocess
from unittest.mock import patch, Mock

import pytest

from testcraft.adapters.io.python_formatters import (FormatterDetector, 
                                                     format_python_content,
                                                     format_with_ruff,
                                                     format_with_black_isort,
                                                     run_black_safe,
                                                     run_formatter_safe,
                                                     run_isort_safe,
                                                     run_ruff_format_safe,
                                                     run_ruff_import_sort_safe)
from testcraft.adapters.io.subprocess_safe import (SubprocessExecutionError,
                                                   SubprocessTimeoutError)


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


class TestFormatterDetector:
    """Test the FormatterDetector class for runtime formatter detection."""
    
    def setup_method(self):
        """Reset cached detection results before each test."""
        FormatterDetector._ruff_available = None
        FormatterDetector._black_available = None 
        FormatterDetector._isort_available = None
    
    @patch('subprocess.run')
    def test_ruff_available_success(self, mock_run):
        """Test Ruff detection when available."""
        mock_run.return_value = Mock(returncode=0)
        
        result = FormatterDetector.is_ruff_available()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["ruff", "--version"], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            timeout=5
        )
    
    @patch('subprocess.run')  
    def test_ruff_not_available(self, mock_run):
        """Test Ruff detection when not available."""
        mock_run.side_effect = FileNotFoundError()
        
        result = FormatterDetector.is_ruff_available()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_ruff_detection_cached(self, mock_run):
        """Test that Ruff detection is cached."""
        mock_run.return_value = Mock(returncode=0)
        
        # First call
        result1 = FormatterDetector.is_ruff_available()
        # Second call
        result2 = FormatterDetector.is_ruff_available()
        
        assert result1 is True
        assert result2 is True
        # Should only call subprocess once due to caching
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_black_available_success(self, mock_run):
        """Test Black detection when available."""
        mock_run.return_value = Mock(returncode=0)
        
        result = FormatterDetector.is_black_available()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["python", "-m", "black", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_isort_available_success(self, mock_run):
        """Test isort detection when available."""
        mock_run.return_value = Mock(returncode=0)
        
        result = FormatterDetector.is_isort_available()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["python", "-m", "isort", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_get_available_formatters(self, mock_run):
        """Test getting summary of all available formatters."""
        # Mock Ruff available, Black available, isort not available
        def mock_subprocess_calls(cmd, **kwargs):
            if cmd[0] == "ruff":
                return Mock(returncode=0)
            elif "black" in cmd:
                return Mock(returncode=0) 
            elif "isort" in cmd:
                raise FileNotFoundError()
                
        mock_run.side_effect = mock_subprocess_calls
        
        result = FormatterDetector.get_available_formatters()
        
        expected = {"ruff": True, "black": True, "isort": False}
        assert result == expected


class TestRuffFormatters:
    """Test Ruff-specific formatting functions."""
    
    @patch("testcraft.adapters.io.python_formatters.run_formatter_safe")
    def test_run_ruff_format_safe(self, mock_formatter):
        """Test the Ruff format wrapper."""
        mock_formatter.return_value = "ruff_formatted"
        
        result = run_ruff_format_safe("code", timeout=60)
        
        mock_formatter.assert_called_once_with(
            ["ruff", "format", "--stdin-filename", "temp.py"], 
            "code", 
            temp_suffix=".py", 
            timeout=60
        )
        assert result == "ruff_formatted"
    
    @patch("testcraft.adapters.io.python_formatters.run_formatter_safe")
    def test_run_ruff_import_sort_safe(self, mock_formatter):
        """Test the Ruff import sort wrapper."""
        mock_formatter.return_value = "ruff_sorted"
        
        result = run_ruff_import_sort_safe("code")
        
        mock_formatter.assert_called_once_with(
            ["ruff", "check", "--select", "I", "--fix", "--stdin-filename", "temp.py"],
            "code",
            temp_suffix=".py",
            timeout=30
        )
        assert result == "ruff_sorted"
    
    @patch("testcraft.adapters.io.python_formatters.run_ruff_format_safe")
    @patch("testcraft.adapters.io.python_formatters.run_ruff_import_sort_safe")
    def test_format_with_ruff(self, mock_import_sort, mock_format):
        """Test combined Ruff formatting (import sort + format)."""
        mock_import_sort.return_value = "imports_sorted"
        mock_format.return_value = "fully_formatted"
        
        result = format_with_ruff("code", timeout=45)
        
        # Should call import sort first, then format
        mock_import_sort.assert_called_once_with("code", 45)
        mock_format.assert_called_once_with("imports_sorted", 45)
        assert result == "fully_formatted"
    
    @patch("testcraft.adapters.io.python_formatters.run_black_safe") 
    @patch("testcraft.adapters.io.python_formatters.run_isort_safe")
    def test_format_with_black_isort(self, mock_isort, mock_black):
        """Test legacy Black + isort formatting."""
        mock_isort.return_value = "isort_formatted"
        mock_black.return_value = "fully_formatted"
        
        result = format_with_black_isort("code", timeout=45)
        
        # Should call isort first, then black
        mock_isort.assert_called_once_with("code", 45)
        mock_black.assert_called_once_with("isort_formatted", 45)
        assert result == "fully_formatted"


class TestSmartFormatterSelection:
    """Test the smart formatter selection logic."""
    
    def setup_method(self):
        """Reset cached detection results before each test."""
        FormatterDetector._ruff_available = None
        FormatterDetector._black_available = None
        FormatterDetector._isort_available = None
    
    @patch("testcraft.adapters.io.python_formatters.format_with_ruff")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_ruff_available")
    def test_format_python_content_uses_ruff_when_available(self, mock_ruff_available, mock_format_ruff):
        """Test that Ruff is used when available (preferred)."""
        mock_ruff_available.return_value = True
        mock_format_ruff.return_value = "ruff_formatted"
        
        result = format_python_content("code")
        
        assert result == "ruff_formatted"
        mock_format_ruff.assert_called_once_with("code", 30)
    
    @patch("testcraft.adapters.io.python_formatters.format_with_black_isort")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_isort_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_black_available") 
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_ruff_available")
    def test_format_python_content_fallback_to_black_isort(
        self, mock_ruff_available, mock_black_available, mock_isort_available, mock_format_black_isort
    ):
        """Test fallback to Black + isort when Ruff not available."""
        mock_ruff_available.return_value = False
        mock_black_available.return_value = True
        mock_isort_available.return_value = True
        mock_format_black_isort.return_value = "black_formatted"
        
        result = format_python_content("code")
        
        assert result == "black_formatted"
        mock_format_black_isort.assert_called_once_with("code", 30)
    
    @patch("testcraft.adapters.io.python_formatters.run_black_safe")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_isort_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_black_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_ruff_available")
    def test_format_python_content_black_only_when_isort_unavailable(
        self, mock_ruff_available, mock_black_available, mock_isort_available, mock_black_safe
    ):
        """Test using Black only when isort is not available."""
        mock_ruff_available.return_value = False
        mock_black_available.return_value = True
        mock_isort_available.return_value = False
        mock_black_safe.return_value = "black_only_formatted"
        
        result = format_python_content("code")
        
        assert result == "black_only_formatted"
        mock_black_safe.assert_called_once_with("code", 30)
    
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.get_available_formatters")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_isort_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_black_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_ruff_available")
    def test_format_python_content_no_formatters_available(
        self, mock_ruff_available, mock_black_available, mock_isort_available, mock_get_available
    ):
        """Test behavior when no formatters are available."""
        mock_ruff_available.return_value = False
        mock_black_available.return_value = False
        mock_isort_available.return_value = False
        mock_get_available.return_value = {"ruff": False, "black": False, "isort": False}
        
        original_content = "import os\ndef test(): pass"
        result = format_python_content(original_content)
        
        # Should return original content when no formatters available
        assert result == original_content
    
    @patch("testcraft.adapters.io.python_formatters.format_with_black_isort")
    @patch("testcraft.adapters.io.python_formatters.format_with_ruff")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_isort_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_black_available")
    @patch("testcraft.adapters.io.python_formatters.FormatterDetector.is_ruff_available")
    def test_format_python_content_ruff_failure_fallback(
        self, mock_ruff_available, mock_black_available, mock_isort_available, mock_format_ruff, mock_format_black_isort
    ):
        """Test fallback to Black + isort when Ruff fails."""
        mock_ruff_available.return_value = True
        mock_black_available.return_value = True
        mock_isort_available.return_value = True
        
        # Ruff fails, should fallback
        mock_format_ruff.side_effect = Exception("Ruff failed")
        mock_format_black_isort.return_value = "black_fallback"
        
        result = format_python_content("code")
        
        assert result == "black_fallback"
        mock_format_ruff.assert_called_once_with("code", 30)
        mock_format_black_isort.assert_called_once_with("code", 30)


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
                
                mock_safe.reset_mock()
                
                # Test Ruff format command construction
                run_ruff_format_safe("code")
                mock_safe.assert_called_with(
                    ["ruff", "format", "--stdin-filename", "temp.py", "/tmp/test.py"], timeout=30
                )
                
                mock_safe.reset_mock()
                
                # Test Ruff import sort command construction  
                run_ruff_import_sort_safe("code")
                mock_safe.assert_called_with(
                    ["ruff", "check", "--select", "I", "--fix", "--stdin-filename", "temp.py", "/tmp/test.py"], timeout=30
                )
