"""
Python code formatting utilities.

This module provides specialized functions for formatting Python code
using popular formatters. It implements a smart formatter selection system
that prioritizes Ruff (modern, fast) and gracefully falls back to Black only.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

from .subprocess_safe import (
    SubprocessExecutionError,
    SubprocessTimeoutError,
    run_subprocess_safe,
)

# Module-level logger
logger = logging.getLogger(__name__)


class FormatterDetector:
    """Detects available code formatters at runtime."""

    _ruff_available: bool | None = None
    _black_available: bool | None = None
    _isort_available: bool | None = None
    _ruff_disabled: bool = False  # Temporary disable if Ruff keeps timing out
    _ruff_failure_count: int = 0  # Track consecutive Ruff failures

    @classmethod
    def is_ruff_available(cls) -> bool:
        """Check if Ruff is available in the current environment."""
        if cls._ruff_available is None:
            try:
                subprocess.run(
                    ["ruff", "--version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                cls._ruff_available = True
                logger.debug("Ruff formatter detected and available")
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                cls._ruff_available = False
                logger.debug("Ruff formatter not available")
        return cls._ruff_available and not cls._ruff_disabled

    @classmethod
    def record_ruff_failure(cls):
        """Record a Ruff formatting failure and disable if too many failures."""
        cls._ruff_failure_count += 1
        if cls._ruff_failure_count >= 3:
            cls._ruff_disabled = True
            logger.warning(
                "Ruff formatter disabled due to repeated failures/timeouts. "
                "Using fallback formatters for this session."
            )

    @classmethod
    def record_ruff_success(cls):
        """Record a Ruff formatting success and reset failure count."""
        cls._ruff_failure_count = 0

    @classmethod
    def is_black_available(cls) -> bool:
        """Check if Black is available in the current environment."""
        if cls._black_available is None:
            try:
                subprocess.run(
                    ["python", "-m", "black", "--version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                cls._black_available = True
                logger.debug("Black formatter detected and available")
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                cls._black_available = False
                logger.debug("Black formatter not available")
        return cls._black_available

    @classmethod
    def is_isort_available(cls) -> bool:
        """Check if isort is available in the current environment."""
        if cls._isort_available is None:
            try:
                subprocess.run(
                    ["python", "-m", "isort", "--version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                cls._isort_available = True
                logger.debug("isort formatter detected and available")
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                cls._isort_available = False
                logger.debug("isort formatter not available")
        return cls._isort_available

    @classmethod
    def get_available_formatters(cls) -> dict[str, bool]:
        """Get a summary of all available formatters."""
        return {
            "ruff": cls.is_ruff_available(),
            "black": cls.is_black_available(),
            "isort": cls.is_isort_available(),
        }


def run_formatter_safe(
    formatter_cmd: list[str], content: str, temp_suffix: str = ".py", timeout: int = 30
) -> str:
    """
    Run a code formatter safely on content using a temporary file.

    This is a specialized function for running code formatters like Black
    and isort that need to operate on files.

    Args:
        formatter_cmd: Command to run (e.g., ['python', '-m', 'black', '--quiet'])
        content: Content to format
        temp_suffix: Suffix for temporary file
        timeout: Maximum time to wait for formatting

    Returns:
        str: Formatted content, or original content if formatting fails

    Example:
        ```python
        formatted = run_formatter_safe(
            ['python', '-m', 'black', '--quiet'],
            "import os\ndef test():pass"
        )
        ```
    """
    try:
        # Create a temporary file for formatting
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=temp_suffix, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            # Add the file path to the formatter command
            cmd = formatter_cmd + [str(temp_path)]

            # Run the formatter
            with run_subprocess_safe(cmd, timeout=timeout):
                pass  # We don't need stdout/stderr for formatting

            # Read the formatted content
            formatted_content = temp_path.read_text(encoding="utf-8")
            return formatted_content

        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)

    except (SubprocessTimeoutError, SubprocessExecutionError, OSError) as e:
        # Only log as debug if it's a module not found error (common in different environments)
        if "No module named" in str(e):
            logger.debug(f"Formatter {formatter_cmd[0]} not available: {e}")
        else:
            logger.warning(f"Formatter {formatter_cmd[0]} failed: {e}")
        return content  # Return original content on failure


def run_black_safe(content: str, timeout: int = 30) -> str:
    """Run Black formatter safely on Python content.

    Prefer invoking via 'python -m black' rather than sys.executable for tests.
    """
    return run_formatter_safe(
        ["python", "-m", "black", "--quiet"],
        content,
        temp_suffix=".py",
        timeout=timeout,
    )


def run_isort_safe(content: str, timeout: int = 30) -> str:
    """Run isort formatter safely on Python content."""
    try:
        return run_formatter_safe(
            ["python", "-m", "isort", "--quiet"],
            content,
            temp_suffix=".py",
            timeout=timeout,
        )
    except Exception as e:
        # If isort isn't installed in the runtime env, log and return original content
        logger.warning(f"isort unavailable or failed: {e}")
        return content


def run_ruff_format_safe(content: str, timeout: int = 30) -> str:
    """Run Ruff formatter safely on Python content.

    Uses a temporary file via run_formatter_safe to match integration tests
    expectations and ensure consistent subprocess handling.
    """
    return run_formatter_safe(
        ["ruff", "format", "--stdin-filename", "temp.py"],
        content,
        temp_suffix=".py",
        timeout=timeout,
    )


def run_ruff_import_sort_safe(content: str, timeout: int = 30) -> str:
    """Run Ruff import sorting safely on Python content.

    Uses a temporary file via run_formatter_safe to match integration tests
    expectations and ensure consistent subprocess handling.
    """
    return run_formatter_safe(
        [
            "ruff",
            "check",
            "--select",
            "I",
            "--fix",
            "--stdin-filename",
            "temp.py",
        ],
        content,
        temp_suffix=".py",
        timeout=timeout,
    )


def format_with_ruff(content: str, timeout: int = 30) -> str:
    """Format Python content using Ruff (imports + format).

    Preferred path: use Ruff for import sorting and code formatting.
    Falls back to raising so caller can choose Black.
    """
    detector = FormatterDetector()

    try:
        # Sort imports with Ruff
        imports_sorted = run_ruff_import_sort_safe(content, timeout)
        # Format code with Ruff
        fully_formatted = run_ruff_format_safe(imports_sorted, timeout)

        # If Ruff produced no changes at all, treat as failure to allow fallback
        if imports_sorted == content and fully_formatted == content:
            raise SubprocessExecutionError(
                "Ruff produced no changes; falling back to Black+isort"
            )

        detector.record_ruff_success()
        return fully_formatted
    except Exception as e:
        detector.record_ruff_failure()
        raise e


def format_with_black_only(content: str, timeout: int = 30) -> str:
    """Format Python content using Black only (fallback)."""
    return run_black_safe(content, timeout)


def format_with_black_isort(content: str, timeout: int = 30) -> str:
    """Legacy function to format with isort first, then Black (for compatibility)."""
    # First apply isort
    isort_formatted = run_isort_safe(content, timeout)
    # Then apply black
    return run_black_safe(isort_formatted, timeout)


def format_python_content(
    content: str, timeout: int = 30, disable_ruff: bool = False
) -> str:
    """
    Format Python content using the best available formatter.

    Uses smart formatter selection that prioritizes reliable formatters to avoid
    hanging during test generation. Skips experimental Ruff features that may hang.

    Args:
        content: Python code content to format
        timeout: Maximum time to wait for each formatter (reduced default)
        disable_ruff: If True, skip Ruff entirely and use Black+isort

    Returns:
        str: Formatted content, or original content if no formatters available
    """
    detector = FormatterDetector()

    # Priority 1: Ruff (imports + format)
    if not disable_ruff and detector.is_ruff_available():
        logger.info("Using Ruff for import sorting and formatting")
        try:
            return format_with_ruff(content, timeout)
        except Exception as e:
            logger.warning(f"Ruff formatting failed, falling back to Black+isort: {e}")

    # Priority 2: Black + isort (legacy, more widely available)
    # Do not gate on detector checks; the safe wrappers handle absence gracefully.
    try:
        return format_with_black_isort(content, timeout)
    except Exception as e:
        logger.warning(f"Black+isort formatting failed: {e}")

    # No formatters available - log helpful message and return original
    available = detector.get_available_formatters()
    logger.warning(
        f"No Python formatters available. Available: {available}. "
        f"Install Ruff with: 'uv add --dev ruff' (preferred) or Black with 'uv add --dev black'"
    )

    return content
