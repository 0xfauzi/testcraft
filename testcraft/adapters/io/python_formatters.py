"""
Python code formatting utilities.

This module provides specialized functions for formatting Python code
using popular formatters like Black and isort, built on top of the
safe subprocess execution utilities.
"""

import logging
from .subprocess_safe import run_subprocess_safe, run_subprocess_simple, SubprocessTimeoutError, SubprocessExecutionError
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Module-level logger
logger = logging.getLogger(__name__)


def run_formatter_safe(
    formatter_cmd: list[str],
    content: str,
    temp_suffix: str = '.py',
    timeout: int = 30
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
        with tempfile.NamedTemporaryFile(mode='w', suffix=temp_suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        
        try:
            # Add the file path to the formatter command
            cmd = formatter_cmd + [str(temp_path)]
            
            # Run the formatter
            with run_subprocess_safe(cmd, timeout=timeout):
                pass  # We don't need stdout/stderr for formatting
            
            # Read the formatted content
            formatted_content = temp_path.read_text(encoding='utf-8')
            return formatted_content
            
        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
            
    except (SubprocessTimeoutError, SubprocessExecutionError, OSError) as e:
        logger.warning(f"Formatter {formatter_cmd[0]} failed: {e}")
        return content  # Return original content on failure


def run_black_safe(content: str, timeout: int = 30) -> str:
    """Run Black formatter safely on Python content."""
    return run_formatter_safe(
        ['python', '-m', 'black', '--quiet'],
        content,
        temp_suffix='.py',
        timeout=timeout
    )


def run_isort_safe(content: str, timeout: int = 30) -> str:
    """Run isort formatter safely on Python content."""
    return run_formatter_safe(
        ['python', '-m', 'isort', '--quiet'],
        content,
        temp_suffix='.py',
        timeout=timeout
    )


def format_python_content(content: str, timeout: int = 30) -> str:
    """
    Format Python content with both isort and Black safely.
    
    Args:
        content: Python code content to format
        timeout: Maximum time to wait for each formatter
        
    Returns:
        str: Formatted content, or original content if formatting fails
    """
    # Apply isort first
    formatted = run_isort_safe(content, timeout)
    
    # Then apply Black
    formatted = run_black_safe(formatted, timeout)
    
    return formatted
