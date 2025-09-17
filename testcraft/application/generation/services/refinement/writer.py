"""
File writing and formatting utilities.

Handles safe file writing with optional formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .....ports.writer_port import WriterPort

logger = logging.getLogger(__name__)


class RefinementWriter:
    """Handles file writing and formatting for refinement operations."""

    def __init__(self, writer_port: WriterPort | None = None):
        """
        Initialize the refinement writer.

        Args:
            writer_port: Optional writer port for safe file operations
        """
        self._writer = writer_port

    async def write_with_writer_port(self, test_file: Path, content: str) -> None:
        """Write using the writer port."""
        if self._writer:
            self._writer.write_file(file_path=test_file, content=content, overwrite=True)

    async def write_with_fallback(self, test_file: Path, content: str) -> None:
        """Fallback writing method."""
        test_file.write_text(content, encoding='utf-8')
        # Optional: format the content
        try:
            from .....adapters.io.python_formatters import format_python_content
            formatted = format_python_content(content)
            if formatted != content:
                test_file.write_text(formatted, encoding='utf-8')
        except Exception as e:
            logger.debug("Could not format Python content: %s", e)

    async def safe_write(self, test_file: Path, content: str) -> None:
        """
        Write content to file using writer port if available, otherwise fallback.

        Args:
            test_file: Path to the file to write
            content: Content to write to the file
        """
        if self._writer:
            try:
                await self.write_with_writer_port(test_file, content)
            except Exception as e:
                logger.warning("WriterPort failed, using fallback: %s", e)
                await self.write_with_fallback(test_file, content)
        else:
            await self.write_with_fallback(test_file, content)
