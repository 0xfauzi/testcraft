"""
Writer adapter that appends content to test files.

This adapter provides simple append functionality for test files,
with formatting using Black and isort.
"""

import logging
from pathlib import Path
from typing import Any

from .python_formatters import format_python_content
from .safety import SafetyError, SafetyPolicies


class WriterAppendError(Exception):
    """Exception raised when append writer operations fail."""

    pass


class WriterAppendAdapter:
    """
    Writer adapter that appends content to test files.

    This adapter implements simple file appending with automatic formatting
    using Black and isort. It includes dry-run support and safety validation.
    """

    def __init__(self, project_root: Path | None = None, dry_run: bool = False):
        """
        Initialize the writer append adapter.

        Args:
            project_root: Optional project root path for validation
            dry_run: Whether to run in dry-run mode (no actual writing)
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)

    def write_file(
        self,
        file_path: str | Path,
        content: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write content to a file by appending.

        Args:
            file_path: Path where the file should be written
            content: Content to write to the file
            overwrite: Whether to overwrite existing files (ignored for append)
            **kwargs: Additional writing parameters

        Returns:
            Dictionary containing write operation results

        Raises:
            WriterAppendError: If file writing fails
        """
        try:
            file_path = Path(file_path)

            # Validate the path and content
            resolved_path = SafetyPolicies.resolve_and_validate_path(
                file_path, self.project_root, is_test_file=True
            )
            SafetyPolicies.validate_file_size(content)
            SafetyPolicies.validate_content_safety(content)
            SafetyPolicies.validate_python_syntax(content)

            # Ensure directory exists
            directory_result = self.ensure_directory(resolved_path.parent)
            if not directory_result["success"]:
                raise WriterAppendError(
                    f"Failed to create directory: {resolved_path.parent}"
                )

            # Read existing content if file exists
            existing_content = ""
            file_existed = resolved_path.exists()
            if file_existed:
                existing_content = resolved_path.read_text(encoding="utf-8")

            # Combine content
            if existing_content and not existing_content.endswith("\n"):
                combined_content = existing_content + "\n" + content
            else:
                combined_content = existing_content + content

            # Format the combined content
            formatted_content = self._format_content(combined_content)

            if self.dry_run:
                # In dry-run mode, show what would be written
                return {
                    "success": True,
                    "dry_run": True,
                    "file_path": str(file_path),
                    "bytes_written": len(formatted_content.encode("utf-8")),
                    "content_preview": (
                        formatted_content[:500] + "..."
                        if len(formatted_content) > 500
                        else formatted_content
                    ),
                    "file_existed": file_existed,
                    "backup_path": None,
                }

            # Create backup if file existed
            backup_path = None
            if file_existed:
                backup_result = self.backup_file(resolved_path)
                if backup_result["success"]:
                    backup_path = backup_result["backup_path"]

            # Write the formatted content
            resolved_path.write_text(formatted_content, encoding="utf-8")

            return {
                "success": True,
                "bytes_written": len(formatted_content.encode("utf-8")),
                "file_path": str(file_path),
                "backup_path": backup_path,
                "file_existed": file_existed,
                "formatted": True,
            }

        except (SafetyError, OSError) as e:
            raise WriterAppendError(f"Failed to write file {file_path}: {e}") from e

    def write_test_file(
        self,
        test_path: str | Path,
        test_content: str,
        source_file: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write a test file by appending content.

        Args:
            test_path: Path where the test file should be written
            test_content: Test content to write
            source_file: Optional path to the source file being tested
            **kwargs: Additional parameters for test file writing

        Returns:
            Dictionary containing test file write results

        Raises:
            WriterAppendError: If test file writing fails
        """
        try:
            # Write the file using the general write_file method
            write_result = self.write_file(test_path, test_content, **kwargs)

            if not write_result["success"]:
                return write_result

            # Parse the content to extract test information
            test_info = self._extract_test_info(test_content)

            return {
                **write_result,
                "test_path": write_result["file_path"],
                "source_file": str(source_file) if source_file else None,
                "imports_added": test_info["imports"],
                "test_functions": test_info["functions"],
            }

        except Exception as e:
            raise WriterAppendError(
                f"Failed to write test file {test_path}: {e}"
            ) from e

    def backup_file(
        self, file_path: str | Path, backup_suffix: str = ".backup"
    ) -> dict[str, Any]:
        """
        Create a backup of an existing file.

        Args:
            file_path: Path of the file to backup
            backup_suffix: Suffix to add to the backup filename

        Returns:
            Dictionary containing backup operation results

        Raises:
            WriterAppendError: If backup creation fails
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return {
                    "success": False,
                    "error": "File does not exist for backup",
                    "original_path": str(file_path),
                    "backup_path": None,
                }

            # Create backup path
            backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)

            if self.dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "original_path": str(file_path),
                    "backup_path": str(backup_path),
                }

            # Copy file content
            content = file_path.read_text(encoding="utf-8")
            backup_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "original_path": str(file_path),
                "backup_path": str(backup_path),
            }

        except OSError as e:
            raise WriterAppendError(f"Failed to backup file {file_path}: {e}") from e

    def ensure_directory(self, directory_path: str | Path) -> dict[str, Any]:
        """
        Ensure that a directory exists, creating it if necessary.

        Args:
            directory_path: Path of the directory to ensure exists

        Returns:
            Dictionary containing directory operation results

        Raises:
            WriterAppendError: If directory creation fails
        """
        try:
            directory_path = Path(directory_path)

            # Validate the directory path
            SafetyPolicies.validate_file_path(directory_path, self.project_root)

            existed = directory_path.exists()

            if self.dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "directory_path": str(directory_path),
                    "created": not existed,
                }

            if not existed:
                directory_path.mkdir(parents=True, exist_ok=True)

            return {
                "success": True,
                "directory_path": str(directory_path),
                "created": not existed,
            }

        except (OSError, SafetyError) as e:
            raise WriterAppendError(
                f"Failed to ensure directory {directory_path}: {e}"
            ) from e

    def _format_content(self, content: str) -> str:
        """
        Format Python content using Black and isort with robust process management.

        Uses the shared subprocess_safe module for safe execution with proper cleanup
        on timeout or interruption. Creates new process groups for better isolation.

        Args:
            content: Python code content to format

        Returns:
            Formatted content

        Raises:
            WriterAppendError: If formatting fails
        """
        return format_python_content(content, timeout=30)

    def _extract_test_info(self, content: str) -> dict[str, Any]:
        """
        Extract test information from content.

        Args:
            content: Test file content to analyze

        Returns:
            Dictionary containing imports and test functions
        """
        import ast
        import re

        try:
            tree = ast.parse(content)

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(
                            f"{module}.{alias.name}" if module else alias.name
                        )

            # Extract test functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    functions.append(node.name)

            return {"imports": imports, "functions": functions}

        except (SyntaxError, ValueError):
            # Fallback to regex if AST parsing fails
            import_pattern = r"^(?:from\s+\S+\s+)?import\s+(.+)$"
            function_pattern = r"^def\s+(test_\w+)\s*\("

            imports = []
            functions = []

            for line in content.split("\n"):
                line = line.strip()

                import_match = re.match(import_pattern, line)
                if import_match:
                    imports.append(import_match.group(1))

                function_match = re.match(function_pattern, line)
                if function_match:
                    functions.append(function_match.group(1))

            return {"imports": imports, "functions": functions}
