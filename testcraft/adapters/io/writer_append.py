"""
Writer adapter that appends content to test files.

This adapter provides simple append functionality for test files,
with formatting using Black and isort.
"""

import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .python_formatters import format_python_content
from .safety import SafetyError, SafetyPolicies

# Cross-platform file locking
try:
    import fcntl  # Unix/Linux
except ImportError:
    fcntl = None  # type: ignore

try:
    import msvcrt  # Windows
except ImportError:
    msvcrt = None  # type: ignore


class WriterAppendError(Exception):
    """Exception raised when append writer operations fail."""

    pass


class WriterAppendAdapter:
    """
    Writer adapter that appends content to test files.

    This adapter implements simple file appending with automatic formatting
    using Black and isort. It includes dry-run support and safety validation.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        dry_run: bool = False,
        strict_formatting: bool = False,
    ) -> None:
        """
        Initialize the writer append adapter.

        Args:
            project_root: Optional project root path for validation
            dry_run: Whether to run in dry-run mode (no actual writing)
            strict_formatting: Whether to raise errors on formatting failures
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.strict_formatting = strict_formatting
        self.logger = logging.getLogger(__name__)
        self._active_locks: dict[Path, Any] = {}  # Track active file locks

    def __del__(self) -> None:
        """Cleanup method to release any remaining file locks."""
        for file_path in list(self._active_locks.keys()):
            try:
                self._release_file_lock(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup lock for {file_path}: {e}")

    def _acquire_file_lock(self, file_path: Path, timeout: int = 10) -> Any:
        """
        Acquire a file lock for the given path with timeout.

        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock in seconds

        Returns:
            Lock object/handle that must be passed to _release_file_lock

        Raises:
            WriterAppendError: If lock cannot be acquired within timeout
        """
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    if platform.system() == "Windows" and msvcrt:
                        # Windows file locking
                        with open(file_path) as f:
                            msvcrt.locking(f.fileno(), 1, 1)  # type: ignore[attr-defined]  # 1 = LK_NBLCK
                        lock_handle = file_path
                    elif fcntl:
                        # Unix/Linux file locking
                        lock_handle = open(file_path)  # type: ignore[assignment]
                        if hasattr(lock_handle, "fileno"):
                            fcntl.flock(
                                lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                            )
                    else:
                        # Fallback: no locking available
                        self.logger.warning(
                            f"File locking not available on {platform.system()}"
                        )
                        return None

                    self._active_locks[file_path] = lock_handle
                    return lock_handle

                except OSError:
                    # Lock not available, wait and retry
                    time.sleep(0.1)
                    continue

            raise WriterAppendError(
                f"Could not acquire lock for {file_path} within {timeout} seconds"
            )

        except WriterAppendError:
            raise
        except Exception as e:
            raise WriterAppendError(
                f"Failed to acquire file lock for {file_path}: {e}"
            ) from e

    def _release_file_lock(self, file_path: Path) -> None:
        """
        Release a previously acquired file lock.

        Args:
            file_path: Path to the file to unlock
        """
        lock_handle = self._active_locks.pop(file_path, None)
        if lock_handle is None:
            return  # No lock to release

        try:
            if platform.system() == "Windows" and msvcrt:
                # Windows file unlocking
                with open(file_path) as f:
                    msvcrt.locking(f.fileno(), 2, 1)  # type: ignore[attr-defined]  # 2 = LK_UNLCK
            elif fcntl and hasattr(lock_handle, "fileno"):
                # Unix/Linux file unlocking
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                if hasattr(lock_handle, "close"):
                    lock_handle.close()
        except Exception as e:
            self.logger.warning(f"Failed to release file lock for {file_path}: {e}")

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

            # Acquire file lock before writing
            lock_handle = None
            if not self.dry_run:
                lock_handle = self._acquire_file_lock(resolved_path)

            try:
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
            finally:
                # Always release the lock
                if lock_handle is not None:
                    self._release_file_lock(resolved_path)

        except (SafetyError, OSError) as e:
            self.logger.error(f"File write failed for {file_path}: {e}")
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

        except (OSError, SafetyError, ValueError) as e:
            self.logger.error(f"Test file write failed for {test_path}: {e}")
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

            # Create backup path with timestamp to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(
                f"{file_path.suffix}.backup.{timestamp}"
            )

            if self.dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "original_path": str(file_path),
                    "backup_path": str(backup_path),
                }

            # Check file size before loading into memory (limit: 100MB)
            file_size = file_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB

            if file_size > max_size:
                self.logger.error(
                    f"File too large for backup: {file_path} ({file_size} bytes)"
                )
                return {
                    "success": False,
                    "error": f"File too large for backup ({file_size} bytes > {max_size} bytes)",
                    "original_path": str(file_path),
                    "backup_path": None,
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
            self.logger.error(f"Backup failed for {file_path}: {e}")
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
            WriterAppendError: If formatting fails and strict_formatting is True
        """
        try:
            return format_python_content(content, timeout=15, disable_ruff=False)
        except Exception as e:
            if self.strict_formatting:
                raise WriterAppendError(f"Formatting failed: {e}") from e
            else:
                self.logger.warning(f"Formatting failed, using original content: {e}")
                return content

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
