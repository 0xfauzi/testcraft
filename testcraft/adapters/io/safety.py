"""
Safety policies for file writing operations.

This module provides safety policies and validation functions to ensure
that file writing operations are performed safely and securely.
"""

import ast
import re
from pathlib import Path


class SafetyError(Exception):
    """Exception raised when safety policies are violated."""

    pass


class SafetyPolicies:
    """
    Safety policies for file writing operations.

    This class provides static methods for validating file operations
    to ensure they are safe and conform to project standards.
    """

    # Maximum file size in bytes (1MB)
    MAX_FILE_SIZE = 1024 * 1024

    # Allowed directories for writing (relative to project root)
    ALLOWED_DIRECTORIES = {"tests", "test"}
    
    # System files that are allowed in project root
    ALLOWED_SYSTEM_FILES = {".testcraft_state.json", ".testcraft.toml"}

    # Dangerous patterns that should not be written to files
    DANGEROUS_PATTERNS = {
        r"__import__\s*\(",  # Dynamic imports
        r"exec\s*\(",  # Code execution
        r"eval\s*\(",  # Expression evaluation
        r"subprocess\.",  # Subprocess calls
        r"os\.system\s*\(",  # System commands
        r"open\s*\([^)]*['\"]w",  # File writing outside of controlled context
        r"shutil\.rmtree\s*\(",  # Directory deletion
        r"os\.remove\s*\(",  # File deletion
        r"os\.unlink\s*\(",  # File unlinking
    }

    @staticmethod
    def validate_file_path(file_path: Path, project_root: Path | None = None) -> None:
        """
        Validate that a file path is safe for writing.

        Args:
            file_path: Path to validate
            project_root: Optional project root path for relative validation

        Raises:
            SafetyError: If the file path is not safe
        """
        # Convert to Path object if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Check if path is absolute and outside allowed directories
        if file_path.is_absolute():
            if project_root:
                try:
                    # Resolve both paths to handle symlinks and /private/ prefixes on macOS
                    resolved_file = file_path.resolve()
                    resolved_root = project_root.resolve()
                    relative_path = resolved_file.relative_to(resolved_root)
                    SafetyPolicies._validate_relative_path(relative_path)
                except ValueError:
                    raise SafetyError(
                        f"File path {file_path} is outside project root {project_root}"
                    )
            else:
                raise SafetyError(
                    f"Absolute paths not allowed without project root: {file_path}"
                )
        else:
            SafetyPolicies._validate_relative_path(file_path)

        # Check for path traversal attempts
        if ".." in file_path.parts:
            raise SafetyError(f"Path traversal not allowed: {file_path}")

        # Check for hidden files/directories (allow system files in project root)
        if any(part.startswith(".") for part in file_path.parts):
            # Allow system files only at project root
            filename = file_path.name
            if filename in SafetyPolicies.ALLOWED_SYSTEM_FILES:
                # Check if file is at project root
                if project_root:
                    # Resolve both paths to handle symlinks
                    resolved_file = file_path.resolve()
                    resolved_root = project_root.resolve()
                    # Check if file is directly in project root (no subdirectories)
                    if resolved_file.parent != resolved_root:
                        raise SafetyError(
                            f"System file {filename} only allowed at project root, "
                            f"not in subdirectory: {file_path}"
                        )
                else:
                    # Without project root, check if it's a single part path
                    if len(file_path.parts) != 1:
                        raise SafetyError(
                            f"System file {filename} only allowed at project root, "
                            f"not in subdirectory: {file_path}"
                        )
            else:
                raise SafetyError(f"Hidden files/directories not allowed: {file_path}")

    @staticmethod
    def _validate_relative_path(relative_path: Path) -> None:
        """Validate that a relative path is within allowed directories."""
        if not relative_path.parts:
            raise SafetyError("Empty path not allowed")

        # Check if it's an allowed system file in project root
        if len(relative_path.parts) == 1 and str(relative_path) in SafetyPolicies.ALLOWED_SYSTEM_FILES:
            return

        # Check if the first part of the path is in allowed directories
        first_dir = relative_path.parts[0]
        if first_dir not in SafetyPolicies.ALLOWED_DIRECTORIES:
            raise SafetyError(
                f"Writing only allowed in {SafetyPolicies.ALLOWED_DIRECTORIES}, "
                f"got: {first_dir}"
            )

    @staticmethod
    def validate_file_size(content: str) -> None:
        """
        Validate that content size is within limits.

        Args:
            content: Content to validate

        Raises:
            SafetyError: If content is too large
        """
        content_size = len(content.encode("utf-8"))
        if content_size > SafetyPolicies.MAX_FILE_SIZE:
            raise SafetyError(
                f"File size {content_size} bytes exceeds limit of "
                f"{SafetyPolicies.MAX_FILE_SIZE} bytes"
            )

    @staticmethod
    def validate_content_safety(content: str) -> None:
        """
        Validate that content doesn't contain dangerous patterns.

        Args:
            content: Content to validate

        Raises:
            SafetyError: If content contains dangerous patterns
        """
        for pattern in SafetyPolicies.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                raise SafetyError(f"Content contains dangerous pattern: {pattern}")

    @staticmethod
    def validate_python_syntax(content: str) -> None:
        """
        Validate that content is valid Python syntax.

        Args:
            content: Python code content to validate

        Raises:
            SafetyError: If content has invalid syntax
        """
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SafetyError(f"Invalid Python syntax: {e}")

    @staticmethod
    def get_allowed_test_extensions() -> set[str]:
        """Get the set of allowed test file extensions."""
        return {".py"}

    @staticmethod
    def validate_test_file_name(file_path: Path) -> None:
        """
        Validate that a file name follows test naming conventions.

        Args:
            file_path: Path to the test file

        Raises:
            SafetyError: If the file name doesn't follow conventions
        """
        filename = file_path.name

        # Check file extension
        if file_path.suffix not in SafetyPolicies.get_allowed_test_extensions():
            raise SafetyError(
                f"Test file must have allowed extension: "
                f"{SafetyPolicies.get_allowed_test_extensions()}, got: {file_path.suffix}"
            )

        # Check test naming convention
        if not (filename.startswith("test_") or filename.endswith("_test.py")):
            raise SafetyError(
                f"Test file must start with 'test_' or end with '_test.py', got: {filename}"
            )

    @staticmethod
    def resolve_and_validate_path(
        file_path: Path, project_root: Path | None = None, is_test_file: bool = True
    ) -> Path:
        """
        Resolve and validate a file path for writing.

        Args:
            file_path: Path to resolve and validate
            project_root: Optional project root for relative validation
            is_test_file: Whether this is a test file (applies test naming validation)

        Returns:
            Resolved and validated path

        Raises:
            SafetyError: If the path is not safe or valid
        """
        # Resolve the path
        resolved_path = file_path.resolve() if file_path.is_absolute() else file_path

        # Validate the path
        SafetyPolicies.validate_file_path(resolved_path, project_root)

        # Additional validation for test files
        if is_test_file:
            SafetyPolicies.validate_test_file_name(resolved_path)

        return resolved_path
