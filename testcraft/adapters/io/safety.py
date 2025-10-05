"""
Safety policies for file writing operations.

This module provides safety policies and validation functions to ensure
that file writing operations are performed safely and securely.
"""

import ast
import fcntl
import os
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
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
    ALLOWED_SYSTEM_FILES = {
        ".testcraft_state.json",
        ".testcraft_evaluation_state.json",
        ".testcraft.toml",
        "custom_state.json",
    }

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
    def validate_file_path(
        file_path: str | Path, project_root: Path | None = None
    ) -> None:
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

        # Normalize macOS /private/ prefix before resolving
        normalized_path = SafetyPolicies._normalize_macos_path(file_path)

        # Resolve the path to handle symlinks and canonicalize
        resolved_path = normalized_path.resolve()

        # Check if path is absolute and outside allowed directories
        if resolved_path.is_absolute():
            if project_root:
                # Normalize and resolve project root too
                normalized_root = SafetyPolicies._normalize_macos_path(project_root)
                resolved_root = normalized_root.resolve()

                try:
                    relative_path = resolved_path.relative_to(resolved_root)
                    SafetyPolicies._validate_relative_path(relative_path)
                except ValueError:
                    raise SafetyError(
                        f"File path {file_path} is outside project root {project_root}"
                    ) from None
            else:
                raise SafetyError(
                    f"Absolute paths not allowed without project root: {file_path}"
                )
        else:
            SafetyPolicies._validate_relative_path(resolved_path)

        # Check for path traversal attempts in the original path components
        if ".." in file_path.parts:
            raise SafetyError(f"Path traversal not allowed: {file_path}")

        # Check for hidden files/directories (allow system files in project root)
        if any(part.startswith(".") for part in file_path.parts):
            # Allow system files only at project root
            filename = file_path.name
            if filename in SafetyPolicies.ALLOWED_SYSTEM_FILES:
                # Check if file is at project root
                if project_root:
                    # Check if file is directly in project root (no subdirectories)
                    if resolved_path.parent != resolved_root:
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
    def _normalize_macos_path(path: Path) -> Path:
        """
        Normalize macOS /private/ prefix for consistent path handling.

        Args:
            path: Path to normalize

        Returns:
            Normalized path
        """
        path_str = str(path)
        # Handle macOS /private/ prefix normalization
        if path_str.startswith("/private/"):
            # Remove /private/ prefix to get the real path
            path_str = path_str[8:]  # Remove "/private/" prefix
        return Path(path_str)

    @staticmethod
    def _validate_relative_path(relative_path: Path) -> None:
        """Validate that a relative path is within allowed directories."""
        if not relative_path.parts:
            raise SafetyError("Empty path not allowed")

        # Check if it's an allowed system file in project root
        if (
            len(relative_path.parts) == 1
            and str(relative_path) in SafetyPolicies.ALLOWED_SYSTEM_FILES
        ):
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
        Validate that content doesn't contain dangerous patterns using AST parsing.

        Args:
            content: Content to validate

        Raises:
            SafetyError: If content contains dangerous patterns
        """
        # First try AST-based validation for Python code
        dangerous_calls = SafetyPolicies._find_dangerous_ast_calls(content)
        if dangerous_calls:
            raise SafetyError(
                f"Content contains dangerous function calls: {dangerous_calls}"
            )

        # Fallback to regex for non-Python content or edge cases
        for pattern in SafetyPolicies.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                # Additional check: make sure it's not in a string/comment context
                if not SafetyPolicies._is_pattern_in_safe_context(content, pattern):
                    raise SafetyError(f"Content contains dangerous pattern: {pattern}")

    @staticmethod
    def _find_dangerous_ast_calls(content: str) -> list[str]:
        """
        Find dangerous function calls using AST parsing.

        Args:
            content: Python code content to analyze

        Returns:
            List of dangerous function call names found
        """
        dangerous_calls = []

        try:
            # Parse the content as Python AST
            tree = ast.parse(content)

            # Walk through all AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if SafetyPolicies._is_dangerous_function(func_name):
                            dangerous_calls.append(func_name)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle cases like os.system, subprocess.call
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            attr_name = node.func.attr
                            full_call = f"{module_name}.{attr_name}"
                            if SafetyPolicies._is_dangerous_function(full_call):
                                dangerous_calls.append(full_call)

        except SyntaxError:
            # If content isn't valid Python, we'll rely on regex fallback
            pass

        return dangerous_calls

    @staticmethod
    def _is_dangerous_function(func_name: str) -> bool:
        """Check if a function name is considered dangerous."""
        dangerous_functions = {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "file",  # File operations
            "os.system",
            "os.popen",
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "shutil.rmtree",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "builtins.eval",
            "builtins.exec",
            "builtins.compile",
            "builtins.__import__",
            "builtins.open",
            "builtins.file",
        }
        return func_name in dangerous_functions

    @staticmethod
    def _is_pattern_in_safe_context(content: str, pattern: str) -> bool:
        """
        Check if a dangerous pattern appears in a safe context (string, comment).

        Args:
            content: Full content to analyze
            pattern: Regex pattern that was matched

        Returns:
            True if pattern is in a safe context, False if it's actual code
        """
        try:
            # Parse as Python to identify string literals and comments
            tree = ast.parse(content)

            # Find all string literals and comments
            safe_ranges = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Str):
                    # String literal
                    safe_ranges.append(
                        (node.lineno, node.col_offset, node.end_col_offset)
                    )
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # String constant (Python 3.8+)
                    safe_ranges.append(
                        (node.lineno, node.col_offset, node.end_col_offset)
                    )

            # Find all comment lines
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    # Full line comment
                    safe_ranges.append((i, 0, len(line)))

            # Check if any match of the pattern falls within safe ranges
            for match in re.finditer(pattern, content, re.IGNORECASE):
                start_pos = match.start()

                # Convert position to line/column for comparison
                lines_before = content[:start_pos].count("\n")
                last_newline = content.rfind("\n", 0, start_pos)
                col_offset = (
                    start_pos - (last_newline + 1) if last_newline != -1 else start_pos
                )

                # Check if this position is in any safe range
                for safe_line, safe_start_col, safe_end_col in safe_ranges:
                    if (
                        safe_line == lines_before + 1
                        and safe_start_col is not None
                        and safe_end_col is not None
                        and safe_start_col <= col_offset < safe_end_col
                    ):
                        return True  # Found in safe context

            return False  # Not in safe context

        except SyntaxError:
            # If we can't parse as Python, assume it's not safe
            return False

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
            raise SafetyError(f"Invalid Python syntax: {e}") from e

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
        # Always resolve the path to handle symlinks and canonicalize
        resolved_path = file_path.resolve()

        # Validate the path
        SafetyPolicies.validate_file_path(resolved_path, project_root)

        # Additional validation for test files
        if is_test_file:
            SafetyPolicies.validate_test_file_name(resolved_path)

        return resolved_path

    @staticmethod
    def validate_path_atomic(
        file_path: Path, project_root: Path | None = None, is_test_file: bool = True
    ) -> Path:
        """
        Perform atomic validation that re-checks path safety immediately before operations.

        This method re-validates the path right before file operations to prevent
        race conditions where the path becomes unsafe between initial validation
        and the actual operation.

        Args:
            file_path: Path to validate atomically
            project_root: Optional project root for validation
            is_test_file: Whether this is a test file

        Returns:
            Re-validated resolved path

        Raises:
            SafetyError: If the path is no longer safe
        """
        # Re-validate the path immediately before use
        resolved_path = SafetyPolicies.resolve_and_validate_path(
            file_path, project_root, is_test_file
        )

        # Additional atomic checks
        SafetyPolicies._validate_path_unchanged(resolved_path, file_path)
        SafetyPolicies._validate_parent_directory_exists(resolved_path)

        # Validate final destination after symlink resolution
        SafetyPolicies.validate_final_destination(resolved_path, project_root)

        return resolved_path

    @staticmethod
    def _validate_path_unchanged(resolved_path: Path, original_path: Path) -> None:
        """Validate that the resolved path matches expectations from the original path."""
        # Ensure the resolved path is still within expected boundaries
        if resolved_path.is_absolute() and original_path.is_absolute():
            # For absolute paths, ensure they resolve to the same location
            if resolved_path != original_path.resolve():
                # This could indicate symlink manipulation
                raise SafetyError(
                    f"Path resolution changed unexpectedly: {original_path} -> {resolved_path}"
                )

    @staticmethod
    def _validate_parent_directory_exists(resolved_path: Path) -> None:
        """Validate that the parent directory exists and is accessible."""
        parent = resolved_path.parent
        if not parent.exists():
            raise SafetyError(f"Parent directory does not exist: {parent}")

        if not parent.is_dir():
            raise SafetyError(f"Parent path is not a directory: {parent}")

        # Check if we can actually write to the parent directory
        if not os.access(parent, os.W_OK):
            raise SafetyError(f"Cannot write to parent directory: {parent}")

    @staticmethod
    @contextmanager
    def atomic_write_context(
        file_path: Path, project_root: Path | None = None, is_test_file: bool = True
    ) -> Generator[Path, None, None]:
        """
        Context manager that provides atomic validation and locking for file write operations.

        This ensures that:
        1. Path is validated immediately before writing
        2. File is locked during the validation-to-write window
        3. All safety checks pass atomically

        Args:
            file_path: Path to write to
            project_root: Optional project root for validation
            is_test_file: Whether this is a test file

        Yields:
            Validated and locked file path

        Raises:
            SafetyError: If validation fails or locking fails
        """
        # Perform atomic path validation
        validated_path = SafetyPolicies.validate_path_atomic(
            file_path, project_root, is_test_file
        )

        # Create a lock file in the same directory
        lock_file = validated_path.parent / f".{validated_path.name}.lock"

        lock_fd = None
        try:
            # Create and acquire lock on the lock file
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
            SafetyPolicies._acquire_file_lock(lock_fd)

            # Double-check path safety after acquiring lock
            current_validated_path = SafetyPolicies.validate_path_atomic(
                file_path, project_root, is_test_file
            )

            if current_validated_path != validated_path:
                raise SafetyError("Path changed during locking operation")

            yield validated_path

        except Exception as e:
            if isinstance(e, SafetyError):
                raise
            raise SafetyError(f"Atomic write context failed: {e}") from e
        finally:
            # Release the lock and clean up
            if lock_fd is not None:
                try:
                    SafetyPolicies._release_file_lock(lock_fd)
                    os.close(lock_fd)
                    # Remove the lock file
                    try:
                        lock_file.unlink(missing_ok=True)
                    except OSError:
                        pass  # Ignore errors when cleaning up lock file
                except OSError:
                    pass  # Ignore errors when releasing lock

    @staticmethod
    def _acquire_file_lock(lock_fd: int, timeout: float = 5.0) -> None:
        """
        Acquire an exclusive file lock with timeout.

        Args:
            lock_fd: File descriptor to lock
            timeout: Maximum time to wait for lock in seconds

        Raises:
            SafetyError: If lock cannot be acquired within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return  # Lock acquired successfully
            except OSError:
                time.sleep(0.01)  # Small delay before retry

        raise SafetyError(f"Could not acquire file lock within {timeout} seconds")

    @staticmethod
    def _release_file_lock(lock_fd: int) -> None:
        """Release a file lock."""
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass  # Ignore errors when releasing lock

    @staticmethod
    def _validate_symlink_chain(resolved_path: Path, max_depth: int = 10) -> None:
        """
        Validate symlink chain to prevent infinite loops and escape attempts.

        Args:
            resolved_path: Path to validate for symlink issues
            max_depth: Maximum symlink resolution depth to prevent infinite loops

        Raises:
            SafetyError: If symlink chain is too deep or contains loops
        """
        current_path = resolved_path
        visited_paths = set()
        depth = 0

        while depth < max_depth and current_path.is_symlink():
            # Check for symlink loops
            resolved_str = str(current_path.resolve())
            if resolved_str in visited_paths:
                raise SafetyError(f"Symlink loop detected: {resolved_path}")
            visited_paths.add(resolved_str)

            # Check if symlink target escapes allowed boundaries
            SafetyPolicies._validate_symlink_target(current_path, resolved_path)

            # Move to the next symlink in the chain
            current_path = current_path.readlink()
            depth += 1

        if depth >= max_depth:
            raise SafetyError(
                f"Symlink chain too deep (max {max_depth}): {resolved_path}"
            )

    @staticmethod
    def _validate_symlink_target(symlink_path: Path, original_path: Path) -> None:
        """
        Validate that a symlink target doesn't escape allowed directories.

        Args:
            symlink_path: The symlink path being validated
            original_path: The original path that led to this symlink

        Raises:
            SafetyError: If symlink target escapes allowed boundaries
        """
        try:
            target_path = symlink_path.readlink()

            # If target is absolute, check if it's within allowed boundaries
            if target_path.is_absolute():
                # This is a security risk - absolute symlinks could escape sandbox
                raise SafetyError(
                    f"Absolute symlink not allowed: {symlink_path} -> {target_path}"
                )

            # For relative symlinks, resolve and check final destination
            resolved_target = (symlink_path.parent / target_path).resolve()

            # Check if the final resolved target is within expected boundaries
            # This prevents symlinks like: allowed_dir/ -> ../../../etc/passwd
            if not SafetyPolicies._is_within_allowed_boundaries(
                resolved_target, original_path
            ):
                raise SafetyError(
                    f"Symlink target escapes allowed boundaries: {symlink_path} -> {target_path}"
                )

        except OSError as e:
            raise SafetyError(f"Cannot read symlink {symlink_path}: {e}") from e

    @staticmethod
    def _is_within_allowed_boundaries(
        resolved_path: Path, reference_path: Path
    ) -> bool:
        """
        Check if a resolved path is within the expected project boundaries.

        Args:
            resolved_path: The fully resolved path to check
            reference_path: The original path that was validated

        Returns:
            True if path is within allowed boundaries
        """
        # For now, this is a basic check - can be enhanced based on project structure
        # The key insight is that we want to ensure symlinks don't escape the project

        # Check if the resolved path is within the project root (if we have one)
        # This is a conservative approach - we can make it more sophisticated later

        # Basic check: ensure the resolved path doesn't contain suspicious patterns
        path_str = str(resolved_path)
        suspicious_patterns = [
            "/etc/",
            "/usr/",
            "/bin/",
            "/sbin/",
            "/var/",
            "/tmp/",
            "/dev/",
        ]

        for pattern in suspicious_patterns:
            if pattern in path_str:
                return False

        return True

    @staticmethod
    def validate_final_destination(
        file_path: Path, project_root: Path | None = None
    ) -> None:
        """
        Validate the final destination after all symlinks are resolved.

        This performs additional security checks on the fully resolved path
        to ensure it hasn't been manipulated through symlinks or other mechanisms.

        Args:
            file_path: Path to validate final destination for
            project_root: Optional project root for boundary validation

        Raises:
            SafetyError: If final destination is not safe
        """
        # First resolve the path completely
        resolved_path = file_path.resolve()

        # Validate symlink chain
        SafetyPolicies._validate_symlink_chain(resolved_path)

        # Additional validation for the final resolved location
        if project_root:
            try:
                # Ensure the final path is still within project boundaries
                resolved_path.relative_to(project_root.resolve())
            except ValueError:
                raise SafetyError(
                    f"Final destination {resolved_path} is outside project root {project_root}"
                ) from None

        # Check for other security issues in the final path
        SafetyPolicies._validate_final_path_security(resolved_path)

    @staticmethod
    def _validate_final_path_security(resolved_path: Path) -> None:
        """
        Perform additional security validations on the final resolved path.

        Args:
            resolved_path: Fully resolved path to validate

        Raises:
            SafetyError: If path has security issues
        """
        # Check for hidden directories that shouldn't be accessible
        parts = resolved_path.parts
        for part in parts:
            if part.startswith(".") and part not in {".", ".."}:
                # Allow specific hidden directories that are expected
                allowed_hidden = {".git", ".github", ".vscode", ".cursor"}
                if part not in allowed_hidden:
                    raise SafetyError(f"Access to hidden directory not allowed: {part}")

        # Additional security checks can be added here

    @staticmethod
    def _comprehensive_validate_path(
        file_path: Path,
        project_root: Path | None = None,
        is_test_file: bool = True,
        validate_content: bool = True,
        content: str | None = None,
    ) -> Path:
        """
        Internal comprehensive validation wrapper that combines all validations in correct order.

        This method ensures consistent security validation across all public methods
        and provides a single point for security policy enforcement.

        Args:
            file_path: Path to validate comprehensively
            project_root: Optional project root for boundary validation
            is_test_file: Whether this is a test file
            validate_content: Whether to validate content safety
            content: Content to validate (if validate_content is True)

        Returns:
            Fully validated and resolved path

        Raises:
            SafetyError: If any validation fails
        """
        try:
            # Step 1: Basic path validation and resolution
            resolved_path = SafetyPolicies.resolve_and_validate_path(
                file_path, project_root, is_test_file
            )

            # Step 2: Atomic validation with enhanced checks
            validated_path = SafetyPolicies.validate_path_atomic(
                file_path, project_root, is_test_file
            )

            # Ensure consistency between different validation methods
            if resolved_path != validated_path:
                raise SafetyError("Path validation inconsistency detected")

            # Step 3: Content validation (if requested)
            if validate_content and content is not None:
                SafetyPolicies.validate_file_size(content)
                SafetyPolicies.validate_content_safety(content)
                SafetyPolicies.validate_python_syntax(content)

            # Step 4: Final destination validation
            SafetyPolicies.validate_final_destination(validated_path, project_root)

            return validated_path

        except SafetyError:
            # Re-raise SafetyError as-is to preserve security context
            raise
        except Exception as e:
            # Log unexpected errors but don't expose internal details
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unexpected validation error for {file_path}: {type(e).__name__}"
            )
            raise SafetyError(f"Validation failed for {file_path}") from e

    @staticmethod
    def validate_path_comprehensive(
        file_path: str | Path,
        project_root: Path | None = None,
        is_test_file: bool = True,
    ) -> Path:
        """
        Comprehensive path validation combining all security checks.

        This is the main public method for path validation that uses the internal
        comprehensive validation wrapper to ensure all security policies are applied.

        Args:
            file_path: Path to validate
            project_root: Optional project root for boundary validation
            is_test_file: Whether this is a test file

        Returns:
            Fully validated and resolved path

        Raises:
            SafetyError: If any validation fails
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        return SafetyPolicies._comprehensive_validate_path(
            file_path=file_path,
            project_root=project_root,
            is_test_file=is_test_file,
            validate_content=False,
        )
