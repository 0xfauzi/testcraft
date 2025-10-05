"""
File Discovery Service - Reusable file discovery and filtering logic.

This module provides a shared service for discovering source and test files
across the project, with configurable patterns and exclusion rules. This
centralizes file discovery logic that was previously duplicated across
multiple use cases.
"""

import logging
import os
from pathlib import Path

from ...config.models import TestPatternConfig

logger = logging.getLogger(__name__)


class FileDiscoveryError(Exception):
    """Exception raised when file discovery fails."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class FileDiscoveryService:
    """
    Service for discovering and filtering files in a project.

    This service provides centralized logic for finding source files, test files,
    and applying exclusion patterns based on configuration. It's designed to be
    reusable across different use cases (generate, coverage, analyze, etc.).
    """

    def __init__(self, config: TestPatternConfig | None = None) -> None:
        """
        Initialize the file discovery service.

        Args:
            config: Test pattern configuration for exclusions and patterns.
                   If None, uses default configuration.

        Raises:
            FileDiscoveryError: If configuration contains invalid patterns.
        """
        self.config = config or TestPatternConfig()

        # Validate patterns in configuration
        self._validate_patterns()

        # Cache for performance optimization
        self._exclude_dirs_set: set[str] | None = None
        self._exclude_patterns_set: set[str] | None = None

    @property
    def exclude_dirs_set(self) -> set[str]:
        """Get cached set of directories to exclude."""
        if self._exclude_dirs_set is None:
            self._exclude_dirs_set = set(self.config.exclude_dirs)
        return self._exclude_dirs_set

    @property
    def exclude_patterns_set(self) -> set[str]:
        """Get cached set of patterns to exclude."""
        if self._exclude_patterns_set is None:
            self._exclude_patterns_set = set(self.config.exclude)
        return self._exclude_patterns_set

    def _validate_patterns(self) -> None:
        """
        Validate patterns in the configuration to catch invalid patterns early.

        Raises:
            FileDiscoveryError: If any pattern is invalid.
        """
        all_patterns = list(self.config.exclude) + list(self.config.test_patterns)

        for pattern in all_patterns:
            if not isinstance(pattern, str):
                raise FileDiscoveryError(
                    f"Pattern must be a string, got {type(pattern)}: {pattern}"
                )

            if not pattern.strip():
                raise FileDiscoveryError(
                    f"Pattern cannot be empty or whitespace: {pattern}"
                )

            # Test pattern with a sample path to catch syntax errors early
            try:
                test_path = Path("/test/path.py")
                # Use our _matches_pattern method to test the pattern
                self._matches_pattern(str(test_path), pattern)
            except Exception as e:
                raise FileDiscoveryError(f"Invalid pattern '{pattern}': {e}") from e

    def discover_source_files(
        self,
        project_path: str | Path,
        file_patterns: list[str] | None = None,
        include_test_files: bool = False,
    ) -> list[str]:
        """
        Discover source files in a project using configured patterns.

        Args:
            project_path: Root path of the project to search
            file_patterns: Custom file patterns to use (defaults to ['*.py'])
            include_test_files: Whether to include test files in results

        Returns:
            List of discovered source file paths (absolute paths)

        Raises:
            FileDiscoveryError: If discovery fails or input validation fails
        """
        # Input validation
        if not project_path:
            raise FileDiscoveryError("Project path cannot be empty")

        try:
            project_path = Path(project_path)
        except (TypeError, ValueError) as e:
            raise FileDiscoveryError(
                f"Invalid project path format: {project_path} - {e}"
            ) from e

        if file_patterns is not None:
            if not isinstance(file_patterns, list):
                raise FileDiscoveryError(
                    f"file_patterns must be a list, got {type(file_patterns)}"
                )
            for pattern in file_patterns:
                if not isinstance(pattern, str):
                    raise FileDiscoveryError(
                        f"All file patterns must be strings, got {type(pattern)}: {pattern}"
                    )

        try:
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            if not project_path.is_dir():
                raise FileDiscoveryError(
                    f"Project path must be a directory: {project_path}"
                )

            # Cache resolved project path for performance
            resolved_project_path = project_path.resolve()

            patterns = file_patterns or ["*.py"]
            source_files = []

            logger.debug(
                f"Discovering source files in {resolved_project_path} with patterns: {patterns}"
            )

            # Use os.walk for efficient directory traversal with exclusion
            total_found = 0
            for root, dirs, files in os.walk(project_path):
                # Filter directories in-place to avoid scanning excluded directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not self._should_exclude_directory(Path(root) / d)
                ]

                # Check each file against patterns
                for filename in files:
                    file_path = Path(root) / filename

                    # Quick check: must match at least one pattern
                    # Match against both filename and project-relative path to support patterns like "tests/**/test_*.py"
                    rel_path = None
                    try:
                        rel_path = str(
                            file_path.resolve().relative_to(resolved_project_path)
                        )
                    except ValueError:
                        # File is not under project path - use filename only
                        rel_path = str(file_path.name)
                        logger.debug(
                            f"File {file_path} is not under project path {resolved_project_path}"
                        )
                    except (OSError, PermissionError) as e:
                        # Critical filesystem errors should be re-raised
                        logger.error(f"Filesystem error accessing {file_path}: {e}")
                        raise FileDiscoveryError(
                            f"Failed to access file {file_path}: {e}", cause=e
                        ) from e

                    matches_pattern = any(
                        self._matches_pattern(filename, pattern)
                        or (
                            rel_path is not None
                            and self._matches_pattern(rel_path, pattern)
                        )
                        for pattern in patterns
                    )
                    if not matches_pattern:
                        continue

                    total_found += 1
                    should_include = self._should_include_file(
                        file_path,
                        resolved_project_path,
                        include_test_files=include_test_files,
                    )

                    logger.debug(f"File {filename}: should_include={should_include}")

                    if should_include:
                        resolved_path = str(file_path.resolve())
                        source_files.append(resolved_path)
                        logger.debug(f"Added file: {resolved_path}")

            logger.debug(
                f"Total files scanned: {total_found}, included: {len(source_files)}"
            )
            logger.info(
                f"Discovered {len(source_files)} source files in {resolved_project_path} (patterns: {patterns})"
            )
            return source_files

        except FileDiscoveryError:
            # Re-raise our own exceptions
            raise
        except (OSError, PermissionError) as e:
            # Critical filesystem errors should be re-raised
            logger.error(
                f"Filesystem error during source file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Source file discovery failed due to filesystem error: {e}", cause=e
            ) from e
        except Exception as e:
            # Log unexpected errors for debugging but wrap them
            logger.exception(
                f"Unexpected error during source file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Source file discovery failed due to unexpected error: {e}", cause=e
            ) from e

    def discover_test_files(
        self,
        project_path: str | Path,
        test_patterns: list[str] | None = None,
        quiet: bool = False,
    ) -> list[str]:
        """
        Discover test files in a project using test-specific patterns.

        Args:
            project_path: Root path of the project to search
            test_patterns: Custom test patterns (defaults to config.test_patterns)
            quiet: If True, reduces logging verbosity for repeated calls

        Returns:
            List of discovered test file paths (absolute paths)

        Raises:
            FileDiscoveryError: If discovery fails or input validation fails
        """
        # Input validation
        if not project_path:
            raise FileDiscoveryError("Project path cannot be empty")

        try:
            project_path = Path(project_path)
        except (TypeError, ValueError) as e:
            raise FileDiscoveryError(
                f"Invalid project path format: {project_path} - {e}"
            ) from e

        if test_patterns is not None:
            if not isinstance(test_patterns, list):
                raise FileDiscoveryError(
                    f"test_patterns must be a list, got {type(test_patterns)}"
                )
            for pattern in test_patterns:
                if not isinstance(pattern, str):
                    raise FileDiscoveryError(
                        f"All test patterns must be strings, got {type(pattern)}: {pattern}"
                    )

        try:
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            if not project_path.is_dir():
                raise FileDiscoveryError(
                    f"Project path must be a directory: {project_path}"
                )

            # Cache resolved project path for performance
            resolved_project_path = project_path.resolve()

            patterns = test_patterns or self.config.test_patterns
            test_files = []

            logger.debug(
                f"Discovering test files in {resolved_project_path} with patterns: {patterns}"
            )

            # Use os.walk for efficient directory traversal with exclusion
            total_found = 0
            for root, dirs, files in os.walk(project_path):
                # Filter directories but keep test directories when discovering tests
                dirs[:] = [
                    d
                    for d in dirs
                    if not self._should_exclude_directory(
                        Path(root) / d, for_tests=True
                    )
                ]

                # Check each file against patterns
                for filename in files:
                    file_path = Path(root) / filename

                    # Quick check: must match at least one pattern
                    matches_pattern = any(
                        self._matches_pattern(filename, pattern) for pattern in patterns
                    )
                    if not matches_pattern:
                        continue

                    total_found += 1
                    if self._should_include_test_file(file_path):
                        test_files.append(str(file_path.resolve()))

            logger.debug(
                f"Total test files scanned: {total_found}, included: {len(test_files)}"
            )
            if quiet:
                logger.debug(
                    f"Discovered {len(test_files)} test files in {resolved_project_path} (patterns: {patterns})"
                )
            else:
                logger.info(
                    f"Discovered {len(test_files)} test files in {resolved_project_path} (patterns: {patterns})"
                )
            return test_files

        except FileDiscoveryError:
            # Re-raise our own exceptions
            raise
        except (OSError, PermissionError) as e:
            # Critical filesystem errors should be re-raised
            logger.error(
                f"Filesystem error during test file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Test file discovery failed due to filesystem error: {e}", cause=e
            ) from e
        except Exception as e:
            # Log unexpected errors for debugging but wrap them
            logger.exception(
                f"Unexpected error during test file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Test file discovery failed due to unexpected error: {e}", cause=e
            ) from e

    def discover_all_python_files(
        self, project_path: str | Path, separate_tests: bool = True
    ) -> dict:
        """
        Discover all Python files, optionally separating source and test files.

        Args:
            project_path: Root path of the project to search
            separate_tests: If True, return separate lists for source and test files

        Returns:
            Dictionary with 'source_files' and 'test_files' keys if separate_tests=True,
            otherwise 'all_files' key with combined list

        Raises:
            FileDiscoveryError: If discovery fails or input validation fails
        """
        # Input validation
        if not project_path:
            raise FileDiscoveryError("Project path cannot be empty")

        try:
            project_path = Path(project_path)
        except (TypeError, ValueError) as e:
            raise FileDiscoveryError(
                f"Invalid project path format: {project_path} - {e}"
            ) from e

        if not isinstance(separate_tests, bool):
            raise FileDiscoveryError(
                f"separate_tests must be a boolean, got {type(separate_tests)}"
            )

        try:
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            if not project_path.is_dir():
                raise FileDiscoveryError(
                    f"Project path must be a directory: {project_path}"
                )

            # Cache resolved project path for performance
            resolved_project_path = project_path.resolve()

            if separate_tests:
                source_files = self.discover_source_files(
                    resolved_project_path, include_test_files=False
                )
                test_files = self.discover_test_files(resolved_project_path)

                return {
                    "source_files": source_files,
                    "test_files": test_files,
                    "total_files": len(source_files) + len(test_files),
                }
            else:
                all_files = self.discover_source_files(
                    resolved_project_path, include_test_files=True
                )
                return {"all_files": all_files, "total_files": len(all_files)}

        except FileDiscoveryError:
            # Re-raise our own exceptions
            raise
        except (OSError, PermissionError) as e:
            # Critical filesystem errors should be re-raised
            logger.error(
                f"Filesystem error during Python file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Python file discovery failed due to filesystem error: {e}", cause=e
            ) from e
        except Exception as e:
            # Log unexpected errors for debugging but wrap them
            logger.exception(
                f"Unexpected error during Python file discovery in {project_path}: {e}"
            )
            raise FileDiscoveryError(
                f"Python file discovery failed due to unexpected error: {e}", cause=e
            ) from e

    def filter_existing_files(
        self, file_paths: list[str | Path], project_path: str | Path | None = None
    ) -> list[str]:
        """
        Filter a list of file paths to only include existing, valid Python files.

        Args:
            file_paths: List of file paths to filter
            project_path: Root path of the project for relative path checking.
                         If None, only basic file extension and pattern checks are performed.

        Returns:
            List of existing, valid Python file paths (absolute paths)

        Raises:
            FileDiscoveryError: If input validation fails
        """
        # Input validation
        if not isinstance(file_paths, list):
            raise FileDiscoveryError(
                f"file_paths must be a list, got {type(file_paths)}"
            )

        if project_path is not None:
            if not project_path:
                raise FileDiscoveryError("Project path cannot be empty when provided")

            try:
                project_path = Path(project_path)
            except (TypeError, ValueError) as e:
                raise FileDiscoveryError(
                    f"Invalid project path format: {project_path} - {e}"
                ) from e

            try:
                if not project_path.exists():
                    raise FileDiscoveryError(
                        f"Project path does not exist: {project_path}"
                    )

                if not project_path.is_dir():
                    raise FileDiscoveryError(
                        f"Project path must be a directory: {project_path}"
                    )
            except (OSError, PermissionError) as e:
                raise FileDiscoveryError(
                    f"Cannot access project path {project_path}: {e}"
                ) from e

        filtered_files = []
        resolved_project_path = None

        # Cache resolved project path if provided
        if project_path is not None:
            try:
                resolved_project_path = Path(project_path).resolve()
            except (OSError, PermissionError) as e:
                # Critical filesystem errors should be re-raised
                logger.error(
                    f"Filesystem error resolving project path {project_path}: {e}"
                )
                raise FileDiscoveryError(
                    f"Failed to resolve project path {project_path}: {e}", cause=e
                ) from e
            except Exception as e:
                # Log unexpected errors for debugging but continue without project path context
                logger.warning(
                    f"Unexpected error resolving project path {project_path}: {e}"
                )
                # Continue without project path context

        for file_path in file_paths:
            path = Path(file_path)

            if path.exists() and self._should_include_file(
                path, resolved_project_path, include_test_files=True
            ):
                filtered_files.append(str(path.resolve()))

        return filtered_files

    def _should_include_file(
        self,
        file_path: Path,
        project_path: Path | None = None,
        include_test_files: bool = False,
    ) -> bool:
        """
        Determine if a file should be included based on exclusion rules.

        Args:
            file_path: Path to the file to check
            project_path: Root path of the project (for relative path checking).
                         If None, only basic file extension and pattern checks are performed.
            include_test_files: Whether to include test files

        Returns:
            True if the file should be included
        """
        # Must be a source code file (allow common source file extensions)
        source_extensions = {".py", ".pyx", ".pyi", ".pyw"}
        if file_path.suffix not in source_extensions:
            return False

        # Check if it's a test file
        is_test_file = self._is_test_file(file_path)
        if is_test_file and not include_test_files:
            return False

        # Check exclude patterns (from config.exclude)
        file_str = str(file_path)
        for pattern in self.exclude_patterns_set:
            if self._matches_pattern(file_str, pattern):
                logger.debug(f"File {file_path} excluded by pattern '{pattern}'")
                return False

        # Check exclude directories (from config.exclude_dirs)
        # Only check directories relative to the project path, not absolute path
        if project_path is not None:
            try:
                relative_path = file_path.relative_to(project_path)
                if any(
                    part in self.exclude_dirs_set for part in relative_path.parts[:-1]
                ):  # Exclude filename part
                    return False
            except ValueError:
                # File is not under project path, exclude it
                return False
        else:
            # No project context - only check immediate directory name
            parent_name = file_path.parent.name
            if parent_name in self.exclude_dirs_set:
                return False

        return True

    def _should_exclude_directory(
        self, dir_path: Path, for_tests: bool = False
    ) -> bool:
        """
        Determine if a directory should be excluded during traversal.

        This is more efficient than checking after scanning all files.

        Args:
            dir_path: Path to the directory to check

        Returns:
            True if the directory should be excluded (skip traversing it)
        """
        # Check if any part of the path is in exclude_dirs
        dir_name = dir_path.name
        if dir_name in self.exclude_dirs_set:
            # Do not exclude standard test directories during test discovery
            if for_tests and dir_name in {"tests", "test"}:
                pass
            else:
                return True

        # Check for patterns that might match directories
        dir_str = str(dir_path)
        for pattern in self.exclude_patterns_set:
            if self._matches_pattern(dir_str, pattern):
                logger.debug(f"Directory {dir_path} excluded by pattern '{pattern}'")
                return True

        # Specific checks for common virtual environment patterns
        if (
            dir_name.startswith(".venv")
            or dir_name.endswith(".egg-info")
            or dir_name.endswith(".dist-info")
            or dir_name == "site-packages"
            or dir_name in {"lib", "lib64", "bin", "Scripts", "include", "share"}
        ):
            return True

        # Check if this looks like a virtual environment based on structure
        # Look for pyvenv.cfg which indicates a virtual environment
        if (dir_path / "pyvenv.cfg").exists():
            return True

        # Check for Conda environment structure
        if (dir_path / "conda-meta").exists():
            return True

        return False

    def _should_include_test_file(self, file_path: Path) -> bool:
        """
        Determine if a test file should be included.

        Args:
            file_path: Path to the test file to check

        Returns:
            True if the test file should be included
        """
        # Must be a source code file (allow common source file extensions)
        source_extensions = {".py", ".pyx", ".pyi", ".pyw"}
        if file_path.suffix not in source_extensions:
            return False

        # Check exclude directories (but allow common test directories)
        if any(
            part in self.exclude_dirs_set and part not in {"tests", "test"}
            for part in file_path.parts
        ):
            return False

        # Check specific exclude patterns (but be more lenient for test files)
        file_str = str(file_path)
        for pattern in self.exclude_patterns_set:
            # Skip test-related exclude patterns for test files
            if any(
                test_keyword in pattern.lower() for test_keyword in ["test_", "_test"]
            ):
                continue
            if self._matches_pattern(file_str, pattern):
                logger.debug(f"Test file {file_path} excluded by pattern '{pattern}'")
                return False

        return True

    def _is_test_file(self, file_path: Path) -> bool:
        """
        Check if a file is a test file based on naming conventions.

        Args:
            file_path: Path to check

        Returns:
            True if the file appears to be a test file
        """
        # Check filename patterns
        filename = file_path.name
        if (
            filename.startswith("test_")
            or filename.endswith("_test.py")
            or filename in ["conftest.py", "pytest.ini"]
        ):
            return True

        # Check if it's in a well-known test directory
        # Only check for exact matches to avoid false positives from temp directories
        parent_parts = file_path.parts[:-1]  # All parts except the filename
        if parent_parts:
            # Check for well-known test directory names (exact matches only)
            for part in parent_parts:
                if part in ["tests", "test"]:
                    return True

        return False

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """
        Check if a file path matches an exclusion pattern.

        This supports basic glob-style patterns with * wildcards.

        Args:
            file_path: File path to check
            pattern: Pattern to match against

        Returns:
            True if the pattern matches
        """
        try:
            # Convert glob pattern to Path.match format
            if "*" in pattern or "?" in pattern or "[" in pattern:
                return Path(file_path).match(pattern)
            else:
                # Simple substring match for non-glob patterns
                return pattern in file_path
        except (ValueError, TypeError) as e:
            # Pattern syntax errors - log and fallback to substring match
            logger.debug(
                f"Pattern '{pattern}' syntax error for path '{file_path}': {e}"
            )
            return pattern in file_path
        except Exception as e:
            # Unexpected errors - log and fallback to substring match
            logger.warning(
                f"Unexpected error matching pattern '{pattern}' against '{file_path}': {e}"
            )
            return pattern in file_path

    def get_discovery_stats(self, project_path: str | Path) -> dict:
        """
        Get statistics about file discovery in a project.

        Args:
            project_path: Root path of the project

        Returns:
            Dictionary with discovery statistics

        Raises:
            FileDiscoveryError: If input validation fails
        """
        # Input validation
        if not project_path:
            raise FileDiscoveryError("Project path cannot be empty")

        try:
            project_path = Path(project_path)
        except (TypeError, ValueError) as e:
            raise FileDiscoveryError(
                f"Invalid project path format: {project_path} - {e}"
            ) from e

        try:
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            if not project_path.is_dir():
                raise FileDiscoveryError(
                    f"Project path must be a directory: {project_path}"
                )

            # Cache resolved project path for performance
            resolved_project_path = project_path.resolve()

            discovery_result = self.discover_all_python_files(
                resolved_project_path, separate_tests=True
            )

            return {
                "project_path": str(resolved_project_path),
                "source_files_count": len(discovery_result["source_files"]),
                "test_files_count": len(discovery_result["test_files"]),
                "total_files": discovery_result["total_files"],
                "config_patterns": {
                    "test_patterns": self.config.test_patterns,
                    "exclude_patterns": list(self.config.exclude),
                    "exclude_dirs_count": len(self.config.exclude_dirs),
                },
            }

        except FileDiscoveryError:
            # Re-raise our own exceptions
            raise
        except (OSError, PermissionError) as e:
            # Critical filesystem errors should be logged and included in error response
            logger.error(
                f"Filesystem error getting discovery stats for {project_path}: {e}"
            )
            return {
                "project_path": str(project_path),
                "error": f"Filesystem error: {e}",
            }
        except Exception as e:
            # Log unexpected errors for debugging
            logger.exception(
                f"Unexpected error getting discovery stats for {project_path}: {e}"
            )
            return {
                "project_path": str(project_path),
                "error": f"Unexpected error: {e}",
            }
