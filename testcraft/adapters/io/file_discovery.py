"""
File Discovery Service - Reusable file discovery and filtering logic.

This module provides a shared service for discovering source and test files
across the project, with configurable patterns and exclusion rules. This
centralizes file discovery logic that was previously duplicated across
multiple use cases.
"""

import fnmatch
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from ...config.models import TestPatternConfig
from ..testing.pytest_collector import PytestCollectionAdapter
from ..parsing.test_classifier import TestFileClassifier
from ..coverage.quick_probe import CoverageQuickProbeAdapter

logger = logging.getLogger(__name__)


class FileDiscoveryError(Exception):
    """Exception raised when file discovery fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class FileDiscoveryService:
    """
    Service for discovering and filtering files in a project.

    This service provides centralized logic for finding source files, test files,
    and applying exclusion patterns based on configuration. It's designed to be
    reusable across different use cases (generate, coverage, analyze, etc.).
    """

    def __init__(self, config: TestPatternConfig | None = None):
        """
        Initialize the file discovery service.

        Args:
            config: Test pattern configuration for exclusions and patterns.
                   If None, uses default configuration.
        """
        self.config = config or TestPatternConfig()

        # Cache for performance optimization
        self._exclude_dirs_set: set[str] | None = None
        self._exclude_patterns_set: set[str] | None = None
        
        # Multi-tier discovery adapters
        self._pytest_collector = PytestCollectionAdapter(self.config.test_discovery)
        self._test_classifier = TestFileClassifier()
        self._coverage_probe = CoverageQuickProbeAdapter(self.config.test_discovery)

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
            FileDiscoveryError: If discovery fails
        """
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            patterns = file_patterns or ["*.py"]
            source_files = []

            logger.debug(
                f"Discovering source files in {project_path} with patterns: {patterns}"
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
                            file_path.resolve().relative_to(project_path.resolve())
                        )
                    except Exception:
                        rel_path = str(file_path.name)

                    matches_pattern = any(
                        fnmatch.fnmatch(filename, pattern)
                        or fnmatch.fnmatch(rel_path, pattern)
                        for pattern in patterns
                    )
                    if not matches_pattern:
                        continue

                    total_found += 1
                    should_include = self._should_include_file(
                        file_path, project_path, include_test_files=include_test_files
                    )

                    logger.debug(f"File {filename}: should_include={should_include}")

                    if should_include:
                        resolved_path = str(file_path.resolve())
                        source_files.append(resolved_path)
                        logger.debug(f"Added file: {resolved_path}")

            logger.debug(
                f"Total files scanned: {total_found}, included: {len(source_files)}"
            )
            logger.info(f"Discovered {len(source_files)} source files in {project_path} (patterns: {patterns})")
            return source_files

        except Exception as e:
            if isinstance(e, FileDiscoveryError):
                raise
            logger.exception(f"Failed to discover source files: {e}")
            raise FileDiscoveryError(f"Source file discovery failed: {e}", cause=e)

    def discover_test_files(
        self, project_path: str | Path, test_patterns: list[str] | None = None, quiet: bool = False
    ) -> list[str]:
        """
        Discover test files in a project using multi-tier discovery.

        Uses the configured test discovery mode to find test files:
        - "auto": Try pytest collector, fall back to AST + glob patterns
        - "pytest-collector": Use only pytest collection
        - "ast": Use AST-based classification + glob patterns
        - "globs": Use legacy glob patterns only

        Args:
            project_path: Root path of the project to search
            test_patterns: Custom test patterns (defaults to config.test_patterns)
            quiet: If True, reduces logging verbosity for repeated calls

        Returns:
            List of discovered test file paths (absolute paths)

        Raises:
            FileDiscoveryError: If discovery fails
        """
        project_path = Path(project_path)
        
        # Get discovery mode
        mode = self.config.test_discovery.mode
        
        if mode == "auto":
            return self._discover_test_files_auto(project_path, test_patterns, quiet)
        elif mode == "pytest-collector":
            return self._discover_test_files_pytest_only(project_path, test_patterns, quiet)
        elif mode == "ast":
            return self._discover_test_files_ast(project_path, test_patterns, quiet)
        elif mode == "globs":
            return self._discover_test_files_legacy(project_path, test_patterns, quiet)
        else:
            logger.warning(f"Unknown test discovery mode '{mode}', falling back to legacy")
            return self._discover_test_files_legacy(project_path, test_patterns, quiet)
    def _discover_test_files_auto(
        self, project_path: Path, test_patterns: list[str] | None = None, quiet: bool = False
    ) -> list[str]:
        """Auto mode: Try pytest collector first, fall back to AST + patterns."""
        start_time = time.time()
        discovery_method = "unknown"
        
        try:
            # Attempt Tier 1: Pytest collection
            if not quiet:
                logger.info("Attempting pytest collection for test discovery")
            
            collection_result = self._pytest_collector.collect(project_path)
            
            if collection_result.success and collection_result.files:
                # Pytest collection succeeded, filter through classifier if enabled
                test_files = [str(f) for f in collection_result.files]
                
                if self.config.test_discovery.classify_support_files:
                    filtered_count = len(test_files)
                    test_files = self._filter_support_files(test_files)
                    logger.debug(f"Filtered {filtered_count - len(test_files)} support files from test list")
                
                discovery_method = "pytest-collection"
                
                # Log discovery metrics
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Test discovery completed via {discovery_method} in {duration_ms:.1f}ms: found {len(test_files)} test files")
                
                return test_files
            else:
                if not quiet:
                    logger.info(f"Pytest collection failed: {collection_result.failure_reason}, falling back to AST+patterns")
        
        except Exception as e:
            if not quiet:
                logger.debug(f"Pytest collection error: {e}, falling back to AST+patterns")
        
        # Fallback to Tier 2A: AST + patterns
        discovery_method = "ast-fallback"
        result = self._discover_test_files_ast(project_path, test_patterns, quiet)
        
        # Log fallback metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Test discovery completed via {discovery_method} in {duration_ms:.1f}ms: found {len(result)} test files")
        
        return result

    def _discover_test_files_pytest_only(
        self, project_path: Path, test_patterns: list[str] | None = None, quiet: bool = False
    ) -> list[str]:
        """Pytest-only mode: Use only pytest collection."""
        try:
            collection_result = self._pytest_collector.collect(project_path)
            
            if collection_result.success:
                test_files = [str(f) for f in collection_result.files]
                
                if self.config.test_discovery.classify_support_files:
                    test_files = self._filter_support_files(test_files)
                
                if not quiet:
                    logger.info(f"Pytest collection found {len(test_files)} test files")
                return test_files
            else:
                raise FileDiscoveryError(f"Pytest collection failed: {collection_result.failure_reason}")
        
        except Exception as e:
            if isinstance(e, FileDiscoveryError):
                raise
            raise FileDiscoveryError(f"Pytest collection failed: {e}", cause=e)

    def _discover_test_files_ast(
        self, project_path: Path, test_patterns: list[str] | None = None, quiet: bool = False
    ) -> list[str]:
        """AST mode: Use AST classification + glob patterns."""
        try:
            # First, find candidate files using patterns
            candidate_files = self._discover_candidate_files_with_patterns(project_path, test_patterns)
            
            # Then classify them using AST analysis
            classified_test_files = []
            
            for file_path in candidate_files:
                classification = self._test_classifier.classify(Path(file_path))
                
                if classification.is_test:
                    classified_test_files.append(file_path)
                elif self.config.test_discovery.classify_support_files and classification.is_support:
                    # Include support files if configured
                    classified_test_files.append(file_path)
            
            if not quiet:
                logger.info(f"AST classification found {len(classified_test_files)} test files from {len(candidate_files)} candidates")
            
            return classified_test_files
        
        except Exception as e:
            raise FileDiscoveryError(f"AST-based test discovery failed: {e}", cause=e)

    def _discover_test_files_legacy(
        self, project_path: Path, test_patterns: list[str] | None = None, quiet: bool = False
    ) -> list[str]:
        """Legacy mode: Use only glob patterns (original implementation)."""
        try:
            if not project_path.exists():
                raise FileDiscoveryError(f"Project path does not exist: {project_path}")

            patterns = test_patterns or self.config.test_patterns
            test_files = []

            logger.debug(
                f"Discovering test files in {project_path} with patterns: {patterns}"
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
                    # Match against both filename and project-relative path to support patterns like "tests/**/test_*.py"
                    rel_path = None
                    try:
                        rel_path = str(
                            file_path.resolve().relative_to(project_path.resolve())
                        )
                    except Exception:
                        rel_path = str(file_path.name)

                    matches_pattern = any(
                        fnmatch.fnmatch(filename, pattern)
                        or fnmatch.fnmatch(rel_path, pattern)
                        for pattern in patterns
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
                logger.debug(f"Discovered {len(test_files)} test files in {project_path} (patterns: {patterns})")
            else:
                logger.info(f"Discovered {len(test_files)} test files in {project_path} (patterns: {patterns})")
            return test_files

        except Exception as e:
            if isinstance(e, FileDiscoveryError):
                raise
            logger.exception(f"Failed to discover test files: {e}")
            raise FileDiscoveryError(f"Test file discovery failed: {e}", cause=e)

    def _discover_candidate_files_with_patterns(
        self, project_path: Path, test_patterns: list[str] | None = None
    ) -> list[str]:
        """Discover candidate files using glob patterns."""
        patterns = test_patterns or self.config.test_patterns
        candidate_files = []

        for root, dirs, files in os.walk(project_path):
            # Filter directories but keep test directories
            dirs[:] = [
                d
                for d in dirs
                if not self._should_exclude_directory(Path(root) / d, for_tests=True)
            ]

            for filename in files:
                file_path = Path(root) / filename

                # Match against both filename and project-relative path
                rel_path = None
                try:
                    rel_path = str(file_path.resolve().relative_to(project_path.resolve()))
                except Exception:
                    rel_path = str(file_path.name)

                matches_pattern = any(
                    fnmatch.fnmatch(filename, pattern)
                    or fnmatch.fnmatch(rel_path, pattern)
                    for pattern in patterns
                )
                
                if matches_pattern and self._should_include_test_file(file_path):
                    candidate_files.append(str(file_path.resolve()))

        return candidate_files

    def _filter_support_files(self, test_files: list[str]) -> list[str]:
        """Filter out support files from test files list."""
        filtered_files = []
        
        for file_path_str in test_files:
            file_path = Path(file_path_str)
            classification = self._test_classifier.classify(file_path)
            
            # Only include files that are actual tests, not just support
            if classification.is_test:
                filtered_files.append(file_path_str)
        
        return filtered_files

    def discover_and_classify_tests(self, project_path: str | Path) -> dict[str, Any]:
        """
        Discover and classify test files, returning both lists and classifications.

        Args:
            project_path: Root path of the project to search

        Returns:
            Dictionary containing:
                - 'test_files': List of discovered test file paths
                - 'classifications': Dict mapping file paths to Classification objects
                - 'discovery_method': Method used for discovery
                - 'stats': Discovery statistics
        """
        try:
            project_path = Path(project_path)
            test_files = self.discover_test_files(project_path)
            
            # Classify all discovered files
            classifications = {}
            for file_path_str in test_files:
                file_path = Path(file_path_str)
                classification = self._test_classifier.classify(file_path)
                classifications[file_path_str] = classification
            
            # Calculate stats
            test_count = sum(1 for c in classifications.values() if c.is_test)
            support_count = sum(1 for c in classifications.values() if c.is_support)
            avg_confidence = sum(c.confidence for c in classifications.values()) / len(classifications) if classifications else 0.0
            
            return {
                'test_files': test_files,
                'classifications': classifications,
                'discovery_method': self.config.test_discovery.mode,
                'stats': {
                    'total_files': len(test_files),
                    'test_files': test_count,
                    'support_files': support_count,
                    'average_confidence': avg_confidence,
                }
            }
            
        except Exception as e:
            raise FileDiscoveryError(f"Test discovery and classification failed: {e}", cause=e)

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
            FileDiscoveryError: If discovery fails
        """
        try:
            if separate_tests:
                source_files = self.discover_source_files(
                    project_path, include_test_files=False
                )
                test_files = self.discover_test_files(project_path)

                return {
                    "source_files": source_files,
                    "test_files": test_files,
                    "total_files": len(source_files) + len(test_files),
                }
            else:
                all_files = self.discover_source_files(
                    project_path, include_test_files=True
                )
                return {"all_files": all_files, "total_files": len(all_files)}

        except Exception as e:
            if isinstance(e, FileDiscoveryError):
                raise
            logger.exception(f"Failed to discover Python files: {e}")
            raise FileDiscoveryError(f"Python file discovery failed: {e}", cause=e)

    def filter_existing_files(self, file_paths: list[str | Path]) -> list[str]:
        """
        Filter a list of file paths to only include existing, valid Python files.

        Args:
            file_paths: List of file paths to filter

        Returns:
            List of existing, valid Python file paths (absolute paths)
        """
        filtered_files = []

        for file_path in file_paths:
            path = Path(file_path)

            if path.exists() and self._should_include_file(
                path, include_test_files=True
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
            if "*" in pattern:
                return Path(file_path).match(pattern)
            else:
                # Simple substring match
                return pattern in file_path
        except Exception:
            # Fallback to simple substring match if Path.match fails
            return pattern in file_path

    def get_discovery_stats(self, project_path: str | Path) -> dict:
        """
        Get statistics about file discovery in a project.

        Args:
            project_path: Root path of the project

        Returns:
            Dictionary with discovery statistics
        """
        try:
            discovery_result = self.discover_all_python_files(
                project_path, separate_tests=True
            )

            return {
                "project_path": str(Path(project_path).resolve()),
                "source_files_count": len(discovery_result["source_files"]),
                "test_files_count": len(discovery_result["test_files"]),
                "total_files": discovery_result["total_files"],
                "config_patterns": {
                    "test_patterns": self.config.test_patterns,
                    "exclude_patterns": list(self.config.exclude),
                    "exclude_dirs_count": len(self.config.exclude_dirs),
                },
            }

        except Exception as e:
            logger.warning(f"Failed to get discovery stats: {e}")
            return {"project_path": str(project_path), "error": str(e)}
