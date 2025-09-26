"""
Import resolver service for generating canonical imports and bootstrap configuration.

This service builds on PackagingDetector to provide the specific import resolution
required by the context assembly pipeline, including bootstrap conftest generation
for different project layouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from .packaging_detector import PackagingDetector, PackagingInfo

logger = logging.getLogger(__name__)


class ImportMap(TypedDict):
    """Import mapping information for a target file."""

    target_import: str
    sys_path_roots: list[str]
    needs_bootstrap: bool
    bootstrap_conftest: str


class ImportResolver:
    """
    Service for resolving canonical imports and bootstrap requirements.

    Provides the exact API required by the context assembly specification:
    resolve(file: Path) -> ImportMap
    """

    def __init__(self) -> None:
        self._packaging_cache: dict[str, PackagingInfo] = {}

    def resolve(self, file_path: Path) -> ImportMap:
        """
        Resolve import configuration for a target file.

        Args:
            file_path: Path to the Python file to resolve imports for

        Returns:
            ImportMap with canonical import string, sys.path roots,
            bootstrap requirements, and conftest.py content
        """
        try:
            # Find project root
            project_root = self._find_project_root(file_path)
            logger.debug(
                "Resolving imports for %s in project %s", file_path, project_root
            )

            # Get packaging information
            packaging_info = self._get_packaging_info(project_root)

            # Get canonical import path - try enhanced resolution first
            canonical_import = self._resolve_canonical_import(
                file_path, packaging_info, project_root
            )
            if not canonical_import:
                raise ValueError(
                    f"Cannot determine canonical import for {file_path} - no package structure found"
                )

            # Validate that we have a proper package structure, not just a standalone file
            if not self._has_valid_package_structure(file_path, packaging_info):
                raise ValueError(
                    f"Cannot determine canonical import for {file_path} - no package structure found"
                )

            # Build target import statement
            target_import = self._build_target_import(canonical_import)

            # Determine sys.path roots (convert to strings)
            sys_path_roots = [
                str(root.resolve()) for root in packaging_info.source_roots
            ]

            # Determine if bootstrap is needed
            needs_bootstrap = self._needs_bootstrap(packaging_info, project_root)

            # Generate bootstrap conftest content
            bootstrap_conftest = (
                self._generate_bootstrap_conftest(sys_path_roots)
                if needs_bootstrap
                else ""
            )

            return ImportMap(
                target_import=target_import,
                sys_path_roots=sys_path_roots,
                needs_bootstrap=needs_bootstrap,
                bootstrap_conftest=bootstrap_conftest,
            )

        except Exception as e:
            logger.error("Failed to resolve imports for %s: %s", file_path, e)
            raise

    def _find_project_root(self, file_path: Path) -> Path:
        """Find project root by looking for common project markers."""
        current = file_path.parent if file_path.is_file() else file_path
        original_start = current

        while current != current.parent:
            # Check for strong project markers
            if any(
                (current / marker).exists()
                for marker in [
                    "pyproject.toml",
                    "setup.py",
                    "setup.cfg",
                    ".git",
                    "requirements.txt",
                    "Pipfile",
                    "poetry.lock",
                ]
            ):
                return current

            # Stop searching if we've gone too far up (beyond reasonable project boundaries)
            # This prevents going all the way to system directories
            if self._is_beyond_project_boundary(current, original_start):
                break

            current = current.parent

        # If we didn't find any project markers, use a conservative fallback:
        # Find the parent directory that contains Python packages but isn't one itself
        current = original_start
        while current != current.parent:
            # If current directory doesn't have __init__.py but contains Python packages,
            # it's likely the project root
            try:
                if not (current / "__init__.py").exists() and any(
                    (child / "__init__.py").exists()
                    for child in current.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                ):
                    return current
            except (PermissionError, OSError):
                # Skip directories we can't access (e.g., system directories)
                pass

            # Stop if we've gone too far
            if self._is_beyond_project_boundary(current, original_start):
                break

            current = current.parent

        # Final conservative fallback: parent of the package directory
        package_dir = original_start
        if (package_dir / "__init__.py").exists():
            return package_dir.parent

        return original_start

    def _is_beyond_project_boundary(self, current: Path, original_start: Path) -> bool:
        """Check if we've gone beyond reasonable project boundaries."""
        try:
            # If we can't make the path relative, we've gone too far
            original_start.relative_to(current)

            # Stop at system directories or very deep nesting
            parts = list(current.parts)
            if len(parts) <= 3:  # e.g., ['/', 'var', 'folders'] - system level
                return True

            # Stop if we're in temp directories but have gone above them
            if any(part.startswith("tmp") for part in parts):
                # If original_start was in a temp dir, don't go above temp dirs
                orig_parts = list(original_start.parts)
                if any(part.startswith("tmp") for part in orig_parts) and not any(
                    part.startswith("tmp") for part in parts[-3:]
                ):
                    return True

            return False
        except ValueError:
            # current is not a parent of original_start
            return True

    def _get_packaging_info(self, project_root: Path) -> PackagingInfo:
        """Get or cache packaging information for a project."""
        cache_key = str(project_root.resolve())

        if cache_key not in self._packaging_cache:
            self._packaging_cache[cache_key] = PackagingDetector.detect_packaging(
                project_root
            )

        return self._packaging_cache[cache_key]

    def _has_valid_package_structure(
        self, file_path: Path, packaging_info: PackagingInfo
    ) -> bool:
        """
        Validate that the file is part of a proper Python package structure.

        Returns False for standalone files that aren't part of any package.
        """
        file_abs = file_path.resolve()

        # If PackagingDetector found package directories, check if our file is in one
        if packaging_info.package_directories:
            for pkg_dir in packaging_info.package_directories:
                pkg_dir_abs = pkg_dir.resolve()
                try:
                    # Check if file is within this package directory
                    file_abs.relative_to(pkg_dir_abs)
                    return True  # File is in a recognized package directory
                except ValueError:
                    continue

        # Also check if the file's directory has __init__.py (direct check)
        if (file_abs.parent / "__init__.py").exists():
            return True

        # If no package structure found, this is likely a standalone file
        # But be lenient - if PackagingDetector found a valid import, allow it
        canonical_import = packaging_info.get_canonical_import(file_path)
        if canonical_import and "." in canonical_import:
            return True  # Has package-like import structure

        return False

    def _resolve_canonical_import(
        self, file_path: Path, packaging_info: PackagingInfo, project_root: Path
    ) -> str | None:
        """
        Resolve canonical import with enhanced logic for edge cases.

        This method provides fallback logic when PackagingDetector fails to
        properly resolve imports for certain layouts (monorepo, heuristic detection).
        """
        # First try the standard packaging detector approach
        canonical_import = packaging_info.get_canonical_import(file_path)

        if canonical_import:
            # Only apply corrections for specific problematic cases
            # Don't correct if it's already a simple, valid package import
            if self._needs_correction(canonical_import, file_path, packaging_info):
                corrected_import = self._correct_import_path(
                    file_path, canonical_import, packaging_info
                )
                if corrected_import:
                    return corrected_import

            return canonical_import

        # Enhanced fallback: manually walk up to find the package root
        return self._manual_import_resolution(
            file_path, packaging_info.source_roots, project_root
        )

    def _needs_correction(
        self, canonical_import: str, file_path: Path, packaging_info: PackagingInfo
    ) -> bool:
        """
        Determine if a canonical import needs correction.

        Only correct imports that include non-package parent directories.
        """
        import_parts = canonical_import.split(".")

        if len(import_parts) <= 1:
            return False

        # Check if the first part of the import corresponds to a non-package directory
        first_part = import_parts[0]

        for source_root in packaging_info.source_roots:
            potential_dir = source_root / first_part
            if potential_dir.exists() and not (potential_dir / "__init__.py").exists():
                # This is a non-package directory, so correction is needed
                return True

        return False

    def _correct_import_path(
        self, file_path: Path, canonical_import: str, packaging_info: PackagingInfo
    ) -> str | None:
        """
        Correct import paths that include non-package parent directories.

        For example, converts 'libs.pkg_a.module_a' to 'pkg_a.module_a' when
        'libs' is not a Python package but 'pkg_a' is.
        """
        file_abs = file_path.resolve()
        import_parts = canonical_import.split(".")

        if len(import_parts) <= 1:
            return None  # Nothing to correct

        # Walk backwards from the file to find the first actual package
        current_dir = file_abs.parent
        package_parts = []

        # Collect all directories with __init__.py (actual packages)
        while current_dir and current_dir != current_dir.parent:
            if (current_dir / "__init__.py").exists():
                package_parts.insert(0, current_dir.name)
                current_dir = current_dir.parent
            else:
                # Stop when we hit a non-package directory
                break

        if not package_parts:
            return None

        # Add the module name if it's not __init__
        module_name = file_abs.stem
        if module_name != "__init__":
            package_parts.append(module_name)

        corrected_import = ".".join(package_parts)

        # Only return the correction if it's different and shorter (removes unwanted parent dirs)
        # The key fix: check that we're actually removing non-package parents
        if corrected_import != canonical_import and len(
            corrected_import.split(".")
        ) < len(canonical_import.split(".")):
            # Verify we're actually fixing a real issue by checking if the first part
            # of the original import corresponds to a non-package directory
            original_first_part = import_parts[0]

            # Find the corresponding directory for the first part
            source_roots = packaging_info.source_roots
            for root in source_roots:
                potential_dir = root / original_first_part
                if (
                    potential_dir.exists()
                    and not (potential_dir / "__init__.py").exists()
                ):
                    # This is a non-package directory, so correction is valid
                    return corrected_import

        return None

    def _manual_import_resolution(
        self, file_path: Path, source_roots: list[Path], project_root: Path
    ) -> str | None:
        """
        Manually resolve import path by walking up from the file to find package boundaries.

        This handles cases where PackagingDetector's automatic detection fails.
        """
        file_abs = file_path.resolve()

        # Try each source root to find the best match
        best_import = None

        for source_root in source_roots:
            root_abs = source_root.resolve()

            # Check if file is under this source root
            try:
                file_abs.relative_to(root_abs)
            except ValueError:
                continue

            # Walk up from file to find the package boundary
            package_parts = []
            current_dir = file_abs.parent

            while current_dir != root_abs and current_dir != current_dir.parent:
                # Check if current directory is a Python package
                if (current_dir / "__init__.py").exists():
                    package_parts.insert(0, current_dir.name)
                    current_dir = current_dir.parent
                else:
                    # If we hit a non-package directory, treat it as a package root
                    # This handles cases like "libs/pkg_a" where "libs" is not a package
                    # but "pkg_a" is the actual package we want to import
                    if package_parts:  # We found at least one package level
                        break
                    current_dir = current_dir.parent

            if package_parts:
                # Add the module name (file without .py extension)
                module_name = file_abs.stem
                if module_name != "__init__":
                    package_parts.append(module_name)

                import_path = ".".join(package_parts)

                # Prefer shorter import paths (closer to actual package roots)
                if best_import is None or len(import_path.split(".")) < len(
                    best_import.split(".")
                ):
                    best_import = import_path

        return best_import

    def _build_target_import(self, canonical_import: str) -> str:
        """
        Build the target import statement from canonical import path.

        Examples:
        - "my_pkg.sub.module" -> "import my_pkg.sub.module as _under_test"
        - "my_pkg.utils" -> "import my_pkg.utils as _under_test"
        """
        return f"import {canonical_import} as _under_test"

    def _needs_bootstrap(
        self, packaging_info: PackagingInfo, project_root: Path
    ) -> bool:
        """
        Determine if sys.path bootstrap is needed.

        Bootstrap is needed when:
        1. Source roots are not the project root (e.g., src/ layout)
        2. Multiple source roots exist
        3. Project uses custom package directories
        """
        # If we have non-standard source roots, we likely need bootstrap
        source_roots = packaging_info.source_roots

        # If project root is the only source root, pytest usually handles this
        if (
            len(source_roots) == 1
            and source_roots[0].resolve() == project_root.resolve()
        ):
            return False

        # If we have src/ layout or multiple roots, we need bootstrap
        if len(source_roots) > 1:
            return True

        # Check if any source root is not the project root
        for root in source_roots:
            if root.resolve() != project_root.resolve():
                return True

        return False

    def _generate_bootstrap_conftest(self, sys_path_roots: list[str]) -> str:
        """
        Generate minimal conftest.py content for sys.path bootstrap.

        Follows the specification format:
        ```python
        import sys, pathlib
        for p in [{{ sys_path_roots }}]:
            p = pathlib.Path(p).resolve()
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        ```
        """
        if not sys_path_roots:
            return ""

        # Format the roots as a Python list
        roots_list = repr(sys_path_roots)

        return f"""import sys
import pathlib

# Auto-generated bootstrap for repository-aware test execution
for p in {roots_list}:
    p = pathlib.Path(p).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
"""
