"""
Directory tree builder service.

Provides project structure building functionality for context generation
and refinement workflows, including module path derivation utilities.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DirectoryTreeBuilder:
    """
    Service for building directory tree representations.

    Provides simplified directory tree building for project structure
    context in both generation and refinement workflows.
    """

    @staticmethod
    def build_tree(project_path: Path) -> dict[str, Any]:
        """
        Build a simplified directory tree representation (backward compatible).

        Args:
            project_path: Root path to build tree from

        Returns:
            Dictionary representing the directory tree structure
        """
        try:
            tree = {"name": project_path.name, "type": "directory", "children": []}

            # Add immediate Python files and directories (simplified)
            for item in project_path.iterdir():
                if item.is_file() and item.suffix == ".py":
                    tree["children"].append(
                        {"name": item.name, "type": "file", "path": str(item)}
                    )
                elif item.is_dir() and not item.name.startswith("."):
                    tree["children"].append({"name": item.name, "type": "directory"})

            return tree

        except Exception as e:
            logger.warning("Failed to build directory tree: %s", e)
            return {}

    @staticmethod
    def build_tree_recursive(
        project_path: Path,
        max_depth: int = 4,
        max_entries_per_dir: int = 200,
        include_py_only: bool = True,
        _current_depth: int = 0,
    ) -> dict[str, Any]:
        """
        Build a recursive directory tree representation with safety limits.

        Args:
            project_path: Root path to build tree from
            max_depth: Maximum directory depth to traverse (safety limit)
            max_entries_per_dir: Maximum files/dirs to include per directory (safety limit)
            include_py_only: If True, only include .py files and directories
            _current_depth: Internal parameter for tracking recursion depth

        Returns:
            Dictionary representing the recursive directory tree structure
        """
        try:
            # Safety check: prevent excessive recursion
            if _current_depth >= max_depth:
                return {
                    "name": project_path.name,
                    "type": "directory",
                    "children": [],
                    "truncated": True,
                }

            tree = {"name": project_path.name, "type": "directory", "children": []}
            entry_count = 0

            # Get directory entries with safety limit
            try:
                items = list(project_path.iterdir())
            except (OSError, PermissionError) as e:
                logger.debug("Cannot read directory %s: %s", project_path, e)
                return tree

            # Sort for deterministic output (files first, then dirs, alphabetically)
            files = [item for item in items if item.is_file()]
            dirs = [item for item in items if item.is_dir()]

            # Filter and sort files
            if include_py_only:
                files = [f for f in files if f.suffix == ".py"]
            files.sort(key=lambda x: x.name.lower())

            # Filter and sort directories (skip hidden directories)
            dirs = [d for d in dirs if not d.name.startswith(".")]
            dirs.sort(key=lambda x: x.name.lower())

            # Add files first (up to the limit)
            for item in files[:max_entries_per_dir]:
                if entry_count >= max_entries_per_dir:
                    tree["truncated"] = True
                    break

                tree["children"].append(
                    {
                        "name": item.name,
                        "type": "file",
                        "path": str(item),
                        "size": item.stat().st_size if item.exists() else 0,
                    }
                )
                entry_count += 1

            # Add directories recursively (remaining space)
            remaining_entries = max_entries_per_dir - entry_count
            for item in dirs[:remaining_entries]:
                if entry_count >= max_entries_per_dir:
                    tree["truncated"] = True
                    break

                # Recurse into subdirectory
                subtree = DirectoryTreeBuilder.build_tree_recursive(
                    item,
                    max_depth,
                    max_entries_per_dir,
                    include_py_only,
                    _current_depth + 1,
                )
                tree["children"].append(subtree)
                entry_count += 1

            # Mark if we had to truncate
            if len(files) + len(dirs) > max_entries_per_dir:
                tree["truncated"] = True

            return tree

        except Exception as e:
            logger.warning(
                "Failed to build recursive directory tree for %s: %s", project_path, e
            )
            return {
                "name": project_path.name if project_path else "unknown",
                "type": "directory",
                "children": [],
            }


class ModulePathDeriver:
    """
    Service for deriving authoritative module paths for Python files.

    Handles src/ layouts, package boundaries, namespace packages, and validates
    import paths to ensure generated tests have correct import statements.
    """

    @staticmethod
    def derive_module_path(
        file_path: Path, project_root: Path | None = None
    ) -> dict[str, Any]:
        """
        Derive the exact dotted module_path for a target source file.

        Args:
            file_path: Path to the Python file
            project_root: Root path of the project (auto-detected if None)

        Returns:
            Dictionary with module_path, validation info, and metadata
        """
        try:
            # Auto-detect project root if not provided
            if project_root is None:
                project_root = ModulePathDeriver._find_project_root(file_path)

            if not project_root or not file_path.is_relative_to(project_root):
                logger.warning(
                    "File %s is not within project root %s", file_path, project_root
                )
                return {
                    "module_path": "",
                    "import_suggestion": "",
                    "validation_status": "failed",
                    "error": "File not in project root",
                    "fallback_paths": [],
                }

            # Get relative path from project root
            rel_path = file_path.relative_to(project_root)

            # Handle different project layouts
            module_candidates = ModulePathDeriver._generate_module_candidates(
                rel_path, project_root
            )

            # Validate candidates by attempting import
            validated_result = ModulePathDeriver._validate_module_paths(
                module_candidates, project_root, file_path
            )

            return validated_result

        except Exception as e:
            logger.warning("Failed to derive module path for %s: %s", file_path, e)
            return {
                "module_path": "",
                "import_suggestion": "",
                "validation_status": "error",
                "error": str(e),
                "fallback_paths": [],
            }

    @staticmethod
    def _find_project_root(file_path: Path) -> Path | None:
        """Find project root by looking for common markers."""
        current = file_path.parent if file_path.is_file() else file_path

        # Look for project markers going up the directory tree
        while current != current.parent:
            # Check for common project markers
            markers = [
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                ".git",
                "requirements.txt",
                "Pipfile",
                "uv.lock",
            ]

            for marker in markers:
                if (current / marker).exists():
                    return current

            current = current.parent

        # Fallback to the directory containing the file
        return file_path.parent if file_path.is_file() else file_path

    @staticmethod
    def _generate_module_candidates(rel_path: Path, project_root: Path) -> list[str]:
        """Generate possible module path candidates."""
        candidates = []

        # Remove .py extension
        if rel_path.suffix == ".py":
            base_path = rel_path.with_suffix("")
        else:
            base_path = rel_path

        parts = base_path.parts

        # Strategy 1: Direct path (e.g., testcraft/cli/main.py -> testcraft.cli.main)
        if parts:
            direct_path = ".".join(parts)
            candidates.append(direct_path)

        # Strategy 2: Handle src/ layouts
        if parts and parts[0] == "src":
            if len(parts) > 1:
                # src/package/module.py -> package.module
                src_path = ".".join(parts[1:])
                candidates.append(src_path)

                # Also try with src prefix: src.package.module
                src_prefix_path = ".".join(parts)
                candidates.append(src_prefix_path)

        # Strategy 3: Package-relative paths (skip common top-level dirs)
        skip_dirs = {"tests", "test", "docs", "examples", "scripts", "tools"}
        if parts and parts[0] not in skip_dirs:
            # Already handled in Strategy 1
            pass
        elif len(parts) > 1:
            # tests/test_module.py -> test_module (for test files)
            no_top_dir = ".".join(parts[1:])
            candidates.append(no_top_dir)

        # Strategy 4: Handle __init__.py files
        if rel_path.name == "__init__.py":
            if len(parts) > 1:
                # Remove __init__.py and use parent directory
                init_candidates = []
                parent_parts = parts[:-1]  # Remove __init__.py

                init_candidates.append(".".join(parent_parts))

                # Handle src/ layout for __init__.py
                if parent_parts and parent_parts[0] == "src" and len(parent_parts) > 1:
                    init_candidates.append(".".join(parent_parts[1:]))

                candidates.extend(init_candidates)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    @staticmethod
    def _validate_module_paths(
        candidates: list[str], project_root: Path, file_path: Path
    ) -> dict[str, Any]:
        """Validate module path candidates by attempting import."""
        if not candidates:
            return {
                "module_path": "",
                "import_suggestion": "",
                "validation_status": "no_candidates",
                "error": "No module path candidates generated",
                "fallback_paths": [],
            }

        # Prepare import paths (project_root and project_root/src)
        import_paths = [str(project_root)]
        src_path = project_root / "src"
        if src_path.exists() and src_path.is_dir():
            import_paths.append(str(src_path))

        validated_paths = []
        failed_paths = []

        # Try each candidate
        for candidate in candidates:
            try:
                # Test import with temporary sys.path modification
                original_path = sys.path.copy()
                try:
                    # Prepend our paths for testing
                    sys.path = import_paths + sys.path

                    # Attempt to import the module
                    try:
                        # Use importlib.util for safer validation
                        spec = importlib.util.find_spec(candidate)
                        if spec and spec.origin:
                            # Verify the spec points to our target file
                            if Path(spec.origin).resolve() == file_path.resolve():
                                validated_paths.append(
                                    {
                                        "module_path": candidate,
                                        "validation_status": "validated",
                                        "spec_origin": spec.origin,
                                    }
                                )
                            else:
                                failed_paths.append(
                                    {
                                        "module_path": candidate,
                                        "validation_status": "wrong_file",
                                        "spec_origin": spec.origin,
                                        "expected_file": str(file_path),
                                    }
                                )
                        else:
                            failed_paths.append(
                                {
                                    "module_path": candidate,
                                    "validation_status": "no_spec",
                                    "error": "Module spec not found",
                                }
                            )
                    except (ImportError, ModuleNotFoundError, ValueError) as e:
                        failed_paths.append(
                            {
                                "module_path": candidate,
                                "validation_status": "import_error",
                                "error": str(e),
                            }
                        )

                finally:
                    # Always restore sys.path
                    sys.path = original_path

            except Exception as e:
                failed_paths.append(
                    {
                        "module_path": candidate,
                        "validation_status": "validation_error",
                        "error": str(e),
                    }
                )

        # Return best result
        if validated_paths:
            best = validated_paths[0]  # First validated path
            # Extract class/function names for import suggestion
            import_suggestion = ModulePathDeriver._generate_import_suggestion(
                best["module_path"], file_path
            )

            return {
                "module_path": best["module_path"],
                "import_suggestion": import_suggestion,
                "validation_status": "validated",
                "spec_origin": best.get("spec_origin"),
                "fallback_paths": [p["module_path"] for p in validated_paths[1:]]
                + [p["module_path"] for p in failed_paths],
            }
        else:
            # No validated paths, return best guess with fallbacks
            best_guess = candidates[0] if candidates else ""
            import_suggestion = (
                ModulePathDeriver._generate_import_suggestion(best_guess, file_path)
                if best_guess
                else ""
            )

            return {
                "module_path": best_guess,
                "import_suggestion": import_suggestion,
                "validation_status": "unvalidated",
                "error": "No candidates could be validated",
                "fallback_paths": candidates[1:] if len(candidates) > 1 else [],
                "failed_validations": failed_paths,
            }

    @staticmethod
    def _generate_import_suggestion(module_path: str, file_path: Path) -> str:
        """Generate import suggestion for the module."""
        if not module_path:
            return ""

        try:
            # Simple heuristic: suggest importing the module name or main classes

            # For __init__.py, suggest importing the package
            if file_path.name == "__init__.py":
                return f"import {module_path}"

            # For regular modules, suggest importing specific items or the module
            # We'll generate both forms for flexibility
            suggestions = [
                f"from {module_path} import {{ClassName}}",
                f"from {module_path} import {{function_name}}",
                f"import {module_path}",
            ]

            # Return the most common pattern
            return suggestions[0]  # from X import Y is most common for specific items

        except Exception as e:
            logger.debug(
                "Could not generate import suggestion for %s: %s", module_path, e
            )
            return f"from {module_path} import ..." if module_path else ""
