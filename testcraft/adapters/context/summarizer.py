"""
Summarization utilities for files and directory structures.

Focuses on:
- Generating a compact directory tree (bounded breadth/depth)
- Extracting top-level imports and class/function signatures
- Enforcing max character budgets for summaries
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _create_error_metadata(
    error_type: str, message: str, recoverable: bool = True
) -> dict[str, Any]:
    """Create structured error metadata for consistent error reporting."""
    return {
        "error": message,
        "error_type": error_type,
        "recoverable": recoverable,
        "timestamp": None,  # Could add timestamp if needed
    }


@dataclass
class SignatureSummary:
    classes: list[str]
    functions: list[str]


class ContextSummarizer:
    def __init__(self) -> None:
        self._ast_cache: dict[str, ast.Module] = {}
        self._import_cache: dict[str, list[str]] = {}

    def _get_cached_ast(self, content: str) -> ast.Module | None:
        """Get cached AST or parse and cache new content."""
        content_hash_str = str(hash(content))
        if content_hash_str in self._ast_cache:
            return self._ast_cache[content_hash_str]

        try:
            tree = ast.parse(content)
            # Only cache if content is reasonably sized to avoid memory issues
            if len(content) < 100000:  # 100KB limit
                self._ast_cache[content_hash_str] = tree
            return tree
        except SyntaxError:
            return None

    def _clear_caches(self) -> None:
        """Clear internal caches to free memory."""
        self._ast_cache.clear()
        self._import_cache.clear()

    def clear_caches(self) -> None:
        """Public method to clear internal caches and free memory."""
        self._clear_caches()

    def _read_file_robust(self, path: Path, max_chars: int) -> str:
        """Read file with multiple encoding attempts and size checks."""
        # File size check to prevent memory issues
        try:
            file_size = path.stat().st_size
            if file_size > max_chars * 10:  # Rough heuristic: 10x max_chars limit
                return f"File too large ({file_size} bytes) to process safely"
        except OSError as e:
            return f"Cannot access file: {e}"

        # Multiple encoding attempts
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                logger.debug(f"Successfully read {path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                logger.debug(f"Failed to read {path} with {encoding} encoding")
                continue
            except OSError as e:
                logger.debug(f"OS error reading {path} with {encoding}: {e}")
                return f"Cannot read file: {e}"

        return "File contains invalid characters in all attempted encodings"

    def summarize_file(
        self,
        path: Path,
        *,
        content: str | None = None,
        summary_type: str = "comprehensive",
        max_chars: int = 2000,
    ) -> dict[str, Any]:
        # Validate inputs
        if max_chars <= 0:
            return {
                "summary": "",
                "key_functions": [],
                "key_classes": [],
                "dependencies": [],
                "summary_metadata": _create_error_metadata(
                    "validation_error", "max_chars must be positive", recoverable=False
                ),
            }

        try:
            if content is None:
                if not path.exists():
                    return {
                        "summary": "",
                        "key_functions": [],
                        "key_classes": [],
                        "dependencies": [],
                        "summary_metadata": _create_error_metadata(
                            "file_not_found",
                            f"File not found: {str(path)}",
                            recoverable=False,
                        ),
                    }

                # Robust file reading with multiple encoding attempts and size checks
                content = self._read_file_robust(path, max_chars)

            # Check if content indicates an error
            if content.startswith(
                (
                    "File too large",
                    "Cannot read file",
                    "Cannot access file",
                    "File contains invalid",
                )
            ):
                return {
                    "summary": "",
                    "key_functions": [],
                    "key_classes": [],
                    "dependencies": [],
                    "summary_metadata": _create_error_metadata(
                        "read_error", content, recoverable=True
                    ),
                }

            imports = self._extract_imports(content)
            sigs = self._extract_signatures(content, extraction_depth="top_level")

            summary_lines: list[str] = []
            summary_lines.append(f"File: {path.name}")

            # Safe bounds checking for list slicing
            safe_imports = sorted(set(imports)) if imports else []
            safe_classes = sigs.classes if sigs.classes else []
            safe_functions = sigs.functions if sigs.functions else []

            if safe_imports:
                summary_lines.append(
                    "Imports: " + ", ".join(safe_imports[: min(10, len(safe_imports))])
                )
            if safe_classes:
                summary_lines.append(
                    "Classes: " + ", ".join(safe_classes[: min(10, len(safe_classes))])
                )
            if safe_functions:
                summary_lines.append(
                    "Functions: "
                    + ", ".join(safe_functions[: min(10, len(safe_functions))])
                )

            return {
                "summary": _truncate("\n".join(summary_lines), max_chars),
                "key_functions": sigs.functions,
                "key_classes": sigs.classes,
                "dependencies": sorted(set(imports)),
                "summary_metadata": {"summary_type": summary_type},
            }

        except Exception as e:
            logger.debug(f"Unexpected error in summarize_file for {path}: {e}")
            return {
                "summary": "",
                "key_functions": [],
                "key_classes": [],
                "dependencies": [],
                "summary_metadata": _create_error_metadata(
                    "unexpected_error", f"Unexpected error: {e}", recoverable=True
                ),
            }

    def summarize_directory_tree(
        self,
        root: Path,
        *,
        max_depth: int = 3,
        max_breadth: int = 10,
        max_chars: int = 4000,
    ) -> str:
        # Input validation
        if max_depth < 0:
            return f"Error: max_depth must be non-negative, got {max_depth}"
        if max_breadth <= 0:
            return f"Error: max_breadth must be positive, got {max_breadth}"
        if max_chars <= 0:
            return f"Error: max_chars must be positive, got {max_chars}"

        def walk(dir_path: Path, depth: int, visited: set[str]) -> list[str]:
            if depth > max_depth:
                return []

            try:
                # Check for symlinks and cycles
                resolved_path = str(dir_path.resolve())
                if resolved_path in visited or dir_path.is_symlink():
                    return ["  " * (depth - 1) + "- [symlink/cycle: {dir_path.name}]"]

                visited.add(resolved_path)

                # Skip common non-source directories
                skip_dirs = {
                    ".git",
                    "__pycache__",
                    "node_modules",
                    ".pytest_cache",
                    ".mypy_cache",
                    "build",
                    "dist",
                }
                if dir_path.name in skip_dirs:
                    return []

                entries = []
                try:
                    # Sort entries for consistent output
                    all_entries = sorted(
                        dir_path.iterdir(), key=lambda p: p.name.lower()
                    )
                    # Filter out hidden files/directories and apply breadth limit
                    for entry in all_entries:
                        if not entry.name.startswith(".") or entry.name in {".github"}:
                            entries.append(entry)
                            if len(entries) >= max_breadth:
                                break
                except (OSError, PermissionError) as e:
                    logger.debug(f"Permission error accessing {dir_path}: {e}")
                    return ["  " * (depth - 1) + "- [access denied: {dir_path.name}]"]

                lines: list[str] = []
                for entry in entries:
                    indent = "  " * (depth - 1)
                    prefix = "- "
                    lines.append(f"{indent}{prefix}{entry.name}")

                    if entry.is_dir():
                        # Check character limit before recursing
                        current_length = len("\n".join(lines))
                        if (
                            current_length > max_chars * 0.7
                        ):  # More aggressive early termination
                            lines.append("  " * depth + "- [truncated...]")
                            break

                        sublines = walk(entry, depth + 1, visited)
                        lines.extend(sublines)

                        # Final check after recursion
                        if len("\n".join(lines)) > max_chars * 0.9:
                            lines.append("  " * depth + "- [truncated...]")
                            break

                return lines

            except (OSError, PermissionError) as e:
                logger.debug(f"Error accessing directory {dir_path}: {e}")
                return ["  " * (depth - 1) + "- [error: {dir_path.name}]"]

        if not root.exists():
            return ""

        visited_paths: set[str] = set()
        try:
            tree = [root.name]
            tree.extend(walk(root, 1, visited_paths))
            return _truncate("\n".join(tree), max_chars)
        except Exception as e:
            logger.debug(f"Error building directory tree for {root}: {e}")
            return f"Error building directory tree: {e}"

    def _extract_imports(self, content: str) -> list[str]:
        """Enhanced import extraction that preserves full module paths and handles relative imports."""
        # Check cache first
        content_hash_str = str(hash(content))
        if content_hash_str in self._import_cache:
            return self._import_cache[content_hash_str]

        try:
            imports: list[str] = []
            tree = self._get_cached_ast(content)
            if tree is None:
                self._import_cache[content_hash_str] = []
                return []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Preserve full module path
                        full_module = alias.name
                        imports.append(full_module)
                        # Also add root module for compatibility
                        root_module = full_module.split(".")[0]
                        if root_module != full_module:
                            imports.append(root_module)

                elif isinstance(node, ast.ImportFrom) and node.module:
                    # Handle relative imports
                    if node.level > 0:
                        # Relative import - store as-is for now
                        relative_module = "." * node.level + (node.module or "")
                        imports.append(relative_module)
                    else:
                        # Absolute import - preserve full module path
                        full_module = node.module
                        imports.append(full_module)
                        # Also add root module for compatibility
                        root_module = full_module.split(".")[0]
                        if root_module != full_module:
                            imports.append(root_module)

            # Intelligent deduplication - prefer full paths over root paths when both exist
            seen = set()
            deduplicated = []
            for imp in imports:
                if imp not in seen:
                    seen.add(imp)
                    deduplicated.append(imp)
                elif "." in imp:
                    # If we already have the root, replace it with the full path
                    root = imp.split(".")[0]
                    if root in seen:
                        # Remove the root and add the full path
                        if root in deduplicated:
                            deduplicated.remove(root)
                        deduplicated.append(imp)
                        seen.add(imp)

            result = sorted(deduplicated)
            self._import_cache[content_hash_str] = result
            return result

        except Exception as e:
            logger.debug(f"Failed to extract imports: {e}")
            self._import_cache[content_hash_str] = []
            return []

    def _extract_signatures(
        self, content: str, *, extraction_depth: str = "top_level"
    ) -> SignatureSummary:
        """Enhanced signature extraction using ast.walk() with support for async functions and methods."""
        classes: list[str] = []
        functions: list[str] = []

        try:
            tree = self._get_cached_ast(content)
            if tree is None:
                return SignatureSummary(classes=[], functions=[])

            # Use ast.walk() to traverse all nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    # Include methods within classes if requested
                    if extraction_depth in ("nested", "comprehensive"):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                                functions.append(f"{node.name}.{item.name}")

                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Only add top-level functions if not including nested
                    # Note: Methods within classes are handled in the ClassDef branch above
                    if extraction_depth == "top_level":
                        # For top-level mode, we add all function names
                        # Methods are already captured with class.method format above
                        functions.append(node.name)
                    elif extraction_depth in ("nested", "comprehensive"):
                        functions.append(node.name)

        except Exception as e:
            logger.debug(f"Failed to extract signatures: {e}")

        return SignatureSummary(classes=classes, functions=functions)
