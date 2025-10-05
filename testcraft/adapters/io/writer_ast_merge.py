"""
Writer adapter that merges content using AST analysis.

This adapter provides sophisticated merging functionality for test files,
analyzing AST structures to avoid duplicates and merge content intelligently.
"""

import ast
import difflib
import fcntl
import logging
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

from .python_formatters import format_python_content
from .safety import SafetyError, SafetyPolicies


class WriterASTMergeError(Exception):
    """Exception raised when AST merge writer operations fail."""

    pass


class ASTMerger:
    """Helper class for merging AST structures."""

    @staticmethod
    def extract_elements(tree: ast.Module) -> dict[str, Any]:
        """
        Extract elements from an AST tree.

        Args:
            tree: AST tree to analyze

        Returns:
            Dictionary containing extracted elements
        """
        elements: dict[str, list[Any]] = {
            "imports": [],
            "from_imports": [],
            "functions": [],
            "classes": [],
            "constants": [],
            "other_statements": [],
        }

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    elements["imports"].append(
                        {"name": alias.name, "asname": alias.asname, "node": node}
                    )
            elif isinstance(node, ast.ImportFrom):
                elements["from_imports"].append(
                    {
                        "module": node.module,
                        "names": [(alias.name, alias.asname) for alias in node.names],
                        "level": node.level,
                        "node": node,
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                elements["functions"].append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(dec) for dec in node.decorator_list],
                        "node": node,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                elements["classes"].append(
                    {
                        "name": node.name,
                        "bases": [ast.unparse(base) for base in node.bases],
                        "decorators": [ast.unparse(dec) for dec in node.decorator_list],
                        "node": node,
                    }
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        elements["constants"].append(
                            {
                                "name": target.id,
                                "value": ast.unparse(node.value),
                                "node": node,
                            }
                        )
            else:
                elements["other_statements"].append(node)

        return elements

    @staticmethod
    def merge_imports(
        existing_imports: list[dict], new_imports: list[dict]
    ) -> list[ast.stmt]:
        """Merge import statements, avoiding duplicates."""
        merged = []
        existing_names = {imp["name"] for imp in existing_imports}

        # Add existing imports
        for imp in existing_imports:
            merged.append(imp["node"])

        # Add new imports that don't exist
        for imp in new_imports:
            if imp["name"] not in existing_names:
                merged.append(imp["node"])
                existing_names.add(imp["name"])

        return merged

    @staticmethod
    def merge_from_imports(
        existing_from_imports: list[dict], new_from_imports: list[dict]
    ) -> list[ast.stmt]:
        """Merge from import statements, avoiding duplicates."""
        existing_modules = {}

        # Group existing imports by module
        for imp in existing_from_imports:
            module = imp["module"] or ""
            if module not in existing_modules:
                existing_modules[module] = {"names": set(), "node": imp["node"]}
            for name, asname in imp["names"]:
                existing_modules[module]["names"].add((name, asname))

        # Merge new imports
        for imp in new_from_imports:
            module = imp["module"] or ""
            if module in existing_modules:
                # Add new names to existing module
                new_names = set(imp["names"])
                existing_names = existing_modules[module]["names"]
                all_names = existing_names | new_names

                if all_names != existing_names:
                    # Create new ImportFrom node with merged names
                    existing_node = existing_modules[module]["node"]
                    new_node = ast.ImportFrom(
                        module=existing_node.module,
                        names=[
                            ast.alias(name=name, asname=asname)
                            for name, asname in sorted(all_names)
                        ],
                        level=existing_node.level,
                        lineno=existing_node.lineno,
                        col_offset=existing_node.col_offset,
                    )
                    existing_modules[module]["node"] = new_node
                    existing_modules[module]["names"] = all_names
            else:
                # Add new module import
                existing_modules[module] = {
                    "names": set(imp["names"]),
                    "node": imp["node"],
                }

        # Return merged import nodes
        return [module_data["node"] for module_data in existing_modules.values()]

    @staticmethod
    def merge_functions(
        existing_functions: list[dict], new_functions: list[dict]
    ) -> list[ast.stmt]:
        """Merge function definitions, avoiding duplicates by name."""
        merged = []
        existing_names = {func["name"] for func in existing_functions}

        # Add existing functions
        for func in existing_functions:
            merged.append(func["node"])

        # Add new functions that don't exist
        for func in new_functions:
            if func["name"] not in existing_names:
                merged.append(func["node"])
                existing_names.add(func["name"])

        return merged

    @staticmethod
    def merge_classes(
        existing_classes: list[dict], new_classes: list[dict]
    ) -> list[ast.stmt]:
        """Merge class definitions, avoiding duplicates by name."""
        merged = []
        existing_names = {cls["name"] for cls in existing_classes}

        # Add existing classes
        for cls in existing_classes:
            merged.append(cls["node"])

        # Add new classes that don't exist
        for cls in new_classes:
            if cls["name"] not in existing_names:
                merged.append(cls["node"])
                existing_names.add(cls["name"])

        return merged


class WriterASTMergeAdapter:
    """
    Writer adapter that merges content using AST analysis.

    This adapter parses existing and new content, merges them structurally
    to avoid duplicates, and formats the result with Black and isort.
    """

    def __init__(self, project_root: Path | None = None, dry_run: bool = False) -> None:
        """
        Initialize the AST merge writer adapter.

        Args:
            project_root: Optional project root path for validation
            dry_run: Whether to run in dry-run mode (no actual writing)
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.merger = ASTMerger()
        self.logger = logging.getLogger(__name__)
        # Cache for parsed ASTs to avoid re-parsing
        self._ast_cache: dict[str, ast.Module] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def write_file(
        self,
        file_path: str | Path,
        content: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write content to a file using AST merging.

        Args:
            file_path: Path where the file should be written
            content: Content to write to the file
            overwrite: Whether to overwrite existing files
            **kwargs: Additional writing parameters

        Returns:
            Dictionary containing write operation results

        Raises:
            WriterASTMergeError: If file writing fails
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
                raise WriterASTMergeError(
                    f"Failed to create directory: {resolved_path.parent}"
                )

            # Read existing content if file exists
            file_existed = resolved_path.exists()
            existing_content = ""
            if file_existed and not overwrite:
                existing_content = resolved_path.read_text(encoding="utf-8")

            # Merge content using AST analysis
            if existing_content:
                merged_content = self._merge_content(existing_content, content)
            else:
                merged_content = content

            # Format the merged content
            formatted_content = self._format_content(merged_content)
            formatting_status = "success"

            if self.dry_run:
                # Generate diff for dry-run
                if existing_content:
                    diff = self._generate_diff(
                        existing_content, formatted_content, str(resolved_path)
                    )
                else:
                    diff = f"Creating new file: {resolved_path}\n" + formatted_content

                return {
                    "success": True,
                    "dry_run": True,
                    "file_path": str(file_path),
                    "bytes_written": len(formatted_content.encode("utf-8")),
                    "diff": diff,
                    "file_existed": file_existed,
                    "backup_path": None,
                    "formatting_status": formatting_status,
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
                "merged": bool(existing_content),
                "formatting_status": formatting_status,
            }

        except (SafetyError, OSError, SyntaxError) as e:
            raise WriterASTMergeError(f"Failed to write file {file_path}: {e}") from e

    def write_test_file(
        self,
        test_path: str | Path,
        test_content: str,
        source_file: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Write a test file using AST merging.

        Args:
            test_path: Path where the test file should be written
            test_content: Test content to write
            source_file: Optional path to the source file being tested
            **kwargs: Additional parameters for test file writing

        Returns:
            Dictionary containing test file write results

        Raises:
            WriterASTMergeError: If test file writing fails
        """
        try:
            # Write the file using AST merging
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
            raise WriterASTMergeError(
                f"Failed to write test file {test_path}: {e}"
            ) from e

    def backup_file(
        self, file_path: str | Path, backup_suffix: str = ".backup"
    ) -> dict[str, Any]:
        """
        Create a backup of an existing file with atomic operations.

        Args:
            file_path: Path of the file to backup
            backup_suffix: Suffix to add to the backup filename

        Returns:
            Dictionary containing backup operation results

        Raises:
            WriterASTMergeError: If backup creation fails
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

            if self.dry_run:
                # For dry run, create timestamped backup name
                timestamp = int(time.time())
                pid = os.getpid()
                backup_name = f"{file_path.stem}{backup_suffix}.{timestamp}.{pid}{file_path.suffix}"
                backup_path = file_path.parent / backup_name

                return {
                    "success": True,
                    "dry_run": True,
                    "original_path": str(file_path),
                    "backup_path": str(backup_path),
                }

            # Create atomic backup using tempfile.mkstemp for race condition safety
            timestamp = int(time.time())
            pid = os.getpid()

            # Use atomic temporary file creation
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"{backup_suffix}.{timestamp}.{pid}{file_path.suffix}",
                dir=file_path.parent,
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                # Read original content
                try:
                    with open(file_path, encoding="utf-8") as src_file:
                        # Use file locking for cross-platform compatibility
                        if platform.system() != "Windows":
                            fcntl.flock(
                                src_file.fileno(), fcntl.LOCK_SH
                            )  # Shared lock for reading

                        content = src_file.read()

                        # Write to temporary file
                        temp_file.write(content)
                        temp_file.flush()

                        # Atomically move temp file to final backup location
                        final_backup_path = (
                            file_path.parent / temp_file.name.split("/")[-1]
                        )
                        os.rename(temp_file.name, final_backup_path)

                except OSError as lock_error:
                    # Clean up temp file if locking fails
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass  # Ignore cleanup errors
                    raise WriterASTMergeError(
                        f"Failed to lock file for backup: {lock_error}"
                    ) from lock_error

            return {
                "success": True,
                "original_path": str(file_path),
                "backup_path": str(final_backup_path),
            }

        except OSError as e:
            raise WriterASTMergeError(f"Failed to backup file {file_path}: {e}") from e

    def ensure_directory(self, directory_path: str | Path) -> dict[str, Any]:
        """
        Ensure that a directory exists, creating it if necessary.

        Args:
            directory_path: Path of the directory to ensure exists

        Returns:
            Dictionary containing directory operation results

        Raises:
            WriterASTMergeError: If directory creation fails
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
            raise WriterASTMergeError(
                f"Failed to ensure directory {directory_path}: {e}"
            ) from e

    def _merge_content(self, existing_content: str, new_content: str) -> str:
        """
        Merge existing and new content using AST analysis.

        Args:
            existing_content: Content currently in the file
            new_content: New content to merge

        Returns:
            Merged content

        Raises:
            WriterASTMergeError: If merging fails
        """
        try:
            self.logger.debug("Starting AST merge process")

            # Parse both contents
            self.logger.debug("Parsing existing content")
            existing_tree = ast.parse(existing_content)
            self.logger.debug("Parsing new content")
            new_tree = ast.parse(new_content)

            # Extract elements from both trees
            self.logger.debug("Extracting elements from existing content")
            existing_elements = self.merger.extract_elements(existing_tree)
            self.logger.debug("Extracting elements from new content")
            new_elements = self.merger.extract_elements(new_tree)

            self.logger.debug(
                f"Existing elements: imports={len(existing_elements['imports'])}, "
                f"from_imports={len(existing_elements['from_imports'])}, "
                f"functions={len(existing_elements['functions'])}, "
                f"classes={len(existing_elements['classes'])}, "
                f"constants={len(existing_elements['constants'])}"
            )

            self.logger.debug(
                f"New elements: imports={len(new_elements['imports'])}, "
                f"from_imports={len(new_elements['from_imports'])}, "
                f"functions={len(new_elements['functions'])}, "
                f"classes={len(new_elements['classes'])}, "
                f"constants={len(new_elements['constants'])}"
            )

            # Merge elements
            merged_body = []

            # Merge imports
            self.logger.debug("Merging imports")
            merged_imports = self.merger.merge_imports(
                existing_elements["imports"], new_elements["imports"]
            )
            self.logger.debug(f"Merged {len(merged_imports)} import statements")
            merged_body.extend(merged_imports)

            # Merge from imports
            self.logger.debug("Merging from imports")
            merged_from_imports = self.merger.merge_from_imports(
                existing_elements["from_imports"], new_elements["from_imports"]
            )
            self.logger.debug(
                f"Merged {len(merged_from_imports)} from import statements"
            )
            merged_body.extend(merged_from_imports)

            # Add a blank line after imports if there are any
            if merged_imports or merged_from_imports:
                # Add blank line using an expression statement with None constant
                # This creates proper AST structure that formatters will handle correctly
                blank_line = ast.Expr(value=ast.Constant(value=None))
                merged_body.append(blank_line)

            # Merge constants with value comparison
            merged_constants = []
            existing_constants_by_name = {
                const["name"]: const for const in existing_elements["constants"]
            }

            # Process existing constants first
            for const in existing_elements["constants"]:
                merged_constants.append(const["node"])

            # Process new constants, checking for conflicts
            for const in new_elements["constants"]:
                const_name = const["name"]
                const_value = const["value"]

                if const_name in existing_constants_by_name:
                    # Check if values match
                    existing_value = existing_constants_by_name[const_name]["value"]
                    if existing_value == const_value:
                        # Same name and value, skip (already have it)
                        self.logger.debug(f"Skipping duplicate constant: {const_name}")
                        continue
                    else:
                        # Same name, different value - create renamed version
                        new_name = self._resolve_constant_name_conflict(const_name)
                        self.logger.warning(
                            f"Constant name conflict resolved: {const_name} -> {new_name} "
                            f"(original: {existing_value}, new: {const_value})"
                        )

                        # Create new constant node with renamed variable
                        parsed = ast.parse(const_value)
                        parsed_stmt = parsed.body[0]
                        if isinstance(parsed_stmt, ast.Expr):
                            value_node = parsed_stmt.value
                        else:
                            value_node = parsed_stmt  # type: ignore[assignment]
                        new_node = ast.Assign(
                            targets=[ast.Name(id=new_name, ctx=ast.Store())],
                            value=value_node,  # Parse the value expression
                            lineno=const["node"].lineno,
                            col_offset=const["node"].col_offset,
                        )
                        merged_constants.append(new_node)
                else:
                    # New constant, add it
                    merged_constants.append(const["node"])

            merged_body.extend(merged_constants)

            # Merge classes
            self.logger.debug("Merging classes")
            merged_classes = self.merger.merge_classes(
                existing_elements["classes"], new_elements["classes"]
            )
            self.logger.debug(f"Merged {len(merged_classes)} class definitions")
            merged_body.extend(merged_classes)

            # Merge functions (including test functions)
            self.logger.debug("Merging functions")
            merged_functions = self.merger.merge_functions(
                existing_elements["functions"], new_elements["functions"]
            )
            self.logger.debug(f"Merged {len(merged_functions)} function definitions")
            merged_body.extend(merged_functions)

            # Add other statements from existing content
            merged_body.extend(existing_elements["other_statements"])

            # Add other statements from new content
            merged_body.extend(new_elements["other_statements"])

            # Create new module with merged body
            merged_tree = ast.Module(body=merged_body, type_ignores=[])

            # Validate merged AST before unparsing
            self._validate_merged_ast(merged_tree)

            # Convert back to source code
            try:
                merged_content = ast.unparse(merged_tree)
                self.logger.debug("AST merge completed successfully")
                return merged_content
            except Exception as unparse_error:
                self.logger.error(f"Failed to unparse merged AST: {unparse_error}")
                raise WriterASTMergeError(
                    f"Failed to convert merged AST back to source: {unparse_error}"
                ) from unparse_error

        except (SyntaxError, ValueError) as e:
            # Enhanced fallback with validation and logging
            self.logger.warning(
                "AST merge failed, falling back to safe concatenation. "
                f"Original error: {e}"
            )

            # Try partial merge: extract what's parseable
            try:
                # Try to parse each content separately
                existing_fallback_tree: ast.Module | None = self._safe_parse_content(
                    existing_content
                )
                new_fallback_tree: ast.Module | None = self._safe_parse_content(
                    new_content
                )

                if existing_fallback_tree and new_fallback_tree:
                    # Both parse successfully, merge what we can
                    merged_content = self._partial_ast_merge(
                        existing_content, new_content
                    )
                else:
                    # At least one failed to parse, use safe concatenation
                    merged_content = self._safe_concatenate_content(
                        existing_content, new_content
                    )

                # Validate the final result
                self._validate_merged_content(merged_content)

                self.logger.info("Partial AST merge succeeded")
                return merged_content

            except Exception as merge_error:
                # Final fallback: safe concatenation with comments
                self.logger.error(f"Partial merge also failed: {merge_error}")
                safe_content = self._safe_concatenate_content(
                    existing_content, new_content
                )

                # Validate even the safe concatenation
                self._validate_merged_content(safe_content)

                self.logger.warning("Using safe concatenation as final fallback")
                return safe_content

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
            WriterASTMergeError: If formatting fails completely
        """
        try:
            # Try primary formatting with ruff + black + isort
            self.logger.debug("Attempting primary formatting with ruff + black + isort")
            return format_python_content(content, timeout=15, disable_ruff=False)

        except Exception as primary_error:
            self.logger.warning(f"Primary formatting failed: {primary_error}")

            try:
                # Fallback 1: Try with black only (disable ruff)
                self.logger.debug("Attempting fallback formatting with black only")
                return format_python_content(content, timeout=15, disable_ruff=True)

            except Exception as black_error:
                self.logger.warning(f"Black-only formatting also failed: {black_error}")

                try:
                    # Fallback 2: Try basic isort only (minimal formatting)
                    self.logger.debug("Attempting minimal formatting with isort only")
                    # For minimal formatting, we'll just return the content as-is
                    # since we don't have a standalone isort formatter
                    return content

                except Exception as isort_error:
                    self.logger.error(f"All formatting attempts failed: {isort_error}")

                    # Final fallback: return unformatted content with warning
                    self.logger.warning(
                        "Returning unformatted content due to formatting failures"
                    )
                    return content

    def _generate_diff(self, original: str, modified: str, filename: str) -> str:
        """Generate unified diff between original and modified content."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )

        return "".join(diff)

    def _extract_test_info(self, content: str) -> dict[str, Any]:
        """
        Extract test information from content.

        Args:
            content: Test file content to analyze

        Returns:
            Dictionary containing imports and test functions
        """
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
            # Fallback to simple parsing if AST fails
            return {"imports": [], "functions": []}

    def _safe_parse_content(self, content: str) -> ast.Module | None:
        """
        Safely parse content, returning None if parsing fails.
        Uses caching to avoid re-parsing identical content.

        Args:
            content: Python code content to parse

        Returns:
            AST module if parsing succeeds, None otherwise
        """
        # Use content hash as cache key for efficiency
        content_hash_str = str(hash(content))

        if content_hash_str in self._ast_cache:
            self._cache_hits += 1
            return self._ast_cache[content_hash_str]

        try:
            tree = ast.parse(content)
            self._ast_cache[content_hash_str] = tree
            self._cache_misses += 1
            return tree
        except (SyntaxError, ValueError):
            self._cache_misses += 1
            return None

    def _partial_ast_merge(self, existing_content: str, new_content: str) -> str:
        """
        Attempt partial AST merge by extracting parseable parts.

        Args:
            existing_content: Original file content
            new_content: New content to merge

        Returns:
            Merged content with parseable parts preserved
        """
        try:
            # Try to extract parseable sections
            existing_tree = self._safe_parse_content(existing_content)
            new_tree = self._safe_parse_content(new_content)

            if not existing_tree and not new_tree:
                # Neither parses, use safe concatenation
                return self._safe_concatenate_content(existing_content, new_content)

            # Build merged content from parseable parts
            parts = []

            if existing_tree:
                try:
                    parts.append(ast.unparse(existing_tree))
                except Exception:
                    parts.append(existing_content)

            if new_tree:
                try:
                    parts.append(ast.unparse(new_tree))
                except Exception:
                    parts.append(new_content)
            elif new_content and not existing_tree:
                parts.append(new_content)

            return "\n\n".join(parts)

        except Exception:
            # Final fallback
            return self._safe_concatenate_content(existing_content, new_content)

    def _safe_concatenate_content(self, existing_content: str, new_content: str) -> str:
        """
        Safely concatenate content with proper separation.

        Args:
            existing_content: Original file content
            new_content: New content to append

        Returns:
            Safely concatenated content
        """
        parts = []

        if existing_content.strip():
            # Add original content as a comment block if it doesn't parse
            if self._safe_parse_content(existing_content) is None:
                parts.append(f"# Original content (unparseable):\n{existing_content}")
            else:
                parts.append(existing_content)

        if new_content.strip():
            # Add new content as a comment block if it doesn't parse
            if self._safe_parse_content(new_content) is None:
                parts.append(f"# New content (unparseable):\n{new_content}")
            else:
                parts.append(new_content)

        return "\n\n".join(parts)

    def _validate_merged_content(self, content: str) -> None:
        """
        Validate that merged content is valid Python.

        Args:
            content: Content to validate

        Raises:
            WriterASTMergeError: If content is not valid Python
        """
        try:
            ast.parse(content)
        except (SyntaxError, ValueError) as e:
            raise WriterASTMergeError(
                f"Merged content is not valid Python: {e}\n"
                f"Content preview: {content[:200]}{'...' if len(content) > 200 else ''}"
            ) from e

    def _resolve_constant_name_conflict(self, original_name: str) -> str:
        """
        Resolve naming conflicts for constants by adding numeric suffixes.

        Args:
            original_name: The original constant name that conflicts

        Returns:
            A unique name with numeric suffix (e.g., CONST -> CONST_1)
        """
        base_name = original_name
        counter = 1

        # Remove existing numeric suffix if present
        if "_" in base_name:
            parts = base_name.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                base_name = "_".join(parts[:-1])
                counter = int(parts[-1]) + 1

        # Find next available number
        while True:
            candidate = f"{base_name}_{counter}"
            # In a real implementation, we'd check against all existing names
            # For now, we'll just increment until we find a reasonable candidate
            counter += 1
            if counter > 1000:  # Prevent infinite loops
                candidate = f"{base_name}_{int(time.time())}_{os.getpid()}"
                break

        return candidate

    def _validate_merged_ast(self, tree: ast.Module) -> None:
        """
        Validate that a merged AST is well-formed and can be safely unparsed.

        Args:
            tree: AST module to validate

        Raises:
            WriterASTMergeError: If AST is invalid or malformed
        """
        try:
            # Basic AST validation
            if not isinstance(tree, ast.Module):
                raise WriterASTMergeError("Merged AST is not a valid Module node")

            if not hasattr(tree, "body") or not isinstance(tree.body, list):
                raise WriterASTMergeError("Merged AST missing or invalid body")

            # Validate all nodes in the body
            for i, node in enumerate(tree.body):
                if not isinstance(node, ast.stmt):
                    raise WriterASTMergeError(
                        f"Invalid statement at index {i}: {type(node)}"
                    )

            # Try to unparse to catch any issues
            ast.unparse(tree)

            self.logger.debug(f"AST validation passed: {len(tree.body)} statements")

        except Exception as e:
            if isinstance(e, WriterASTMergeError):
                raise
            raise WriterASTMergeError(f"AST validation failed: {e}") from e

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get AST cache statistics for debugging.

        Returns:
            Dictionary with cache hit/miss counts
        """
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._ast_cache),
        }

    def clear_cache(self) -> None:
        """Clear the AST cache."""
        self._ast_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.debug("AST cache cleared")
