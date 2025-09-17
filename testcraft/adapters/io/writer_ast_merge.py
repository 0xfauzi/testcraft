"""
Writer adapter that merges content using AST analysis.

This adapter provides sophisticated merging functionality for test files,
analyzing AST structures to avoid duplicates and merge content intelligently.
"""

import ast
import difflib
import logging
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
    def extract_elements(tree: ast.AST) -> dict[str, Any]:
        """
        Extract elements from an AST tree.

        Args:
            tree: AST tree to analyze

        Returns:
            Dictionary containing extracted elements
        """
        elements = {
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
                    # Update the import with merged names
                    node = existing_modules[module]["node"]
                    node.names = [
                        ast.alias(name=name, asname=asname)
                        for name, asname in sorted(all_names)
                    ]
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

    def __init__(self, project_root: Path | None = None, dry_run: bool = False):
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

            # Allow callers to disable Ruff formatting via kwargs
            disable_ruff = bool(kwargs.get("disable_ruff_format", False))
            # Format the merged content
            formatted_content = self._format_content(merged_content, disable_ruff=disable_ruff)

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
        Create a backup of an existing file.

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
            # Parse both contents
            existing_tree = ast.parse(existing_content)
            new_tree = ast.parse(new_content)

            # Extract elements from both trees
            existing_elements = self.merger.extract_elements(existing_tree)
            new_elements = self.merger.extract_elements(new_tree)

            # Merge elements
            merged_body = []

            # Merge imports
            merged_imports = self.merger.merge_imports(
                existing_elements["imports"], new_elements["imports"]
            )
            merged_body.extend(merged_imports)

            # Merge from imports
            merged_from_imports = self.merger.merge_from_imports(
                existing_elements["from_imports"], new_elements["from_imports"]
            )
            merged_body.extend(merged_from_imports)

            # Add a blank line after imports if there are any
            if merged_imports or merged_from_imports:
                # Add blank line (represented as a Pass statement that will be formatted away)
                pass

            # Merge constants
            merged_constants = []
            existing_constant_names = {
                const["name"] for const in existing_elements["constants"]
            }

            # Add existing constants
            for const in existing_elements["constants"]:
                merged_constants.append(const["node"])

            # Add new constants that don't exist
            for const in new_elements["constants"]:
                if const["name"] not in existing_constant_names:
                    merged_constants.append(const["node"])

            merged_body.extend(merged_constants)

            # Merge classes
            merged_classes = self.merger.merge_classes(
                existing_elements["classes"], new_elements["classes"]
            )
            merged_body.extend(merged_classes)

            # Merge functions (including test functions)
            merged_functions = self.merger.merge_functions(
                existing_elements["functions"], new_elements["functions"]
            )
            merged_body.extend(merged_functions)

            # Add other statements from existing content
            merged_body.extend(existing_elements["other_statements"])

            # Add other statements from new content
            merged_body.extend(new_elements["other_statements"])

            # Create new module with merged body
            merged_tree = ast.Module(body=merged_body, type_ignores=[])

            # Convert back to source code
            return ast.unparse(merged_tree)

        except (SyntaxError, ValueError):
            # Fallback to simple concatenation if AST parsing fails
            return existing_content + "\n\n" + new_content

    def _format_content(self, content: str, *, disable_ruff: bool = False) -> str:
        """
        Format Python content using Black and isort with robust process management.

        Uses the shared subprocess_safe module for safe execution with proper cleanup
        on timeout or interruption. Creates new process groups for better isolation.

        Args:
            content: Python code content to format

        Returns:
            Formatted content

        Raises:
            WriterASTMergeError: If formatting fails
        """
        return format_python_content(content, timeout=15, disable_ruff=disable_ruff)

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
