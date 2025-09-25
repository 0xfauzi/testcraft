"""
ContextPack builder service for repository-aware test generation.

This service builds a complete ContextPack according to the context assembly
specification, composing components from ImportResolver, EnrichedContextBuilder,
and other existing services.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from ....domain.models import (
    Budget,
    ContextPack,
    Conventions,
    Focal,
    ImportMap,
    PropertyContext,
    ResolvedDef,
    Target,
)
from ....ports.parser_port import ParserPort
from .enhanced_context_builder import EnrichedContextBuilder
from .import_resolver import ImportResolver

logger = logging.getLogger(__name__)


class ContextPackBuilder:
    """
    Builder for complete ContextPack objects per the context assembly specification.

    Composes import_map (from ImportResolver), focal code parsing, resolved_defs
    for on-demand symbols, property_context with ranked methods and G/W/T snippets,
    and reuses EnrichedContextBuilder for contracts, deps, fixtures, side-effects.
    """

    def __init__(
        self,
        import_resolver: ImportResolver | None = None,
        enriched_context_builder: EnrichedContextBuilder | None = None,
        parser: ParserPort | None = None,
    ) -> None:
        """
        Initialize the ContextPack builder.

        Args:
            import_resolver: Service for resolving canonical imports and bootstrap
            enriched_context_builder: Service for enriched context with safety rules
            parser: Parser service for extracting focal code information
        """
        self._import_resolver = import_resolver or ImportResolver()
        self._enriched_context_builder = (
            enriched_context_builder or EnrichedContextBuilder()
        )
        self._parser = parser
        self._cache: dict[str, Any] = {}

    def build_context_pack(
        self,
        target_file: Path,
        target_object: str,
        project_root: Path | None = None,
        conventions: Conventions | None = None,
        budget: Budget | None = None,
    ) -> ContextPack:
        """
        Build a complete ContextPack for the target.

        Args:
            target_file: Path to the module file containing the target
            target_object: Target object (Class.method, function, etc.)
            project_root: Project root directory (auto-detected if None)
            conventions: Test conventions (uses defaults if None)
            budget: Token budget configuration (uses defaults if None)

        Returns:
            Complete ContextPack matching the specification schema

        Raises:
            ValueError: If target cannot be resolved or parsed
            Exception: If any component fails to build
        """
        try:
            logger.info("Building ContextPack for %s in %s", target_object, target_file)

            # Resolve project root if not provided
            if project_root is None:
                project_root = self._find_project_root(target_file)

            # Build target information
            target = Target(
                module_file=str(target_file.resolve()),
                object=target_object,
            )

            # Build import_map component using ImportResolver
            try:
                import_map_data = self._import_resolver.resolve(target_file)
                import_map = ImportMap(
                    target_import=import_map_data["target_import"],
                    sys_path_roots=import_map_data["sys_path_roots"],
                    needs_bootstrap=import_map_data["needs_bootstrap"],
                    bootstrap_conftest=import_map_data["bootstrap_conftest"],
                )
            except ValueError as e:
                # Fallback for files without proper package structure (e.g., test files)
                logger.warning(
                    "Import resolution failed for %s: %s. Using fallback import.",
                    target_file,
                    e,
                )
                module_name = target_file.stem
                import_map = ImportMap(
                    target_import=f"import {module_name} as _under_test",
                    sys_path_roots=[str(target_file.parent.resolve())],
                    needs_bootstrap=True,
                    bootstrap_conftest=f"""import sys
import pathlib

# Fallback bootstrap for standalone file
p = pathlib.Path(r"{target_file.parent.resolve()}").resolve()
if str(p) not in sys.path:
    sys.path.insert(0, str(p))
""",
                )

            # Build focal code component
            focal = self._build_focal_component(target_file, target_object)

            # Build resolved_defs component (placeholder for now)
            resolved_defs = self._build_resolved_defs_component(target_file)

            # Build property_context component (placeholder for now)
            property_context = self._build_property_context_component(
                target_file, target_object
            )

            # Use provided or default conventions/budget
            final_conventions = conventions or Conventions()
            final_budget = budget or Budget()

            # Assemble the ContextPack
            context_pack = ContextPack(
                target=target,
                import_map=import_map,
                focal=focal,
                resolved_defs=resolved_defs,
                property_context=property_context,
                conventions=final_conventions,
                budget=final_budget,
            )

            logger.debug("Built ContextPack successfully for %s", target_object)
            return context_pack

        except Exception as e:
            logger.error(
                "Failed to build ContextPack for %s in %s: %s",
                target_object,
                target_file,
                e,
            )
            raise

    def _find_project_root(self, file_path: Path) -> Path:
        """Find project root by looking for common project markers."""
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
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

        return file_path.parent if file_path.is_file() else file_path

    def _build_focal_component(self, file_path: Path, target_object: str) -> Focal:
        """
        Build the focal code component with source, signature, and docstring.

        Args:
            file_path: Path to the source file
            target_object: Target object identifier

        Returns:
            Focal component with source code information
        """
        try:
            # Read and parse the file
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Extract the target object
            if "." in target_object:
                # Handle class.method case
                class_name, method_name = target_object.split(".", 1)
                node_info = self._extract_class_method(tree, class_name, method_name)
            else:
                # Handle function case
                node_info = self._extract_function(tree, target_object)

            if not node_info:
                # Fallback: return the entire file as focal
                return Focal(
                    source=content,
                    signature=f"# Target: {target_object}",
                    docstring=None,
                )

            return Focal(
                source=node_info["source"],
                signature=node_info["signature"],
                docstring=node_info["docstring"],
            )

        except Exception as e:
            logger.warning("Failed to parse focal code for %s: %s", target_object, e)
            # Fallback to basic information
            try:
                content = file_path.read_text(encoding="utf-8")
                return Focal(
                    source=content[:2000],  # Limit size as fallback
                    signature=f"# Target: {target_object}",
                    docstring=None,
                )
            except Exception:
                # Final fallback
                return Focal(
                    source=f"# Could not read {file_path}",
                    signature=f"# Target: {target_object}",
                    docstring=None,
                )

    def _extract_function(self, tree: ast.AST, func_name: str) -> dict[str, Any] | None:
        """Extract function information from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                source = ast.get_source_segment(
                    # We need the original source for get_source_segment
                    "",  # This will not work without original source
                    node,
                ) or self._reconstruct_function_source(node)

                signature = self._build_signature(node)
                docstring = ast.get_docstring(node)

                return {
                    "source": source,
                    "signature": signature,
                    "docstring": docstring,
                }
        return None

    def _extract_class_method(
        self, tree: ast.AST, class_name: str, method_name: str
    ) -> dict[str, Any] | None:
        """Extract class method information from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        source = self._reconstruct_function_source(item)
                        signature = self._build_signature(item)
                        docstring = ast.get_docstring(item)

                        return {
                            "source": source,
                            "signature": signature,
                            "docstring": docstring,
                        }
        return None

    def _build_signature(self, func_node: ast.FunctionDef) -> str:
        """Build function signature from AST node."""
        try:
            args = []

            # Handle positional arguments
            for arg in func_node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        # Skip annotation if unparsing fails
                        pass
                args.append(arg_str)

            # Handle default arguments
            defaults = func_node.args.defaults
            if defaults:
                # Apply defaults to the last N arguments where N = len(defaults)
                num_defaults = len(defaults)
                for i, default in enumerate(defaults):
                    arg_index = len(args) - num_defaults + i
                    if 0 <= arg_index < len(args):
                        try:
                            default_str = ast.unparse(default)
                            args[arg_index] += f" = {default_str}"
                        except Exception:
                            # Skip default if unparsing fails
                            pass

            # Handle varargs
            if func_node.args.vararg:
                vararg_str = f"*{func_node.args.vararg.arg}"
                if func_node.args.vararg.annotation:
                    try:
                        vararg_str += (
                            f": {ast.unparse(func_node.args.vararg.annotation)}"
                        )
                    except Exception:
                        pass
                args.append(vararg_str)

            # Handle keyword arguments
            for kwarg in func_node.args.kwonlyargs:
                kwarg_str = kwarg.arg
                if kwarg.annotation:
                    try:
                        kwarg_str += f": {ast.unparse(kwarg.annotation)}"
                    except Exception:
                        pass
                args.append(kwarg_str)

            # Handle kwargs
            if func_node.args.kwarg:
                kwarg_str = f"**{func_node.args.kwarg.arg}"
                if func_node.args.kwarg.annotation:
                    try:
                        kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}"
                    except Exception:
                        pass
                args.append(kwarg_str)

            # Build return annotation
            return_annotation = ""
            if func_node.returns:
                try:
                    return_annotation = f" -> {ast.unparse(func_node.returns)}"
                except Exception:
                    pass

            return f"def {func_node.name}({', '.join(args)}){return_annotation}:"

        except Exception as e:
            logger.debug("Failed to build signature for %s: %s", func_node.name, e)
            # Fallback to simple signature
            return f"def {func_node.name}(...):"

    def _reconstruct_function_source(self, func_node: ast.FunctionDef) -> str:
        """Reconstruct function source from AST node."""
        try:
            # Use ast.unparse if available (Python 3.9+)
            return ast.unparse(func_node)
        except AttributeError:
            # Fallback for older Python versions
            lines = [self._build_signature(func_node)]
            if ast.get_docstring(func_node):
                lines.append(f'    """{ast.get_docstring(func_node)}"""')
            lines.append("    # Implementation details...")
            return "\n".join(lines)

    def _build_resolved_defs_component(self, file_path: Path) -> list[ResolvedDef]:
        """
        Build resolved_defs component for on-demand symbol definitions.

        TODO: This is a placeholder implementation. Full implementation should:
        1. Analyze the focal code for symbols that might be called
        2. Resolve their definitions using Jedi or similar
        3. Include minimal bodies only when essential
        4. Support on-demand resolution during planning/generation
        """
        # Placeholder implementation
        resolved_defs = []

        try:
            # Basic implementation: extract imports and classes from the file
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Extract some basic symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    resolved_defs.append(
                        ResolvedDef(
                            name=node.name,
                            kind="class",
                            signature=f"class {node.name}:",
                            doc=ast.get_docstring(node),
                            body="omitted",
                        )
                    )
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith(
                    "_"
                ):
                    resolved_defs.append(
                        ResolvedDef(
                            name=node.name,
                            kind="func",
                            signature=self._build_signature(node),
                            doc=ast.get_docstring(node),
                            body="omitted",
                        )
                    )

        except Exception as e:
            logger.debug("Failed to build resolved_defs for %s: %s", file_path, e)

        return resolved_defs[:10]  # Limit to avoid bloat

    def _build_property_context_component(
        self, file_path: Path, target_object: str
    ) -> PropertyContext:
        """
        Build property_context component with ranked methods and G/W/T snippets.

        TODO: This is a placeholder implementation. Full implementation should:
        1. Use PropertyAnalyzer (APT-style) to identify GIVEN/WHEN/THEN relationships
        2. Rank methods by intra-class complete > intra-class G/W/T > repo-level complete
        3. Extract minimal G/W/T snippets from related code
        4. Build test bundles from existing tests
        """
        # Placeholder implementation - returns empty context
        return PropertyContext()

    def build_enriched_context(
        self,
        source_file: Path,
        project_root: Path | None = None,
        existing_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Build enriched context using EnrichedContextBuilder.

        This method delegates to the existing EnrichedContextBuilder for
        contracts, dependencies, fixtures, and side-effects detection.
        """
        return self._enriched_context_builder.build_enriched_context(
            source_file=source_file,
            project_root=project_root,
            existing_context=existing_context,
        )
