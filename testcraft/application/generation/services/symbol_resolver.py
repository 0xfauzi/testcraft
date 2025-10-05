"""
Symbol resolver service for missing_symbols resolution loop.

This service implements the missing_symbols resolution loop as specified in the
context assembly specification. It uses ParserPort to fetch precise definitions
on demand when the LLM reports missing symbols in PLAN or REFINE stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....domain.models import ResolvedDef, TestElementType
from ....ports.parser_port import ParserPort
from .import_resolver import ImportResolver

logger = logging.getLogger(__name__)


class SymbolResolver:
    """
    Service for resolving missing symbols using ParserPort.

    Takes missing_symbols from LLM responses and uses ParserPort to extract
    precise definitions (signatures, docstrings, minimal bodies) for on-demand
    context injection.
    """

    def __init__(
        self,
        parser_port: ParserPort,
        import_resolver: ImportResolver | None = None,
    ) -> None:
        """
        Initialize the symbol resolver.

        Args:
            parser_port: Parser port for extracting symbol definitions
            import_resolver: Service for resolving import paths
        """
        self._parser = parser_port
        self._import_resolver = import_resolver or ImportResolver()
        self._cache: dict[str, ResolvedDef | None] = {}

    def resolve_symbols(
        self, missing_symbols: list[str], project_root: Path | None = None
    ) -> list[ResolvedDef]:
        """
        Resolve a list of missing symbols.

        Args:
            missing_symbols: List of symbol names to resolve
            project_root: Project root directory for path resolution

        Returns:
            List of resolved symbol definitions
        """
        resolved_defs = []

        for symbol in missing_symbols:
            resolved = self.resolve_single_symbol(symbol, project_root)
            if resolved is not None:
                resolved_defs.append(resolved)

        logger.info(
            "Resolved %d out of %d symbols", len(resolved_defs), len(missing_symbols)
        )
        return resolved_defs

    def resolve_single_symbol(
        self, symbol: str, project_root: Path | None = None
    ) -> ResolvedDef | None:
        """
        Resolve a single missing symbol.

        Args:
            symbol: Symbol name to resolve (e.g., "module.Class", "module.function")
            project_root: Project root directory for path resolution

        Returns:
            ResolvedDef if found, None otherwise
        """
        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        try:
            # Parse symbol name to get module, class, and name components
            module_name, class_name, symbol_name = self._parse_symbol_name(symbol)

            # Find the module file
            module_path = self._find_module_file(module_name, project_root)
            if module_path is None:
                logger.warning("Could not find module file for symbol: %s", symbol)
                self._cache[symbol] = None
                return None

            # Search for the symbol in the module file
            symbol_info = self._find_symbol_in_file(
                module_path, module_name, class_name, symbol_name
            )
            if symbol_info is None:
                logger.warning(
                    "Could not find symbol %s in module %s", symbol, module_path
                )
                self._cache[symbol] = None
                return None

            # Create ResolvedDef
            resolved_def = ResolvedDef(
                name=symbol,
                kind=symbol_info["kind"],
                signature=symbol_info["signature"],
                doc=symbol_info["docstring"],
                body=symbol_info["body"],
            )

            self._cache[symbol] = resolved_def
            return resolved_def

        except Exception as e:
            logger.exception("Error resolving symbol %s: %s", symbol, e)
            self._cache[symbol] = None
            return None

    def _parse_symbol_name(self, symbol: str) -> tuple[str, str | None, str]:
        """
        Parse a symbol name into module, class, and name components.

        Args:
            symbol: Symbol name (e.g., "module.Class.method", "module.function")

        Returns:
            Tuple of (module_name, class_name, symbol_name)
        """
        # Handle fully qualified names (module.Class.method)
        if "." in symbol:
            parts = symbol.split(".")
            if len(parts) == 2:
                # module.function
                return parts[0], None, parts[1]
            elif len(parts) == 3:
                # module.Class.method
                return parts[0], parts[1], parts[2]
            else:
                # More than 3 parts - treat last part as name, middle parts as class
                return ".".join(parts[:-2]), parts[-2], parts[-1]
        else:
            # Just a name - assume it's in the current module
            return "", None, symbol

    def _find_module_file(
        self, module_name: str, project_root: Path | None = None
    ) -> Path | None:
        """
        Find the file for a given module name.

        Args:
            module_name: Module name to find
            project_root: Project root directory

        Returns:
            Path to the module file if found, None otherwise
        """
        if not project_root:
            # Try to infer from current context or use import resolver
            # For now, assume we're in the project root
            project_root = Path.cwd()

        # Convert dotted module name to path (e.g., "testcraft.domain.models" -> "testcraft/domain/models")
        module_path_parts = module_name.replace(".", "/")

        # Try common module locations
        candidate_paths = [
            project_root / f"{module_path_parts}.py",
            project_root / f"{module_path_parts}/__init__.py",
        ]

        # Try src/ layout
        if (project_root / "src").exists():
            candidate_paths.extend(
                [
                    project_root / "src" / f"{module_path_parts}.py",
                    project_root / "src" / f"{module_path_parts}/__init__.py",
                ]
            )

        for path in candidate_paths:
            if path.exists():
                return path

        # Try to resolve using import resolver as a fallback
        try:
            # The import resolver can help locate the module file
            # Note: ImportResolver.resolve() expects a file path, so we need to
            # construct a potential path first
            potential_path = project_root / f"{module_path_parts}.py"
            import_map = self._import_resolver.resolve(potential_path)
            # Extract the actual module file path from sys_path_roots if available
            if import_map and import_map.get("sys_path_roots"):
                # Try to find the module in the resolved sys.path roots
                for sys_path_root in import_map["sys_path_roots"]:
                    root = Path(sys_path_root)
                    for candidate in [
                        root / f"{module_path_parts}.py",
                        root / f"{module_path_parts}/__init__.py",
                    ]:
                        if candidate.exists():
                            return candidate
        except Exception as e:
            logger.debug("Import resolver failed for %s: %s", module_name, e)

        return None

    def _find_symbol_in_file(
        self,
        module_path: Path,
        module_name: str,
        class_name: str | None,
        symbol_name: str,
    ) -> dict[str, Any] | None:
        """
        Find a symbol in a module file.

        Args:
            module_path: Path to the module file
            module_name: Module name for context
            class_name: Class name if symbol is a method
            symbol_name: Name of the symbol to find

        Returns:
            Dictionary with symbol information if found, None otherwise
        """
        try:
            # Parse the file using ParserPort
            parse_result = self._parser.parse_file(module_path)

            # Look for the symbol in the parsed elements
            for element in parse_result.get("elements", []):
                if self._matches_symbol(element, module_name, class_name, symbol_name):
                    return self._extract_symbol_info(element, parse_result)

            return None

        except Exception as e:
            logger.exception(
                "Error finding symbol %s in file %s: %s", symbol_name, module_path, e
            )
            return None

    def _matches_symbol(
        self, element: Any, module_name: str, class_name: str | None, symbol_name: str
    ) -> bool:
        """
        Check if an element matches the symbol we're looking for.

        Args:
            element: Parsed element from ParserPort
            module_name: Module name for context
            class_name: Class name if looking for a method
            symbol_name: Symbol name to match

        Returns:
            True if the element matches
        """
        # Get element name and type with defensive checks
        element_name = getattr(element, "name", "")
        element_type = getattr(element, "type", "")

        # Validate that element has required attributes
        if not element_name or not element_type:
            logger.debug("Element missing name or type attributes")
            return False

        # Check if this is the symbol we're looking for
        if class_name:
            # Looking for a method: element_name should be "Class.method"
            expected_name = f"{class_name}.{symbol_name}"

            # Try exact match first (handles cases where ParserPort includes class prefix)
            if element_name == expected_name and (
                element_type == TestElementType.METHOD or element_type == "method"
            ):
                return True

            # Fallback: check if element_name matches just the symbol_name
            # (handles cases where ParserPort doesn't include class prefix)
            if element_name == symbol_name and (
                element_type == TestElementType.METHOD or element_type == "method"
            ):
                return True

            return False
        else:
            # Looking for a function: element_name should be "function"
            return element_name == symbol_name and (
                element_type == TestElementType.FUNCTION or element_type == "function"
            )

    def _extract_symbol_info(
        self, element: Any, parse_result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract symbol information from a parsed element.

        Args:
            element: Parsed element
            parse_result: Full parse result from ParserPort

        Returns:
            Dictionary with symbol information
        """
        # Get element name and type
        element_name = getattr(element, "name", "")
        element_type = getattr(element, "type", "")

        # Determine the kind of symbol
        kind = "func"  # Default
        if element_type == TestElementType.CLASS or element_type == "class":
            kind = "class"
        elif element_type == TestElementType.METHOD or element_type == "method":
            kind = "func"  # Methods are functions in the domain model
        elif element_type == TestElementType.FUNCTION or element_type == "function":
            kind = "func"

        # Get signature and docstring
        signature = self._generate_signature(element)
        docstring = getattr(element, "docstring", None)

        # Use "omitted" as placeholder for body content to keep context size manageable
        body = "omitted"

        return {
            "name": element_name,
            "kind": kind,
            "signature": signature,
            "docstring": docstring,
            "body": body,
        }

    def _generate_signature(self, element: Any) -> str:
        """
        Generate a signature string for an element.

        Args:
            element: Parsed element

        Returns:
            String representation of the element's signature
        """
        # This is a simplified signature generation
        # In a real implementation, you'd use the AST to generate proper signatures
        element_name = getattr(element, "name", "")
        element_type = getattr(element, "type", "")

        if element_type == "class":
            return f"class {element_name}:"
        elif element_type == "function":
            return f"def {element_name}(...):"
        elif element_type == "method":
            # Extract method name from "Class.method" format
            method_name = (
                element_name.split(".")[-1] if "." in element_name else element_name
            )
            return f"def {method_name}(self, ...):"
        else:
            return f"{element_name}"

    def clear_cache(self) -> None:
        """Clear the symbol resolution cache."""
        self._cache.clear()
