"""
Symbol resolver service for missing_symbols resolution loop.

This service implements the missing_symbols resolution loop as specified in the
context assembly specification. It uses ParserPort to fetch precise definitions
on demand when the LLM reports missing symbols in PLAN or REFINE stages.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
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
        allow_runtime_imports: bool = False,
        runtime_timeout_sec: float | None = None,
        external_modules: set[str] | None = None,
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
        self._module_cache: dict[str, dict[str, Any] | None] = {}
        self._resolution_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        self._allow_runtime_imports = allow_runtime_imports
        self._runtime_timeout_sec = (
            float(runtime_timeout_sec)
            if runtime_timeout_sec and runtime_timeout_sec > 0
            else 10.0
        )
        self._external_modules: set[str] = set()
        self._external_modules_lower: set[str] = set()
        self._stdlib_modules: set[str] = set()
        self._external_symbol_cache: set[str] = set()

        if external_modules:
            self.update_environment_modules(external_modules=external_modules)

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

        if symbol in self._external_symbol_cache:
            return None

        try:
            # Parse symbol name to get module, class, and name components
            module_name, class_name, symbol_name = self._parse_symbol_name(symbol)
            failure_reasons: list[str] = []

            if not module_name:
                logger.debug(
                    "Symbol %s did not include module information; skipping", symbol
                )
                failure_reasons.append("missing module qualifier")
                self._cache[symbol] = None
                return None

            # Find the module file
            module_path = self._find_module_file(module_name, project_root)
            if module_path is None:
                if self._is_external_module(module_name):
                    logger.debug(
                        "Skipping resolution for external module %s (symbol %s)",
                        module_name,
                        symbol,
                    )
                    self._external_symbol_cache.add(symbol)
                    self._cache[symbol] = None
                    return None
                failure_reasons.append("module file not found")
                logger.warning("Could not find module file for symbol: %s", symbol)
                self._cache[symbol] = None
                return None

            # Search for the symbol in the module file
            symbol_info = self._find_symbol_in_file(
                module_path,
                module_name,
                class_name,
                symbol_name,
                project_root,
                failure_reasons,
                visited=set(),
            )
            if symbol_info is None and self._allow_runtime_imports:
                runtime_info = self._resolve_via_runtime_import(
                    module_name,
                    class_name,
                    symbol_name,
                    module_path,
                    failure_reasons,
                )
                if runtime_info is not None:
                    resolved_def = ResolvedDef(
                        name=symbol,
                        kind=runtime_info["kind"],
                        signature=runtime_info["signature"],
                        doc=runtime_info.get("docstring"),
                        body=runtime_info.get("body", "omitted"),
                    )
                    self._cache[symbol] = resolved_def
                    return resolved_def

            if symbol_info is None:
                if symbol in self._external_symbol_cache:
                    logger.debug(
                        "Skipping warning for external symbol %s (%s)",
                        symbol,
                        "; ".join(failure_reasons) if failure_reasons else "external",
                    )
                    self._cache[symbol] = None
                    return None

                has_external_reason = any(
                    "external module" in reason for reason in failure_reasons
                )
                if has_external_reason:
                    logger.debug(
                        "Skipping warning for external symbol %s (%s)",
                        symbol,
                        "; ".join(failure_reasons),
                    )
                    self._external_symbol_cache.add(symbol)
                else:
                    reason_text = (
                        "; ".join(failure_reasons)
                        if failure_reasons
                        else "unknown reason"
                    )
                    location = str(module_path)
                    logger.warning(
                        "Could not find symbol %s in module %s (%s)",
                        symbol,
                        location,
                        reason_text,
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

    def update_environment_modules(
        self,
        *,
        external_modules: set[str] | None = None,
        stdlib_modules: set[str] | None = None,
    ) -> None:
        """Update cached knowledge of external and stdlib modules."""

        if external_modules:
            for module in external_modules:
                if not module:
                    continue
                root = module.split(".")[0]
                self._external_modules.add(root)
                self._external_modules_lower.add(root.lower())

        if stdlib_modules:
            for module in stdlib_modules:
                if not module:
                    continue
                root = module.split(".")[0]
                self._stdlib_modules.add(root)
                self._external_modules_lower.add(root.lower())

    def _parse_symbol_name(self, symbol: str) -> tuple[str, str | None, str]:
        """
        Parse a symbol name into module, class, and name components.

        Args:
            symbol: Symbol name (e.g., "module.Class.method", "module.function")

        Returns:
            Tuple of (module_name, class_name, symbol_name)
        """
        # Handle fully qualified names (module.Class.method) with submodules
        if "." not in symbol:
            return "", None, symbol

        parts = symbol.split(".")
        symbol_name = parts[-1]

        if len(parts) == 2:
            # module.symbol
            module_name = parts[0]
            return module_name, None, symbol_name

        potential_class = parts[-2]
        if potential_class and potential_class[0].isupper():
            module_name = ".".join(parts[:-2]) or ""
            class_name = potential_class
            return module_name, class_name, symbol_name

        module_name = ".".join(parts[:-1])
        return module_name, None, symbol_name

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

        if self._is_external_module(module_name):
            logger.debug(
                "Treating %s as external/stdlib module; skipping filesystem lookup",
                module_name,
            )
            return None

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

    def _is_external_module(self, module_name: str) -> bool:
        """Check if a module is known to be external or part of the stdlib."""

        if not module_name:
            return False

        root = module_name.split(".")[0]
        root_lower = root.lower()

        if root in self._stdlib_modules:
            return True
        if root_lower in self._external_modules_lower:
            return True
        return False

    def _find_symbol_in_file(
        self,
        module_path: Path,
        module_name: str,
        class_name: str | None,
        symbol_name: str,
        project_root: Path | None,
        diagnostics: list[str],
        visited: set[tuple[str, str, str]],
    ) -> dict[str, Any] | None:
        """
        Find a symbol in a module file, following imports as needed.

        Args:
            module_path: Path to the module file
            module_name: Module name for context
            class_name: Class name if symbol is a method
            symbol_name: Name of the symbol to find
            project_root: Project root directory for resolution
            diagnostics: Collector for failure reasons
            visited: Cycle detector set for recursive lookups

        Returns:
            Dictionary with symbol information if found, None otherwise
        """

        cache_key = (module_name, class_name or "", symbol_name)

        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        if cache_key in visited:
            diagnostics.append(
                f"circular resolution detected for {module_name or '<module>'}.{class_name or symbol_name}"
            )
            return None

        visited.add(cache_key)
        try:
            parse_result = self._get_parse_result(module_path)
            if parse_result is None:
                diagnostics.append(f"failed to parse {module_path}")
                self._resolution_cache[cache_key] = None
                return None

            elements = parse_result.get("elements", []) or []
            for element in elements:
                if self._matches_symbol(element, module_name, class_name, symbol_name):
                    info = self._extract_symbol_info(element, parse_result)
                    self._resolution_cache[cache_key] = info
                    return info

            diagnostics.append(
                f"no direct definition for {symbol_name} in {module_name or module_path.name}"
            )

            imports_info = parse_result.get("imports") or []
            alias_map, star_imports = self._build_import_alias_map(
                module_name, imports_info
            )

            if self._should_skip_external_singleton(symbol_name, alias_map):
                diagnostics.append(
                    f"{symbol_name}: treated as external singleton (skipped)"
                )
                self._external_symbol_cache.add(f"{module_name}.{symbol_name}")
                self._resolution_cache[cache_key] = None
                return None

            if class_name:
                method_info = self._resolve_imported_method(
                    module_name,
                    class_name,
                    symbol_name,
                    alias_map,
                    star_imports,
                    project_root,
                    diagnostics,
                    visited,
                )
                if method_info is not None:
                    self._resolution_cache[cache_key] = method_info
                    return method_info

            symbol_info = self._resolve_imported_symbol(
                module_name,
                symbol_name,
                alias_map,
                star_imports,
                project_root,
                diagnostics,
                visited,
            )
            if symbol_info is not None:
                self._resolution_cache[cache_key] = symbol_info
                return symbol_info

            diagnostics.append(
                f"{symbol_name} not found after following imports from {module_name or module_path.name}"
            )
            self._resolution_cache[cache_key] = None
            return None
        finally:
            visited.discard(cache_key)

    def _get_parse_result(self, module_path: Path) -> dict[str, Any] | None:
        """Fetch and cache the parser output for a module."""
        cache_key = str(module_path)
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        try:
            result = self._parser.parse_file(module_path)
            self._module_cache[cache_key] = result
            return result
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("Parser failed for %s: %s", module_path, exc)
            self._module_cache[cache_key] = None
            return None

    def _build_import_alias_map(
        self, current_module: str, imports: list[dict[str, Any]]
    ) -> tuple[dict[str, tuple[str | None, str | None]], list[str]]:
        """Build a map of import aliases to their originating modules."""

        alias_map: dict[str, tuple[str | None, str | None]] = {}
        star_imports: list[str] = []

        for entry in imports:
            entry_type = entry.get("type")
            if entry_type == "import":
                module = entry.get("module")
                alias = entry.get("alias") or (
                    module.split(".")[0] if isinstance(module, str) and module else None
                )
                if alias and module:
                    alias_map.setdefault(alias, (module, None))
            elif entry_type == "from_import":
                name = entry.get("name")
                alias_name = entry.get("alias") or name
                module = entry.get("module") or ""
                level = entry.get("level", 0) or 0
                resolved_module = self._normalize_import_module(
                    current_module, module, level
                )

                if name == "*":
                    star_imports.append(resolved_module)
                    continue

                if alias_name:
                    alias_map.setdefault(alias_name, (resolved_module, name))

        return alias_map, star_imports

    def _should_skip_external_singleton(
        self,
        symbol_name: str,
        alias_map: dict[str, tuple[str | None, str | None]],
    ) -> bool:
        """Heuristic to skip common external singletons like logger/console."""

        lower_name = symbol_name.casefold()

        if lower_name == "logger":
            for alias, (module, _attr) in alias_map.items():
                root = (module or "").split(".")[0].casefold() if module else ""
                if alias.casefold() == "logging" or root == "logging":
                    return True

        if lower_name == "console":
            for alias, (module, attr) in alias_map.items():
                root = (module or "").split(".")[0].casefold() if module else ""
                if root in {"rich", "console"}:
                    if alias.casefold().startswith("console") or (
                        attr and attr.casefold().startswith("console")
                    ):
                        return True

        return False

    def _normalize_import_module(
        self, current_module: str, imported_module: str, level: int
    ) -> str:
        """Normalise relative import targets to absolute module names."""

        if level <= 0:
            return imported_module

        current_parts = current_module.split(".") if current_module else []
        # Guard against relative levels that walk past the package root
        cutoff = len(current_parts) - level
        if cutoff < 0:
            cutoff = 0
        base_parts = current_parts[:cutoff]

        if imported_module:
            base_parts.extend(part for part in imported_module.split(".") if part)

        return ".".join(base_parts)

    def _resolve_imported_method(
        self,
        module_name: str,
        class_name: str,
        symbol_name: str,
        alias_map: dict[str, tuple[str | None, str | None]],
        star_imports: list[str],
        project_root: Path | None,
        diagnostics: list[str],
        visited: set[tuple[str, str, str]],
    ) -> dict[str, Any] | None:
        """Attempt to resolve a method on a class imported from another module."""

        found_candidate = False
        target = alias_map.get(class_name)
        if target:
            found_candidate = True
            target_module, target_attr = target
            lookup_class = target_attr or class_name
            result = self._resolve_import_target(
                alias_label=class_name,
                target_module=target_module,
                target_attr=lookup_class,
                lookup_class_name=lookup_class,
                lookup_symbol_name=symbol_name,
                project_root=project_root,
                diagnostics=diagnostics,
                visited=visited,
            )
            if result is not None:
                return result

        for star_module in star_imports:
            found_candidate = True
            result = self._resolve_import_target(
                alias_label=f"* from {star_module or '<root>'}",
                target_module=star_module,
                target_attr=class_name,
                lookup_class_name=class_name,
                lookup_symbol_name=symbol_name,
                project_root=project_root,
                diagnostics=diagnostics,
                visited=visited,
            )
            if result is not None:
                return result

        if not found_candidate:
            diagnostics.append(
                f"no import alias for class {class_name} in {module_name or '<module>'}"
            )

        return None

    def _resolve_imported_symbol(
        self,
        module_name: str,
        symbol_name: str,
        alias_map: dict[str, tuple[str | None, str | None]],
        star_imports: list[str],
        project_root: Path | None,
        diagnostics: list[str],
        visited: set[tuple[str, str, str]],
    ) -> dict[str, Any] | None:
        """Attempt to resolve a symbol imported from another module."""

        found_candidate = False
        target = alias_map.get(symbol_name)
        if target:
            found_candidate = True
            target_module, target_attr = target
            lookup_symbol = target_attr or symbol_name
            result = self._resolve_import_target(
                alias_label=symbol_name,
                target_module=target_module,
                target_attr=target_attr,
                lookup_class_name=None,
                lookup_symbol_name=lookup_symbol,
                project_root=project_root,
                diagnostics=diagnostics,
                visited=visited,
            )
            if result is not None:
                return result

        for star_module in star_imports:
            found_candidate = True
            result = self._resolve_import_target(
                alias_label=f"* from {star_module or '<root>'}",
                target_module=star_module,
                target_attr=None,
                lookup_class_name=None,
                lookup_symbol_name=symbol_name,
                project_root=project_root,
                diagnostics=diagnostics,
                visited=visited,
            )
            if result is not None:
                return result

        if not found_candidate:
            diagnostics.append(
                f"no import alias for {symbol_name} in {module_name or '<module>'}"
            )

        return None

    def _resolve_import_target(
        self,
        *,
        alias_label: str,
        target_module: str | None,
        target_attr: str | None,
        lookup_class_name: str | None,
        lookup_symbol_name: str,
        project_root: Path | None,
        diagnostics: list[str],
        visited: set[tuple[str, str, str]],
    ) -> dict[str, Any] | None:
        """Resolve a concrete import target by delegating to _find_symbol_in_file."""

        if not target_module:
            diagnostics.append(f"{alias_label}: missing target module reference")
            return None

        module_path = self._find_module_file(target_module, project_root)
        if module_path is None:
            if self._is_external_module(target_module):
                logger.debug(
                    "Skipping external module %s while resolving %s",
                    target_module,
                    lookup_symbol_name,
                )
                self._external_symbol_cache.add(f"{target_module}.{lookup_symbol_name}")
                diagnostics.append(
                    f"{alias_label}: external module {target_module} (skipped)"
                )
                return None
            diagnostics.append(
                f"{alias_label}: module {target_module} not found on disk"
            )
            return None

        result = self._find_symbol_in_file(
            module_path,
            target_module,
            lookup_class_name,
            lookup_symbol_name,
            project_root,
            diagnostics,
            visited,
        )

        if result is None:
            diagnostics.append(
                f"{alias_label}: {lookup_symbol_name} not found in {target_module}"
            )

        return result

    def _resolve_via_runtime_import(
        self,
        module_name: str,
        class_name: str | None,
        symbol_name: str,
        module_path: Path,
        diagnostics: list[str],
    ) -> dict[str, Any] | None:
        """Attempt to resolve a symbol by importing the module at runtime."""

        if not module_name:
            diagnostics.append("runtime import skipped: empty module name")
            return None

        try:
            import_map = self._import_resolver.resolve(module_path)
            sys_path_roots = import_map.get("sys_path_roots", []) if import_map else []
        except Exception as exc:  # pragma: no cover - import resolution fallback
            diagnostics.append(f"runtime import path resolution failed: {exc}")
            logger.debug(
                "Runtime import path resolution failed for %s: %s", module_path, exc
            )
            return None

        runtime_script = dedent(
            """
            import importlib
            import inspect
            import json
            import sys
            import traceback

            sys.path[:0] = {sys_path_roots}

            module_name = {module_name}
            class_name = {class_name}
            symbol_name = {symbol_name}

            def build_signature(obj, kind):
                if kind == "class":
                    return f"class {{obj.__name__}}:"
                if kind == "func":
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(...)"
                    display_name = symbol_name if class_name else getattr(
                        obj, "__name__", symbol_name
                    )
                    return f"def {{display_name}}{{sig}}:"
                return f"{{symbol_name}} = <{{type(obj).__name__}}>"

            try:
                module = importlib.import_module(module_name)
                if class_name:
                    cls = getattr(module, class_name)
                    attr = getattr(cls, symbol_name)
                    kind = "func"
                else:
                    attr = getattr(module, symbol_name)
                    if inspect.isclass(attr):
                        kind = "class"
                    elif inspect.isfunction(attr) or inspect.ismethod(attr):
                        kind = "func"
                    else:
                        kind = "const"

                payload = {{
                    "status": "ok",
                    "info": {{
                        "kind": kind,
                        "signature": build_signature(attr, kind),
                        "docstring": inspect.getdoc(attr),
                        "body": "omitted",
                    }},
                }}
            except Exception:
                payload = {{
                    "status": "error",
                    "error": traceback.format_exc(),
                }}

            print(json.dumps(payload))
            """
        ).format(
            sys_path_roots=repr(sys_path_roots),
            module_name=repr(module_name),
            class_name=repr(class_name),
            symbol_name=repr(symbol_name),
        )

        try:
            process = subprocess.run(
                [sys.executable, "-c", runtime_script],
                capture_output=True,
                text=True,
                timeout=self._runtime_timeout_sec,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            diagnostics.append(f"runtime import execution error: {exc}")
            logger.debug("Runtime import execution error for %s: %s", module_name, exc)
            return None

        output_text = process.stdout.strip() or process.stderr.strip()
        if not output_text:
            diagnostics.append(
                f"runtime import produced no output (exit code {process.returncode})"
            )
            return None

        payload_text = output_text.splitlines()[-1]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - unexpected output
            diagnostics.append(f"runtime import returned non-JSON output: {exc}")
            logger.debug("Runtime import output for %s: %s", module_name, output_text)
            return None

        if payload.get("status") != "ok":
            error_msg = payload.get("error", "unknown error")
            diagnostics.append(f"runtime import error: {error_msg}")
            logger.debug("Runtime import error for %s: %s", module_name, error_msg)
            return None

        info = payload.get("info") or {}
        info.setdefault("body", "omitted")
        diagnostics.append("resolved via runtime import")
        return info

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

        # Normalise for case-insensitive comparisons
        element_name_cf = element_name.casefold()
        symbol_name_cf = symbol_name.casefold()

        # Check if this is the symbol we're looking for
        if class_name:
            # Looking for a method: element_name should be "Class.method"
            expected_name = f"{class_name}.{symbol_name}"
            expected_name_cf = expected_name.casefold()

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

            # Case-insensitive variants
            if element_name_cf == expected_name_cf and (
                element_type == TestElementType.METHOD or element_type == "method"
            ):
                return True

            if element_name_cf == symbol_name_cf and (
                element_type == TestElementType.METHOD or element_type == "method"
            ):
                return True

            return False
        else:
            # Looking for a function or class at module scope
            if element_name == symbol_name:
                if element_type in (TestElementType.CLASS, "class"):
                    return True
                if element_type in (TestElementType.FUNCTION, "function"):
                    return True
                if element_type in (TestElementType.MODULE, "module"):
                    return True
                return False

            if element_name_cf != symbol_name_cf:
                return False

            if element_type in (TestElementType.CLASS, "class"):
                return True
            if element_type in (TestElementType.FUNCTION, "function"):
                return True
            if element_type in (TestElementType.MODULE, "module"):
                return True
            return False

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
        self._module_cache.clear()
        self._resolution_cache.clear()
        self._external_symbol_cache.clear()
