"""
Enhanced context builder for test generation.

Builds comprehensive enriched context including:
- Packaging and import information
- Entity interface manifests
- Test safety rules
- Side-effect boundaries
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from .packaging_detector import (
    EntityInterfaceDetector,
    PackagingDetector,
    PackagingInfo,
)

logger = logging.getLogger(__name__)


class EnrichedContextBuilder:
    """
    Builder for enriched context with packaging awareness and test safety rules.

    Combines packaging detection, entity analysis, and safety rules to provide
    comprehensive context for test generation.
    """

    def __init__(self) -> None:
        self._packaging_cache: dict[str, PackagingInfo] = {}

    def build_enriched_context(
        self,
        source_file: Path,
        project_root: Path | None = None,
        existing_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Build enriched context for a source file.

        Args:
            source_file: Path to the source file being tested
            project_root: Project root (auto-detected if None)
            existing_context: Existing context to enhance

        Returns:
            Dictionary with enriched context information
        """
        try:
            # Auto-detect project root if not provided
            if project_root is None:
                project_root = self._find_project_root(source_file)

            # Get or detect packaging information
            packaging_info = self._get_packaging_info(project_root)

            # Detect entities in the source file
            entity_info = EntityInterfaceDetector.detect_entities(source_file)

            # Build canonical import information
            import_info = self._build_import_info(source_file, packaging_info)

            # Build test safety rules
            safety_rules = self._build_safety_rules(packaging_info, entity_info)

            # Build side-effect boundaries
            boundaries = self._build_boundaries_info(source_file, entity_info)

            # Assemble enriched context
            enriched_context = {
                "packaging": {
                    "project_root": str(project_root),
                    "source_roots": [str(sr) for sr in packaging_info.source_roots],
                    "src_is_package": packaging_info.src_is_package,
                    "disallowed_import_prefixes": packaging_info.disallowed_import_prefixes,
                },
                "imports": import_info,
                "entities": entity_info["entities"],
                "boundaries_to_mock": boundaries,
                "test_safety_rules": safety_rules,
                "existing_context": existing_context,
            }

            logger.debug(
                "Built enriched context for %s: packaging=%s, entities=%d, safety_rules=%d",
                source_file,
                packaging_info.src_is_package,
                len(entity_info["entities"]),
                len(safety_rules),
            )

            return enriched_context

        except Exception as e:
            logger.warning(
                "Failed to build enriched context for %s: %s", source_file, e
            )
            return {
                "packaging": {},
                "imports": {},
                "entities": {},
                "boundaries_to_mock": {},
                "test_safety_rules": [],
                "existing_context": existing_context,
            }

    def format_for_llm(self, enriched_context: dict[str, Any]) -> str:
        """
        Format enriched context for LLM consumption.

        Args:
            enriched_context: Dictionary from build_enriched_context

        Returns:
            Formatted string for inclusion in LLM prompt
        """
        try:
            sections = []

            # Packaging and imports section
            packaging = enriched_context.get("packaging", {})
            imports = enriched_context.get("imports", {})

            if packaging or imports:
                pkg_lines = ["# Module Import Information"]

                if imports.get("canonical_import"):
                    pkg_lines.append(f"import_statement: {imports['canonical_import']}")

                if imports.get("module_path"):
                    pkg_lines.append(f"module_path: {imports['module_path']}")

                if imports.get("validation_status"):
                    pkg_lines.append(f"import_status: {imports['validation_status']}")

                if packaging.get("disallowed_import_prefixes"):
                    prefixes = packaging["disallowed_import_prefixes"]
                    pkg_lines.append(f"disallowed_prefixes: {prefixes}")

                if len(pkg_lines) > 1:
                    sections.append("\n".join(pkg_lines))

            # Entities section
            entities = enriched_context.get("entities", {})
            if entities:
                entity_lines = ["# Entity Interface Manifest"]

                for name, info in entities.items():
                    entity_desc = [f"{name}:"]
                    entity_desc.append(f"  kind: {info.get('kind', 'regular_class')}")
                    entity_desc.append(
                        f"  instantiate_real: {info.get('instantiate_real', True)}"
                    )

                    attributes = info.get("attributes_read_by_uut", [])
                    if attributes:
                        entity_desc.append(
                            f"  attributes: {attributes[:10]}"
                        )  # Limit for brevity

                    entity_lines.extend(entity_desc)

                if len(entity_lines) > 1:
                    sections.append("\n".join(entity_lines))

            # Boundaries section
            boundaries = enriched_context.get("boundaries_to_mock", {})
            if boundaries:
                boundary_lines = ["# Side-Effect Boundaries to Mock"]

                for category, items in boundaries.items():
                    if items:
                        boundary_lines.append(f"{category}: {items}")

                if len(boundary_lines) > 1:
                    sections.append("\n".join(boundary_lines))

            # Safety rules section
            safety_rules = enriched_context.get("test_safety_rules", [])
            if safety_rules:
                safety_lines = ["# Test Safety Rules"]
                for i, rule in enumerate(safety_rules[:8], 1):  # Limit to 8 rules
                    safety_lines.append(f"{i}. {rule}")

                sections.append("\n".join(safety_lines))

            # Combine with existing context
            existing = enriched_context.get("existing_context")
            if existing:
                sections.insert(0, existing)

            return "\n\n".join(sections) if sections else ""

        except Exception as e:
            logger.warning("Failed to format enriched context: %s", e)
            return enriched_context.get("existing_context", "")

    def _get_packaging_info(self, project_root: Path) -> PackagingInfo:
        """Get packaging information, using cache if available."""
        try:
            cache_key = str(project_root.resolve())
        except Exception:
            # Fallback to string representation if resolve fails
            cache_key = str(project_root)

        if cache_key not in self._packaging_cache:
            self._packaging_cache[cache_key] = PackagingDetector.detect_packaging(
                project_root
            )

        return self._packaging_cache[cache_key]

    def _find_project_root(self, file_path: Path) -> Path:
        """Find project root by looking for common markers."""
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

    def _safe_read_file(self, file_path: Path) -> str:
        """Read file with multiple encoding fallbacks."""
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug("Failed to read %s with %s: %s", file_path, encoding, e)
                break

        # If all encodings fail, raise the last exception
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "All encodings failed")

    def _build_import_info(
        self, source_file: Path, packaging_info: PackagingInfo
    ) -> dict[str, Any]:
        """Build canonical import information for the source file."""
        import_info = {}

        try:
            # Get canonical import path
            canonical_import = packaging_info.get_canonical_import(source_file)

            if canonical_import:
                import_info["module_path"] = canonical_import

                # Build import statement
                if source_file.name == "__init__.py":
                    import_info["canonical_import"] = f"import {canonical_import}"
                else:
                    # Try to detect main classes/functions for better import suggestions
                    try:
                        content = self._safe_read_file(source_file)

                        # Parse AST with specific error handling
                        try:
                            tree = ast.parse(content)
                        except SyntaxError as e:
                            logger.debug(
                                "Syntax error in %s (line %d): %s - using module import fallback",
                                source_file,
                                e.lineno,
                                e.msg,
                            )
                            # Fallback to module import for syntax errors
                            import_info["canonical_import"] = (
                                f"import {canonical_import}"
                            )
                            import_info["validation_status"] = "syntax_error"
                            import_info["syntax_error"] = f"Line {e.lineno}: {e.msg}"
                        except Exception as e:
                            logger.debug(
                                "AST parsing error in %s: %s - using module import fallback",
                                source_file,
                                e,
                            )
                            # Fallback to module import for other AST errors
                            import_info["canonical_import"] = (
                                f"import {canonical_import}"
                            )
                            import_info["validation_status"] = "parse_error"
                            import_info["parse_error"] = str(e)

                        classes = []
                        functions = []

                        for node in tree.body:
                            if isinstance(node, ast.ClassDef):
                                classes.append(node.name)
                            elif isinstance(
                                node, ast.FunctionDef
                            ) and not node.name.startswith("_"):
                                functions.append(node.name)

                        if classes:
                            # Prefer class imports
                            main_class = classes[0]  # Use first class as primary
                            import_info["canonical_import"] = (
                                f"from {canonical_import} import {main_class}"
                            )
                            import_info["suggested_imports"] = [
                                f"from {canonical_import} import {cls}"
                                for cls in classes[:3]
                            ]
                        elif functions:
                            # Function imports
                            main_func = functions[0]
                            import_info["canonical_import"] = (
                                f"from {canonical_import} import {main_func}"
                            )
                            import_info["suggested_imports"] = [
                                f"from {canonical_import} import {func}"
                                for func in functions[:3]
                            ]
                        else:
                            # Module import as fallback
                            import_info["canonical_import"] = (
                                f"import {canonical_import}"
                            )

                    except UnicodeDecodeError:
                        # File reading failed - fallback to module import
                        import_info["canonical_import"] = f"import {canonical_import}"
                        import_info["validation_status"] = "encoding_error"
                        import_info["encoding_error"] = (
                            "Could not decode file with available encodings"
                        )
                    except Exception as e:
                        # Other errors - fallback to module import
                        logger.debug(
                            "Error analyzing %s for import suggestions: %s",
                            source_file,
                            e,
                        )
                        import_info["canonical_import"] = f"import {canonical_import}"
                        import_info["validation_status"] = "analysis_error"
                        import_info["analysis_error"] = str(e)

                # Validate import
                if packaging_info.is_import_allowed(canonical_import):
                    import_info["validation_status"] = "validated"
                else:
                    import_info["validation_status"] = "disallowed"
            else:
                import_info["validation_status"] = "failed"
                import_info["error"] = "Could not determine canonical import path"

        except Exception as e:
            import_info["validation_status"] = "error"
            import_info["error"] = str(e)

        return import_info

    def _build_safety_rules(
        self, packaging_info: PackagingInfo, entity_info: dict[str, Any]
    ) -> list[str]:
        """Build test safety rules based on detected patterns."""
        rules = []

        # Import safety rules
        if packaging_info.disallowed_import_prefixes:
            prefixes_str = ", ".join(
                f"'{p}'" for p in packaging_info.disallowed_import_prefixes
            )
            rules.append(f"Never use import prefixes: {prefixes_str}")

        # Entity safety rules
        entities = entity_info.get("entities", {})
        orm_models = [
            name
            for name, info in entities.items()
            if not info.get("instantiate_real", True)
        ]

        if orm_models:
            rules.append(
                f"Do not instantiate ORM models in @pytest.mark.parametrize: {orm_models[:3]}"
            )
            rules.append(
                "Use duck-typed stubs for ORM models when only attribute access is needed"
            )

        # General safety rules
        rules.extend(
            [
                "Never create domain objects inside @pytest.mark.parametrize decorators",
                "Use flags or sentinels in parametrization, construct objects in test body after mocking",
                "Break infinite loops deterministically by patching time.sleep or toggling state",
                "Mock all external dependencies: database, HTTP, filesystem, time operations",
            ]
        )

        # Packaging-specific rules
        if not packaging_info.src_is_package:
            rules.append(
                "Use the provided canonical import statement; 'src' is a source root, not a package"
            )

        return rules

    def _build_boundaries_info(
        self, source_file: Path, entity_info: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Build information about side-effect boundaries that need mocking using AST analysis."""
        boundaries = {
            "database": [],
            "http": [],
            "filesystem": [],
            "time": [],
            "external_processes": [],
        }

        try:
            content = self._safe_read_file(source_file)
            tree = ast.parse(content)

            # Database boundaries - look for ORM patterns and database operations
            db_patterns = {
                "session",
                "query",
                "commit",
                "rollback",
                "execute",
                "get_db",
                "database",
                "Session",
                "sessionmaker",
                "transaction",
                "save",
                "delete",
                "update",
                "insert",
                "select",
                "connection",
                "cursor",
            }

            # HTTP boundaries - look for HTTP clients and requests
            http_patterns = {
                "requests",
                "httpx",
                "urllib",
                "http",
                "fetch",
                "APIClient",
                "HttpClient",
                "RestClient",
                "client",
                "session",
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "head",
                "options",
            }

            # Filesystem boundaries - look for file operations
            fs_patterns = {
                "open",
                "read_text",
                "write_text",
                "Path",
                "os.path",
                "shutil",
                "tempfile",
                "glob",
                "pathlib",
                "exists",
                "is_file",
                "is_dir",
                "mkdir",
                "makedirs",
                "remove",
                "unlink",
                "rename",
                "copy",
                "move",
            }

            # Time boundaries - look for time and scheduling operations
            time_patterns = {
                "time.sleep",
                "datetime.now",
                "time.time",
                "schedule",
                "asyncio.sleep",
                "datetime",
                "time",
                "sleep",
                "delay",
                "wait",
                "timeout",
            }

            # External processes - look for subprocess operations
            process_patterns = {
                "subprocess",
                "os.system",
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
                "os.exec",
                "os.spawn",
                "os.fork",
                "multiprocessing",
            }

            for node in ast.walk(tree):
                # Check function calls
                if isinstance(node, ast.Call):
                    # Check function name
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        self._categorize_boundary(
                            boundaries,
                            func_name,
                            db_patterns,
                            http_patterns,
                            fs_patterns,
                            time_patterns,
                            process_patterns,
                        )
                    elif isinstance(node.func, ast.Attribute):
                        # Handle module.function() calls
                        self._categorize_boundary_attr(
                            boundaries,
                            node.func,
                            db_patterns,
                            http_patterns,
                            fs_patterns,
                            time_patterns,
                            process_patterns,
                        )

                # Check attribute access
                elif isinstance(node, ast.Attribute):
                    self._categorize_boundary_attr(
                        boundaries,
                        node,
                        db_patterns,
                        http_patterns,
                        fs_patterns,
                        time_patterns,
                        process_patterns,
                    )

                # Check imports
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    self._categorize_import_boundary(
                        boundaries,
                        node,
                        db_patterns,
                        http_patterns,
                        fs_patterns,
                        time_patterns,
                        process_patterns,
                    )

            # Remove duplicates and limit items
            for category in boundaries:
                boundaries[category] = list(set(boundaries[category]))[:5]

        except Exception as e:
            logger.debug("Failed to analyze boundaries for %s: %s", source_file, e)

        return boundaries

    def _categorize_boundary(
        self,
        boundaries: dict,
        name: str,
        db_patterns: set,
        http_patterns: set,
        fs_patterns: set,
        time_patterns: set,
        process_patterns: set,
    ) -> None:
        """Categorize a function name into appropriate boundary category."""
        if name in db_patterns:
            boundaries["database"].append(name)
        elif name in http_patterns:
            boundaries["http"].append(name)
        elif name in fs_patterns:
            boundaries["filesystem"].append(name)
        elif name in time_patterns:
            boundaries["time"].append(name)
        elif name in process_patterns:
            boundaries["external_processes"].append(name)

    def _categorize_boundary_attr(
        self,
        boundaries: dict,
        node: ast.Attribute,
        db_patterns: set,
        http_patterns: set,
        fs_patterns: set,
        time_patterns: set,
        process_patterns: set,
    ) -> None:
        """Categorize attribute access patterns."""
        # Handle cases like module.function or object.method
        if isinstance(node.value, ast.Name):
            module_attr = f"{node.value.id}.{node.attr}"
            if module_attr in time_patterns:
                boundaries["time"].append(module_attr)
            elif module_attr in process_patterns:
                boundaries["external_processes"].append(module_attr)
            elif module_attr in fs_patterns:
                boundaries["filesystem"].append(module_attr)
            elif node.value.id in {"requests", "httpx", "urllib"}:
                boundaries["http"].append(f"{node.value.id}.{node.attr}")
            elif node.value.id in {"os", "shutil", "tempfile", "glob", "pathlib"}:
                boundaries["filesystem"].append(f"{node.value.id}.{node.attr}")
            elif node.value.id == "subprocess":
                boundaries["external_processes"].append(f"subprocess.{node.attr}")
        elif isinstance(node.value, ast.Attribute):
            # Handle nested attributes like db.session.commit
            if isinstance(node.value.value, ast.Name):
                nested_attr = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
                if "session" in nested_attr or "query" in nested_attr:
                    boundaries["database"].append(nested_attr)

    def _categorize_import_boundary(
        self,
        boundaries: dict,
        node: ast.expr,
        db_patterns: set,
        http_patterns: set,
        fs_patterns: set,
        time_patterns: set,
        process_patterns: set,
    ) -> None:
        """Categorize import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]  # Get root module
                if module_name in db_patterns:
                    boundaries["database"].append(module_name)
                elif module_name in http_patterns:
                    boundaries["http"].append(module_name)
                elif module_name in fs_patterns:
                    boundaries["filesystem"].append(module_name)
                elif module_name in time_patterns:
                    boundaries["time"].append(module_name)
                elif module_name in process_patterns:
                    boundaries["external_processes"].append(module_name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]  # Get root module
                if module_name in db_patterns:
                    boundaries["database"].append(module_name)
                elif module_name in http_patterns:
                    boundaries["http"].append(module_name)
                elif module_name in fs_patterns:
                    boundaries["filesystem"].append(module_name)
                elif module_name in time_patterns:
                    boundaries["time"].append(module_name)
                elif module_name in process_patterns:
                    boundaries["external_processes"].append(module_name)
