"""
ContextPack builder service for repository-aware test generation.

This service builds a complete ContextPack according to the context assembly
specification, composing components from ImportResolver, EnrichedContextBuilder,
and other existing services.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None  # type: ignore[assignment]
from typing import Any

from ....adapters.io.file_discovery import FileDiscoveryError, FileDiscoveryService
from ....domain.models import (
    Budget,
    ContextPack,
    Conventions,
    DependencyMetadata,
    Focal,
    ImportMap,
    PropertyContext,
    ResolvedDef,
    Target,
    TestElement,
    TestElementType,
    TestGenerationPlan,
)
from ....ports.parser_port import ParserPort
from .context_assembler import ContextAssembler
from .enhanced_context_builder import EnrichedContextBuilder
from .import_resolver import ImportResolver
from .structure import ModulePathDeriver

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
        context_assembler: ContextAssembler | None = None,
        file_discovery_service: FileDiscoveryService | None = None,
    ) -> None:
        """
        Initialize the ContextPack builder.

        Args:
            import_resolver: Service for resolving canonical imports and bootstrap
            enriched_context_builder: Service for enriched context with safety rules
            parser: Parser service for extracting focal code information
            context_assembler: Service for context assembly and AST analysis
        """
        self._import_resolver = import_resolver or ImportResolver()
        self._enriched_context_builder = (
            enriched_context_builder or EnrichedContextBuilder()
        )
        self._parser = parser
        # Avoid instantiating Protocol types directly in test environments
        effective_parser = parser
        try:
            # If no parser provided, defer creation to caller to avoid Protocol instantiation
            if effective_parser is None:
                effective_parser = None
        except Exception:
            effective_parser = None

        self._context_assembler = context_assembler or ContextAssembler(
            context_port=None,  # Will be injected by caller if needed
            parser_port=effective_parser,  # Do not instantiate ParserPort() here
            config={},  # Will be injected by caller if needed
            import_resolver=self._import_resolver,
        )
        self._file_discovery = file_discovery_service or FileDiscoveryService()
        self._cache: dict[str, Any] = {}
        self._dependency_metadata_cache: DependencyMetadata | None = None

    def _validate_target_file(
        self, target_file: Path, project_root: Path | None
    ) -> Path:
        """Ensure the target file is eligible per discovery configuration."""

        try:
            resolved_target = target_file.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"Target file {target_file} does not exist") from exc

        resolved_root: Path | None = None
        if project_root is not None:
            try:
                resolved_root = project_root.resolve()
            except OSError as exc:
                raise ValueError(
                    f"Unable to resolve project root {project_root}: {exc}"
                ) from exc

            try:
                resolved_target.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"Target file {resolved_target} is outside the project root {resolved_root}"
                ) from exc

        try:
            allowed = self._file_discovery.filter_existing_files(
                [resolved_target], resolved_root
            )
        except FileDiscoveryError as exc:
            raise ValueError(
                f"Failed to validate target file {resolved_target}: {exc}"
            ) from exc

        if not allowed:
            raise ValueError(
                f"Target file {resolved_target} is excluded by discovery configuration"
            )

        return Path(allowed[0])

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

            # Resolve canonical module details for richer target metadata
            resolved_target_file = self._validate_target_file(target_file, project_root)
            project_root, module_path = self._resolve_project_metadata(
                resolved_target_file, project_root
            )

            relative_path: str | None = None
            if project_root:
                try:
                    relative_path = str(resolved_target_file.relative_to(project_root))
                except ValueError:
                    relative_path = str(resolved_target_file)
            else:
                relative_path = str(resolved_target_file)

            is_method = "." in target_object
            class_name = target_object.split(".")[0] if is_method else None
            method_name = target_object.split(".")[1] if is_method else None

            target = Target(
                module_file=str(resolved_target_file),
                object=target_object,
                module_path=module_path,
                relative_path=relative_path,
                class_name=class_name,
                method_name=method_name,
                is_method=is_method,
                exists_in_source=resolved_target_file.exists(),
            )

            # Build import_map component using ImportResolver with enhanced fallback
            import_map = None
            try:
                import_map = self._import_resolver.resolve(resolved_target_file)
            except ValueError as e:
                # Enhanced fallback using context_assembler's import analysis
                logger.warning(
                    "Import resolution failed for %s: %s. Using enhanced fallback.",
                    target_file,
                    e,
                )

                try:
                    # Try to get project root using context_assembler's method
                    project_root = self._context_assembler._find_project_root(
                        resolved_target_file
                    )

                    # Use context_assembler's import resolver as fallback
                    fallback_import_map = (
                        self._context_assembler._import_resolver.resolve(
                            resolved_target_file
                        )
                    )

                    # If context_assembler's resolver also fails, create enhanced fallback
                    if fallback_import_map is None:
                        raise ValueError(
                            "Context assembler import resolution also failed"
                        )

                    import_map = fallback_import_map

                except Exception as fallback_error:
                    logger.debug(
                        "Context assembler fallback also failed: %s", fallback_error
                    )

                    # Create enhanced fallback with better bootstrap logic
                    module_import = module_path

                    if not module_import:
                        try:
                            module_info = ModulePathDeriver.derive_module_path(
                                resolved_target_file, project_root
                            )
                            module_import = module_info.get("module_path")
                        except Exception:
                            module_import = None

                    if not module_import and project_root:
                        try:
                            relative_path = resolved_target_file.relative_to(
                                project_root
                            )
                            parts = list(relative_path.with_suffix("").parts)
                            if parts and parts[-1] == "__init__":
                                parts = parts[:-1]
                            module_import = ".".join(part for part in parts if part)
                        except Exception:
                            module_import = None

                    if not module_import:
                        if resolved_target_file.name == "__init__.py":
                            module_import = resolved_target_file.parent.name or ""
                        else:
                            module_import = target_file.stem

                    if module_import in ("", "__init__"):
                        parent_name = resolved_target_file.parent.name
                        module_import = parent_name or module_import

                    # Use context_assembler's project root detection for better sys.path setup
                    try:
                        project_root = self._context_assembler._find_project_root(
                            resolved_target_file
                        )
                        sys_path_root = str(project_root)
                    except Exception:
                        sys_path_root = str(resolved_target_file.parent.resolve())

                    import_map = ImportMap(
                        target_import=f"import {module_import} as _under_test",
                        sys_path_roots=[sys_path_root],
                        needs_bootstrap=True,
                        bootstrap_conftest=f"""import sys
import pathlib

# Enhanced fallback bootstrap using context_assembler patterns
p = pathlib.Path(r"{sys_path_root}").resolve()
if str(p) not in sys.path:
    sys.path.insert(0, str(p))

# Add current directory as fallback
current_dir = pathlib.Path(r"{resolved_target_file.parent.resolve()}").resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
""",
                    )

            # Build focal code component
            focal = self._build_focal_component(
                resolved_target_file, target_object, project_root
            )

            # Build resolved_defs component (placeholder for now)
            resolved_defs = self._build_resolved_defs_component(
                resolved_target_file, target_object
            )

            # Build property_context component (placeholder for now)
            property_context = self._build_property_context_component(
                resolved_target_file, target_object
            )

            # Use provided or default conventions/budget
            final_conventions = conventions or Conventions()
            final_budget = budget or Budget()

            dependency_metadata = self._collect_dependency_metadata(project_root)

            # Assemble the ContextPack
            context_pack = ContextPack(
                target=target,
                import_map=import_map,
                focal=focal,
                resolved_defs=resolved_defs,
                property_context=property_context,
                conventions=final_conventions,
                budget=final_budget,
                context="",  # Will be set later by context assembler if needed
                dependency_metadata=dependency_metadata,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "ContextPack payload:\n%s",
                    json.dumps(
                        context_pack.model_dump(),
                        indent=2,
                        sort_keys=True,
                    ),
                )

            logger.debug("Built ContextPack successfully for %s", target_object)
            return context_pack

        except ValueError as e:
            logger.error(
                "Invalid input for ContextPack building - %s in %s: %s",
                target_object,
                target_file,
                e,
            )
            raise
        except OSError as e:
            logger.error(
                "File system error building ContextPack for %s in %s: %s",
                target_object,
                target_file,
                e,
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error building ContextPack for %s in %s: %s",
                target_object,
                target_file,
                e,
            )
            # Don't re-raise unexpected errors, return None instead
            return None

    def _collect_dependency_metadata(
        self, project_root: Path | None
    ) -> DependencyMetadata:
        """Collect dependency metadata for prompt and resolver context."""

        if self._dependency_metadata_cache is not None:
            return self._dependency_metadata_cache

        external_modules: set[str] = set()
        distributions: dict[str, str] = {}

        if project_root is not None:
            try:
                self._augment_with_declared_dependencies(
                    project_root=project_root,
                    external_modules=external_modules,
                    distributions=distributions,
                )
            except Exception:
                logger.debug(
                    "Unable to augment dependency metadata from project files",
                    exc_info=True,
                )
        else:
            logger.debug("Project root not provided; skipping dependency metadata")

        dependency_metadata = DependencyMetadata(
            external_modules=sorted(external_modules),
            distributions=dict(sorted(distributions.items())),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Dependency metadata snapshot: external=%s distributions=%s",
                dependency_metadata.external_modules,
                dependency_metadata.distributions,
            )

        self._dependency_metadata_cache = dependency_metadata
        return dependency_metadata

    def _augment_with_declared_dependencies(
        self,
        project_root: Path | None,
        external_modules: set[str],
        distributions: dict[str, str],
    ) -> None:
        """Augment dependency metadata using project lockfiles/configuration."""

        if not project_root:
            return

        project_root = Path(project_root)

        def _add_name(name: str, version: str | None = None) -> None:
            if not name:
                return
            normalized = name.replace("-", "_")
            external_modules.add(normalized)
            external_modules.add(name)
            if version is not None:
                distributions[name] = version
            elif name not in distributions:
                distributions.setdefault(name, "")

        # pyproject dependencies
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                text = pyproject_path.read_text(encoding="utf-8")
                if tomllib is not None:
                    data = tomllib.loads(text)
                    project_section = data.get("project") or {}
                    deps = project_section.get("dependencies") or []
                    opt_deps = project_section.get("optional-dependencies") or {}
                    dep_groups = data.get("dependency-groups") or {}

                    def _process_dependency(dep: str) -> None:
                        if not dep:
                            return
                        name = dep
                        if Requirement is not None:
                            try:
                                requirement = Requirement(dep)
                                name = requirement.name
                                spec = str(requirement.specifier) or None
                            except Exception:
                                name = dep.split()[0]
                                spec = None
                        else:
                            name = dep.split()[0]
                            spec = None
                        _add_name(name, spec)

                    for dep in deps:
                        if isinstance(dep, str):
                            _process_dependency(dep)

                    if isinstance(opt_deps, dict):
                        for dep_list in opt_deps.values():
                            if isinstance(dep_list, list):
                                for dep in dep_list:
                                    if isinstance(dep, str):
                                        _process_dependency(dep)

                    if isinstance(dep_groups, dict):
                        for group in dep_groups.values():
                            if isinstance(group, list):
                                for dep in group:
                                    if isinstance(dep, str):
                                        _process_dependency(dep)
            except Exception:
                logger.debug(
                    "Failed to parse pyproject.toml for dependency metadata",
                    exc_info=True,
                )

        requirements_txt = project_root / "requirements.txt"
        if requirements_txt.exists():
            try:
                text = requirements_txt.read_text(encoding="utf-8")
                for line in text.splitlines():
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("#")
                        or line.startswith(("-", "--"))
                        or line.lower().startswith("git+")
                    ):
                        continue
                    name = line
                    if Requirement is not None:
                        try:
                            requirement = Requirement(line)
                            name = requirement.name
                            spec = str(requirement.specifier) or None
                        except Exception:
                            name = line.split()[0]
                            spec = None
                    else:
                        name = line.split()[0]
                        spec = None
                    _add_name(name, spec)
            except Exception:
                logger.debug(
                    "Failed to parse requirements.txt for dependency metadata",
                    exc_info=True,
                )

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

    def _build_focal_component(
        self, file_path: Path, target_object: str, project_root: Path | None = None
    ) -> Focal:
        """
        Build the focal code component with source, signature, and docstring.

        Args:
            file_path: Path to the source file
            target_object: Target object identifier

        Returns:
            Focal component with source code information
        """
        try:
            # Use context_assembler to extract focal information
            resolved_project_root, module_path = self._resolve_project_metadata(
                file_path, project_root
            )

            element = TestElement(
                name=target_object,
                type=self._determine_element_type(target_object),
                line_range=(1, 1),  # Minimal valid range; context assembler refines it
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                existing_tests=[],
                coverage_before=None,
                file_path=file_path,
                project_root=resolved_project_root or project_root,
                module_path=module_path,
            )

            # Use context_assembler's focal building logic
            focal = self._context_assembler._build_focal_object(
                source_path=file_path,
                plan=plan,
                target=Target(
                    module_file=str(file_path),
                    object=target_object,
                ),
            )

            if focal:
                return focal

            # Fallback: return basic focal information
            return Focal(
                source=f"# Target: {target_object}",
                signature=f"# Target: {target_object}",
                docstring=None,
                is_placeholder=True,
                placeholder_reason="context_pack_basic_fallback",
            )

        except (ValueError, TypeError) as e:
            logger.warning("Invalid input for focal component %s: %s", target_object, e)
            # Fallback to basic information
            try:
                content = file_path.read_text(encoding="utf-8")
                return Focal(
                    source=content[:2000],  # Limit size as fallback
                    signature=f"# Target: {target_object}",
                    docstring=None,
                    is_placeholder=True,
                    placeholder_reason="context_pack_file_read_fallback",
                )
            except OSError:
                # File read error fallback
                return Focal(
                    source=f"# Could not read {file_path}",
                    signature=f"# Target: {target_object}",
                    docstring=None,
                    is_placeholder=True,
                    placeholder_reason="context_pack_file_read_error",
                )
            except Exception as e:
                logger.debug("Unexpected error in focal fallback: %s", e)
                # Final fallback
                return Focal(
                    source=f"# Error reading {file_path}",
                    signature=f"# Target: {target_object}",
                    docstring=None,
                    is_placeholder=True,
                    placeholder_reason="context_pack_unexpected_error",
                )
        except OSError as e:
            logger.warning(
                "File error building focal component for %s: %s", target_object, e
            )
            # File system error fallback
            return Focal(
                source=f"# File system error: {file_path}",
                signature=f"# Target: {target_object}",
                docstring=None,
                is_placeholder=True,
                placeholder_reason="context_pack_filesystem_error",
            )
        except Exception as e:
            logger.warning(
                "Unexpected error building focal component for %s: %s", target_object, e
            )
            # Unexpected error fallback
            return Focal(
                source=f"# Error processing {file_path}",
                signature=f"# Target: {target_object}",
                docstring=None,
                is_placeholder=True,
                placeholder_reason="context_pack_processing_error",
            )

    def _build_resolved_defs_component(
        self, file_path: Path, target_object: str
    ) -> list[ResolvedDef]:
        """
        Build resolved_defs component for on-demand symbol definitions.

        Delegates to context_assembler for robust symbol extraction and resolution.
        """
        try:
            # Use context_assembler's resolved definitions logic
            from ....domain.models import TestElement, TestGenerationPlan

            # Create a minimal plan for symbol extraction
            resolved_project_root, module_path = self._resolve_project_metadata(
                file_path, None
            )

            element = TestElement(
                name=target_object,
                type=self._determine_element_type(target_object),
                line_range=(1, 1),
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                existing_tests=[],
                coverage_before=None,
                file_path=file_path,
                project_root=resolved_project_root,
                module_path=module_path,
            )

            resolved_def_dicts = self._context_assembler._populate_resolved_definitions(
                source_path=file_path,
                plan=plan,
                context_data={},
            )

            # Transform context_assembler results to ResolvedDef objects
            resolved_defs = []
            for def_dict in resolved_def_dicts:
                try:
                    kind = def_dict.get("kind") or def_dict.get("type", "unknown")
                    if kind == "function":
                        kind = "func"
                    resolved_defs.append(
                        ResolvedDef(
                            name=def_dict.get("name", "unknown"),
                            kind=kind,
                            signature=def_dict.get("signature", ""),
                            doc=def_dict.get("doc") or def_dict.get("docstring"),
                            body=def_dict.get("body")
                            or def_dict.get("source", "omitted"),
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to transform resolved def: %s", e)
                    continue

            return resolved_defs[:10]  # Limit to avoid bloat

        except Exception as e:
            logger.debug("Failed to build resolved_defs for %s: %s", file_path, e)
            # Return empty list as fallback
            return []

    def _build_property_context_component(
        self, file_path: Path, target_object: str
    ) -> PropertyContext:
        """
        Build property_context component with ranked methods and G/W/T snippets.

        Delegates to context_assembler for comprehensive property analysis.
        """
        try:
            # Create a minimal plan for property context analysis
            resolved_project_root, module_path = self._resolve_project_metadata(
                file_path, None
            )

            element = TestElement(
                name=target_object,
                type=self._determine_element_type(target_object),
                line_range=(1, 1),
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                existing_tests=[],
                coverage_before=None,
                file_path=file_path,
                project_root=resolved_project_root,
                module_path=module_path,
            )

            # Use context_assembler's property context building
            property_context = self._context_assembler._build_property_context(
                plan=plan,
                context_data={},
            )

            return property_context

        except Exception as e:
            logger.debug(
                "Failed to build property context for %s: %s", target_object, e
            )
            # Return empty context as fallback
            return PropertyContext()

    def build_enriched_context(
        self,
        source_file: Path,
        project_root: Path | None = None,
        existing_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Build enriched context using context_assembler's enhanced context building.

        Delegates to context_assembler for comprehensive context enrichment
        including contracts, dependencies, fixtures, and side-effects detection.
        """
        try:
            # Use context_assembler's enhanced context building
            enriched_context = (
                self._context_assembler._build_enriched_context_for_generation(
                    source_path=source_file,
                    base_context=existing_context,
                    import_map=None,  # Will be resolved by context assembler if needed
                )
            )

            # Convert string result to dict format expected by callers
            if isinstance(enriched_context, str):
                return {"context": enriched_context}
            elif isinstance(enriched_context, dict):
                return enriched_context
            else:
                # Fallback to EnrichedContextBuilder if context_assembler fails
                return self._enriched_context_builder.build_enriched_context(
                    source_file=source_file,
                    project_root=project_root,
                    existing_context=existing_context,
                )

        except Exception as e:
            logger.debug(
                "Failed to build enriched context via context_assembler: %s", e
            )
            # Fallback to original EnrichedContextBuilder
            return self._enriched_context_builder.build_enriched_context(
                source_file=source_file,
                project_root=project_root,
                existing_context=existing_context,
            )

    def _resolve_project_metadata(
        self, file_path: Path, project_root: Path | None
    ) -> tuple[Path | None, str | None]:
        """
        Resolve project root and module path metadata for a source file.

        Args:
            file_path: Path to the Python file
            project_root: Optional pre-resolved project root

        Returns:
            Tuple of (project_root, module_path)
        """
        resolved_root = project_root
        if resolved_root is None:
            try:
                resolved_root = self._find_project_root(file_path)
            except Exception as root_error:
                logger.debug(
                    "Project root detection failed for %s: %s", file_path, root_error
                )
                resolved_root = None

        module_path: str | None = None
        try:
            module_info = ModulePathDeriver.derive_module_path(file_path, resolved_root)
            module_path = module_info.get("module_path")
        except Exception as module_error:
            logger.debug(
                "Module path derivation failed when resolving metadata for %s: %s",
                file_path,
                module_error,
            )

        return resolved_root, module_path

    @staticmethod
    def _determine_element_type(target_object: str) -> TestElementType:
        """Best-effort inference of element type based on target identifier."""
        normalized = (target_object or "").strip()
        if not normalized or normalized.lower() in {"module", "__module__"}:
            return TestElementType.MODULE
        if "." in target_object:
            return TestElementType.METHOD
        if target_object and target_object[0].isupper():
            return TestElementType.CLASS
        return TestElementType.FUNCTION
