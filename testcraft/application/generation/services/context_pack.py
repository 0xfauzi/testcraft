"""
ContextPack builder service for repository-aware test generation.

This service builds a complete ContextPack according to the context assembly
specification, composing components from ImportResolver, EnrichedContextBuilder,
and other existing services.
"""

from __future__ import annotations

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
from .context_assembler import ContextAssembler
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
        context_assembler: ContextAssembler | None = None,
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
        self._context_assembler = context_assembler or ContextAssembler(
            context_port=None,  # Will be injected by caller if needed
            parser_port=parser or ParserPort(),  # Use provided parser or default
            config={},  # Will be injected by caller if needed
            import_resolver=self._import_resolver,
        )
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

            # Build import_map component using ImportResolver with enhanced fallback
            import_map = None
            try:
                import_map = self._import_resolver.resolve(target_file)
            except ValueError as e:
                # Enhanced fallback using context_assembler's import analysis
                logger.warning(
                    "Import resolution failed for %s: %s. Using enhanced fallback.",
                    target_file,
                    e,
                )

                # Use context_assembler's import analysis for better fallback
                try:
                    # Try to get project root using context_assembler's method
                    project_root = self._context_assembler._find_project_root(
                        target_file
                    )

                    # Use context_assembler's import resolver as fallback
                    fallback_import_map = (
                        self._context_assembler._import_resolver.resolve(target_file)
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
                    module_name = target_file.stem

                    # Use context_assembler's project root detection for better sys.path setup
                    try:
                        project_root = self._context_assembler._find_project_root(
                            target_file
                        )
                        sys_path_root = str(project_root)
                    except Exception:
                        sys_path_root = str(target_file.parent.resolve())

                    import_map = ImportMap(
                        target_import=f"import {module_name} as _under_test",
                        sys_path_roots=[sys_path_root],
                        needs_bootstrap=True,
                        bootstrap_conftest=f"""import sys
import pathlib

# Enhanced fallback bootstrap using context_assembler patterns
p = pathlib.Path(r"{sys_path_root}").resolve()
if str(p) not in sys.path:
    sys.path.insert(0, str(p))

# Add current directory as fallback
current_dir = pathlib.Path(r"{target_file.parent.resolve()}").resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
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
                context="",  # Will be set later by context assembler if needed
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
            # Use context_assembler to extract focal information
            # Create a minimal plan for the context assembler
            from ....domain.models import TestElement, TestGenerationPlan

            element = TestElement(
                name=target_object,
                type="function",  # Default type, will be determined by context assembler
                line_range=(0, 0),  # Will be determined by context assembler
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                test_file_path=str(file_path),
                source_file_path=str(file_path),
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
                )
            except OSError:
                # File read error fallback
                return Focal(
                    source=f"# Could not read {file_path}",
                    signature=f"# Target: {target_object}",
                    docstring=None,
                )
            except Exception as e:
                logger.debug("Unexpected error in focal fallback: %s", e)
                # Final fallback
                return Focal(
                    source=f"# Error reading {file_path}",
                    signature=f"# Target: {target_object}",
                    docstring=None,
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
            )

    def _build_resolved_defs_component(self, file_path: Path) -> list[ResolvedDef]:
        """
        Build resolved_defs component for on-demand symbol definitions.

        Delegates to context_assembler for robust symbol extraction and resolution.
        """
        try:
            # Use context_assembler's resolved definitions logic
            from ....domain.models import TestElement, TestGenerationPlan

            # Create a minimal plan for symbol extraction
            element = TestElement(
                name="module_symbols",  # Generic name for symbol extraction
                type="module",
                line_range=(0, 0),
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                test_file_path=str(file_path),
                source_file_path=str(file_path),
            )

            # Use context_assembler's resolved definitions population
            resolved_def_dicts = self._context_assembler._populate_resolved_definitions(
                source_path=file_path,
                plan=plan,
                context_data={},
            )

            # Transform context_assembler results to ResolvedDef objects
            resolved_defs = []
            for def_dict in resolved_def_dicts:
                try:
                    resolved_defs.append(
                        ResolvedDef(
                            name=def_dict.get("name", "unknown"),
                            kind=def_dict.get("type", "unknown"),
                            signature=def_dict.get("signature", ""),
                            doc=def_dict.get("doc"),
                            body=def_dict.get("source", "omitted"),
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
            # Use context_assembler's property context building logic
            from ....domain.models import TestElement, TestGenerationPlan

            # Create a minimal plan for property context analysis
            element = TestElement(
                name=target_object,
                type="function",  # Will be refined by context assembler
                line_range=(0, 0),
            )

            plan = TestGenerationPlan(
                elements_to_test=[element],
                test_file_path=str(file_path),
                source_file_path=str(file_path),
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
