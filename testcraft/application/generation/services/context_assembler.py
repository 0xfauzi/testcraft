"""
Context assembler service for test generation and refinement.

Unifies context building logic for both generation and refinement workflows,
including project context gathering, snippet retrieval, and enrichment integration.
"""

from __future__ import annotations

import ast
import functools
import itertools
import logging
import os
import re
import signal
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ....domain.models import (
    Budget,
    ContextPack,
    Conventions,
    Focal,
    ImportMap,
    PropertyContext,
    Target,
    TestGenerationPlan,
)
from ....ports.context_port import ContextPort
from ....ports.parser_port import ParserPort
from .enhanced_context_builder import EnrichedContextBuilder
from .enrichment_detectors import EnrichmentDetectors
from .import_resolver import ImportResolver
from .structure import DirectoryTreeBuilder, ModulePathDeriver

logger = logging.getLogger(__name__)


# Configuration constants for security and performance
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_MAX_AST_NODES = 50000
DEFAULT_MAX_PROJECT_ROOT_ITERATIONS = 20
DEFAULT_MAX_CONTEXT_SECTIONS = 1000
DEFAULT_FILE_OPERATION_TIMEOUT = 5.0  # seconds
DEFAULT_METHOD_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3


def timeout_decorator(seconds: float):
    """Decorator to add timeout to functions using signal.alarm on Unix systems."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds"
                )

            # Only works on Unix systems
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel the alarm
                    return result
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # On Windows or systems without SIGALRM, just call the function
                return func(*args, **kwargs)

        return wrapper

    return decorator


def retry_with_backoff(max_retries: int = DEFAULT_MAX_RETRIES, base_delay: float = 1.0):
    """Decorator to retry operations with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except OSError as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.debug(
                            f"Retrying {func.__name__} in {delay}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            f"Failed {func.__name__} after {max_retries + 1} attempts"
                        )
            raise last_exception

        return wrapper

    return decorator


def safe_file_read(
    file_path: Path, max_size: int = DEFAULT_MAX_FILE_SIZE
) -> str | None:
    """Safely read a file with size validation and error handling."""
    try:
        # Check file size before reading
        if file_path.stat().st_size > max_size:
            logger.warning(
                f"File {file_path} is too large ({file_path.stat().st_size} bytes > {max_size} bytes)"
            )
            return None

        # Use timeout for file operation
        result = {"content": None, "error": None}

        @retry_with_backoff()
        def read_file():
            try:
                result["content"] = file_path.read_text(encoding="utf-8")
            except Exception as e:
                result["error"] = e

        timer = threading.Timer(DEFAULT_FILE_OPERATION_TIMEOUT, lambda: None)
        timer.start()
        read_file()
        timer.cancel()

        if result["error"]:
            raise result["error"]

        return result["content"]
    except Exception as e:
        logger.debug(f"Failed to read file {file_path}: {e}")
        return None


def validate_ast_content(content: str, max_nodes: int = DEFAULT_MAX_AST_NODES) -> bool:
    """Validate AST content for security and size constraints."""
    try:
        # Check for obviously malicious patterns
        dangerous_patterns = [
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"open\s*\(",
            r"os\.(system|popen|spawn)",
            r"subprocess\.",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(
                    f"Potentially dangerous pattern found in content: {pattern}"
                )
                return False

        # Parse AST and check node count with timeout protection
        @timeout_decorator(5.0)  # 5 second timeout for AST parsing
        def parse_and_count():
            tree = ast.parse(content)

            def count_nodes(node):
                count = 1
                for child in ast.iter_child_nodes(node):
                    count += count_nodes(child)
                return count

            node_count = count_nodes(tree)
            if node_count > max_nodes:
                logger.warning(f"AST node count {node_count} exceeds limit {max_nodes}")
                return False

            return True

        return parse_and_count()
    except TimeoutError as e:
        logger.warning(f"AST parsing timed out: {e}")
        return False
    except SyntaxError as e:
        logger.debug(f"AST validation failed due to syntax error: {e}")
        return False
    except Exception as e:
        logger.warning(f"AST validation failed: {e}")
        return False


def safe_ast_parse(content: str) -> ast.AST | None:
    """Safely parse AST content with timeout and validation."""
    try:
        if not validate_ast_content(content):
            return None

        @timeout_decorator(5.0)
        def parse_ast():
            return ast.parse(content)

        return parse_ast()
    except Exception as e:
        logger.debug(f"AST parsing failed: {e}")
        return None


class ContextAssembler:
    """
    Service for assembling context for test generation and refinement.

    Provides unified context gathering including project structure, snippet
    retrieval, enrichment detection, and context assembly for LLM consumption.
    """

    def __init__(
        self,
        context_port: ContextPort,
        parser_port: ParserPort,
        config: dict[str, Any],
        import_resolver: ImportResolver | None = None,
    ):
        """
        Initialize the context assembler.

        Args:
            context_port: Port for context operations
            parser_port: Port for code parsing
            config: Configuration with context settings and budgets
            import_resolver: Service for resolving canonical imports and bootstrap requirements
        """
        self._context = context_port
        self._parser = parser_port
        self._config = config

        # Reuse shared services
        self._enrichment = EnrichmentDetectors()
        self._structure_builder = DirectoryTreeBuilder()
        self._enhanced_context_builder = EnrichedContextBuilder()
        self._import_resolver = import_resolver or ImportResolver()

    def gather_project_context(
        self, project_path: Path, files_to_process: list[Path]
    ) -> dict[str, Any]:
        """
        Gather project context including directory tree and codebase information.

        Args:
            project_path: Root path of the project
            files_to_process: Files being processed

        Returns:
            Dictionary containing context information
        """
        # Build project context graph
        context_graph = None
        try:
            context_graph = self._context.build_context_graph(project_path)
        except Exception as e:
            logger.warning("Failed to build context graph: %s", e)

        # Index files for context retrieval
        indexed_files = {}
        for file_path in files_to_process:
            try:
                index_result = self._context.index(file_path)
                indexed_files[str(file_path)] = index_result
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)

        # Use recursive directory tree with configuration-based limits
        project_structure = None
        try:
            directory_config = self._config.get("context_budgets", {}).get(
                "directory_tree", {}
            )
            max_depth = directory_config.get("max_depth", 4)
            max_entries_per_dir = directory_config.get("max_entries_per_dir", 200)
            include_py_only = directory_config.get("include_py_only", True)

            project_structure = self._structure_builder.build_tree_recursive(
                project_path, max_depth, max_entries_per_dir, include_py_only
            )
        except Exception as e:
            logger.warning("Failed to gather project context: %s", e)

        # Return whatever context we were able to gather
        result = {}
        if context_graph is not None:
            result["context_graph"] = context_graph
        if indexed_files:
            result["indexed_files"] = indexed_files
        if project_structure is not None:
            result["project_structure"] = project_structure

        return result

    @timeout_decorator(DEFAULT_METHOD_TIMEOUT)
    def context_for_generation(
        self, plan: TestGenerationPlan, source_path: Path | None = None
    ) -> ContextPack | None:
        """
        Get relevant context for test generation.

        Implements snippet-based retrieval and merges import-graph neighbors
        discovered via ContextPort.get_related_context. Returns complete ContextPack
        objects instead of just context strings for full integration.

        Args:
            plan: The test generation plan
            source_path: Optional source file path for the plan

        Returns:
            Complete ContextPack object or None if no useful context
        """
        if not self._config.get("enable_context", True):
            return None

        if not plan.elements_to_test:
            logger.debug("No elements to test in plan, returning None")
            return None

        try:
            # Track progress and check resource usage
            start_time = time.time()
            last_checkpoint = start_time

            def check_progress_and_resources(step_name: str):
                """Check progress and resource usage at key points."""
                nonlocal last_checkpoint
                current_time = time.time()
                elapsed = current_time - start_time

                # Check if we're approaching timeout
                if elapsed > DEFAULT_METHOD_TIMEOUT - 5:  # 5 second buffer
                    logger.warning(
                        f"Approaching timeout ({elapsed:.1f}s) at step: {step_name}"
                    )
                    raise TimeoutError(f"Method timeout at step: {step_name}")

                # Check memory usage if psutil is available
                if HAS_PSUTIL:
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > 500:  # 500MB threshold
                            logger.warning(
                                f"High memory usage ({memory_mb:.1f}MB) at step: {step_name}"
                            )
                    except Exception as e:
                        logger.debug(f"Failed to check memory usage: {e}")

                last_checkpoint = current_time
                logger.debug(f"Completed step: {step_name} (elapsed: {elapsed:.1f}s)")

            # Resolve imports for the source file if available
            import_map = None
            if source_path is not None:
                try:
                    import_map = self._import_resolver.resolve(source_path)
                    logger.debug("Resolved imports for %s: %s", source_path, import_map)
                except Exception as e:
                    logger.warning(
                        "Failed to resolve imports for %s: %s", source_path, e
                    )
                    import_map = None
            check_progress_and_resources("import_resolution")
            # Build lightweight context query from top plan elements
            query_parts = [element.name for element in plan.elements_to_test[:3]]
            query = " ".join(query_parts)

            # 1) Retrieve top-ranked symbol-aware snippets
            snippet_items = self._retrieve_snippets(query, limit=5)
            check_progress_and_resources("snippet_retrieval")

            # 2) Merge import-graph neighbors via ContextPort.get_related_context
            neighbor_items = self._get_neighbor_context(source_path)
            check_progress_and_resources("neighbor_context")

            # 3) Extract concise exemplars from existing tests
            exemplar_items = self._get_test_exemplars(source_path, plan)
            check_progress_and_resources("test_exemplars")

            # 4) Extract concise API contracts/invariants for target elements
            contract_items = self._get_contract_context(source_path, plan)
            check_progress_and_resources("contract_context")

            # 5) Detect dependencies/config surfaces and available pytest fixtures
            deps_cfg_fixture_items = self._get_deps_config_fixtures(source_path)
            check_progress_and_resources("deps_config_fixtures")

            # 6) Get individual advanced context types for proper section capping
            coverage_hints = (
                self._get_coverage_hints(source_path) if source_path else []
            )
            callgraph_items = (
                self._get_callgraph_neighbors(source_path) if source_path else []
            )
            error_items = (
                self._get_error_paths(source_path, plan) if source_path else []
            )
            usage_items = (
                self._get_usage_examples(source_path, plan) if source_path else []
            )
            pytest_settings = (
                self._get_pytest_settings_context(source_path) if source_path else []
            )
            side_effects = (
                self._get_side_effects_context(source_path) if source_path else []
            )
            path_constraints = (
                self._get_path_constraints_context(source_path, plan)
                if source_path
                else []
            )
            check_progress_and_resources("advanced_context")

            # Validate context data integrity before assembly
            context_sections = [
                snippet_items,
                neighbor_items,
                exemplar_items,
                contract_items,
                deps_cfg_fixture_items,
                coverage_hints,
                callgraph_items,
                error_items,
                usage_items,
                pytest_settings,
                side_effects,
                path_constraints,
            ]

            # Check for corrupted context sections
            for i, section in enumerate(context_sections):
                if not isinstance(section, list):
                    logger.error(
                        f"Context section {i} is not a list, got {type(section)}"
                    )
                    return None

            # 7) Assemble bounded context with deterministic ordering and de-dupe
            base_context = self._assemble_final_context(context_sections)
            check_progress_and_resources("context_assembly")

            # 9) Enhance with packaging and safety information
            enriched_context_string = self._build_enriched_context_for_generation(
                source_path, base_context, import_map
            )
            check_progress_and_resources("enriched_context")

            # Validate enriched context before proceeding
            if enriched_context_string is not None and not isinstance(
                enriched_context_string, str
            ):
                logger.error("Enriched context is not a string, returning None")
                return None

            # 10) Build complete ContextPack with all components
            # We can proceed even if import_map is None (e.g., for files without proper package structure)
            if enriched_context_string is not None:
                # Build complete ContextPack using structured context data
                context_pack = self._build_complete_context_pack(
                    plan,
                    source_path,
                    import_map,
                    {
                        "contract_items": contract_items,
                        "error_items": error_items,
                        "side_effects": side_effects,
                        "deps_cfg_fixture_items": deps_cfg_fixture_items,
                        "exemplar_items": exemplar_items,
                        "usage_items": usage_items,
                        "pytest_settings": pytest_settings,
                        "path_constraints": path_constraints,
                        "neighbor_items": neighbor_items,
                        "snippet_items": snippet_items,
                    },
                )
                check_progress_and_resources("context_pack_building")

                if context_pack:
                    # Add the formatted context string
                    context_pack.context = enriched_context_string
                    check_progress_and_resources("final_validation")
                    return context_pack

            # Return None if we don't have sufficient information for a ContextPack
            return None

        except Exception as e:
            logger.warning("Failed to retrieve context: %s", e)
            return None

    def _build_complete_context_pack(
        self,
        plan: TestGenerationPlan,
        source_path: Path | None,
        import_map: dict[str, Any] | ImportMap | None,
        context_data: dict[str, list[str]],
    ) -> ContextPack | None:
        """
        Build a complete ContextPack with all components properly populated.

        Args:
            plan: The test generation plan
            source_path: Source file path
            import_map: Import mapping information
            context_data: Structured context data from helper methods

        Returns:
            Complete ContextPack or None if insufficient data
        """
        try:
            # 1. Extract and validate target information
            target = self._extract_target_info(source_path, plan)
            if not target:
                logger.warning("Could not extract valid target information")
                return None

            # 2. Build complete focal object
            focal = self._build_focal_object(source_path, plan, target)
            if not focal:
                logger.warning("Could not build focal object")
                return None

            # 3. Populate resolved definitions
            resolved_defs = self._populate_resolved_definitions(
                source_path, plan, context_data
            )

            # 4. Build property context
            property_context = self._build_property_context(plan, context_data)

            # 5. Populate conventions
            conventions = self._populate_conventions(source_path, context_data)

            # 6. Build budget object
            budget = self._build_budget_object()

            # 7. Assemble complete ContextPack
            context_pack = ContextPack(
                target=target,
                import_map=import_map,
                focal=focal,
                resolved_defs=resolved_defs,
                property_context=property_context,
                conventions=conventions,
                budget=budget,
                context="",  # Will be set by caller
            )

            # 8. Validate the ContextPack
            if not self._validate_context_pack(context_pack):
                logger.warning("ContextPack validation failed, returning None")
                return None

            logger.debug("Built complete ContextPack for %s", source_path)
            return context_pack

        except Exception as e:
            logger.warning("Failed to build complete ContextPack: %s", e)
            return None

    def _extract_target_info(
        self, source_path: Path | None, plan: TestGenerationPlan
    ) -> Target | None:
        """
        Extract and validate real target information.

        Args:
            source_path: Path to the source file
            plan: The test generation plan

        Returns:
            Target object with validated information or None if invalid
        """
        try:
            if not source_path or not source_path.exists():
                logger.warning("Invalid or missing source path: %s", source_path)
                return None

            if not plan.elements_to_test:
                logger.warning("No elements to test in plan")
                return None

            # Get the primary element being tested
            primary_element = plan.elements_to_test[0]
            object_name = primary_element.name

            if not object_name:
                logger.warning("Empty object name in plan element")
                return None

            # Validate that the target exists in the source file
            try:
                parse_result = self._parser.parse_file(source_path)
                ast_tree = parse_result.get("ast")

                if not ast_tree:
                    logger.warning("Could not parse AST for %s", source_path)
                    # Continue with basic target info even if AST parsing fails
                else:
                    # Check if the object exists in the AST
                    found = False
                    for node in ast.walk(ast_tree):
                        if isinstance(
                            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                        ):
                            if node.name == object_name:
                                found = True
                                break
                        # Handle method case: "ClassName.method_name"
                        elif "." in object_name:
                            class_name, method_name = object_name.split(".", 1)
                            if (
                                isinstance(node, ast.ClassDef)
                                and node.name == class_name
                            ):
                                for item in node.body:
                                    if isinstance(
                                        item, ast.FunctionDef | ast.AsyncFunctionDef
                                    ):
                                        if item.name == method_name:
                                            found = True
                                            break

                    if not found:
                        logger.warning(
                            "Target %s not found in %s", object_name, source_path
                        )

            except Exception as e:
                logger.warning("AST validation failed for %s: %s", source_path, e)
                # Continue with basic target info

            # Build target with comprehensive information
            try:
                # Get module path using existing ModulePathDeriver
                module_path_info = ModulePathDeriver.derive_module_path(source_path)
                module_path = module_path_info.get("module_path", "")

                # Get relative path from project root
                try:
                    project_root = self._find_project_root(source_path)
                    relative_path = (
                        str(source_path.relative_to(project_root))
                        if project_root
                        else str(source_path)
                    )
                except Exception as e:
                    logger.debug(
                        "Failed to find project root for %s: %s", source_path, e
                    )
                    relative_path = str(source_path)

                # Determine if this is a method (contains dot)
                is_method = "." in object_name
                class_name = object_name.split(".")[0] if is_method else None
                method_name = object_name.split(".")[1] if is_method else None

                # Create enhanced Target object
                target = Target(
                    module_file=str(source_path),
                    module_path=module_path,
                    relative_path=relative_path,
                    object=object_name,
                    class_name=class_name,
                    method_name=method_name,
                    is_method=is_method,
                    exists_in_source=True,  # We validated this above
                )

                logger.debug("Extracted target info: %s", target)
                return target

            except Exception as e:
                logger.warning("Failed to build target object: %s", e)
                # Fallback to basic target
                return Target(
                    module_file=str(source_path) if source_path else "unknown",
                    object=object_name,
                )

        except Exception as e:
            logger.warning("Failed to extract target info: %s", e)
            return None

    def _find_project_root(self, source_path: Path) -> Path | None:
        """Find project root by looking for common project markers with safety limits."""
        try:
            if not source_path or not source_path.exists():
                return None

            current = source_path.parent
            max_iterations = self._config.get(
                "max_project_root_iterations", DEFAULT_MAX_PROJECT_ROOT_ITERATIONS
            )

            for _iteration in range(max_iterations):
                # Check for filesystem boundary (mount point)
                if os.path.ismount(str(current)):
                    logger.debug(f"Reached filesystem boundary at {current}")
                    break

                # Check for project markers
                if any(
                    (current / marker).exists()
                    for marker in ["pyproject.toml", "setup.py", ".git"]
                ):
                    return current

                # Move to parent directory
                parent = current.parent
                if parent == current:
                    # Reached filesystem root
                    break
                current = parent

            logger.debug(
                f"Project root not found after {max_iterations} iterations for {source_path}"
            )
        except Exception as e:
            logger.debug("Failed to find project root for %s: %s", source_path, e)

        return None

    def _validate_ast_node(self, node: Any, expected_types: tuple[type, ...]) -> bool:
        """Validate AST node type and required attributes."""
        if not node or not isinstance(node, expected_types):
            return False
        # Check for required attributes
        if hasattr(node, "name") and not hasattr(node, "name"):
            return False
        return True

    def _build_focal_object(
        self, source_path: Path | None, plan: TestGenerationPlan, target: Target
    ) -> Focal | None:
        """
        Build complete focal object with signature, parameters, and metadata.

        Args:
            source_path: Path to the source file
            plan: The test generation plan
            target: Target object with location information

        Returns:
            Complete Focal object or None if insufficient data
        """
        try:
            if not plan.elements_to_test:
                logger.warning("No elements to test in plan")
                return None

            primary_element = plan.elements_to_test[0]
            object_name = primary_element.name

            # Basic focal information (already available)
            source = object_name
            docstring = primary_element.docstring

            # Extract complete signature using AST parsing
            signature = self._extract_complete_signature(source_path, primary_element)

            # Extract parameter information
            parameters = self._extract_parameter_info(source_path, primary_element)

            # Extract return type information
            return_type = self._extract_return_type(source_path, primary_element)

            # Get source location information
            line_number = (
                getattr(primary_element, "line_range", (0, 0))[0]
                if hasattr(primary_element, "line_range")
                else 0
            )

            # Extract decorators if any
            decorators = self._extract_decorators(source_path, primary_element)

            # Build comprehensive focal object
            focal = Focal(
                source=source,
                signature=signature,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                line_number=line_number,
                decorators=decorators,
                is_method=target.is_method,
                class_name=target.class_name,
                method_name=target.method_name,
            )

            logger.debug("Built focal object: %s", focal)
            return focal

        except Exception as e:
            logger.warning("Failed to build focal object: %s", e)
            # Return basic focal as fallback
            try:
                primary_element = plan.elements_to_test[0]
                return Focal(
                    source=primary_element.name,
                    signature=f"def {primary_element.name}(...):",
                    docstring=primary_element.docstring,
                )
            except Exception as e:
                logger.debug("Failed to create fallback focal object: %s", e)
                return None

    def _extract_complete_signature(
        self, source_path: Path | None, element: Any
    ) -> str:
        """Extract complete signature from AST parsing."""
        try:
            if not source_path or not source_path.exists():
                return f"def {element.name}(...):"

            # Use existing _get_signature method as a starting point
            node = self._find_node_for_element_from_plan(source_path, element)
            if node:
                return self._get_signature(node, element, [])

            # Fallback to basic signature
            return f"def {element.name}(...):"

        except Exception as e:
            logger.debug("Failed to extract complete signature: %s", e)
            return f"def {element.name}(...):"

    def _extract_parameter_info(
        self, source_path: Path | None, element: Any
    ) -> list[dict[str, Any]]:
        """Extract parameter information from AST."""
        parameters = []

        try:
            if not source_path:
                return parameters

            node = self._find_node_for_element_from_plan(source_path, element)
            if not node or not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return parameters

            # Extract parameter information from AST
            for arg in node.args.args:
                param_info = {
                    "name": arg.arg,
                    "type": None,  # Would need to extract from type comments or annotations
                    "default": None,  # Would need to extract default values
                    "kind": "positional_or_keyword",
                }

                # Try to get type annotation
                if arg.annotation:
                    try:
                        param_info["type"] = ast.unparse(arg.annotation)
                    except Exception as e:
                        logger.debug("Failed to unparse parameter annotation: %s", e)
                        pass

                parameters.append(param_info)

            # Handle *args and **kwargs
            if node.args.vararg:
                parameters.append(
                    {
                        "name": node.args.vararg.arg,
                        "type": None,
                        "default": None,
                        "kind": "var_positional",
                    }
                )

            if node.args.kwarg:
                parameters.append(
                    {
                        "name": node.args.kwarg.arg,
                        "type": None,
                        "default": None,
                        "kind": "var_keyword",
                    }
                )

        except Exception as e:
            logger.debug("Failed to extract parameter info: %s", e)

        return parameters

    def _extract_return_type(
        self, source_path: Path | None, element: Any
    ) -> str | None:
        """Extract return type information from AST."""
        try:
            if not source_path:
                return None

            node = self._find_node_for_element_from_plan(source_path, element)
            if not node or not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return None

            # Try to get return annotation
            if node.returns:
                return ast.unparse(node.returns)

        except Exception as e:
            logger.debug("Failed to extract return type: %s", e)

        return None

    def _extract_decorators(self, source_path: Path | None, element: Any) -> list[str]:
        """Extract decorator information from AST."""
        decorators = []

        try:
            if not source_path:
                return decorators

            node = self._find_node_for_element_from_plan(source_path, element)
            if not node:
                return decorators

            # Extract decorator names
            for decorator in getattr(node, "decorator_list", []):
                try:
                    if hasattr(decorator, "id"):
                        decorators.append(decorator.id)
                    elif (
                        hasattr(decorator, "attr")
                        and hasattr(decorator, "value")
                        and hasattr(decorator.value, "id")
                    ):
                        decorators.append(f"{decorator.value.id}.{decorator.attr}")
                    else:
                        decorators.append(ast.unparse(decorator))
                except Exception as e:
                    logger.debug("Failed to extract decorator info: %s", e)
                    continue

        except Exception as e:
            logger.debug("Failed to extract decorators: %s", e)

        return decorators

    def _find_node_for_element_from_plan(
        self, source_path: Path, element: Any
    ) -> Any | None:
        """Find AST node for a plan element."""
        try:
            parse_result = self._parser.parse_file(source_path)
            ast_tree = parse_result.get("ast")

            if not ast_tree:
                return None

            object_name = element.name

            # Handle method case: "ClassName.method_name"
            if "." in object_name:
                class_name, method_name = object_name.split(".", 1)
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                                if item.name == method_name:
                                    return item
            else:
                # Handle function or class at module level
                for node in ast.walk(ast_tree):
                    if isinstance(
                        node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                    ):
                        if node.name == object_name:
                            return node

        except Exception as e:
            logger.debug("Failed to find node for element %s: %s", element.name, e)

        return None

    def _populate_resolved_definitions(
        self,
        source_path: Path | None,
        plan: TestGenerationPlan,
        context_data: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """
        Populate resolved definitions using AST nodes and dependencies.

        Args:
            source_path: Path to the source file
            plan: The test generation plan
            context_data: Structured context data from helper methods

        Returns:
            List of resolved definition dictionaries
        """
        resolved_defs = []

        try:
            if not source_path or not plan.elements_to_test:
                return resolved_defs

            # Get the primary element being tested
            primary_element = plan.elements_to_test[0]

            # Extract the main definition
            main_def = self._extract_main_definition(source_path, primary_element)
            if main_def:
                resolved_defs.append(main_def)

            # Extract related definitions from context data
            related_defs = self._extract_related_definitions(source_path, context_data)
            resolved_defs.extend(related_defs)

            # Extract dependency definitions
            dependency_defs = self._extract_dependency_definitions(
                source_path, context_data
            )
            resolved_defs.extend(dependency_defs)

            # Remove duplicates while preserving order
            seen_signatures = set()
            unique_defs = []
            for def_info in resolved_defs:
                signature = def_info.get("signature", "")
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_defs.append(def_info)

            logger.debug("Populated %d resolved definitions", len(unique_defs))
            return unique_defs

        except Exception as e:
            logger.warning("Failed to populate resolved definitions: %s", e)
            return []

    def _extract_main_definition(
        self, source_path: Path, element: Any
    ) -> dict[str, Any] | None:
        """Extract the main definition for the target element."""
        try:
            node = self._find_node_for_element_from_plan(source_path, element)
            if not node:
                return None

            # Get source code for the definition with safety checks
            source_code = safe_file_read(source_path)
            if source_code is None:
                return None

            # Extract the definition source
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                def_source = self._extract_function_source(node, source_code)
            elif isinstance(node, ast.ClassDef):
                def_source = self._extract_class_source(node, source_code)
            else:
                return None

            # Extract dependencies used by this definition
            dependencies = self._extract_definition_dependencies(node, source_code)

            # Extract type annotations
            type_annotations = self._extract_type_annotations(node)

            # Extract decorators
            decorators = []
            for decorator in getattr(node, "decorator_list", []):
                try:
                    decorators.append(ast.unparse(decorator))
                except Exception as e:
                    logger.debug("Failed to unparse decorator: %s", e)
                    continue

            return {
                "name": node.name,
                "type": "function"
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                else "class",
                "signature": ast.unparse(node)
                if hasattr(ast, "unparse")
                else str(node),
                "source": def_source,
                "dependencies": dependencies,
                "type_annotations": type_annotations,
                "decorators": decorators,
                "line_start": getattr(node, "lineno", 0),
                "line_end": getattr(node, "end_lineno", 0),
                "is_main_target": True,
            }

        except Exception as e:
            logger.debug("Failed to extract main definition: %s", e)
            return None

    def _extract_related_definitions(
        self, source_path: Path, context_data: dict[str, list[str]]
    ) -> list[dict[str, Any]]:
        """Extract related definitions from context data."""
        related_defs = []

        try:
            # Use neighbor context to find related files
            neighbor_items = context_data.get("neighbor_items", [])
            for item in neighbor_items[:3]:  # Limit to prevent excessive processing
                if isinstance(item, str) and "# Related:" in item:
                    # Extract file path from neighbor context
                    try:
                        related_path = Path(item.replace("# Related: ", "").strip())
                        if related_path.exists() and related_path != source_path:
                            related_def = self._extract_definition_from_file(
                                related_path
                            )
                            if related_def:
                                related_defs.append(related_def)
                    except Exception as e:
                        logger.debug(
                            "Failed to process related file %s: %s", related_path, e
                        )
                        continue

        except Exception as e:
            logger.debug("Failed to extract related definitions: %s", e)

        return related_defs

    def _extract_dependency_definitions(
        self, source_path: Path, context_data: dict[str, list[str]]
    ) -> list[dict[str, Any]]:
        """Extract definitions for dependencies found in context."""
        dependency_defs = []

        try:
            # Use contract context and other data to find dependency patterns
            contract_items = context_data.get("contract_items", [])
            for item in contract_items[:3]:
                if isinstance(item, str) and "# Contract:" in item:
                    # Parse contract information to find dependencies
                    dep_def = self._parse_contract_for_dependencies(item)
                    if dep_def:
                        dependency_defs.append(dep_def)

        except Exception as e:
            logger.debug("Failed to extract dependency definitions: %s", e)

        return dependency_defs

    def _extract_function_source(self, node: ast.FunctionDef, source_code: str) -> str:
        """Extract source code for a function definition."""
        try:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                lines = source_code.splitlines()
                start_line = node.lineno - 1
                end_line = node.end_lineno

                if start_line < len(lines) and end_line <= len(lines):
                    return "\n".join(lines[start_line:end_line])
        except Exception as e:
            logger.debug("Failed to extract function source: %s", e)
            pass

        # Fallback: return the unparsed AST
        try:
            return ast.unparse(node) if hasattr(ast, "unparse") else str(node)
        except Exception as e:
            logger.debug("Failed to unparse function AST: %s", e)
            return f"def {node.name}(...): ..."

    def _extract_class_source(self, node: ast.ClassDef, source_code: str) -> str:
        """Extract source code for a class definition."""
        try:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                lines = source_code.splitlines()
                start_line = node.lineno - 1
                end_line = node.end_lineno

                if start_line < len(lines) and end_line <= len(lines):
                    return "\n".join(lines[start_line:end_line])
        except Exception as e:
            logger.debug("Failed to extract class source: %s", e)
            pass

        # Fallback: return the unparsed AST
        try:
            return ast.unparse(node) if hasattr(ast, "unparse") else str(node)
        except Exception as e:
            logger.debug("Failed to unparse class AST: %s", e)
            return f"class {node.name}: ..."

    def _extract_definition_dependencies(
        self, node: Any, source_code: str
    ) -> list[str]:
        """Extract dependencies used by a definition."""
        dependencies = []

        try:
            # Find all Name nodes that reference external dependencies
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    # Check if this is an import or external reference
                    name = child.id
                    # Skip built-ins and common keywords
                    if name not in {
                        "self",
                        "cls",
                        "True",
                        "False",
                        "None",
                        "int",
                        "str",
                        "float",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                        "len",
                        "range",
                        "enumerate",
                        "zip",
                        "map",
                        "filter",
                        "sum",
                        "max",
                        "min",
                        "abs",
                        "round",
                    }:
                        dependencies.append(name)

        except Exception as e:
            logger.debug("Failed to extract dependencies: %s", e)

        return list(set(dependencies))  # Remove duplicates

    def _extract_type_annotations(self, node: Any) -> list[dict[str, Any]]:
        """Extract type annotations from a definition."""
        annotations = []

        try:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Extract parameter type annotations
                for arg in node.args.args:
                    if arg.annotation:
                        annotations.append(
                            {
                                "type": "parameter",
                                "name": arg.arg,
                                "annotation": ast.unparse(arg.annotation)
                                if hasattr(ast, "unparse")
                                else str(arg.annotation),
                            }
                        )

                # Extract return type annotation
                if node.returns:
                    annotations.append(
                        {
                            "type": "return",
                            "annotation": ast.unparse(node.returns)
                            if hasattr(ast, "unparse")
                            else str(node.returns),
                        }
                    )

        except Exception as e:
            logger.debug("Failed to extract type annotations: %s", e)

        return annotations

    def _extract_definition_from_file(self, file_path: Path) -> dict[str, Any] | None:
        """Extract a representative definition from a related file."""
        try:
            if not file_path.exists():
                return None

            source_code = safe_file_read(file_path)
            if source_code is None:
                return None
            tree = safe_ast_parse(source_code)
            if tree is None:
                return None

            # Find the first function or class definition
            for node in ast.walk(tree):
                if isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    return {
                        "name": node.name,
                        "type": "function"
                        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                        else "class",
                        "signature": ast.unparse(node)
                        if hasattr(ast, "unparse")
                        else str(node),
                        "source": self._extract_function_source(node, source_code)
                        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                        else self._extract_class_source(node, source_code),
                        "file_path": str(file_path),
                        "line_start": getattr(node, "lineno", 0),
                        "is_main_target": False,
                    }

        except Exception as e:
            logger.debug("Failed to extract definition from %s: %s", file_path, e)

        return None

    def _parse_contract_for_dependencies(
        self, contract_item: str
    ) -> dict[str, Any] | None:
        """Parse contract information to extract dependency definitions."""
        try:
            # This is a simplified parser - in practice, would use more sophisticated parsing
            lines = contract_item.split("\n")
            for line in lines:
                if "signature:" in line:
                    # Extract signature information
                    signature = line.replace("signature:", "").strip()
                    if signature and not signature.startswith("def ..."):
                        return {
                            "name": "contract_dependency",
                            "type": "function",
                            "signature": signature,
                            "source": "# Contract dependency",
                            "is_main_target": False,
                        }

        except Exception as e:
            logger.debug("Failed to parse contract for dependencies: %s", e)

        return None

    def _build_property_context(
        self, plan: TestGenerationPlan, context_data: dict[str, list[str]]
    ) -> PropertyContext:
        """
        Build property context from contracts, invariants, and constraints.

        Args:
            plan: The test generation plan
            context_data: Structured context data from helper methods

        Returns:
            PropertyContext object with comprehensive property information
        """
        # Initialize with empty/default values
        invariants = []
        pre_conditions = []
        post_conditions = []
        error_conditions = []
        side_effects = []
        constraints = []

        try:
            # Extract invariants from contract context
            invariants = self._extract_invariants_from_contracts(context_data)

            # Extract pre/post conditions from docstrings and assertions
            pre_conditions, post_conditions = self._extract_pre_post_conditions(plan)

            # Extract error conditions from error paths
            error_conditions = self._extract_error_conditions_from_paths(context_data)

            # Extract side effects
            side_effects = self._extract_side_effects_from_context(context_data)

            # Extract constraints from path analysis
            constraints = self._extract_path_constraints(context_data)

        except Exception as e:
            logger.warning("Failed to build property context: %s", e)

        # Create comprehensive PropertyContext
        return PropertyContext(
            invariants=invariants,
            pre_conditions=pre_conditions,
            post_conditions=post_conditions,
            error_conditions=error_conditions,
            side_effects=side_effects,
            constraints=constraints,
        )

    def _extract_invariants_from_contracts(
        self, context_data: dict[str, list[str]]
    ) -> list[str]:
        """Extract invariants from contract context data."""
        invariants = []

        try:
            contract_items = context_data.get("contract_items", [])
            for item in contract_items[:5]:  # Limit processing
                if isinstance(item, str) and "invariants:" in item:
                    # Parse invariants from contract text
                    lines = item.split("\n")
                    for line in lines:
                        if "invariants:" in line:
                            invariant_text = line.replace("invariants:", "").strip()
                            if invariant_text.startswith(
                                "["
                            ) and invariant_text.endswith("]"):
                                # Parse list format: [condition1, condition2]
                                invariant_list = invariant_text[1:-1].split(",")
                                for invariant in invariant_list:
                                    clean_invariant = invariant.strip().strip("\"'")
                                    if clean_invariant and len(clean_invariant) > 3:
                                        invariants.append(clean_invariant)
                            else:
                                # Single invariant
                                if invariant_text and len(invariant_text) > 3:
                                    invariants.append(invariant_text)

        except Exception as e:
            logger.debug("Failed to extract invariants from contracts: %s", e)

        return invariants

    def _extract_pre_post_conditions(
        self, plan: TestGenerationPlan
    ) -> tuple[list[str], list[str]]:
        """Extract pre/post conditions from plan elements."""
        pre_conditions = []
        post_conditions = []

        try:
            if not plan.elements_to_test:
                return pre_conditions, post_conditions

            primary_element = plan.elements_to_test[0]
            docstring = getattr(primary_element, "docstring", "") or ""

            if docstring:
                # Parse docstring for pre/post conditions
                doc_info = self._parse_docstring_for_conditions(docstring)
                pre_conditions.extend(doc_info.get("pre_conditions", []))
                post_conditions.extend(doc_info.get("post_conditions", []))

        except Exception as e:
            logger.debug("Failed to extract pre/post conditions: %s", e)

        return pre_conditions, post_conditions

    def _extract_error_conditions_from_paths(
        self, context_data: dict[str, list[str]]
    ) -> list[str]:
        """Extract error conditions from error paths context."""
        error_conditions = []

        try:
            error_items = context_data.get("error_items", [])
            for item in error_items[:5]:  # Limit processing
                if isinstance(item, str) and "# Error paths:" in item:
                    # Parse error paths
                    error_text = item.replace("# Error paths:", "").strip()
                    if error_text.startswith("[") and error_text.endswith("]"):
                        # Parse list format
                        error_list = error_text[1:-1].split(",")
                        for error in error_list:
                            clean_error = error.strip().strip("\"'")
                            if clean_error and len(clean_error) > 2:
                                error_conditions.append(clean_error)

        except Exception as e:
            logger.debug("Failed to extract error conditions: %s", e)

        return error_conditions

    def _extract_side_effects_from_context(
        self, context_data: dict[str, list[str]]
    ) -> list[str]:
        """Extract side effects from context data."""
        side_effects = []

        try:
            # Check for side effects in various context sections
            side_effect_sections = [
                context_data.get("side_effects", []),
                context_data.get("deps_cfg_fixture_items", []),
            ]

            for section in side_effect_sections:
                for item in section[:3]:  # Limit processing
                    if isinstance(item, str):
                        # Look for side effect indicators
                        if "_effects:" in item:
                            effects_text = item.split("_effects:")[1].strip()
                            if effects_text.startswith("[") and effects_text.endswith(
                                "]"
                            ):
                                effects_list = effects_text[1:-1].split(",")
                                for effect in effects_list:
                                    clean_effect = effect.strip().strip("\"'")
                                    if clean_effect and len(clean_effect) > 2:
                                        side_effects.append(clean_effect)

        except Exception as e:
            logger.debug("Failed to extract side effects: %s", e)

        return side_effects

    def _extract_path_constraints(
        self, context_data: dict[str, list[str]]
    ) -> list[str]:
        """Extract path constraints from context data."""
        constraints = []

        try:
            path_constraint_items = context_data.get("path_constraints", [])
            for item in path_constraint_items[:3]:  # Limit processing
                if isinstance(item, str) and "# Path constraints:" in item:
                    constraint_text = item.replace("# Path constraints:", "").strip()
                    if constraint_text and len(constraint_text) > 5:
                        constraints.append(constraint_text)

        except Exception as e:
            logger.debug("Failed to extract path constraints: %s", e)

        return constraints

    def _parse_docstring_for_conditions(self, docstring: str) -> dict[str, list[str]]:
        """Parse docstring for pre/post conditions."""
        conditions = {
            "pre_conditions": [],
            "post_conditions": [],
        }

        try:
            lines = docstring.split("\n")

            # Look for Args/Parameters section
            in_args = False
            for line in lines:
                line = line.strip()
                if line.lower().startswith("args:") or line.lower().startswith(
                    "parameters:"
                ):
                    in_args = True
                    continue
                elif in_args and line and not line.startswith(" "):
                    # End of args section
                    break
                elif in_args and line.startswith("    ") or line.startswith("  "):
                    # This is a parameter description - could contain preconditions
                    param_desc = line.strip()
                    if ":" in param_desc:
                        param_name, description = param_desc.split(":", 1)
                        if description.strip():
                            conditions["pre_conditions"].append(
                                f"{param_name.strip()}: {description.strip()}"
                            )

            # Look for Returns section
            in_returns = False
            for line in lines:
                line = line.strip()
                if line.lower().startswith("returns:") or line.lower().startswith(
                    "return:"
                ):
                    in_returns = True
                    continue
                elif in_returns and line and not line.startswith(" "):
                    break
                elif in_returns and line.startswith("    ") or line.startswith("  "):
                    return_desc = line.strip()
                    if return_desc:
                        conditions["post_conditions"].append(f"Returns: {return_desc}")

        except Exception as e:
            logger.debug("Failed to parse docstring for conditions: %s", e)

        return conditions

    def _populate_conventions(
        self, source_path: Path | None, context_data: dict[str, list[str]]
    ) -> Conventions:
        """
        Extract conventions from codebase patterns and existing analysis.

        Args:
            source_path: Path to the source file
            context_data: Structured context data from helper methods

        Returns:
            Conventions object with extracted patterns and styles
        """
        # Initialize with defaults
        naming_patterns = {}
        pytest_patterns = {}
        coding_styles = {}
        test_patterns = {}

        try:
            # Extract naming patterns from codebase analysis
            naming_patterns = self._extract_naming_patterns(source_path, context_data)

            # Extract pytest fixture usage patterns
            pytest_patterns = self._extract_pytest_patterns(context_data)

            # Extract coding style information
            coding_styles = self._extract_coding_styles(source_path, context_data)

            # Extract test patterns from exemplars
            test_patterns = self._extract_test_patterns(context_data)

        except Exception as e:
            logger.warning("Failed to populate conventions: %s", e)

        # Create comprehensive Conventions object
        return Conventions(
            naming_patterns=naming_patterns,
            pytest_patterns=pytest_patterns,
            coding_styles=coding_styles,
            test_patterns=test_patterns,
        )

    def _extract_naming_patterns(
        self, source_path: Path | None, context_data: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Extract naming patterns from codebase analysis."""
        patterns = {
            "function_patterns": [],
            "class_patterns": [],
            "variable_patterns": [],
            "test_patterns": [],
        }

        try:
            if not source_path:
                return patterns

            # Analyze the source file for naming patterns
            try:
                source_code = safe_file_read(source_path)
                if source_code is None:
                    return patterns
                tree = safe_ast_parse(source_code)
                if tree is None:
                    return patterns

                # Extract function naming patterns
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        functions.append(node.name)

                # Analyze function naming patterns
                if functions:
                    # Snake case patterns
                    snake_case = [
                        f
                        for f in functions
                        if f.replace("_", "").isalnum() and "_" in f
                    ]
                    if snake_case:
                        patterns["function_patterns"].extend(
                            [f"snake_case: {snake_case[:3]}"]
                        )

                    # Test function patterns
                    test_functions = [f for f in functions if f.startswith("test_")]
                    if test_functions:
                        patterns["test_patterns"].extend(
                            [f"test_prefix: {test_functions[:3]}"]
                        )

                # Extract class naming patterns
                classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(node.name)

                if classes:
                    # Pascal case patterns
                    pascal_case = [
                        c for c in classes if c[0].isupper() and "_" not in c
                    ]
                    if pascal_case:
                        patterns["class_patterns"].extend(
                            [f"PascalCase: {pascal_case[:3]}"]
                        )

            except Exception as e:
                logger.debug("Failed to analyze source file for naming patterns: %s", e)

            # Extract patterns from context data
            exemplar_items = context_data.get("exemplar_items", [])
            for item in exemplar_items[:3]:
                if isinstance(item, str) and "fixtures=" in item:
                    # Extract fixture naming patterns
                    fixtures_text = item.split("fixtures=")[1].split(",")[0].strip()
                    if fixtures_text and len(fixtures_text) > 2:
                        patterns["variable_patterns"].append(
                            f"fixture_pattern: {fixtures_text}"
                        )

        except Exception as e:
            logger.debug("Failed to extract naming patterns: %s", e)

        return patterns

    def _extract_pytest_patterns(
        self, context_data: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Extract pytest fixture usage patterns."""
        patterns = {
            "fixture_usage": [],
            "marker_usage": [],
            "assertion_patterns": [],
        }

        try:
            # Extract from exemplar items
            exemplar_items = context_data.get("exemplar_items", [])
            for item in exemplar_items[:5]:
                if isinstance(item, str):
                    # Extract fixture usage
                    if "fixtures=" in item:
                        fixtures_text = (
                            item.split("fixtures=")[1].split(",")[0].strip("[]")
                        )
                        if fixtures_text and len(fixtures_text) > 2:
                            patterns["fixture_usage"].append(fixtures_text)

                    # Extract marker usage
                    if "markers=" in item:
                        markers_text = (
                            item.split("markers=")[1].split(",")[0].strip("[]")
                        )
                        if markers_text and len(markers_text) > 2:
                            patterns["marker_usage"].append(markers_text)

            # Extract from deps/config/fixtures items
            deps_items = context_data.get("deps_cfg_fixture_items", [])
            for item in deps_items[:3]:
                if isinstance(item, str):
                    # Look for pytest settings
                    if "pytest_settings:" in item:
                        settings_text = item.replace("pytest_settings:", "").strip()
                        if settings_text.startswith("[") and settings_text.endswith(
                            "]"
                        ):
                            settings_list = settings_text[1:-1].split(",")
                            for setting in settings_list:
                                clean_setting = setting.strip().strip("\"'")
                                if clean_setting and len(clean_setting) > 2:
                                    patterns["fixture_usage"].append(
                                        f"pytest_setting: {clean_setting}"
                                    )

        except Exception as e:
            logger.debug("Failed to extract pytest patterns: %s", e)

        return patterns

    def _extract_coding_styles(
        self, source_path: Path | None, context_data: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Extract coding style information."""
        styles = {
            "indentation": "unknown",
            "docstring_format": "unknown",
            "import_style": "unknown",
            "line_length": "unknown",
        }

        try:
            if not source_path:
                return styles

            # Analyze source file for coding style
            try:
                source_code = source_path.read_text(encoding="utf-8")

                # Detect indentation style
                lines = source_code.split("\n")
                indented_lines = [
                    line
                    for line in lines
                    if line.startswith("    ") or line.startswith("\t")
                ]

                if indented_lines:
                    # Check for spaces vs tabs
                    space_count = sum(
                        1 for line in indented_lines if line.startswith("    ")
                    )
                    tab_count = sum(
                        1 for line in indented_lines if line.startswith("\t")
                    )

                    if space_count > tab_count:
                        styles["indentation"] = "4_spaces"
                    elif tab_count > space_count:
                        styles["indentation"] = "tabs"

                # Detect docstring format
                if '"""' in source_code:
                    styles["docstring_format"] = "triple_double_quotes"
                elif "'''" in source_code:
                    styles["docstring_format"] = "triple_single_quotes"

                # Detect import style
                import_lines = [
                    line
                    for line in lines
                    if line.strip().startswith("import ")
                    or line.strip().startswith("from ")
                ]
                if import_lines:
                    # Check for import grouping
                    import_count = sum(
                        1 for line in import_lines if line.strip().startswith("import ")
                    )
                    from_count = sum(
                        1 for line in import_lines if line.strip().startswith("from ")
                    )

                    if import_count > 0 and from_count > 0:
                        styles["import_style"] = "mixed"
                    elif import_count > 0:
                        styles["import_style"] = "direct_import"
                    elif from_count > 0:
                        styles["import_style"] = "from_import"

                # Estimate line length patterns
                non_empty_lines = [line for line in lines if line.strip()]
                if non_empty_lines:
                    avg_length = sum(len(line) for line in non_empty_lines) / len(
                        non_empty_lines
                    )
                    if avg_length < 80:
                        styles["line_length"] = "short"
                    elif avg_length < 120:
                        styles["line_length"] = "medium"
                    else:
                        styles["line_length"] = "long"

            except Exception as e:
                logger.debug("Failed to analyze coding style: %s", e)

        except Exception as e:
            logger.debug("Failed to extract coding styles: %s", e)

        return styles

    def _extract_test_patterns(
        self, context_data: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Extract test patterns from exemplar data."""
        patterns = {
            "assertion_styles": [],
            "fixture_usage": [],
            "setup_patterns": [],
        }

        try:
            # Extract from exemplar items
            exemplar_items = context_data.get("exemplar_items", [])
            for item in exemplar_items[:5]:
                if isinstance(item, str):
                    # Extract assertion patterns
                    if "asserts=" in item:
                        asserts_text = item.split("asserts=")[1].split(",")[0].strip()
                        if asserts_text.isdigit():
                            assert_count = int(asserts_text)
                            if assert_count > 0:
                                patterns["assertion_styles"].append(
                                    f"assertions_per_test: {asserts_text}"
                                )

                    # Extract fixture usage patterns
                    if "fixtures=" in item:
                        fixtures_text = (
                            item.split("fixtures=")[1].split(",")[0].strip("[]")
                        )
                        if fixtures_text and len(fixtures_text) > 2:
                            patterns["fixture_usage"].append(fixtures_text)

                    # Extract raises patterns
                    if "raises=" in item:
                        raises_text = item.split("raises=")[1].split(",")[0].strip()
                        if raises_text.isdigit():
                            raises_count = int(raises_text)
                            if raises_count > 0:
                                patterns["setup_patterns"].append(
                                    f"exception_testing: {raises_text}"
                                )

        except Exception as e:
            logger.debug("Failed to extract test patterns: %s", e)

        return patterns

    def _build_budget_object(self) -> Budget:
        """
        Build budget object from existing configuration values.

        Returns:
            Budget object with limits and constraints from configuration
        """
        # Get budget configuration from existing config
        prompt_budgets = self._config.get("prompt_budgets", {})
        context_budgets = self._config.get("context_budgets", {})

        # Extract budget values with sensible defaults
        context_limits = {
            "max_context_chars": int(prompt_budgets.get("total_chars", 4000)),
            "max_section_chars": int(prompt_budgets.get("per_item_chars", 600)),
            "max_sections": len(prompt_budgets.get("section_caps", {})) or 10,
        }

        token_limits = {
            "max_input_tokens": int(prompt_budgets.get("max_input_tokens", 4000)),
            "max_output_tokens": int(prompt_budgets.get("max_output_tokens", 1000)),
            "reserve_tokens": int(prompt_budgets.get("reserve_tokens", 500)),
        }

        performance_limits = {
            "max_processing_time": int(
                self._config.get("max_processing_time", 30)
            ),  # seconds
            "max_files_to_process": int(context_budgets.get("max_files", 50)),
            "max_depth_per_dir": int(
                context_budgets.get("directory_tree", {}).get("max_depth", 4)
            ),
            "max_entries_per_dir": int(
                context_budgets.get("directory_tree", {}).get(
                    "max_entries_per_dir", 200
                )
            ),
        }

        # Extract section caps
        section_caps = {}
        caps_config = prompt_budgets.get("section_caps", {})
        if isinstance(caps_config, dict):
            for section, cap in caps_config.items():
                try:
                    section_caps[section] = int(cap)
                except (ValueError, TypeError):
                    section_caps[section] = 5  # Default cap

        # Extract context categories
        context_categories = self._config.get("context_categories", {})
        if not isinstance(context_categories, dict):
            context_categories = {}

        # Build comprehensive Budget object
        return Budget(
            context_limits=context_limits,
            token_limits=token_limits,
            performance_limits=performance_limits,
            section_caps=section_caps,
            context_categories=context_categories,
        )

    def _validate_context_pack(self, context_pack: ContextPack) -> bool:
        """
        Validate that all ContextPack fields are properly populated.

        Args:
            context_pack: The ContextPack to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate target
            if not context_pack.target or not context_pack.target.module_file:
                logger.warning("ContextPack target is missing or invalid")
                return False

            # Validate focal
            if not context_pack.focal or not context_pack.focal.source:
                logger.warning("ContextPack focal is missing or invalid")
                return False

            # Validate resolved definitions (can be empty but should be a list)
            if not isinstance(context_pack.resolved_defs, list):
                logger.warning("ContextPack resolved_defs is not a list")
                return False

            # Validate property context (can be empty but should have expected structure)
            if not context_pack.property_context:
                logger.warning("ContextPack property_context is missing")
                return False

            # Validate conventions (can be empty but should have expected structure)
            if not context_pack.conventions:
                logger.warning("ContextPack conventions is missing")
                return False

            # Validate budget (can be empty but should have expected structure)
            if not context_pack.budget:
                logger.warning("ContextPack budget is missing")
                return False

            # Validate import map (can be None)
            # Import map is optional, so we don't validate it strictly

            # Validate context (should be a string)
            if not isinstance(context_pack.context, str):
                logger.warning("ContextPack context is not a string")
                return False

            logger.debug("ContextPack validation passed")
            return True

        except Exception as e:
            logger.warning("ContextPack validation failed: %s", e)
            return False

    def context_for_refinement(
        self, test_file: Path, test_content: str
    ) -> dict[str, Any] | None:
        """
        Build source context for test refinement using AST analysis and context discovery.

        Args:
            test_file: Path to the test file being refined
            test_content: Content of the test file

        Returns:
            Dictionary with comprehensive source context information or None if unavailable
        """
        try:
            context = {
                "test_file_path": str(test_file),
                "test_content": test_content,
                "related_source_files": [],
                "imports_context": [],
                "dependency_analysis": {},
                "retrieved_context": [],
                "project_structure": {},
                "import_map": None,
            }

            # Resolve imports for the test file to understand its structure and dependencies
            try:
                import_map = self._import_resolver.resolve(test_file)
                context["import_map"] = import_map
                logger.debug(
                    "Resolved imports for test file %s: %s", test_file, import_map
                )
            except Exception as e:
                logger.warning(
                    "Failed to resolve imports for test file %s: %s", test_file, e
                )

            # Skip context gathering if not enabled
            if not self._config.get("enable_context", True):
                return context

            try:
                # Step 1: Use AST analysis to find test file dependencies
                dependency_analysis = self._parser.analyze_dependencies(test_file)
                context["dependency_analysis"] = dependency_analysis

                # Extract import information
                imports = dependency_analysis.get("imports", [])
                internal_deps = dependency_analysis.get("internal_deps", [])

                # If internal_deps is empty, derive modules from test_content via AST only (no regex fallback)
                if not internal_deps:
                    try:
                        derived = self._derive_modules_from_test_ast(test_content)
                        # Merge in a deterministic way without duplicates
                        merged = list(dict.fromkeys(list(internal_deps) + derived))
                        internal_deps = merged
                        # Also reflect in dependency_analysis for downstream use
                        dependency_analysis["internal_deps"] = internal_deps
                        context["dependency_analysis"] = dependency_analysis
                    except Exception as e:
                        logger.debug("Failed to derive modules from test AST: %s", e)
                        pass

                # Step 2: Index the test file for context relationships
                try:
                    # Index the test file if not already indexed
                    index_result = self._context.index(test_file, content=test_content)
                    logger.debug("Indexed test file %s: %s", test_file, index_result)

                    # Step 3: Use context port to find related files
                    related_context = self._context.get_related_context(
                        test_file, relationship_type="all"
                    )

                    # Add related files found through context relationships
                    for related_file_path in related_context.get("related_files", []):
                        related_path = Path(related_file_path)
                        if related_path.exists() and related_path.suffix == ".py":
                            try:
                                # Limit content size for performance
                                source_content = safe_file_read(related_path)
                                if source_content is None:
                                    continue
                                context["related_source_files"].append(
                                    {
                                        "path": str(related_path),
                                        "content": source_content[:2000],
                                        "relationship": "context_analysis",
                                    }
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to read related file %s: %s",
                                    related_path,
                                    e,
                                )

                    # Step 4: Build intelligent retrieval queries from test context
                    retrieval_queries = self._extract_test_context_queries(
                        test_file, test_content
                    )

                    for query in retrieval_queries[:3]:  # Limit queries for performance
                        try:
                            retrieval_result = self._context.retrieve(
                                query=query, context_type="general", limit=3
                            )

                            if retrieval_result.get("results"):
                                context["retrieved_context"].append(
                                    {
                                        "query": query,
                                        "results": retrieval_result["results"][
                                            :2
                                        ],  # Limit results
                                    }
                                )

                        except Exception as e:
                            logger.warning(
                                "Context retrieval failed for query '%s': %s", query, e
                            )

                    # Step 5: Add import-based source file discovery
                    for dep in internal_deps:
                        potential_source_paths = self._find_source_files_for_module(
                            test_file, dep
                        )
                        for source_path in potential_source_paths:
                            if source_path.exists():
                                try:
                                    source_content = safe_file_read(source_path)
                                    if source_content is None:
                                        continue
                                    context["related_source_files"].append(
                                        {
                                            "path": str(source_path),
                                            "content": source_content[:2000],
                                            "relationship": f"import_dependency: {dep}",
                                        }
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to read source file %s: %s",
                                        source_path,
                                        e,
                                    )

                    # Step 6: Add imports context for better LLM understanding
                    for import_info in imports:
                        context["imports_context"].append(
                            {
                                "module": import_info.get("module", ""),
                                "items": import_info.get("items", []),
                                "alias": import_info.get("alias", ""),
                                "is_internal": import_info.get("module", "")
                                in internal_deps,
                            }
                        )

                except Exception as e:
                    logger.warning(
                        "Context port analysis failed for %s: %s", test_file, e
                    )
                    # Continue with basic context even if context port fails

            except Exception as e:
                logger.warning("AST/Context analysis failed for %s: %s", test_file, e)
                # Fall back to basic context if advanced analysis fails

            # Step 7: Add project structure context using recursive tree builder
            try:
                # Determine appropriate project root for refinement context
                project_root = (
                    test_file.parent.parent
                    if test_file.parent != test_file.parent.parent
                    else test_file.parent
                )

                # Use recursive directory tree with configuration-based limits
                directory_config = self._config.get("context_budgets", {}).get(
                    "directory_tree", {}
                )
                max_depth = directory_config.get(
                    "max_depth", 3
                )  # Slightly smaller for refinement
                max_entries_per_dir = directory_config.get("max_entries_per_dir", 150)
                include_py_only = directory_config.get("include_py_only", True)

                context["project_structure"] = (
                    self._structure_builder.build_tree_recursive(
                        project_root, max_depth, max_entries_per_dir, include_py_only
                    )
                )
            except Exception as e:
                logger.warning("Failed to build project structure context: %s", e)

            return context

        except Exception as e:
            logger.warning("Failed to build source context for %s: %s", test_file, e)
            return None

    def _derive_modules_from_test_ast(self, test_content: str) -> list[str]:
        """
        Use AST to derive likely internal module import paths from the test file content.
        Rules:
        - from X import Y -> add X
        - import X.Y as Z -> add X.Y; import X -> add X
        - Filter out obvious stdlib/pytest/mocks/infra modules
        - Return unique, ordered list
        """
        modules: list[str] = []
        try:
            tree = ast.parse(test_content)
        except Exception as e:
            logger.debug("Failed to parse test content AST: %s", e)
            return modules

        def add_module(mod: str) -> None:
            if not isinstance(mod, str):
                return
            mod = mod.strip()
            if not mod:
                return
            top = mod.split(".")[0]
            filtered = {
                "pytest",
                "unittest",
                "json",
                "re",
                "os",
                "sys",
                "pathlib",
                "typing",
                "datetime",
                "time",
                "collections",
                "itertools",
                "functools",
                "math",
                "rich",
                "logging",
                "schedule",
            }
            if top in filtered:
                return
            if mod not in modules:
                modules.append(mod)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if getattr(node, "level", 0) == 0 and isinstance(
                    getattr(node, "module", None), str
                ):
                    add_module(node.module)
            elif isinstance(node, ast.Import):
                for alias in getattr(node, "names", []) or []:
                    name = getattr(alias, "name", None)
                    if isinstance(name, str):
                        add_module(name)

        # Prefer dotted module paths by stable ordering: dotted first, then single-segment
        dotted = [m for m in modules if "." in m]
        single = [m for m in modules if "." not in m]
        return dotted + single

    def _retrieve_snippets(self, query: str, limit: int = 5) -> list[str]:
        """Retrieve top-ranked symbol-aware snippets."""
        snippet_items = []
        try:
            retrieval = self._context.retrieve(
                query=query, context_type="general", limit=limit
            )

            for item in retrieval.get("results", [])[:limit]:
                # The retriever exposes a `snippet` field; prefer that over content
                if isinstance(item, dict):
                    snippet = item.get("snippet")
                    if isinstance(snippet, str) and snippet.strip():
                        # Enforce a conservative per-snippet cap
                        snippet_items.append(snippet[:400])
        except Exception as e:
            logger.debug("Failed to retrieve snippets: %s", e)
            pass

        return snippet_items

    def _get_neighbor_context(self, source_path: Path | None) -> list[str]:
        """Get import-graph neighbors via ContextPort.get_related_context."""
        neighbor_items = []
        if source_path is not None:
            try:
                related = self._context.get_related_context(
                    source_path, relationship_type="all"
                )
                for related_path in related.get("related_files", [])[:3]:
                    try:
                        p = Path(related_path)
                        if p.exists() and p.suffix == ".py":
                            # Add only a small header + leading lines as context
                            content = p.read_text(encoding="utf-8")
                            header = f"# Related: {p.name}"
                            neighbor_items.append(header)
                            neighbor_items.append(content[:600])
                    except Exception as e:
                        # Ignore read errors for neighbors
                        logger.debug("Failed to read neighbor file %s: %s", p, e)
                        continue
            except Exception as e:
                logger.debug("Failed to get neighbor context: %s", e)
                pass

        return neighbor_items

    def _get_test_exemplars(
        self, source_path: Path | None, plan: TestGenerationPlan
    ) -> list[str]:
        """Extract concise exemplars from existing tests (assertions/fixtures)."""
        exemplar_items = []
        try:
            if source_path is not None:
                existing_tests = self._find_existing_test_files(source_path)
                for test_path_str in existing_tests[:3]:  # limit exemplar sources
                    tp = Path(test_path_str)
                    if not tp.exists() or tp.suffix != ".py":
                        continue
                    content = safe_file_read(tp)
                    if content is None:
                        continue
                    try:
                        tree = safe_ast_parse(content)
                        if tree is None:
                            continue
                        asserts = 0
                        raises = 0
                        fixtures_used: set[str] = set()
                        markers_used: set[str] = set()
                        for node in ast.walk(tree):
                            # Count assertions
                            if isinstance(node, ast.Assert):
                                asserts += 1
                            # Count pytest.raises context managers
                            if isinstance(node, ast.With):
                                for item in node.items:
                                    expr = getattr(item, "context_expr", None)
                                    src = getattr(
                                        getattr(expr, "attr", None), "lower", None
                                    )
                                    if src and getattr(expr, "attr", "") == "raises":
                                        raises += 1
                            # Track fixture arguments (function args)
                            if isinstance(
                                node, ast.FunctionDef
                            ) and node.name.startswith("test_"):
                                for arg in getattr(node.args, "args", [])[:5]:
                                    if isinstance(arg, ast.arg):
                                        fixtures_used.add(arg.arg)
                            # Track pytest.mark.* decorators
                            if isinstance(node, ast.FunctionDef):
                                for dec in getattr(node, "decorator_list", []):
                                    name = getattr(dec, "attr", None) or getattr(
                                        dec, "id", None
                                    )
                                    if isinstance(name, str) and name:
                                        markers_used.add(name)

                        header = (
                            f"# Exemplars from {tp.name}: asserts={asserts}, "
                            f"raises={raises}, fixtures={sorted(fixtures_used)[:5]}, "
                            f"markers={sorted(markers_used)[:5]}"
                        )
                        exemplar_items.append(header[:600])
                    except Exception as e:
                        logger.debug("Failed to process test exemplar: %s", e)
                        continue
        except Exception as e:
            logger.debug("Failed to get test exemplars: %s", e)
            pass

        return exemplar_items

    def _get_contract_context(
        self, source_path: Path | None, plan: TestGenerationPlan
    ) -> list[str]:
        """Extract concise API contracts/invariants for target elements."""
        contract_items = []
        try:
            if source_path is not None:
                parse_result = self._parser.parse_file(source_path)
                tree = parse_result.get("ast")
                source_lines = parse_result.get("source_lines", []) or []
                source_code = "\n".join(source_lines) if source_lines else ""

                for element in plan.elements_to_test[:5]:
                    header = f"# Contract: {getattr(element.type, 'value', str(element.type))} {element.name}"
                    parts = [header]

                    node = self._find_node_for_element(tree, element)
                    signature = self._get_signature(node, element, source_lines)

                    if signature:
                        parts.append(f"signature: {signature[:200]}")

                    # Extract invariants and exceptions from function/method bodies
                    invariants, raises = self._extract_invariants_and_raises(
                        node, source_code
                    )

                    # Docstring parsing
                    doc = getattr(element, "docstring", None) or ""
                    if doc:
                        info = self._parse_docstring(doc)
                    else:
                        info = {
                            "summary": "",
                            "params": [],
                            "returns": "",
                            "raises": [],
                        }

                    if info.get("params"):
                        parts.append(f"params: {list(info['params'])[:8]}")
                    if info.get("returns"):
                        parts.append(f"returns: {str(info['returns'])[:120]}")

                    # Prefer docstring-declared raises; fall back to AST-detected raises
                    doc_raises = info.get("raises") or []
                    if doc_raises or raises:
                        combined_raises = list(
                            dict.fromkeys(list(doc_raises) + raises)
                        )[:8]
                        if combined_raises:
                            parts.append(f"raises: {combined_raises}")
                    if invariants:
                        parts.append("invariants: [" + ", ".join(invariants[:3]) + "]")
                    if info.get("summary"):
                        parts.append(f"doc: {info['summary'][:300]}")

                    contract_items.append("\n".join(parts)[:600])
        except Exception as e:
            logger.debug("Failed to get contract context: %s", e)
            pass

        return contract_items

    def _get_deps_config_fixtures(self, source_path: Path | None) -> list[str]:
        """Detect dependencies/config surfaces and available pytest fixtures."""
        deps_cfg_fixture_items = []
        try:
            if source_path is not None:
                # Get enrichment config and check if features are enabled
                enrichment_cfg = self._config.get("context_enrichment", {})

                # Read source file and parse AST if needed
                try:
                    parse_result = self._parser.parse_file(source_path)
                    src_text = (
                        "\n".join(parse_result.get("source_lines", []))
                        if hasattr(parse_result, "source_lines")
                        else None
                    )
                    ast_tree = (
                        parse_result.get("ast")
                        if hasattr(parse_result, "ast")
                        else None
                    )

                    # Fallback to direct file reading if needed
                    if src_text is None:
                        src_text = safe_file_read(source_path)
                    if ast_tree is None and src_text is not None:
                        ast_tree = safe_ast_parse(src_text)
                except Exception as e:
                    logger.debug("Failed to parse source file %s: %s", source_path, e)
                    src_text, ast_tree = "", None

                # Build summary sections based on enabled features
                summary = ["# Deps/Config/Fixtures"]

                # Environment/config detection
                if enrichment_cfg.get("enable_env_detection", True) and src_text:
                    env_config_data = self._enrichment.detect_env_config_usage(
                        src_text, ast_tree
                    )
                    if env_config_data["env_vars"]:
                        summary.append(f"env_vars: {env_config_data['env_vars']}")
                    if env_config_data["config_keys"]:
                        summary.append(f"config_keys: {env_config_data['config_keys']}")

                # Database/HTTP client boundaries
                if (
                    enrichment_cfg.get("enable_db_boundary_detection", True)
                    or enrichment_cfg.get("enable_http_boundary_detection", True)
                ) and src_text:
                    client_data = self._enrichment.detect_client_boundaries(
                        src_text, ast_tree
                    )
                    if (
                        enrichment_cfg.get("enable_db_boundary_detection", True)
                        and client_data["database_clients"]
                    ):
                        summary.append(f"db_clients: {client_data['database_clients']}")
                    if (
                        enrichment_cfg.get("enable_http_boundary_detection", True)
                        and client_data["http_clients"]
                    ):
                        summary.append(f"http_clients: {client_data['http_clients']}")

                # Comprehensive fixtures discovery
                if enrichment_cfg.get("enable_comprehensive_fixtures", True):
                    project_root = source_path.parent
                    # Find project root by looking for common project markers
                    while project_root.parent != project_root:
                        if any(
                            (project_root / marker).exists()
                            for marker in ["pyproject.toml", "setup.py", ".git"]
                        ):
                            break
                        project_root = project_root.parent

                    fixture_data = self._enrichment.discover_comprehensive_fixtures(
                        project_root
                    )

                    # Format fixtures with scope info where available
                    fixture_lines = []
                    if fixture_data["builtin"]:
                        fixture_lines.append(f"builtin: {fixture_data['builtin']}")
                    if fixture_data["custom"]:
                        custom_with_scope = [
                            f"{name}({scope})"
                            for name, scope in fixture_data["custom"].items()
                        ]
                        fixture_lines.append(f"custom: {custom_with_scope}")
                    if fixture_data["third_party"]:
                        fixture_lines.append(
                            f"third_party: {fixture_data['third_party']}"
                        )

                    if fixture_lines:
                        summary.extend(fixture_lines)

                # Side-effect boundaries
                if (
                    enrichment_cfg.get("enable_side_effect_detection", True)
                    and src_text
                ):
                    side_effect_data = self._enrichment.detect_side_effect_boundaries(
                        src_text, ast_tree
                    )
                    if side_effect_data:
                        for category, effects in side_effect_data.items():
                            if effects:
                                summary.append(f"{category}_effects: {effects}")

                # Keep existing pytest settings from pyproject if present
                pytest_settings = self._get_pytest_settings(source_path)
                if pytest_settings:
                    summary.append(f"pytest_settings: {pytest_settings[:5]}")

                # Add summary block if we have content beyond the header
                if len(summary) > 1:
                    deps_cfg_fixture_items.append("\n".join(summary)[:600])
        except Exception as e:
            logger.debug("Failed to get deps/config/fixtures: %s", e)
            pass

        return deps_cfg_fixture_items

    def _assemble_final_context(self, context_sections: list[list[str]]) -> str | None:
        """Assemble bounded context with deterministic ordering and de-dupe with memory safety."""
        try:
            # Respect section caps from config
            caps = self._config.get("prompt_budgets", {}).get("section_caps", {})
            per_item_cap = int(
                self._config.get("prompt_budgets", {}).get("per_item_chars", 600)
            )
            total_cap = int(
                self._config.get("prompt_budgets", {}).get("total_chars", 4000)
            )
            max_sections = self._config.get(
                "max_context_sections", DEFAULT_MAX_CONTEXT_SECTIONS
            )

            # Early validation: check if we have too many sections
            if len(context_sections) > max_sections:
                logger.warning(
                    f"Too many context sections ({len(context_sections)} > {max_sections}), truncating"
                )
                context_sections = context_sections[:max_sections]

            def _take(items: list[str], key: str) -> list[str]:
                limit = (
                    int(caps.get(key, len(items)))
                    if isinstance(caps, dict)
                    else len(items)
                )
                # Use bounded iteration to prevent memory issues
                out = []
                for it in itertools.islice(
                    items, min(limit, 100)
                ):  # Hard limit per section
                    if isinstance(it, str) and it.strip():
                        # Apply per-item cap
                        capped_item = it[:per_item_cap]
                        out.append(capped_item)
                        # Early termination if we're approaching memory limits
                        if len(out) >= 50:  # Reasonable limit per section type
                            break
                return out

            # Build section keys dynamically from configuration for maintainability
            default_section_order = [
                "snippets",
                "neighbors",
                "test_exemplars",
                "contracts",
                "deps_config_fixtures",
                "coverage_hints",
                "callgraph",
                "error_paths",
                "usage_examples",
                "pytest_settings",
                "side_effects",
            ]

            # Get configured sections and their order (allows for customization)
            section_caps = caps if isinstance(caps, dict) else {}
            configured_sections = set(section_caps.keys()) if section_caps else set()

            # Use configured sections if available, fall back to defaults
            section_keys = []
            for section in default_section_order:
                if not configured_sections or section in configured_sections:
                    section_keys.append(section)

            # Add any additional configured sections not in default order
            for section in configured_sections:
                if section not in section_keys:
                    section_keys.append(section)
                    logger.debug("Adding configured section '%s' to ordering", section)

            # Validate that we have the right number of sections
            if len(context_sections) != len(section_keys):
                raise ValueError(
                    f"Section count mismatch: expected {len(section_keys)} sections {section_keys[:5]}, "
                    f"got {len(context_sections)} context_sections. Context may be corrupted."
                )

            # Track cumulative size to prevent memory exhaustion
            ordered_sections = []
            cumulative_size = 0
            max_cumulative_size = total_cap * 2  # Allow some buffer

            for i, section in enumerate(context_sections):
                if cumulative_size >= max_cumulative_size:
                    logger.warning(
                        f"Approaching memory limit ({cumulative_size} >= {max_cumulative_size}), terminating early"
                    )
                    break

                key = section_keys[i] if i < len(section_keys) else "other"
                section_items = _take(section, key)

                for item in section_items:
                    if cumulative_size >= max_cumulative_size:
                        break
                    ordered_sections.append(item)
                    cumulative_size += len(item)

            # Deduplicate while preserving order
            seen = set()
            ordered = []
            for block in itertools.islice(ordered_sections, 200):  # Limit total items
                if not isinstance(block, str):
                    continue
                key = block.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                ordered.append(block)

            if not ordered:
                return None

            # Apply total cap accounting for separators with size tracking
            total = []
            acc = 0
            separator = "\n\n"

            for i, block in enumerate(
                itertools.islice(ordered, 100)
            ):  # Limit final output
                if acc >= total_cap:
                    break

                # Account for separator length (except for first item)
                sep_len = len(separator) if i > 0 else 0
                available = total_cap - acc - sep_len

                if available <= 0:
                    break

                if len(block) <= available:
                    piece = block
                else:
                    # Ensure we have room for truncation marker
                    marker = "\n# [snipped]"
                    if available > len(marker):
                        piece = block[: available - len(marker)] + marker
                    else:
                        # Not enough room for even a truncated version
                        break

                total.append(piece)
                acc += len(piece) + sep_len

            return separator.join(total) if total else None

        except Exception as e:
            logger.warning(f"Failed to assemble final context: {e}")
            return None

    # Helper methods extracted from original complex methods

    def _find_existing_test_files(self, source_file: Path) -> list[str]:
        """Find existing test files for a source file."""
        existing_tests = []

        # Common test file patterns
        test_patterns = [
            source_file.parent / f"test_{source_file.name}",
            source_file.parent / f"{source_file.stem}_test.py",
            source_file.parent.parent / "tests" / f"test_{source_file.name}",
        ]

        for pattern in test_patterns:
            if pattern.exists():
                existing_tests.append(str(pattern))

        return existing_tests

    def _extract_test_context_queries(
        self, test_file: Path, test_content: str
    ) -> list[str]:
        """Extract intelligent search queries from test file content."""
        queries = []

        try:
            # Parse test content using AST to find test functions and their patterns
            try:
                tree = safe_ast_parse(test_content)
                if tree is None:
                    # Fallback to filename-based queries
                    return [test_file.stem.replace("test_", "").replace("_", " ")]

                # Extract test function names and build queries from them
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith(
                        "test_"
                    ):
                        # Convert test function name to search query
                        clean_name = node.name[5:]  # Remove "test_" prefix
                        query_words = re.findall(r"\w+", clean_name.lower())
                        if query_words:
                            queries.append(" ".join(query_words))

                        # Extract string literals from test body as additional context
                        for child in ast.walk(node):
                            if isinstance(child, ast.Constant) and isinstance(
                                child.value, str
                            ):
                                # Skip very short or very long strings
                                if (
                                    3 <= len(child.value) <= 30
                                    and child.value.replace("_", "").isalnum()
                                ):
                                    queries.append(child.value)

            except SyntaxError as e:
                logger.warning(
                    "Could not parse test file %s for query extraction: %s",
                    test_file,
                    e,
                )

            # Fallback: extract keywords from filename
            if not queries:
                # Convert test_foo_bar.py -> "foo bar"
                test_name = test_file.stem
                if test_name.startswith("test_"):
                    clean_name = test_name[5:]
                    query_words = re.findall(r"\w+", clean_name.lower())
                    if query_words:
                        queries.append(" ".join(query_words))

            # Deduplicate and limit queries
            unique_queries = list(
                dict.fromkeys(queries)
            )  # Preserve order while deduplicating
            return unique_queries[:5]  # Limit to 5 most relevant queries

        except Exception as e:
            logger.warning(
                "Failed to extract context queries from %s: %s", test_file, e
            )
            return [test_file.stem.replace("test_", "").replace("_", " ")]

    def _find_source_files_for_module(
        self, test_file: Path, module_name: str
    ) -> list[Path]:
        """Find potential source files for an imported module."""
        potential_paths = []

        try:
            # Convert module.submodule to file paths
            module_parts = module_name.split(".")

            # Strategy 1: Relative to test file location
            base_dirs = [
                test_file.parent,  # Same directory as test
                test_file.parent.parent,  # Parent directory
            ]

            # Strategy 2: Common project structures
            # If test is in tests/ directory, look in parallel source directories
            if "tests" in test_file.parts:
                tests_index = None
                for i, part in enumerate(test_file.parts):
                    if part == "tests":
                        tests_index = i
                        break

                if tests_index is not None and tests_index > 0:
                    project_root = Path(*test_file.parts[:tests_index])
                    base_dirs.extend(
                        [
                            project_root,  # Project root
                            project_root / test_file.parts[0],  # Main package directory
                            project_root / "src",  # Common src/ directory
                            project_root / "lib",  # Common lib/ directory
                        ]
                    )

            # Strategy 3: Build potential file paths from module name
            for base_dir in base_dirs:
                if not base_dir or not base_dir.exists():
                    continue

                # Direct module file: mymodule -> mymodule.py
                direct_file = base_dir / f"{module_parts[-1]}.py"
                if direct_file.exists():
                    potential_paths.append(direct_file)

                # Package module: mypackage.mymodule -> mypackage/mymodule.py
                if len(module_parts) > 1:
                    package_file = base_dir
                    for part in module_parts:
                        package_file = package_file / part
                    package_file = package_file.with_suffix(".py")
                    if package_file.exists():
                        potential_paths.append(package_file)

                # Package init: mypackage -> mypackage/__init__.py
                package_init = base_dir / module_parts[0] / "__init__.py"
                if package_init.exists():
                    potential_paths.append(package_init)

            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for path in potential_paths:
                path_str = str(path)
                if path_str not in seen:
                    seen.add(path_str)
                    unique_paths.append(path)

            return unique_paths

        except Exception as e:
            logger.warning(
                "Failed to find source files for module %s: %s", module_name, e
            )
            return []

    def _find_node_for_element(self, ast_tree: Any, elem: Any) -> Any | None:
        """Find AST node for a given element."""
        try:
            name = getattr(elem, "name", "")
            start = getattr(elem, "line_range", (0, 0))[0]
            if not ast_tree:
                return None
            # Method: "ClassName.method"
            if "." in name:
                cls_name, meth_name = name.split(".", 1)
                for node in getattr(ast_tree, "body", []):
                    if isinstance(node, ast.ClassDef) and node.name == cls_name:
                        for sub in node.body:
                            if (
                                isinstance(sub, ast.FunctionDef)
                                and sub.name == meth_name
                            ):
                                return sub
            # Function or Class at module level
            for node in getattr(ast_tree, "body", []):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return node
                if isinstance(node, ast.ClassDef) and node.name == name:
                    return node
            # Fallback: match by start line
            for node in ast.walk(ast_tree):
                if hasattr(node, "lineno") and getattr(node, "lineno", -1) == start:
                    return node
        except Exception as e:
            logger.debug("Failed to find node for element: %s", e)
            return None
        return None

    def _get_signature(self, node: Any, element: Any, source_lines: list[str]) -> str:
        """Get signature string for a node."""
        try:
            if isinstance(node, ast.FunctionDef):
                return self._signature_of_function(node, source_lines)
            elif isinstance(node, ast.AsyncFunctionDef):
                sig_core = self._signature_of_function(node, source_lines)
                return ("async " + sig_core) if sig_core else ""
            elif isinstance(node, ast.ClassDef):
                return self._signature_of_class(node, source_lines)
        except Exception as e:
            logger.debug("Failed to get signature: %s", e)
            pass

        # Fallback to first source line
        try:
            start = getattr(element, "line_range", (0, 0))[0]
            if start and start - 1 < len(source_lines):
                return source_lines[start - 1].strip()
        except Exception as e:
            logger.debug("Failed to get fallback signature: %s", e)
            pass

        return ""

    def _signature_of_function(
        self, fn: ast.FunctionDef, source_lines: list[str]
    ) -> str:
        """Build function signature string."""
        try:
            # Simplified signature extraction - would implement full AST signature parsing
            if (
                source_lines
                and hasattr(fn, "lineno")
                and fn.lineno - 1 < len(source_lines)
            ):
                return source_lines[fn.lineno - 1].strip()
        except Exception as e:
            logger.debug("Failed to extract function signature: %s", e)
            pass
        return f"def {fn.name}(...):"

    def _signature_of_class(self, cls: ast.ClassDef, source_lines: list[str]) -> str:
        """Build class signature string."""
        try:
            if (
                source_lines
                and hasattr(cls, "lineno")
                and cls.lineno - 1 < len(source_lines)
            ):
                return source_lines[cls.lineno - 1].strip()
        except Exception as e:
            logger.debug("Failed to extract class signature: %s", e)
            pass
        return f"class {cls.name}:"

    def _extract_invariants_and_raises(
        self, node: Any, source_code: str
    ) -> tuple[list[str], list[str]]:
        """Extract invariants and exceptions from function/method bodies."""
        invariants = []
        raises = []

        try:
            target_fn = (
                node
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                else None
            )
            if target_fn is not None:
                for sub in ast.walk(target_fn):
                    if isinstance(sub, ast.Assert) and len(invariants) < 3:
                        try:
                            expr_txt = (
                                ast.get_source_segment(source_code, sub.test) or ""
                            )
                            if expr_txt:
                                invariants.append(expr_txt[:60])
                        except Exception as e:
                            logger.debug("Failed to extract assert expression: %s", e)
                            pass
                    if isinstance(sub, ast.Raise) and sub.exc is not None:
                        try:
                            exc_txt = ast.get_source_segment(source_code, sub.exc) or ""
                            if exc_txt:
                                # Keep only class/identifier part
                                exc_name = exc_txt.split("(")[0].strip()
                                raises.append(exc_name)
                        except Exception as e:
                            logger.debug("Failed to extract raise expression: %s", e)
                            pass

                # De-duplicate
                raises = list(dict.fromkeys([r for r in raises if r]))[:5]
                invariants = list(dict.fromkeys([i for i in invariants if i]))
        except Exception as e:
            logger.debug("Failed to extract invariants and raises: %s", e)
            pass

        return invariants, raises

    def _parse_docstring(self, doc: str) -> dict[str, Any]:
        """Parse docstring for params, returns, raises information."""
        info = {
            "summary": "",
            "params": [],
            "returns": "",
            "raises": [],
        }
        if not isinstance(doc, str) or not doc.strip():
            return info

        try:
            lines = [line.rstrip() for line in doc.splitlines()]
            # Summary: first non-empty line
            for ln in lines:
                if ln.strip():
                    info["summary"] = re.sub(r"\s+", " ", ln.strip())
                    break

            # Sphinx-style
            for m in re.finditer(r":param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", doc):
                info["params"].append(m.group(1))
            m = re.search(r":return[s]?\s*:\s*(.+)", doc)
            if m:
                info["returns"] = re.sub(r"\s+", " ", m.group(1)).strip()
            for m in re.finditer(r":raise[s]?\s+([A-Za-z_][A-Za-z0-9_]*)\s*:?", doc):
                info["raises"].append(m.group(1))

            # Google-style sections
            def _collect_section(header: str) -> list[str]:
                items = []
                try:
                    idx = next(
                        i
                        for i, line in enumerate(lines)
                        if line.strip().lower().startswith(header)
                    )
                except StopIteration:
                    return items
                for line in lines[idx + 1 : idx + 12]:  # scan a few lines
                    if not line.strip():
                        break
                    m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:\(]", line)
                    if m2:
                        items.append(m2.group(1))
                return items

            if not info["params"]:
                info["params"] = _collect_section("args:") or _collect_section(
                    "parameters:"
                )
            if not info["raises"]:
                info["raises"] = _collect_section("raises:")

            # NumPy-style Returns
            if not info["returns"]:
                try:
                    idx = next(
                        i
                        for i, line in enumerate(lines)
                        if line.strip().lower().startswith("returns")
                    )
                    if idx + 1 < len(lines):
                        info["returns"] = re.sub(r"\s+", " ", lines[idx + 1].strip())
                except StopIteration:
                    pass
        except Exception:
            pass

        return info

    def _get_pytest_settings(self, source_path: Path) -> list[str]:
        """Get pytest settings from pyproject.toml."""
        pytest_settings = []
        try:
            # Find project root
            project_root = source_path.parent
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent

            pyproject = project_root / "pyproject.toml"
            if pyproject.exists():
                try:
                    with open(pyproject, "rb") as f:
                        data = tomllib.load(f)
                    ini_opts = (
                        data.get("tool", {}).get("pytest", {}).get("ini_options", {})
                    )
                    if isinstance(ini_opts, dict):
                        for k, v in ini_opts.items():
                            pytest_settings.append(f"{k}={v}")
                except Exception:
                    pass
        except Exception:
            pass

        return pytest_settings

    def _get_coverage_hints(self, source_path: Path) -> list[str]:
        """Get per-file coverage hints from CoveragePort if available."""
        coverage_hints = []
        try:
            # Check if coverage hints are enabled
            enrichment_cfg = self._config.get("context_enrichment", {})
            if not enrichment_cfg.get("enable_coverage_hints", True):
                return coverage_hints

            # Note: This would need integration with CoverageEvaluator to get per-file data
            # For now, return placeholder that could be wired to actual coverage data
            # Future: Pass coverage data from CoverageEvaluator to context assembler
            pass  # Placeholder for coverage integration
        except Exception:
            pass
        return coverage_hints

    def _get_callgraph_neighbors(self, source_path: Path) -> list[str]:
        """Get call-graph neighbors using ContextPort relationships."""
        callgraph_items = []
        try:
            # Check if callgraph analysis is enabled
            enrichment_cfg = self._config.get("context_enrichment", {})
            if not enrichment_cfg.get("enable_callgraph", True):
                return callgraph_items

            rel = self._context.get_related_context(
                source_path, relationship_type="all"
            )

            # Extract structured relationship information
            relationships = rel.get("relationships", [])
            related_files = rel.get("related_files", [])

            if relationships or related_files:
                edges = []

                # Add relationship edges if available
                if isinstance(relationships, list):
                    edges.extend(str(r)[:100] for r in relationships[:5])

                # Add import neighbors from related files
                for rf in related_files[:3]:
                    try:
                        rf_path = Path(rf)
                        if rf_path.exists() and rf_path.suffix == ".py":
                            edges.append(f"import:{rf_path.name}")
                    except Exception:
                        continue

                if edges:
                    callgraph_items.append(f"# Call-graph edges: {edges[:8]}")
        except Exception:
            pass
        return callgraph_items

    def _get_error_paths(
        self, source_path: Path, plan: TestGenerationPlan
    ) -> list[str]:
        """Get error paths combining docstring analysis with AST scanning."""
        error_items = []
        try:
            # Check if error path analysis is enabled
            enrichment_cfg = self._config.get("context_enrichment", {})
            if not enrichment_cfg.get("enable_error_paths", True):
                return error_items

            # Combine docstring raises with AST-detected exceptions
            docstring_raises = set()
            ast_raises = set()

            # Get raises from element docstrings
            for element in plan.elements_to_test[:3]:
                doc = getattr(element, "docstring", "") or ""
                if doc:
                    doc_info = self._parse_docstring(doc)
                    docstring_raises.update(doc_info.get("raises", []))

            # Get raises from source code AST with safety checks
            try:
                text = safe_file_read(source_path)
                if text is not None:
                    ast_raises.update(
                        re.findall(r"raise\s+([A-Za-z_][A-Za-z0-9_]*)", text)
                    )
            except Exception as e:
                logger.debug("Failed to read source file for raises analysis: %s", e)
                pass

            # Combine and format
            all_raises = sorted(docstring_raises | ast_raises)[:8]
            if all_raises:
                error_items.append(f"# Error paths: {all_raises}")
        except Exception:
            pass
        return error_items

    def _get_usage_examples(
        self, source_path: Path, plan: TestGenerationPlan
    ) -> list[str]:
        """Get usage examples with enhanced module-qualified queries and deduplication."""
        usage_items = []
        try:
            # Check if usage examples are enabled
            enrichment_cfg = self._config.get("context_enrichment", {})
            if not enrichment_cfg.get("enable_usage_examples", True):
                return usage_items

            seen_snippets = set()
            file_snippet_count = {}

            # Derive module path for better import pattern queries
            module_path_info = {}
            if source_path:
                try:
                    module_path_info = ModulePathDeriver.derive_module_path(source_path)
                    logger.debug(
                        "Derived module path for usage examples: %s -> %s",
                        source_path,
                        module_path_info.get("module_path", "none"),
                    )
                except Exception as e:
                    logger.debug(
                        "Could not derive module path for usage examples: %s", e
                    )
                    module_path_info = {}

            # Build enhanced queries from plan elements using module path
            for element in plan.elements_to_test[:3]:
                name = element.name.split(".")[-1]  # Get base name
                module_path = module_path_info.get("module_path", "")

                # Build module-qualified query strategies (prioritized)
                queries = []

                # Strategy 1: Module-qualified import patterns (highest priority)
                if module_path:
                    queries.extend(
                        [
                            f"from {module_path} import {name}",  # Exact module import
                            f"from {module_path} import",  # Module import context
                            f"{module_path}.{name}(",  # Qualified call pattern
                            f"import {module_path}",  # Module import
                        ]
                    )

                # Strategy 2: Fallback to file-based patterns
                queries.extend(
                    [
                        f"from {source_path.stem} import {name}",  # File-based import
                        f"{name}(",  # Function call pattern
                        f"{name} usage",  # Usage context
                        f"{name} example",  # Example usage
                    ]
                )

                # Strategy 3: Class/method specific patterns
                if "." in element.name:
                    class_name = element.name.split(".")[0]
                    queries.extend(
                        [
                            f"{class_name}().{name}(",  # Method call pattern
                            f"{class_name}.{name}",  # Static method pattern
                        ]
                    )

                # Execute queries in priority order
                for query in queries:
                    try:
                        res = self._context.retrieve(
                            query=query, context_type="usage", limit=3
                        )

                        # Look for high-quality usage examples
                        found_good_example = False
                        for item in res.get("results", [])[:2]:
                            if not isinstance(item, dict):
                                continue

                            snippet = item.get("snippet", "")
                            item_path = item.get("path", "unknown")

                            if not snippet or snippet in seen_snippets:
                                continue

                            # Limit snippets per file for diversity
                            file_count = file_snippet_count.get(item_path, 0)
                            if file_count >= 2:
                                continue

                            # Score snippets for quality
                            snippet_score = 0
                            if module_path and module_path in snippet:
                                snippet_score += 3  # Module-qualified usage
                            if "import" in snippet:
                                snippet_score += 2  # Import statements
                            if "(" in snippet and "=" in snippet:
                                snippet_score += 2  # Call with assignment
                            elif "(" in snippet:
                                snippet_score += 1  # Function call

                            # Only include high-quality examples
                            if snippet_score >= 1:
                                # Format with quality indicator
                                quality_indicator = (
                                    "module-qualified"
                                    if snippet_score >= 3
                                    else "standard"
                                )
                                usage_items.append(
                                    f"# Usage {name} ({quality_indicator}): {snippet[:200]}"
                                )
                                seen_snippets.add(snippet)
                                file_snippet_count[item_path] = file_count + 1
                                found_good_example = True

                        # If we found a good example from a high-priority query, move on
                        if (
                            found_good_example
                            and len(queries) > 4
                            and queries.index(query) < 4
                        ):
                            break  # Prioritize module-qualified results

                    except Exception:
                        continue

                # Limit total usage examples
                if len(usage_items) >= 5:
                    break

        except Exception:
            pass
        return usage_items

    def _get_pytest_settings_context(self, source_path: Path) -> list[str]:
        """Get pytest settings context with configuration check."""
        pytest_context = []
        try:
            # Check if pytest settings context is needed based on feature flags
            self._config.get("context_enrichment", {})
            context_cats = self._config.get("context_categories", {})

            if not context_cats.get("pytest_settings", True):
                return pytest_context

            pytest_settings = self._get_pytest_settings(source_path)
            if pytest_settings:
                header = f"# pytest settings: {pytest_settings[:5]}"
                pytest_context.append(header)
        except Exception:
            pass
        return pytest_context

    def _get_side_effects_context(self, source_path: Path) -> list[str]:
        """Get side effects context with configuration check."""
        side_effects_context = []
        try:
            # Check if side effects analysis is enabled
            enrichment_cfg = self._config.get("context_enrichment", {})
            if not enrichment_cfg.get("enable_side_effect_detection", True):
                return side_effects_context

            # Read source text and parse AST if needed for side effect detection
            try:
                parse_result = self._parser.parse_file(source_path)
                src_text = (
                    "\n".join(parse_result.get("source_lines", []))
                    if parse_result.get("source_lines")
                    else None
                )
                ast_tree = parse_result.get("ast") if parse_result else None

                # Fallback to direct file reading if needed
                if src_text is None:
                    src_text = safe_file_read(source_path)
                if ast_tree is None and src_text:
                    ast_tree = safe_ast_parse(src_text)
            except Exception:
                src_text, ast_tree = "", None

            if src_text:
                side_effect_data = self._enrichment.detect_side_effect_boundaries(
                    src_text, ast_tree
                )
                if side_effect_data:
                    summary_parts = []
                    for category, effects in side_effect_data.items():
                        if effects:
                            summary_parts.append(f"{category}_effects: {effects[:3]}")
                    if summary_parts:
                        header = f"# Side effects: {', '.join(summary_parts)}"
                        side_effects_context.append(header[:600])
        except Exception:
            pass
        return side_effects_context

    def _get_path_constraints_context(
        self, source_path: Path, plan: TestGenerationPlan
    ) -> list[str]:
        """Get path constraints context for conditional logic and branching analysis."""
        path_constraints = []
        try:
            # Check if path constraints analysis is enabled
            context_cats = self._config.get("context_categories", {})
            if not context_cats.get("path_constraints", True):
                return path_constraints

            # Analyze conditional branches and path constraints in the source code
            try:
                parse_result = self._parser.parse_file(source_path)
                ast_tree = parse_result.get("ast") if parse_result else None

                if ast_tree:
                    conditions = []
                    branches = []

                    # Walk AST to find conditional logic
                    for node in ast.walk(ast_tree):
                        # If statements
                        if isinstance(node, ast.If):
                            # Extract condition text if possible
                            try:
                                src_text = safe_file_read(source_path)
                                if src_text is not None and hasattr(node, "lineno"):
                                    lines = src_text.split("\n")
                                    if node.lineno - 1 < len(lines):
                                        condition_line = lines[node.lineno - 1].strip()
                                        # Clean up the condition
                                        if condition_line.startswith("if "):
                                            condition = (
                                                condition_line[3:].rstrip(":").strip()
                                            )
                                            conditions.append(condition[:100])
                            except Exception:
                                conditions.append("conditional_branch")

                        # Match statements (Python 3.10+)
                        elif isinstance(node, ast.Match):
                            branches.append("match_statement")

                        # Try/except blocks
                        elif isinstance(node, ast.Try):
                            for handler in node.handlers:
                                if handler.type:
                                    try:
                                        exc_name = (
                                            handler.type.id
                                            if hasattr(handler.type, "id")
                                            else str(handler.type)
                                        )
                                        branches.append(f"except_{exc_name}")
                                    except Exception:
                                        branches.append("except_clause")

                    # Format path constraints summary
                    summary_parts = []
                    if conditions:
                        summary_parts.append(f"conditions: {conditions[:5]}")
                    if branches:
                        summary_parts.append(f"branches: {branches[:3]}")

                    if summary_parts:
                        header = f"# Path constraints: {', '.join(summary_parts)}"
                        path_constraints.append(header[:600])

            except Exception:
                # Fallback: basic conditional detection via text patterns
                try:
                    src_text = safe_file_read(source_path)
                    if src_text is not None:
                        if_count = src_text.count("if ")
                        elif_count = src_text.count("elif ")
                        try_count = src_text.count("try:")
                        match_count = src_text.count("match ")

                        if if_count + elif_count + try_count + match_count > 0:
                            summary = f"# Path constraints: if={if_count}, elif={elif_count}, try={try_count}, match={match_count}"
                            path_constraints.append(summary)
                except Exception:
                    pass

        except Exception:
            pass
        return path_constraints

    def _build_enriched_context_for_generation(
        self,
        source_path: Path | None,
        base_context: str | None,
        import_map: dict[str, Any] | ImportMap | None = None,
    ) -> str | None:
        """
        Build enriched context with packaging and safety information.

        Args:
            source_path: Path to the source file being tested
            base_context: Base context from traditional context assembly
            import_map: Import mapping information from ImportResolver

        Returns:
            Enhanced context string with packaging and safety information
        """
        if source_path is None:
            return base_context

        try:
            # Build enriched context using the enhanced context builder
            enriched_context = self._enhanced_context_builder.build_enriched_context(
                source_file=source_path,
                existing_context=base_context,
            )

            # Add import map information to enriched context if available
            if import_map is not None:
                import_context_lines = []

                # Handle both dict and ImportMap objects
                if isinstance(import_map, dict):
                    target_import = import_map.get("target_import")
                    sys_path_roots = import_map.get("sys_path_roots")
                    needs_bootstrap = import_map.get("needs_bootstrap")
                    bootstrap_conftest = import_map.get("bootstrap_conftest")
                else:
                    # Assume it's an ImportMap object
                    target_import = getattr(import_map, "target_import", None)
                    sys_path_roots = getattr(import_map, "sys_path_roots", None)
                    needs_bootstrap = getattr(import_map, "needs_bootstrap", None)
                    bootstrap_conftest = getattr(import_map, "bootstrap_conftest", None)

                # Add canonical import line
                if target_import:
                    import_context_lines.append(f"# Canonical import: {target_import}")

                # Add sys.path roots information
                if sys_path_roots:
                    import_context_lines.append(f"# Sys.path roots: {sys_path_roots}")

                # Add bootstrap requirements
                if needs_bootstrap:
                    import_context_lines.append(
                        "# Bootstrap: conftest.py setup required"
                    )
                    if bootstrap_conftest:
                        import_context_lines.append("# Bootstrap content available")

                # Prepend import context to enriched context if we have it
                if import_context_lines:
                    import_context_str = "\n".join(import_context_lines)
                    if enriched_context and "context" in enriched_context:
                        enriched_context["context"] = (
                            f"{import_context_str}\n\n{enriched_context['context']}"
                        )
                    elif base_context:
                        # Fallback: prepend to base context
                        base_context = f"{import_context_str}\n\n{base_context}"

            # Format for LLM consumption
            formatted_context = self._enhanced_context_builder.format_for_llm(
                enriched_context
            )

            # Store enriched context for potential use in validation
            if hasattr(self, "_last_enriched_context"):
                self._last_enriched_context = enriched_context
            else:
                # Add as instance variable for access by other methods
                self._last_enriched_context = enriched_context

            return formatted_context if formatted_context else base_context

        except Exception as e:
            logger.warning(
                "Failed to build enriched context for %s: %s", source_path, e
            )
            return base_context

    def get_last_enriched_context(self) -> dict[str, Any] | None:
        """
        Get the last enriched context built for validation purposes.

        Returns:
            Dictionary with enriched context information or None
        """
        return getattr(self, "_last_enriched_context", None)
