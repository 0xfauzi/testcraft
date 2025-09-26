"""
Context assembler service for test generation and refinement.

Unifies context building logic for both generation and refinement workflows,
including project context gathering, snippet retrieval, and enrichment integration.
"""

from __future__ import annotations

import ast
import logging
import re
import tomllib
from pathlib import Path
from typing import Any

from ....domain.models import ContextPack, ImportMap, TestGenerationPlan
from ....ports.context_port import ContextPort
from ....ports.parser_port import ParserPort
from .enhanced_context_builder import EnrichedContextBuilder
from .enrichment_detectors import EnrichmentDetectors
from .import_resolver import ImportResolver
from .structure import DirectoryTreeBuilder, ModulePathDeriver

logger = logging.getLogger(__name__)


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

        try:
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
            # Build lightweight context query from top plan elements
            query_parts = [element.name for element in plan.elements_to_test[:3]]
            query = " ".join(query_parts)

            # 1) Retrieve top-ranked symbol-aware snippets
            snippet_items = self._retrieve_snippets(query, limit=5)

            # 2) Merge import-graph neighbors via ContextPort.get_related_context
            neighbor_items = self._get_neighbor_context(source_path)

            # 3) Extract concise exemplars from existing tests
            exemplar_items = self._get_test_exemplars(source_path, plan)

            # 4) Extract concise API contracts/invariants for target elements
            contract_items = self._get_contract_context(source_path, plan)

            # 5) Detect dependencies/config surfaces and available pytest fixtures
            deps_cfg_fixture_items = self._get_deps_config_fixtures(source_path)

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

            # 7) Assemble bounded context with deterministic ordering and de-dupe
            # 8) Build enriched context with packaging and safety information
            base_context = self._assemble_final_context(
                [
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
            )

            # 9) Enhance with packaging and safety information
            enriched_context_string = self._build_enriched_context_for_generation(
                source_path, base_context, import_map
            )

            # 10) Build complete ContextPack with all components
            # We can proceed even if import_map is None (e.g., for files without proper package structure)
            if enriched_context_string is not None:
                # Create a minimal ContextPack for context provision
                # TODO: This should be enhanced to build full ContextPack with all components
                from ....domain.models import (
                    Budget,
                    Conventions,
                    Focal,
                    PropertyContext,
                    Target,
                )

                # For now, create a basic ContextPack with available information
                # Full ContextPack building is handled by ContextPackBuilder
                target = Target(
                    module_file=str(source_path) if source_path else "unknown",
                    object=plan.elements_to_test[0].name
                    if plan.elements_to_test
                    else "unknown",
                )

                focal = Focal(
                    source=plan.elements_to_test[0].name
                    if plan.elements_to_test
                    else "unknown",
                    signature="def "
                    + (
                        plan.elements_to_test[0].name.split(".")[-1]
                        if plan.elements_to_test
                        else "unknown"
                    )
                    + "(...):",
                    docstring=plan.elements_to_test[0].docstring
                    if plan.elements_to_test
                    else None,
                )

                context_pack = ContextPack(
                    target=target,
                    import_map=import_map,  # Can be None if import resolution failed
                    focal=focal,
                    resolved_defs=[],  # TODO: Build resolved definitions
                    property_context=PropertyContext(),  # TODO: Build property context
                    conventions=Conventions(),
                    budget=Budget(),
                    context=enriched_context_string,
                )

                return context_pack

            # Return None if we don't have sufficient information for a ContextPack
            return None

        except Exception as e:
            logger.warning("Failed to retrieve context: %s", e)
            return None

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
                    except Exception:
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
                                source_content = related_path.read_text(
                                    encoding="utf-8"
                                )
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
                                    source_content = source_path.read_text(
                                        encoding="utf-8"
                                    )
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
        except Exception:
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
        except Exception:
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
                    except Exception:
                        # Ignore read errors for neighbors
                        continue
            except Exception:
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
                    content = tp.read_text(encoding="utf-8")
                    try:
                        tree = ast.parse(content)
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
                                for dec in node.decorator_list:
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
                    except Exception:
                        continue
        except Exception:
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
        except Exception:
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
                        src_text = source_path.read_text(encoding="utf-8")
                    if ast_tree is None:
                        ast_tree = ast.parse(src_text) if src_text else None
                except Exception:
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
        except Exception:
            pass

        return deps_cfg_fixture_items

    def _assemble_final_context(self, context_sections: list[list[str]]) -> str | None:
        """Assemble bounded context with deterministic ordering and de-dupe."""
        # Respect section caps from config
        caps = self._config.get("prompt_budgets", {}).get("section_caps", {})
        per_item_cap = int(
            self._config.get("prompt_budgets", {}).get("per_item_chars", 600)
        )
        total_cap = int(self._config.get("prompt_budgets", {}).get("total_chars", 4000))

        def _take(items: list[str], key: str) -> list[str]:
            limit = (
                int(caps.get(key, len(items))) if isinstance(caps, dict) else len(items)
            )
            out = []
            for it in items[:limit]:
                if isinstance(it, str) and it.strip():
                    out.append(it[:per_item_cap])
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
            logger.warning(
                "Section count mismatch: expected %d sections %s, got %d context_sections. "
                "Some context may be miscategorized.",
                len(section_keys),
                section_keys[:5],
                len(context_sections),
            )

        ordered_sections = []
        for i, section in enumerate(context_sections):
            key = section_keys[i] if i < len(section_keys) else "other"
            ordered_sections.extend(_take(section, key))

        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for block in ordered_sections:
            if not isinstance(block, str):
                continue
            key = block.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(block)

        if not ordered:
            return None

        # Apply total cap accounting for separators
        total = []
        acc = 0
        separator = "\n\n"

        for i, block in enumerate(ordered):
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
                tree = ast.parse(test_content)

                # Extract test function names and build queries from them
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith(
                        "test_"
                    ):
                        # Convert test function name to search query
                        clean_name = node.name[5:]  # Remove "test_" prefix
                        query_words = re.findall(r"[a-z]+", clean_name.lower())
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
                    query_words = re.findall(r"[a-z]+", clean_name.lower())
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
        except Exception:
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
        except Exception:
            pass

        # Fallback to first source line
        try:
            start = getattr(element, "line_range", (0, 0))[0]
            if start and start - 1 < len(source_lines):
                return source_lines[start - 1].strip()
        except Exception:
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
        except Exception:
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
        except Exception:
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
                        except Exception:
                            pass
                    if isinstance(sub, ast.Raise) and sub.exc is not None:
                        try:
                            exc_txt = ast.get_source_segment(source_code, sub.exc) or ""
                            if exc_txt:
                                # Keep only class/identifier part
                                exc_name = exc_txt.split("(")[0].strip()
                                raises.append(exc_name)
                        except Exception:
                            pass

                # De-duplicate
                raises = list(dict.fromkeys([r for r in raises if r]))[:5]
                invariants = list(dict.fromkeys([i for i in invariants if i]))
        except Exception:
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

            # Get raises from source code AST
            try:
                text = source_path.read_text(encoding="utf-8")
                ast_raises.update(re.findall(r"raise\s+([A-Za-z_][A-Za-z0-9_]*)", text))
            except Exception:
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
                    src_text = source_path.read_text(encoding="utf-8")
                if ast_tree is None and src_text:
                    ast_tree = ast.parse(src_text)
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
                                src_text = source_path.read_text(encoding="utf-8")
                                if hasattr(node, "lineno"):
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
                    src_text = source_path.read_text(encoding="utf-8")
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
