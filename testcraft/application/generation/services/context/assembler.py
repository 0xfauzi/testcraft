from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .....domain.models import TestGenerationPlan
from .....ports.context_port import ContextPort
from .....ports.parser_port import ParserPort

from .retrieval import retrieve_snippets, get_neighbor_context
from .exemplars import get_test_exemplars
from .contracts import get_contract_context
from .deps_fixtures import get_deps_config_fixtures
from .coverage_hints import get_coverage_hints
from .neighbors import get_callgraph_neighbors
from .error_paths import get_error_paths
from .usage_examples import get_usage_examples
from .pytest_settings import get_pytest_settings_context
from .side_effects import get_side_effects_context
from .path_constraints import get_path_constraints_context
from .assemble import assemble_final_context
from ..enhanced_context_builder import EnrichedContextBuilder
from ..structure import DirectoryTreeBuilder


logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Facade for assembling context for test generation and refinement.

    Thin orchestrator delegating to cohesive context modules.
    """

    def __init__(
        self,
        context_port: ContextPort,
        parser_port: ParserPort,
        config: dict[str, Any],
    ) -> None:
        self._context = context_port
        self._parser = parser_port
        self._config = config

        self._structure_builder = DirectoryTreeBuilder()
        self._enhanced_context_builder = EnrichedContextBuilder()

    def gather_project_context(
        self, project_path: Path, files_to_process: list[Path]
    ) -> dict[str, Any]:
        try:
            context_graph = self._context.build_context_graph(project_path)

            indexed_files: dict[str, Any] = {}
            for file_path in files_to_process:
                try:
                    index_result = self._context.index(file_path)
                    indexed_files[str(file_path)] = index_result
                except Exception as e:  # pragma: no cover - warning path
                    logger.warning("Failed to index %s: %s", file_path, e)

            directory_config = self._config.get("context_budgets", {}).get(
                "directory_tree", {}
            )
            max_depth = directory_config.get("max_depth", 4)
            max_entries_per_dir = directory_config.get("max_entries_per_dir", 200)
            include_py_only = directory_config.get("include_py_only", True)

            return {
                "context_graph": context_graph,
                "indexed_files": indexed_files,
                "project_structure": self._structure_builder.build_tree_recursive(
                    project_path, max_depth, max_entries_per_dir, include_py_only
                ),
            }
        except Exception as e:  # pragma: no cover - warning path
            logger.warning("Failed to gather project context: %s", e)
            return {}

    def context_for_generation(
        self, plan: TestGenerationPlan, source_path: Path | None = None
    ) -> str | None:
        if not self._config.get("enable_context", True):
            return None

        try:
            query_parts = [element.name for element in plan.elements_to_test[:3]]
            query = " ".join(query_parts)

            snippet_items = retrieve_snippets(self._context, query, limit=5)
            neighbor_items = get_neighbor_context(self._context, source_path)
            exemplar_items = get_test_exemplars(self._parser, source_path, plan)
            contract_items = get_contract_context(self._parser, source_path, plan)
            deps_cfg_fixture_items = get_deps_config_fixtures(
                self._parser, self._config, source_path
            )

            coverage_hints = (
                get_coverage_hints(self._config, source_path) if source_path else []
            )
            callgraph_items = (
                get_callgraph_neighbors(self._context, self._config, source_path)
                if source_path
                else []
            )
            error_items = (
                get_error_paths(self._parser, self._config, source_path, plan)
                if source_path
                else []
            )
            usage_items = (
                get_usage_examples(self._context, source_path, plan) if source_path else []
            )
            pytest_settings = (
                get_pytest_settings_context(self._config, source_path)
                if source_path
                else []
            )
            side_effects = (
                get_side_effects_context(self._parser, self._config, source_path)
                if source_path
                else []
            )
            path_constraints = (
                get_path_constraints_context(self._parser, self._config, source_path, plan)
                if source_path
                else []
            )

            base_context = assemble_final_context(
                self._config,
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
                ],
            )

            return self._build_enriched_context_for_generation(source_path, base_context)
        except Exception as e:  # pragma: no cover - warning path
            logger.warning("Failed to retrieve context: %s", e)
            return None

    def context_for_refinement(self, test_file: Path, test_content: str) -> dict[str, Any] | None:
        try:
            from .refine_source import build_refinement_context  # lazy import to avoid cycles

            base = build_refinement_context(
                parser=self._parser,
                context_port=self._context,
                config=self._config,
                test_file=test_file,
                test_content=test_content,
                structure_builder=self._structure_builder,
            )
            return base
        except Exception as e:  # pragma: no cover - warning path
            logger.warning("Failed to build source context for %s: %s", test_file, e)
            return None

    def _build_enriched_context_for_generation(
        self, source_path: Path | None, base_context: str | None
    ) -> str | None:
        if source_path is None:
            return base_context
        try:
            enriched = self._enhanced_context_builder.build_enriched_context(
                source_file=source_path, existing_context=base_context
            )
            formatted = self._enhanced_context_builder.format_for_llm(enriched)
            self._last_enriched_context = enriched
            return formatted if formatted else base_context
        except Exception as e:  # pragma: no cover - warning path
            logger.warning("Failed to build enriched context for %s: %s", source_path, e)
            return base_context

    def get_last_enriched_context(self) -> dict[str, Any] | None:
        return getattr(self, "_last_enriched_context", None)

    # Private methods expected by tests - delegate to modular functions
    def _get_coverage_hints(self, source_path: Path) -> list[str]:
        return get_coverage_hints(self._config, source_path)

    def _get_callgraph_neighbors(self, source_path: Path) -> list[str]:
        return get_callgraph_neighbors(self._context, self._config, source_path)

    def _get_error_paths(self, source_path: Path, plan: TestGenerationPlan) -> list[str]:
        return get_error_paths(self._parser, self._config, source_path, plan)

    def _get_usage_examples(self, source_path: Path, plan: TestGenerationPlan) -> list[str]:
        return get_usage_examples(self._context, source_path, plan)

    def _get_pytest_settings_context(self, source_path: Path) -> list[str]:
        return get_pytest_settings_context(self._config, source_path)

    def _get_side_effects_context(self, source_path: Path) -> list[str]:
        return get_side_effects_context(self._parser, self._config, source_path)

    def _assemble_final_context(self, context_sections: list[list[str]]) -> str:
        return assemble_final_context(self._config, context_sections)



