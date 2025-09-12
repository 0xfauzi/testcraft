"""
Content builder service for test generation.

Handles source code content extraction from test generation plans
and test file path determination with per-run caching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....domain.models import TestGenerationPlan
from ....ports.parser_port import ParserPort

logger = logging.getLogger(__name__)


class ContentBuilder:
    """
    Service for building test content and determining test file paths.

    Provides source code extraction from TestGenerationPlan elements
    with import header inclusion and size capping, plus test path determination.
    """

    def __init__(self, parser_port: ParserPort):
        """
        Initialize the content builder.

        Args:
            parser_port: Port for code parsing operations
        """
        self._parser = parser_port
        # Lightweight per-run cache to avoid re-parsing files
        self._parse_cache: dict[Path, dict[str, Any]] = {}

    def build_code_content(
        self, plan: TestGenerationPlan, source_path: Path | None = None
    ) -> str:
        """
        Build combined code content from test elements in the plan.

        This implementation pulls the actual source code for each element from the
        parser, and includes import lines from the module so the LLM has full
        context to generate runnable tests. Content size is capped to avoid
        excessive prompt length.

        Args:
            plan: The test generation plan
            source_path: Optional source file path for the plan

        Returns:
            Combined code content string for LLM consumption
        """
        code_parts: list[str] = []

        source_lines: list[str] | None = None
        source_content_map: dict[str, str] = {}

        if source_path is not None:
            try:
                parse_result = self._get_parse_result_cached(source_path)
                source_lines = parse_result.get("source_lines", [])
                source_content_map = parse_result.get("source_content", {}) or {}

                # Add import lines at the top to provide dependency context
                import_lines: list[str] = []
                for line in source_lines:
                    stripped = line.strip()
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        import_lines.append(line)
                if import_lines:
                    code_parts.append("# Module imports for context")
                    code_parts.extend(import_lines[:100])  # cap number of import lines
                    code_parts.append("")
            except Exception as e:
                logger.debug("Failed to parse source file for plan: %s", e)

        # Cap sizes to keep prompt manageable
        PER_ELEMENT_CHAR_LIMIT = 4000
        TOTAL_CHAR_LIMIT = 50000

        # Append actual source for each requested element
        total_chars = sum(len(p) for p in code_parts)
        for element in plan.elements_to_test:
            if total_chars >= TOTAL_CHAR_LIMIT:
                code_parts.append(
                    "\n# [Truncated additional elements due to size limits]"
                )
                break

            header = (
                f"# {getattr(element.type, 'value', str(element.type))}: {element.name}"
            )
            code_parts.append(header)

            element_source = ""
            # Prefer exact source captured during parsing
            if element.name in source_content_map:
                element_source = source_content_map.get(element.name, "")
            # Fallback: slice from source_lines using recorded line_range
            elif source_lines is not None:
                try:
                    start, end = element.line_range
                    # line_range is inclusive
                    snippet = "\n".join(source_lines[start - 1 : end])
                    element_source = snippet
                except Exception:
                    element_source = ""

            # Final fallback to prior minimal stub if we couldn't recover source
            if not element_source:
                if element.docstring:
                    element_source = f'"""{element.docstring}"""'
                else:
                    element_source = "pass"

            # Enforce per-element cap
            if len(element_source) > PER_ELEMENT_CHAR_LIMIT:
                element_source = (
                    element_source[:PER_ELEMENT_CHAR_LIMIT] + "\n# [snipped]"
                )

            code_parts.append(element_source)
            code_parts.append("")
            total_chars += len(header) + 1 + len(element_source)

        return "\n".join(code_parts)

    def determine_test_path(self, plan: TestGenerationPlan) -> str:
        """
        Determine the output path for the test file.

        Args:
            plan: The test generation plan

        Returns:
            String path for the output test file
        """
        # Simplified implementation - would use actual source file paths from plan
        if plan.elements_to_test:
            # Use first element to determine naming
            element = plan.elements_to_test[0]
            return f"tests/test_{element.name.lower()}.py"
        return "tests/test_generated.py"

    def _get_parse_result_cached(self, file_path: Path) -> dict[str, Any]:
        """
        Get parse result with per-run caching to avoid repeated parsing.

        Args:
            file_path: Path to file to parse

        Returns:
            Parse result dictionary
        """
        if file_path not in self._parse_cache:
            self._parse_cache[file_path] = self._parser.parse_file(file_path)
        return self._parse_cache[file_path]

    def clear_cache(self):
        """Clear the parse cache for new generation runs."""
        self._parse_cache.clear()
