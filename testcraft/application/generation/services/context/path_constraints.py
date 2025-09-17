from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .....domain.models import TestGenerationPlan


def get_path_constraints_context(parser_port: Any, config: dict, source_path: Path, plan: TestGenerationPlan) -> list[str]:
    items: list[str] = []
    try:
        context_cats = config.get("context_categories", {})
        if not context_cats.get("path_constraints", True):
            return items
        try:
            parse_result = parser_port.parse_file(source_path)
            ast_tree = parse_result.get("ast") if parse_result else None
            if ast_tree:
                conditions: list[str] = []
                branches: list[str] = []
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.If):
                        try:
                            src_text = source_path.read_text(encoding="utf-8")
                            if hasattr(node, "lineno"):
                                lines = src_text.split("\n")
                                if node.lineno - 1 < len(lines):
                                    condition_line = lines[node.lineno - 1].strip()
                                    if condition_line.startswith("if "):
                                        condition = condition_line[3:].rstrip(":").strip()
                                        conditions.append(condition[:100])
                        except Exception:
                            conditions.append("conditional_branch")
                    elif isinstance(node, ast.Match):
                        branches.append("match_statement")
                    elif isinstance(node, ast.Try):
                        for handler in node.handlers:
                            if handler.type:
                                try:
                                    exc_name = handler.type.id if hasattr(handler.type, 'id') else str(handler.type)
                                    branches.append(f"except_{exc_name}")
                                except Exception:
                                    branches.append("except_clause")
                summary_parts: list[str] = []
                if conditions:
                    summary_parts.append(f"conditions: {conditions[:5]}")
                if branches:
                    summary_parts.append(f"branches: {branches[:3]}")
                if summary_parts:
                    items.append(f"# Path constraints: {', '.join(summary_parts)}")
        except Exception:
            try:
                src_text = source_path.read_text(encoding="utf-8")
                if_count = src_text.count("if ")
                elif_count = src_text.count("elif ")
                try_count = src_text.count("try:")
                match_count = src_text.count("match ")
                if if_count + elif_count + try_count + match_count > 0:
                    items.append(
                        f"# Path constraints: if={if_count}, elif={elif_count}, try={try_count}, match={match_count}"
                    )
            except Exception:
                pass
    except Exception:
        pass
    return items



