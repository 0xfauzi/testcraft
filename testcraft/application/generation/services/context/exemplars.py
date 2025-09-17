from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .....domain.models import TestGenerationPlan


def _find_existing_test_files(source_file: Path) -> list[str]:
    existing_tests: list[str] = []
    patterns = [
        source_file.parent / f"test_{source_file.name}",
        source_file.parent / f"{source_file.stem}_test.py",
        source_file.parent.parent / "tests" / f"test_{source_file.name}",
    ]
    for pattern in patterns:
        if pattern.exists():
            existing_tests.append(str(pattern))
    return existing_tests


def get_test_exemplars(parser_port: Any, source_path: Path | None, plan: TestGenerationPlan) -> list[str]:
    items: list[str] = []
    try:
        if source_path is None:
            return items
        existing_tests = _find_existing_test_files(source_path)
        for test_path_str in existing_tests[:3]:
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
                    if isinstance(node, ast.Assert):
                        asserts += 1
                    if isinstance(node, ast.With):
                        for item in node.items:
                            expr = getattr(item, "context_expr", None)
                            if getattr(expr, "attr", "") == "raises":
                                raises += 1
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        for arg in getattr(node.args, "args", [])[:5]:
                            if isinstance(arg, ast.arg):
                                fixtures_used.add(arg.arg)
                    if isinstance(node, ast.FunctionDef):
                        for dec in node.decorator_list:
                            name = getattr(dec, "attr", None) or getattr(dec, "id", None)
                            if isinstance(name, str) and name:
                                markers_used.add(name)
                header = (
                    f"# Exemplars from {tp.name}: asserts={asserts}, "
                    f"raises={raises}, fixtures={sorted(list(fixtures_used))[:5]}, "
                    f"markers={sorted(list(markers_used))[:5]}"
                )
                items.append(header[:600])
            except Exception:
                continue
    except Exception:
        pass
    return items



