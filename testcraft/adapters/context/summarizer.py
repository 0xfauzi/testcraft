"""
Summarization utilities for files and directory structures.

Focuses on:
- Generating a compact directory tree (bounded breadth/depth)
- Extracting top-level imports and class/function signatures
- Enforcing max character budgets for summaries
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


@dataclass
class SignatureSummary:
    classes: list[str]
    functions: list[str]


class ContextSummarizer:
    def summarize_file(
        self,
        path: Path,
        *,
        content: str | None = None,
        summary_type: str = "comprehensive",
        max_chars: int = 2000,
    ) -> dict[str, Any]:
        if content is None:
            if not path.exists():
                return {
                    "summary": "",
                    "key_functions": [],
                    "key_classes": [],
                    "dependencies": [],
                    "summary_metadata": {"error": f"File not found: {str(path)}"},
                }
            content = path.read_text(encoding="utf-8")

        imports = self._extract_imports(content)
        sigs = self._extract_signatures(content)

        summary_lines: list[str] = []
        summary_lines.append(f"File: {path.name}")
        if imports:
            summary_lines.append("Imports: " + ", ".join(sorted(set(imports))[:10]))
        if sigs.classes:
            summary_lines.append("Classes: " + ", ".join(sigs.classes[:10]))
        if sigs.functions:
            summary_lines.append("Functions: " + ", ".join(sigs.functions[:10]))

        return {
            "summary": _truncate("\n".join(summary_lines), max_chars),
            "key_functions": sigs.functions,
            "key_classes": sigs.classes,
            "dependencies": sorted(set(imports)),
            "summary_metadata": {"summary_type": summary_type},
        }

    def summarize_directory_tree(
        self,
        root: Path,
        *,
        max_depth: int = 3,
        max_breadth: int = 10,
        max_chars: int = 4000,
    ) -> str:
        def walk(dir_path: Path, depth: int) -> list[str]:
            if depth > max_depth:
                return []
            entries = sorted(
                [p for p in dir_path.iterdir() if not p.name.startswith(".")]
            )[:max_breadth]
            lines: list[str] = []
            for entry in entries:
                indent = "  " * (depth - 1)
                prefix = "- "
                lines.append(f"{indent}{prefix}{entry.name}")
                if entry.is_dir():
                    lines.extend(walk(entry, depth + 1))
            return lines

        if not root.exists():
            return ""
        tree = [root.name]
        tree.extend(walk(root, 1))
        return _truncate("\n".join(tree), max_chars)

    def _extract_imports(self, content: str) -> list[str]:
        try:
            imports: list[str] = []
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split(".")[0])
            return imports
        except Exception:
            return []

    def _extract_signatures(self, content: str) -> SignatureSummary:
        classes: list[str] = []
        functions: list[str] = []
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except Exception:
            pass
        return SignatureSummary(classes=classes, functions=functions)
