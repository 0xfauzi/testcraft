from __future__ import annotations

from pathlib import Path
from typing import Any

from .....domain.models import TestGenerationPlan
from ..structure import ModulePathDeriver
from .....ports.context_port import ContextPort


def get_usage_examples(context_port: ContextPort, source_path: Path, plan: TestGenerationPlan) -> list[str]:
    items: list[str] = []
    try:
        seen_snippets: set[str] = set()
        file_snippet_count: dict[str, int] = {}

        module_path_info: dict[str, Any] = {}
        try:
            module_path_info = ModulePathDeriver.derive_module_path(source_path)
        except Exception:
            module_path_info = {}

        for element in plan.elements_to_test[:3]:
            name = element.name.split(".")[-1]
            module_path = module_path_info.get("module_path", "")

            queries: list[str] = []
            if module_path:
                queries.extend(
                    [
                        f"from {module_path} import {name}",
                        f"from {module_path} import",
                        f"{module_path}.{name}(",
                        f"import {module_path}",
                    ]
                )
            queries.extend(
                [
                    f"from {source_path.stem} import {name}",
                    f"{name}(",
                    f"{name} usage",
                    f"{name} example",
                ]
            )
            if "." in element.name:
                class_name = element.name.split(".")[0]
                queries.extend([f"{class_name}().{name}(", f"{class_name}.{name}"])

            for query in queries:
                try:
                    res = context_port.retrieve(query=query, context_type="usage", limit=3)
                    found_good_example = False
                    for item in res.get("results", [])[:2]:
                        if not isinstance(item, dict):
                            continue
                        snippet = item.get("snippet", "")
                        item_path = item.get("path", "unknown")
                        if not snippet or snippet in seen_snippets:
                            continue
                        file_count = file_snippet_count.get(item_path, 0)
                        if file_count >= 2:
                            continue
                        snippet_score = 0
                        if module_path and module_path in snippet:
                            snippet_score += 3
                        if "import" in snippet:
                            snippet_score += 2
                        if "(" in snippet and "=" in snippet:
                            snippet_score += 2
                        elif "(" in snippet:
                            snippet_score += 1
                        if snippet_score >= 1:
                            quality_indicator = "module-qualified" if snippet_score >= 3 else "standard"
                            items.append(f"# Usage {name} ({quality_indicator}): {snippet[:200]}")
                            seen_snippets.add(snippet)
                            file_snippet_count[item_path] = file_count + 1
                            found_good_example = True
                            break
                    if found_good_example and len(queries) > 4 and queries.index(query) < 4:
                        break
                except Exception:
                    continue
            if len(items) >= 5:
                break
    except Exception:
        pass
    return items



