from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .....domain.models import TestGenerationPlan
from .contracts import _parse_docstring


def get_error_paths(parser_port: Any, config: dict, source_path: Path, plan: TestGenerationPlan) -> list[str]:
    items: list[str] = []
    try:
        enrichment_cfg = config.get("context_enrichment", {})
        if not enrichment_cfg.get("enable_error_paths", True):
            return items
        docstring_raises: set[str] = set()
        ast_raises: set[str] = set()
        for element in plan.elements_to_test[:3]:
            doc = getattr(element, "docstring", "") or ""
            if doc:
                doc_info = _parse_docstring(doc)
                docstring_raises.update(doc_info.get("raises", []))
        try:
            text = source_path.read_text(encoding="utf-8")
            ast_raises.update(re.findall(r"raise\s+([A-Za-z_][A-Za-z0-9_]*)", text))
        except Exception:
            pass
        all_raises = sorted(list(docstring_raises | ast_raises))[:8]
        if all_raises:
            items.append(f"# Error paths: {all_raises}")
    except Exception:
        pass
    return items



