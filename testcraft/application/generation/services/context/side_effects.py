from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..enrichment_detectors import EnrichmentDetectors


def get_side_effects_context(parser_port: Any, config: dict, source_path: Path) -> list[str]:
    items: list[str] = []
    try:
        enrichment_cfg = config.get("context_enrichment", {})
        if not enrichment_cfg.get("enable_side_effect_detection", True):
            return items
        try:
            parse_result = parser_port.parse_file(source_path)
            src_text = (
                "\n".join(parse_result.get("source_lines", []))
                if parse_result.get("source_lines")
                else None
            )
            ast_tree = parse_result.get("ast") if parse_result else None
            if src_text is None:
                src_text = source_path.read_text(encoding="utf-8")
            if ast_tree is None and src_text:
                ast_tree = ast.parse(src_text)
        except Exception:
            src_text, ast_tree = "", None
        if src_text:
            enrich = EnrichmentDetectors()
            side_effect_data = enrich.detect_side_effect_boundaries(src_text, ast_tree)
            if side_effect_data:
                summary_parts: list[str] = []
                for category, effects in side_effect_data.items():
                    if effects:
                        summary_parts.append(f"{category}_effects: {effects[:3]}")
                if summary_parts:
                    items.append(f"# Side effects: {', '.join(summary_parts)}"[:600])
    except Exception:
        pass
    return items



