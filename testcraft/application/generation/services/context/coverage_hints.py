from __future__ import annotations

from pathlib import Path


def get_coverage_hints(config: dict, source_path: Path) -> list[str]:
    items: list[str] = []
    try:
        enrichment_cfg = config.get("context_enrichment", {})
        if not enrichment_cfg.get("enable_coverage_hints", True):
            return items
        # Placeholder for future integration with CoverageEvaluator
        # Intentionally returns empty list to preserve behavior
    except Exception:
        pass
    return items



