from __future__ import annotations

from pathlib import Path

from .....ports.context_port import ContextPort


def get_callgraph_neighbors(context_port: ContextPort, config: dict, source_path: Path) -> list[str]:
    items: list[str] = []
    try:
        enrichment_cfg = config.get("context_enrichment", {})
        if not enrichment_cfg.get("enable_callgraph", True):
            return items
        rel = context_port.get_related_context(source_path, relationship_type="all")
        relationships = rel.get("relationships", [])
        related_files = rel.get("related_files", [])
        if relationships or related_files:
            edges: list[str] = []
            if isinstance(relationships, list):
                edges.extend(str(r)[:100] for r in relationships[:5])
            for rf in related_files[:3]:
                try:
                    rf_path = Path(rf)
                    if rf_path.exists() and rf_path.suffix == ".py":
                        edges.append(f"import:{rf_path.name}")
                except Exception:
                    continue
            if edges:
                items.append(f"# Call-graph edges: {edges[:8]}")
    except Exception:
        pass
    return items



