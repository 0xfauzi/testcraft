from __future__ import annotations

from pathlib import Path
from typing import Any

from .....ports.context_port import ContextPort


def retrieve_snippets(context_port: ContextPort, query: str, limit: int = 5) -> list[str]:
    items: list[str] = []
    try:
        retrieval = context_port.retrieve(query=query, context_type="general", limit=limit)
        for item in retrieval.get("results", [])[:limit]:
            if isinstance(item, dict):
                snippet = item.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    items.append(snippet[:400])
    except Exception:
        pass
    return items


def get_neighbor_context(context_port: ContextPort, source_path: Path | None) -> list[str]:
    items: list[str] = []
    if source_path is None:
        return items
    try:
        related = context_port.get_related_context(source_path, relationship_type="all")
        for related_path in related.get("related_files", [])[:3]:
            try:
                p = Path(related_path)
                if p.exists() and p.suffix == ".py":
                    content = p.read_text(encoding="utf-8")
                    items.append(f"# Related: {p.name}")
                    items.append(content[:600])
            except Exception:
                continue
    except Exception:
        pass
    return items



