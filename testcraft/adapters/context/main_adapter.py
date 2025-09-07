"""
Main context adapter implementing the ContextPort interface.

Provides in-memory indexing, simple retrieval, and summarization utilities
to satisfy the `ContextPort` Protocol without external dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ...ports.context_port import ContextPort
from .indexer import InMemoryHybridIndexer, IndexedChunk
from .retriever import SimpleContextRetriever
from .summarizer import ContextSummarizer


logger = logging.getLogger(__name__)


class TestcraftContextAdapter(ContextPort):
    """
    Main context adapter implementing ContextPort.

    - Indexes files into adaptive chunks with lightweight metadata
    - Retrieves relevant snippets using a simple hybrid scoring
    - Summarizes files using AST and import analysis
    - Builds a basic context graph based on imports
    """

    def __init__(self) -> None:
        self._indexer = InMemoryHybridIndexer()
        self._retriever = SimpleContextRetriever(self._indexer)
        self._summarizer = ContextSummarizer()

    # ContextPort
    def index(
        self,
        file_path: Union[str, Path],
        content: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            logger.info("Indexing file: %s", str(path))
            index_id, chunks = self._indexer.index_file(path, content=content, **kwargs)

            context_summary = self._summarizer.summarize_file(path, content=content)
            return {
                "index_id": index_id,
                "elements_indexed": len(chunks),
                "context_summary": context_summary.get("summary", ""),
                "metadata": {
                    "path": str(path),
                    "chunk_count": len(chunks),
                    "imports": context_summary.get("dependencies", []),
                },
            }
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception("Indexing failed for %s: %s", str(file_path), exc)
            raise

    def retrieve(
        self,
        query: str,
        context_type: str = "general",
        limit: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            results = self._retriever.retrieve(query, limit=limit, context_type=context_type, **kwargs)
            return results
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception("Retrieval failed: %s", exc)
            raise

    def summarize(
        self,
        file_path: Union[str, Path],
        summary_type: str = "comprehensive",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return self._summarizer.summarize_file(Path(file_path), summary_type=summary_type, **kwargs)
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception("Summarization failed for %s: %s", str(file_path), exc)
            raise

    def get_related_context(
        self,
        file_path: Union[str, Path],
        relationship_type: str = "all",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        indexed = self._indexer.get_index_by_path(path)
        if not indexed:
            # Not indexed yet; nothing to relate
            return {
                "related_files": [],
                "relationships": [],
                "usage_context": [],
                "dependency_context": [],
            }

        imports = set(indexed.imports)
        related_files: List[str] = []
        relationships: List[Tuple[str, str]] = []  # (from, to)

        for other in self._indexer.iter_documents():
            if other.path == indexed.path:
                continue
            # If this file imports other, or other imports this
            if Path(other.path).stem in imports:
                related_files.append(other.path)
                relationships.append((indexed.path, other.path))
            if Path(indexed.path).stem in set(other.imports):
                related_files.append(other.path)
                relationships.append((other.path, indexed.path))

        # De-duplicate
        related_files = sorted(set(related_files))
        relationships = sorted(set(relationships))

        return {
            "related_files": related_files,
            "relationships": [
                {"from": f, "to": t, "type": "import"} for f, t in relationships
            ],
            "usage_context": [],
            "dependency_context": list(imports),
        }

    def build_context_graph(
        self, project_root: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        nodes: List[str] = []
        edges: List[Tuple[str, str]] = []
        for doc in self._indexer.iter_documents():
            nodes.append(doc.path)
            for imp in doc.imports:
                # Map import name to known file by stem match if present
                for other in self._indexer.iter_documents():
                    if Path(other.path).stem == imp:
                        edges.append((doc.path, other.path))

        # De-duplicate
        nodes = sorted(set(nodes))
        edges = sorted(set(set(edges)))

        return {
            "graph": {"directed": True},
            "nodes": nodes,
            "edges": [{"from": f, "to": t, "type": "import"} for f, t in edges],
            "graph_metadata": {"project_root": str(project_root), "node_count": len(nodes), "edge_count": len(edges)},
        }


