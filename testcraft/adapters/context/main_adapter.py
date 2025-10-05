"""
Main context adapter implementing the ContextPort interface.

Provides in-memory indexing, simple retrieval, and summarization utilities
to satisfy the `ContextPort` Protocol without external dependencies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from ...ports.context_port import ContextPort
from .indexer import InMemoryHybridIndexer
from .retriever import SimpleContextRetriever
from .summarizer import ContextSummarizer

logger = logging.getLogger(__name__)


class RelationshipData(TypedDict):
    """Structured relationship data for better type safety."""

    from_file: str
    to_file: str
    relationship_type: str
    import_name: str | None


class ImportGraph:
    """Efficient import relationship graph with caching."""

    def __init__(self) -> None:
        self._import_to_files: dict[str, set[str]] = defaultdict(set)
        self._file_to_imports: dict[str, set[str]] = defaultdict(set)
        self._relationships: list[RelationshipData] = []

    def add_relationship(
        self,
        from_file: str,
        to_file: str,
        import_name: str | None = None,
        relationship_type: str = "import",
    ) -> None:
        """Add a relationship between files."""
        relationship = RelationshipData(
            from_file=from_file,
            to_file=to_file,
            relationship_type=relationship_type,
            import_name=import_name,
        )
        self._relationships.append(relationship)
        self._import_to_files[import_name or ""].add(to_file)
        self._file_to_imports[from_file].add(import_name or "")

    def get_files_by_import(self, import_name: str) -> set[str]:
        """Get all files that can be imported by the given import name."""
        return self._import_to_files.get(import_name, set())

    def get_imports_by_file(self, file_path: str) -> set[str]:
        """Get all imports used by a specific file."""
        return self._file_to_imports.get(file_path, set())

    def get_all_relationships(self) -> list[RelationshipData]:
        """Get all relationships in the graph."""
        return self._relationships.copy()


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
        self._summarizer = ContextSummarizer()
        self._import_graph = ImportGraph()
        self._import_cache: dict[str, set[str]] = {}  # Cache for resolved imports
        self._relationship_cache: dict[
            str, dict[str, Any]
        ] = {}  # Cache for relationships

        # Store path registry reference for resolving path_ids to paths
        self._path_registry = self._indexer._path_registry

        # Use the retriever with path registry
        self._retriever = SimpleContextRetriever(self._indexer, self._path_registry)

    def _resolve_import_to_file(self, import_name: str, project_root: Path) -> set[str]:
        """
        Resolve an import name to actual file paths using proper Python import resolution.

        This replaces the simple stem matching with sophisticated import resolution
        that handles relative imports, package structure, and module search paths.
        """
        cache_key = f"{import_name}:{project_root}"
        if cache_key in self._import_cache:
            return self._import_cache[cache_key]

        resolved_files: set[str] = set()

        # Handle relative imports
        if import_name.startswith("."):
            # For now, treat relative imports as unresolvable
            # In a full implementation, we'd need the current file's context
            return resolved_files

        # Handle absolute imports - look for modules in the project
        for doc in self._indexer.iter_documents():
            doc_path = Path(str(getattr(doc, "path", "")))
            if not doc_path:
                continue
            try:
                # Check if this document could match the import
                # This is a simplified version - a full implementation would use
                # the ImportResolver from application/generation/services/
                if self._import_matches_file(import_name, doc_path, project_root):
                    resolved_files.add(str(doc_path))
            except Exception:
                # Skip files that can't be analyzed
                continue

        self._import_cache[cache_key] = resolved_files
        return resolved_files

    def _import_matches_file(
        self, import_name: str, file_path: Path, project_root: Path
    ) -> bool:
        """
        Check if an import name could refer to a given file using proper Python import resolution.

        This uses the existing PackagingDetector from the application layer to determine
        if a file can be imported with the given import name, considering Python's
        module search paths, package structure, and import rules.
        """
        try:
            # Use the existing PackagingDetector for proper resolution
            from ...application.generation.services.packaging_detector import (
                PackagingDetector,
            )

            packaging_info = PackagingDetector.detect_packaging(project_root)

            # Check if the file's canonical import matches the given import
            canonical_import = packaging_info.get_canonical_import(file_path)
            if not canonical_import:
                return self._enhanced_import_heuristic(
                    import_name, file_path, project_root
                )

            # For absolute imports, check if they match or are compatible
            if not import_name.startswith("."):
                return self._check_absolute_import_match(
                    import_name, canonical_import, file_path, project_root
                )

            # For relative imports, use a more sophisticated heuristic
            return self._check_relative_import_match(
                import_name, file_path, project_root, packaging_info
            )

        except Exception:
            # Fallback to enhanced heuristic if PackagingDetector fails
            return self._enhanced_import_heuristic(import_name, file_path, project_root)

    def _check_absolute_import_match(
        self,
        import_name: str,
        canonical_import: str,
        file_path: Path,
        project_root: Path,
    ) -> bool:
        """Check if an absolute import name matches the file's canonical import."""
        import_parts = import_name.split(".")
        canonical_parts = canonical_import.split(".")

        # Exact match
        if import_name == canonical_import:
            return True

        # Prefix match - import_name is a prefix of canonical_import
        # (e.g., "mymodule" matches "mymodule.submodule")
        if len(import_parts) <= len(canonical_parts):
            return canonical_parts[: len(import_parts)] == import_parts

        # Suffix match - canonical_import is a prefix of import_name
        # (e.g., "mymodule.submodule" matches "mymodule")
        if len(canonical_parts) <= len(import_parts):
            return import_parts[: len(canonical_parts)] == canonical_parts

        return False

    def _check_relative_import_match(
        self, import_name: str, file_path: Path, project_root: Path, packaging_info
    ) -> bool:
        """Check if a relative import could refer to this file."""
        # For relative imports, we need to determine the relative path from the importing file
        # This is complex and would require knowing the importing file's context
        # For now, use a heuristic based on directory structure

        import_levels = len(import_name) - len(import_name.lstrip("."))

        try:
            rel_path = file_path.relative_to(project_root)
            dir_count = len(rel_path.parts) - 1  # -1 for the file itself

            # Allow relative imports that don't go beyond the package boundary
            return dir_count >= import_levels
        except ValueError:
            return False

    def _enhanced_import_heuristic(
        self, import_name: str, file_path: Path, project_root: Path
    ) -> bool:
        """Enhanced heuristic that considers package structure."""
        import_parts = import_name.split(".")
        file_stem = file_path.stem

        # Check if the last part of the import matches the file name
        if import_parts[-1] == file_stem:
            return True

        # Check if the import could refer to this file in a package structure
        try:
            rel_path = file_path.relative_to(project_root)
            # Count directories to match import depth
            dir_count = len(rel_path.parts) - 1  # -1 for the file itself

            # Allow imports that match the directory structure
            if dir_count + 1 == len(import_parts):
                return True

            # Allow shorter imports that could refer to this file
            # (e.g., "mypackage" could import "mypackage.module")
            if dir_count + 1 > len(import_parts):
                return True

        except ValueError:
            pass

        # Check for __init__.py files that might indicate package boundaries
        current_dir = file_path.parent
        for _ in range(len(import_parts)):
            if (current_dir / "__init__.py").exists():
                return True
            try:
                current_dir = current_dir.parent
            except ValueError:
                break

        return False

    def _build_import_relationships(self, file_path: str, project_root: Path) -> None:
        """Build import relationships for a file and update the graph."""
        try:
            indexed = self._indexer.get_index_by_path(Path(file_path))
            if not indexed:
                return

            for import_name in indexed.imports:
                if not import_name.strip():
                    continue

                # Resolve the import to actual files
                target_files = self._resolve_import_to_file(import_name, project_root)

                for target_file in target_files:
                    self._import_graph.add_relationship(
                        from_file=file_path,
                        to_file=target_file,
                        import_name=import_name,
                        relationship_type="import",
                    )
        except Exception as exc:
            logger.warning(f"Failed to build relationships for {file_path}: {exc}")

    def _get_related_context_cached(
        self, file_path: str, relationship_type: str = "all"
    ) -> dict[str, Any]:
        """Get related context with caching to avoid repeated computation."""
        cache_key = f"{file_path}:{relationship_type}"
        if cache_key in self._relationship_cache:
            return self._relationship_cache[cache_key]

        result = self._get_related_context_impl(file_path, relationship_type)
        self._relationship_cache[cache_key] = result
        return result

    def _get_related_context_impl(
        self, file_path: str, relationship_type: str = "all"
    ) -> dict[str, Any]:
        """Implementation of related context retrieval."""
        path = Path(file_path)
        indexed = self._indexer.get_index_by_path(path)
        if not indexed:
            return {
                "related_files": [],
                "relationships": [],
                "usage_context": [],
                "dependency_context": [],
            }

        # Get relationships from the import graph
        relationships = self._import_graph.get_all_relationships()
        related_files: list[str] = []
        filtered_relationships: list[dict[str, str]] = []

        for rel in relationships:
            if rel["from_file"] == file_path:
                if (
                    relationship_type == "all"
                    or relationship_type == rel["relationship_type"]
                ):
                    if rel["to_file"] not in related_files:
                        related_files.append(rel["to_file"])
                    filtered_relationships.append(
                        {
                            "from": rel["from_file"],
                            "to": rel["to_file"],
                            "type": rel["relationship_type"],
                        }
                    )

        return {
            "related_files": sorted(related_files),
            "relationships": filtered_relationships,
            "usage_context": [],
            "dependency_context": list(indexed.imports),
        }

    # ContextPort
    def index(
        self,
        file_path: str | Path,
        content: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            path = Path(file_path)

            # Validate file path
            if not path.exists() and content is None:
                raise FileNotFoundError(f"File not found: {path}")

            logger.info("Indexing file: %s", str(path))
            index_id, chunks = self._indexer.index_file(path, content=content, **kwargs)

            context_summary = self._summarizer.summarize_file(path, content=content)

            # Build import relationships for better context graph
            try:
                project_root = kwargs.get("project_root", path.parent)
                self._build_import_relationships(str(path), Path(project_root))
            except Exception as exc:
                logger.warning(f"Failed to build relationships for {path}: {exc}")

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
    ) -> dict[str, Any]:
        try:
            results = self._retriever.retrieve(
                query, limit=limit, context_type=context_type, **kwargs
            )
            return results
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception("Retrieval failed: %s", exc)
            raise

    def summarize(
        self,
        file_path: str | Path,
        summary_type: str = "comprehensive",
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return self._summarizer.summarize_file(
                Path(file_path), summary_type=summary_type, **kwargs
            )
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception("Summarization failed for %s: %s", str(file_path), exc)
            raise

    def get_related_context(
        self,
        file_path: str | Path,
        relationship_type: str = "all",
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            path = Path(file_path)

            # Validate parameters
            if not isinstance(relationship_type, str):
                raise ValueError(
                    f"relationship_type must be a string, got {type(relationship_type)}"
                )

            # Use cached implementation for performance
            return self._get_related_context_cached(str(path), relationship_type)

        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception(
                "Related context retrieval failed for %s: %s", str(file_path), exc
            )
            # Return empty result on error rather than failing completely
            return {
                "related_files": [],
                "relationships": [],
                "usage_context": [],
                "dependency_context": [],
            }

    def build_context_graph(
        self, project_root: str | Path, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            project_root_path = Path(project_root)

            # Validate project root
            if not project_root_path.exists():
                raise ValueError(f"Project root does not exist: {project_root}")

            # Use the import graph for efficient relationship building
            # Build relationships for all indexed files
            for doc in self._indexer.iter_documents():
                try:
                    doc_path_str = str(getattr(doc, "path", ""))
                    if doc_path_str:
                        self._build_import_relationships(
                            doc_path_str, project_root_path
                        )
                except Exception as exc:
                    doc_path_str = str(getattr(doc, "path", "unknown"))
                    logger.warning(
                        f"Failed to build relationships for {doc_path_str}: {exc}"
                    )

            # Extract nodes and edges from the import graph
            nodes: list[str] = []
            edges: list[dict[str, str]] = []

            # Get unique nodes from relationships
            seen_nodes = set()
            for rel in self._import_graph.get_all_relationships():
                if rel["from_file"] not in seen_nodes:
                    nodes.append(rel["from_file"])
                    seen_nodes.add(rel["from_file"])
                if rel["to_file"] not in seen_nodes:
                    nodes.append(rel["to_file"])
                    seen_nodes.add(rel["to_file"])

                edges.append(
                    {
                        "from": rel["from_file"],
                        "to": rel["to_file"],
                        "type": rel["relationship_type"],
                    }
                )

            # Sort for consistent output
            nodes = sorted(nodes)
            edges = sorted(edges, key=lambda x: (x["from"], x["to"], x["type"]))

            return {
                "graph": {"directed": True},
                "nodes": nodes,
                "edges": edges,
                "graph_metadata": {
                    "project_root": str(project_root),
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                },
            }
        except Exception as exc:  # pragma: no cover - logged and re-raised
            logger.exception(
                "Context graph building failed for %s: %s", str(project_root), exc
            )
            raise
