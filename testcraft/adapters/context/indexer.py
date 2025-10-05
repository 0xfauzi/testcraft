"""
In-memory hybrid indexer with adaptive chunking and simple metadata.

This provides a minimal, dependency-free scaffold suitable for unit tests and
future enhancement (BM25/dense placeholders omitted for now).
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedChunk:
    """A single indexed chunk with metadata."""

    __slots__ = ("doc_id", "chunk_id", "text", "path_id", "symbol", "imports")

    doc_id: str
    chunk_id: int
    text: str
    path_id: str
    symbol: str | None
    imports: tuple[str, ...]


@dataclass
class IndexedDocument:
    """Indexed document container for quick traversal."""

    __slots__ = ("index_id", "path_id", "chunks", "imports")

    index_id: str
    path_id: str
    chunks: list[IndexedChunk]
    imports: tuple[str, ...]


class PathRegistry:
    """Registry for path normalization and deduplication."""

    def __init__(self) -> None:
        self._paths: dict[str, str] = {}  # path_id -> resolved_path
        self._path_to_id: dict[str, str] = {}  # resolved_path -> path_id

    def register_path(self, path: Path) -> str:
        """Register a path and return its unique ID."""
        resolved = str(path.resolve())
        if resolved in self._path_to_id:
            return self._path_to_id[resolved]

        path_id = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
        self._paths[path_id] = resolved
        self._path_to_id[resolved] = path_id
        return path_id

    def get_path(self, path_id: str) -> str | None:
        """Get resolved path by ID."""
        return self._paths.get(path_id)


class InMemoryHybridIndexer:
    """
    Lightweight in-memory indexer.

    - Splits content into adaptive chunks (function/class-level if possible,
      otherwise paragraph/line blocks)
    - Extracts imports using AST parsing with regex fallback
    - Generates stable `index_id` keyed by absolute file path
    """

    def __init__(self) -> None:
        self._docs: dict[str, IndexedDocument] = {}
        self._path_registry = PathRegistry()

    def _make_index_id(self, path: Path) -> str:
        absolute = str(path.resolve())
        return hashlib.sha1(absolute.encode("utf-8")).hexdigest()[:16]

    def index_file(
        self, path: Path, *, content: str | None = None, max_chunk_chars: int = 1200
    ) -> tuple[str, list[IndexedChunk]]:
        """Index a file and return its ID and chunks."""
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")

        if content is None:
            if not path.exists():
                raise FileNotFoundError(str(path))
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError) as e:
                logger.warning(f"Failed to read {path}: {e}")
                raise

        if not content.strip():
            logger.debug(f"Empty or whitespace-only file: {path}")
            content = content or ""

        index_id = self._make_index_id(path)
        path_id = self._path_registry.register_path(path)
        imports = tuple(self._extract_imports(content))
        chunks = self._chunk_content(
            index_id, path_id, content, max_chunk_chars=max_chunk_chars
        )

        resolved_path = str(path.resolve())
        self._docs[resolved_path] = IndexedDocument(
            index_id=index_id, path_id=path_id, chunks=chunks, imports=imports
        )

        return index_id, chunks

    def _extract_imports(self, content: str) -> Iterable[str]:
        """Extract imports using AST parsing with regex fallback."""
        try:
            tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Relative import - store as-is for now
                            module = "." * node.level + (node.module or "")
                        else:
                            module = node.module.split(".")[0]
                        imports.add(module)

            return sorted(imports)

        except SyntaxError:
            logger.debug("AST parsing failed, falling back to regex")
            return self._extract_imports_regex(content)

    def _extract_imports_regex(self, content: str) -> Iterable[str]:
        """Fallback regex-based import extraction."""
        pattern = re.compile(
            r"^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE
        )
        imports = set()
        for match in pattern.finditer(content):
            module = match.group(1) or match.group(2)
            if module:
                imports.add(module.split(".")[0])
        return sorted(imports)

    def _chunk_content(
        self, index_id: str, path_id: str, content: str, *, max_chunk_chars: int
    ) -> list[IndexedChunk]:
        """Chunk content using AST-based symbol detection with fallback."""
        # Try AST-based symbol boundaries first
        try:
            boundaries = list(self._find_symbol_blocks_ast(content))
            if boundaries:
                chunks: list[IndexedChunk] = []
                for i, (symbol, start, end) in enumerate(boundaries):
                    if start >= len(content) or end > len(content) or start >= end:
                        logger.warning(
                            f"Invalid chunk boundaries: {start}-{end} for content length {len(content)}"
                        )
                        continue

                    text = content[start:end]
                    if not text.strip():
                        continue

                    chunks.append(
                        IndexedChunk(
                            doc_id=index_id,
                            chunk_id=i,
                            text=text,
                            path_id=path_id,
                            symbol=symbol,
                            imports=tuple(self._extract_imports(text)),
                        )
                    )
                return chunks
        except SyntaxError:
            logger.debug("AST-based chunking failed, using fallback")

        # Fallback: split by blank lines and size budget
        return self._chunk_content_fallback(index_id, path_id, content, max_chunk_chars)

    def _find_symbol_blocks_ast(self, content: str) -> Iterable[tuple[str, int, int]]:
        """Find symbol blocks using AST parsing."""
        tree = ast.parse(content)
        lines = content.splitlines(True)

        # Collect all top-level symbols
        symbols = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                symbol_name = node.name
                start_line = node.lineno - 1  # AST uses 1-based line numbers

                # Calculate byte offset for start
                start_offset = sum(len(line) for line in lines[:start_line])

                # Find end line by looking at the next symbol or end of file
                end_line = len(lines)
                for other_node in tree.body:
                    if (
                        isinstance(
                            other_node,
                            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                        )
                        and other_node.lineno > node.lineno
                    ):
                        end_line = min(end_line, other_node.lineno - 1)

                end_offset = sum(len(line) for line in lines[:end_line])
                symbols.append((symbol_name, start_offset, end_offset))

        return symbols

    def _chunk_content_fallback(
        self, index_id: str, path_id: str, content: str, max_chunk_chars: int
    ) -> list[IndexedChunk]:
        """Fallback chunking by lines and size budget."""
        current: list[str] = []
        acc = 0
        chunk_id = 0
        chunks: list[IndexedChunk] = []

        for line in content.splitlines(keepends=True):
            if acc + len(line) > max_chunk_chars and current:
                text = "".join(current)
                chunks.append(
                    IndexedChunk(
                        doc_id=index_id,
                        chunk_id=chunk_id,
                        text=text,
                        path_id=path_id,
                        symbol=None,
                        imports=tuple(self._extract_imports(text)),
                    )
                )
                current = []
                acc = 0
                chunk_id += 1
            current.append(line)
            acc += len(line)

        if current:
            text = "".join(current)
            chunks.append(
                IndexedChunk(
                    doc_id=index_id,
                    chunk_id=chunk_id,
                    text=text,
                    path_id=path_id,
                    symbol=None,
                    imports=tuple(self._extract_imports(text)),
                )
            )

        return chunks

    def reindex_file(
        self, path: Path, *, content: str | None = None, max_chunk_chars: int = 1200
    ) -> tuple[str, list[IndexedChunk]]:
        """Reindex a single file, replacing any existing index."""
        resolved_path = str(path.resolve())
        if resolved_path in self._docs:
            del self._docs[resolved_path]
        return self.index_file(path, content=content, max_chunk_chars=max_chunk_chars)

    # Utilities for retriever/summarizer
    def iter_documents(self) -> Iterable[IndexedDocument]:
        return list(self._docs.values())

    def get_index_by_path(self, path: Path) -> IndexedDocument | None:
        return self._docs.get(str(path.resolve()))
