"""
In-memory hybrid indexer with adaptive chunking and simple metadata.

This provides a minimal, dependency-free scaffold suitable for unit tests and
future enhancement (BM25/dense placeholders omitted for now).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class IndexedChunk:
    """A single indexed chunk with metadata."""

    doc_id: str
    chunk_id: int
    text: str
    path: str
    symbol: Optional[str]
    imports: Tuple[str, ...]


@dataclass
class IndexedDocument:
    """Indexed document container for quick traversal."""

    index_id: str
    path: str
    chunks: List[IndexedChunk]
    imports: Tuple[str, ...]


class InMemoryHybridIndexer:
    """
    Lightweight in-memory indexer.

    - Splits content into adaptive chunks (function/class-level if possible,
      otherwise paragraph/line blocks)
    - Extracts naive `imports` by regex
    - Generates stable `index_id` keyed by absolute file path
    """

    def __init__(self) -> None:
        self._docs: Dict[str, IndexedDocument] = {}

    def _make_index_id(self, path: Path) -> str:
        absolute = str(path.resolve())
        return hashlib.sha1(absolute.encode("utf-8")).hexdigest()[:16]

    def index_file(
        self, path: Path, *, content: Optional[str] = None, max_chunk_chars: int = 1200
    ) -> Tuple[str, List[IndexedChunk]]:
        if content is None:
            if not path.exists():
                raise FileNotFoundError(str(path))
            content = path.read_text(encoding="utf-8")

        index_id = self._make_index_id(path)
        imports = tuple(self._extract_imports(content))
        chunks = self._chunk_content(index_id, path, content, max_chunk_chars=max_chunk_chars)

        self._docs[str(path.resolve())] = IndexedDocument(
            index_id=index_id, path=str(path.resolve()), chunks=chunks, imports=imports
        )

        return index_id, chunks

    def _extract_imports(self, content: str) -> Iterable[str]:
        pattern = re.compile(r"^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)
        for match in pattern.finditer(content):
            module = match.group(1) or match.group(2)
            if module:
                yield module.split(".")[0]

    def _chunk_content(
        self, index_id: str, path: Path, content: str, *, max_chunk_chars: int
    ) -> List[IndexedChunk]:
        # Try function/class boundaries first
        boundaries = list(self._find_symbol_blocks(content))
        chunks: List[IndexedChunk] = []
        if boundaries:
            for i, (symbol, start, end) in enumerate(boundaries):
                text = content[start:end]
                if not text.strip():
                    continue
                chunks.append(
                    IndexedChunk(
                        doc_id=index_id,
                        chunk_id=i,
                        text=text,
                        path=str(path.resolve()),
                        symbol=symbol,
                        imports=tuple(self._extract_imports(text)),
                    )
                )
            return chunks

        # Fallback: split by blank lines and size budget
        current: List[str] = []
        acc = 0
        chunk_id = 0
        for line in content.splitlines(keepends=True):
            if acc + len(line) > max_chunk_chars and current:
                text = "".join(current)
                chunks.append(
                    IndexedChunk(
                        doc_id=index_id,
                        chunk_id=chunk_id,
                        text=text,
                        path=str(path.resolve()),
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
                    path=str(path.resolve()),
                    symbol=None,
                    imports=tuple(self._extract_imports(text)),
                )
            )

        return chunks

    def _find_symbol_blocks(self, content: str) -> Iterable[Tuple[str, int, int]]:
        # Very lightweight: find "def" and "class" blocks using regex and indentation
        lines = content.splitlines(keepends=True)
        header_pattern = re.compile(r"^(def|class)\s+([\w_]+)\s*\(?.*", re.MULTILINE)
        indices = [i for i, line in enumerate(lines) if header_pattern.match(line)]
        for idx, start_line in enumerate(indices):
            header_match = header_pattern.match(lines[start_line])
            assert header_match is not None
            symbol = header_match.group(2)
            start = sum(len(l) for l in lines[:start_line])
            end_line = indices[idx + 1] if idx + 1 < len(indices) else len(lines)
            end = sum(len(l) for l in lines[:end_line])
            yield symbol, start, end

    # Utilities for retriever/summarizer
    def iter_documents(self) -> Iterable[IndexedDocument]:
        return list(self._docs.values())

    def get_index_by_path(self, path: Path) -> Optional[IndexedDocument]:
        return self._docs.get(str(path.resolve()))


