"""
Simple context retriever that ranks chunks via naive hybrid scores.

Currently uses:
- BM25-like token overlap (approximated via term frequency scoring)
- Symbol name boosts when present
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Tuple

from .indexer import InMemoryHybridIndexer, IndexedChunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


class SimpleContextRetriever:
    def __init__(self, indexer: InMemoryHybridIndexer) -> None:
        self._indexer = indexer

    def retrieve(self, query: str, *, limit: int = 10, context_type: str = "general", **kwargs: Any) -> Dict[str, Any]:
        q_tokens = _tokenize(query)
        scored: List[Tuple[float, IndexedChunk]] = []

        # Precompute doc frequencies for a tiny BM25-ish scoring
        df: Dict[str, int] = {}
        for doc in self._indexer.iter_documents():
            seen_tokens = set()
            for chunk in doc.chunks:
                tokens = set(_tokenize(chunk.text))
                seen_tokens.update(tokens)
            for tok in seen_tokens:
                df[tok] = df.get(tok, 0) + 1

        corpus_docs = max(1, len(list(self._indexer.iter_documents())))

        for doc in self._indexer.iter_documents():
            for chunk in doc.chunks:
                score = 0.0
                c_tokens = _tokenize(chunk.text)
                length_norm = 1.0 + math.log(1 + len(c_tokens))

                for tok in q_tokens:
                    tf = c_tokens.count(tok)
                    if tf == 0:
                        continue
                    idf = math.log(1 + corpus_docs / (1 + df.get(tok, 0)))
                    score += (tf * idf) / length_norm

                # Boost if symbol name appears
                if chunk.symbol:
                    sym = chunk.symbol.lower()
                    if sym in q_tokens or any(sym in t for t in q_tokens):
                        score *= 1.5

                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for s, c in scored[:limit] if s > 0]

        return {
            "results": [
                {
                    "path": c.path,
                    "chunk_id": c.chunk_id,
                    "symbol": c.symbol,
                    "snippet": c.text[:400],
                    "score": s,
                }
                for s, c in scored[:limit]
                if s > 0
            ],
            "relevance_scores": [s for s, _ in scored[:limit] if s > 0],
            "total_found": len(top),
            "query_metadata": {"context_type": context_type, "query": query},
        }


