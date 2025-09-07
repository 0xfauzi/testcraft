"""
Context adapters for indexing, retrieval, and summarization.

This package provides simple, in-memory implementations to satisfy the
`ContextPort` interface without external dependencies. They are designed
to be lightweight scaffolds that can be evolved as requirements grow.
"""

from .indexer import InMemoryHybridIndexer
from .retriever import SimpleContextRetriever
from .summarizer import ContextSummarizer
from .main_adapter import TestcraftContextAdapter

__all__ = [
    "InMemoryHybridIndexer",
    "SimpleContextRetriever",
    "ContextSummarizer",
    "TestcraftContextAdapter",
]


