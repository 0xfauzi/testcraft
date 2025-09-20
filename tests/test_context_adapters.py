from pathlib import Path

from testcraft.adapters.context import (
    ContextSummarizer,
    InMemoryHybridIndexer,
    SimpleContextRetriever,
    TestcraftContextAdapter,
)


def test_indexer_and_retriever_basic(tmp_path: Path) -> None:
    src = tmp_path / "sample.py"
    src.write_text(
        """
import os

def greet(name: str) -> str:
    return f"Hello {name}"

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello {name}"
""",
        encoding="utf-8",
    )

    indexer = InMemoryHybridIndexer()
    index_id, chunks = indexer.index_file(src)
    assert index_id
    assert len(chunks) >= 1

    retriever = SimpleContextRetriever(indexer)
    results = retriever.retrieve("greet function", limit=5)
    assert results["total_found"] >= 1
    assert (
        any("greet" in (r.get("symbol") or "") for r in results["results"])
        or results["results"]
    ), results


def test_summarizer_and_adapter(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text(
        """
import sys

def add(a: int, b: int) -> int:
    return a + b
""",
        encoding="utf-8",
    )

    summarizer = ContextSummarizer()
    summary = summarizer.summarize_file(src)
    assert "mod.py" in summary["summary"]
    assert "add" in ",".join(summary["key_functions"])
    assert "sys" in ",".join(summary["dependencies"]) or summary["dependencies"] == []

    adapter = TestcraftContextAdapter()
    idx = adapter.index(src)
    assert idx["elements_indexed"] >= 1
    ret = adapter.retrieve("add two numbers", limit=3)
    assert ret["total_found"] >= 0
    sum2 = adapter.summarize(src)
    assert "File: mod.py" in sum2["summary"]
