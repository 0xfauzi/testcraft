from testcraft.adapters.llm.router import route_model
from testcraft.adapters.llm.stream import stream_text, StreamConfig


def test_route_model_thresholds():
    low = "def a():\n  return 1\n"
    mid = "\n".join(["def f(): pass" for _ in range(6)])
    high = "x" * 4000

    r1 = route_model(low)
    r2 = route_model(mid)
    r3 = route_model(high)

    assert r1.provider in {"openai", "claude", "azure", "bedrock"}
    assert r2.provider in {"openai", "claude", "azure", "bedrock"}
    assert r3.provider in {"openai", "claude", "azure", "bedrock"}


def test_stream_text_chunks():
    text = "a" * 300
    chunks = list(stream_text(text, config=StreamConfig(chunk_size=100)))
    assert len(chunks) == 3
    assert "".join(chunks) == text


