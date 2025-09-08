from testcraft.adapters.llm.router import LLMRouter
from testcraft.adapters.llm.stream import StreamConfig, stream_text


def test_llm_router_uses_config():
    """Test that LLMRouter uses configuration instead of complexity-based routing."""
    config = {"default_provider": "openai", "openai_model": "o4-mini"}
    router = LLMRouter(config)

    # The router should always use the configured provider, not complexity-based routing
    assert router.default_provider == "openai"


def test_stream_text_chunks():
    text = "a" * 300
    chunks = list(stream_text(text, config=StreamConfig(chunk_size=100)))
    assert len(chunks) == 3
    assert "".join(chunks) == text
