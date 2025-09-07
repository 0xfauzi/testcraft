from testcraft.adapters.llm.openai import OpenAIAdapter
from testcraft.adapters.llm.claude import ClaudeAdapter
from testcraft.adapters.llm.azure import AzureOpenAIAdapter
from testcraft.adapters.llm.bedrock import BedrockAdapter


def test_openai_generate_tests():
    adapter = OpenAIAdapter()
    out = adapter.generate_tests("def f(): pass")
    assert "tests" in out and "metadata" in out


def test_claude_analyze_code():
    adapter = ClaudeAdapter()
    out = adapter.analyze_code("def g(): pass")
    assert "analysis" in out and out["metadata"]["model"]


def test_azure_refine_content():
    adapter = AzureOpenAIAdapter()
    out = adapter.refine_content("print(1)", "improve")
    assert "refined_content" in out


def test_bedrock_generate_tests():
    adapter = BedrockAdapter()
    out = adapter.generate_tests("def h(): pass")
    assert "tests" in out and out["metadata"]["model_id"]
