from testcraft.adapters.llm.azure import AzureOpenAIAdapter
from testcraft.adapters.llm.bedrock import BedrockAdapter
from testcraft.adapters.llm.claude import ClaudeAdapter
from testcraft.adapters.llm.openai import OpenAIAdapter


def test_openai_generate_tests():
    adapter = OpenAIAdapter()
    out = adapter.generate_tests("def f(): pass")
    assert "tests" in out and "metadata" in out


def test_claude_analyze_code():
    adapter = ClaudeAdapter()
    out = adapter.analyze_code("def g(): pass")
    assert "testability_score" in out and out["metadata"]["model"]


def test_azure_refine_content():
    adapter = AzureOpenAIAdapter()
    out = adapter.refine_content("print(1)", "improve")
    assert "refined_content" in out


def test_bedrock_generate_tests():
    adapter = BedrockAdapter()
    out = adapter.generate_tests("def h(): pass")
    assert "tests" in out and out["metadata"]["model_id"]


def test_generate_usecase_context_budgets_defaults():
    from unittest.mock import MagicMock

    from testcraft.application.generate_usecase import GenerateUseCase

    mock = MagicMock()
    uc = GenerateUseCase(
        llm_port=mock,
        writer_port=mock,
        coverage_port=mock,
        refine_port=mock,
        context_port=mock,
        parser_port=mock,
        state_port=mock,
        telemetry_port=mock,
    )
    cfg = uc._config
    assert cfg["prompt_budgets"]["per_item_chars"] == 1500
    assert cfg["prompt_budgets"]["total_chars"] == 10000
    assert "section_caps" in cfg["prompt_budgets"]
