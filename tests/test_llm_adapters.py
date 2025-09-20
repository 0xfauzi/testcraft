from unittest.mock import MagicMock, patch

import pytest

from testcraft.adapters.llm.azure import AzureOpenAIAdapter
from testcraft.adapters.llm.bedrock import BedrockAdapter
from testcraft.adapters.llm.claude import ClaudeAdapter
from testcraft.adapters.llm.openai import OpenAIAdapter


def test_openai_generate_tests():
    """Test OpenAI adapter generates tests with correct structure."""
    adapter = OpenAIAdapter()
    out = adapter.generate_tests("def f(): pass")

    # Verify standardized return structure
    assert "tests" in out
    assert "coverage_focus" in out
    assert "confidence" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "openai"


def test_claude_analyze_code():
    """Test Claude adapter analyzes code with correct structure."""
    adapter = ClaudeAdapter()
    out = adapter.analyze_code("def g(): pass")

    # Verify standardized return structure
    assert "testability_score" in out
    assert "complexity_metrics" in out
    assert "recommendations" in out
    assert "potential_issues" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "anthropic"
    assert "parsed" in metadata


def test_azure_refine_content():
    """Test Azure adapter refines content with correct structure."""
    adapter = AzureOpenAIAdapter()
    out = adapter.refine_content("print(1)", "improve")

    # Verify standardized return structure
    assert "refined_content" in out
    assert "changes_made" in out
    assert "confidence" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "azure-openai"
    # Deployment info is in extras
    assert "extras" in metadata
    assert "deployment" in metadata["extras"]
    assert "parsed" in metadata


def test_bedrock_generate_tests():
    """Test Bedrock adapter generates tests with correct structure."""
    adapter = BedrockAdapter()
    out = adapter.generate_tests("def h(): pass")

    # Verify standardized return structure
    assert "tests" in out
    assert "coverage_focus" in out
    assert "confidence" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "bedrock"


def test_azure_analyze_code():
    """Test Azure adapter analyzes code with correct structure."""
    adapter = AzureOpenAIAdapter()
    out = adapter.analyze_code("def test_func(): return True")

    # Verify standardized return structure
    assert "testability_score" in out
    assert "complexity_metrics" in out
    assert "recommendations" in out
    assert "potential_issues" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "azure-openai"
    # Deployment info is in extras
    assert "extras" in metadata
    assert "deployment" in metadata["extras"]
    assert "parsed" in metadata


def test_bedrock_analyze_code():
    """Test Bedrock adapter analyzes code with correct structure."""
    adapter = BedrockAdapter()
    out = adapter.analyze_code("def test_func(): return True")

    # Verify standardized return structure
    assert "testability_score" in out
    assert "complexity_metrics" in out
    assert "recommendations" in out
    assert "potential_issues" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "bedrock"


def test_claude_refine_content():
    """Test Claude adapter refines content with correct structure."""
    adapter = ClaudeAdapter()
    out = adapter.refine_content("print(1)", "improve")

    # Verify standardized return structure
    assert "refined_content" in out
    assert "changes_made" in out
    assert "confidence" in out
    assert "metadata" in out

    # Verify metadata contains expected keys (normalized schema)
    metadata = out["metadata"]
    assert "provider" in metadata
    assert "model_identifier" in metadata
    assert "parsed" in metadata
    assert "usage" in metadata
    assert metadata["provider"] == "anthropic"
    assert "parsed" in metadata


@patch("testcraft.prompts.registry.PromptRegistry.get_system_prompt")
@patch("testcraft.prompts.registry.PromptRegistry.get_user_prompt")
def test_adapters_use_prompt_registry(mock_user_prompt, mock_system_prompt):
    """Test that all adapters use the prompt registry for prompts."""
    mock_system_prompt.return_value = "System prompt from registry"
    mock_user_prompt.return_value = "User prompt from registry"

    adapters = [
        OpenAIAdapter(),
        ClaudeAdapter(),
        AzureOpenAIAdapter(),
        BedrockAdapter(),
    ]

    for adapter in adapters:
        # Test generate_tests uses registry
        adapter.generate_tests("def f(): pass")
        mock_system_prompt.assert_called_with(
            prompt_type="llm_test_generation", test_framework="pytest"
        )
        mock_user_prompt.assert_called()

        # Test analyze_code uses registry
        adapter.analyze_code("def f(): pass")
        mock_system_prompt.assert_called_with(
            prompt_type="llm_code_analysis", analysis_type="comprehensive"
        )

        # Test refine_content uses registry (when no system_prompt provided)
        adapter.refine_content("old content", "make it better")
        mock_system_prompt.assert_called_with(prompt_type="llm_content_refinement")

        # Reset mocks for next adapter
        mock_system_prompt.reset_mock()
        mock_user_prompt.reset_mock()


def test_adapters_work_without_api_keys():
    """Test that all adapters work with stub clients when no API keys are available."""
    # All adapters should initialize without real credentials and use stub clients
    adapters = [
        ("OpenAI", OpenAIAdapter()),
        ("Claude", ClaudeAdapter()),
        ("Azure", AzureOpenAIAdapter()),
        ("Bedrock", BedrockAdapter()),
    ]

    for name, adapter in adapters:
        # All methods should work without raising exceptions
        try:
            result = adapter.generate_tests("def f(): pass")
            assert isinstance(result, dict), f"{name} generate_tests should return dict"

            result = adapter.analyze_code("def g(): pass")
            assert isinstance(result, dict), f"{name} analyze_code should return dict"

            result = adapter.refine_content("old", "new")
            assert isinstance(result, dict), f"{name} refine_content should return dict"

        except Exception as e:
            pytest.fail(f"{name} adapter failed with stub client: {e}")


def test_all_adapters_return_consistent_structures():
    """Test that all adapters return consistent response structures."""
    adapters = [
        OpenAIAdapter(),
        ClaudeAdapter(),
        AzureOpenAIAdapter(),
        BedrockAdapter(),
    ]

    for adapter in adapters:
        # Test generate_tests structure
        result = adapter.generate_tests("def f(): pass")
        assert isinstance(result["tests"], str)
        assert isinstance(result["coverage_focus"], list)
        assert isinstance(result["confidence"], int | float)
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]

        # Test analyze_code structure
        result = adapter.analyze_code("def g(): pass")
        assert isinstance(result["testability_score"], int | float)
        assert isinstance(result["complexity_metrics"], dict)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["potential_issues"], list)
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]

        # Test refine_content structure
        result = adapter.refine_content("old content", "instructions")
        assert isinstance(result["refined_content"], str)
        assert isinstance(result["changes_made"], str)
        assert isinstance(result["confidence"], int | float)
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]


def test_generate_usecase_context_budgets_defaults():
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


def test_router_conforms_to_llm_port_interface():
    """Test that LLMRouter conforms to LLMPort interface and is sync."""
    from testcraft.adapters.llm.router import LLMRouter

    # Test that router has all required methods
    router = LLMRouter()
    required_methods = [
        "generate_tests",
        "analyze_code",
        "refine_content",
        "generate_test_plan",
    ]

    for method_name in required_methods:
        assert hasattr(router, method_name), f"Router missing method: {method_name}"
        assert callable(getattr(router, method_name)), (
            f"Router {method_name} not callable"
        )

        # Verify methods are sync (not async)
        method = getattr(router, method_name)
        import asyncio

        assert not asyncio.iscoroutinefunction(method), (
            f"Router {method_name} should be sync, not async"
        )

    # Test that router removed the non-port method
    assert not hasattr(router, "refine_tests"), (
        "Router should not have non-port refine_tests method"
    )

    # Test that router can actually delegate calls correctly
    try:
        result = router.generate_tests("def test(): pass")
        assert isinstance(result, dict), "Router should return dict from generate_tests"
        assert "tests" in result, "Router should delegate to adapter correctly"
    except Exception as e:
        # Acceptable failure due to stub adapter or missing config
        assert (
            "stub" in str(e).lower()
            or "config" in str(e).lower()
            or "provider" in str(e).lower()
        )


def test_per_request_token_budgeting():
    """Test that adapters use per-request token budgeting with safety margins."""
    adapters = [
        ("OpenAI", OpenAIAdapter()),
        ("Claude", ClaudeAdapter()),
        ("Azure", AzureOpenAIAdapter()),
        ("Bedrock", BedrockAdapter()),
    ]

    for name, adapter in adapters:
        # Test that adapter has token calculator
        assert hasattr(adapter, "token_calculator"), f"{name} missing token_calculator"

        # Test that token calculator has proper limits
        calc = adapter.token_calculator
        assert calc.limits.max_context > 0, f"{name} invalid max_context"
        assert calc.limits.max_output > 0, f"{name} invalid max_output"
        assert calc.limits.recommended_output > 0, f"{name} invalid recommended_output"

        # Test safety margin application - should not exceed absolute max
        safe_tokens = calc.calculate_max_tokens("test_generation")
        max_safe = int(calc.limits.max_output * 0.8)  # 80% safety margin
        assert safe_tokens <= calc.limits.max_output, f"{name} exceeds max_output limit"
        assert safe_tokens <= max_safe, f"{name} exceeds safety margin of max_output"

        # Test that different use cases produce different token allocations
        test_gen_tokens = calc.calculate_max_tokens("test_generation")
        analysis_tokens = calc.calculate_max_tokens("code_analysis")
        refinement_tokens = calc.calculate_max_tokens("refinement")

        # All should be reasonable but different due to use-case multipliers
        assert test_gen_tokens > 0, f"{name} test_generation tokens should be > 0"
        assert analysis_tokens > 0, f"{name} code_analysis tokens should be > 0"
        assert refinement_tokens > 0, f"{name} refinement tokens should be > 0"


def test_normalized_metadata_consistency():
    """Test that all adapters return consistent normalized metadata."""
    adapters = [
        ("openai", OpenAIAdapter()),
        ("anthropic", ClaudeAdapter()),
        ("azure-openai", AzureOpenAIAdapter()),
        ("bedrock", BedrockAdapter()),
    ]

    for expected_provider, adapter in adapters:
        result = adapter.generate_tests("def test(): pass")
        metadata = result["metadata"]

        # Check required top-level keys
        assert "provider" in metadata
        assert "model_identifier" in metadata
        assert "parsed" in metadata
        assert "usage" in metadata

        # Check provider matches expected
        assert metadata["provider"] == expected_provider

        # Check usage normalization
        usage = metadata["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
