import pytest
from unittest.mock import patch, MagicMock

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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "model" in metadata
    assert "parsed" in metadata


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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "model" in metadata
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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "deployment" in metadata
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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "model_id" in metadata
    assert "parsed" in metadata


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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "deployment" in metadata
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
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "model_id" in metadata
    assert "parsed" in metadata


def test_claude_refine_content():
    """Test Claude adapter refines content with correct structure."""
    adapter = ClaudeAdapter()
    out = adapter.refine_content("print(1)", "improve")
    
    # Verify standardized return structure
    assert "refined_content" in out
    assert "changes_made" in out
    assert "confidence" in out
    assert "metadata" in out
    
    # Verify metadata contains expected keys
    metadata = out["metadata"]
    assert "model" in metadata
    assert "parsed" in metadata


@patch('testcraft.prompts.registry.PromptRegistry.get_system_prompt')
@patch('testcraft.prompts.registry.PromptRegistry.get_user_prompt')
def test_adapters_use_prompt_registry(mock_user_prompt, mock_system_prompt):
    """Test that all adapters use the prompt registry for prompts."""
    mock_system_prompt.return_value = "System prompt from registry"
    mock_user_prompt.return_value = "User prompt from registry"
    
    adapters = [
        OpenAIAdapter(),
        ClaudeAdapter(), 
        AzureOpenAIAdapter(),
        BedrockAdapter()
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
        mock_system_prompt.assert_called_with(
            prompt_type="llm_content_refinement"
        )
        
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
        ("Bedrock", BedrockAdapter())
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
        BedrockAdapter()
    ]
    
    for adapter in adapters:
        # Test generate_tests structure
        result = adapter.generate_tests("def f(): pass")
        assert isinstance(result["tests"], str)
        assert isinstance(result["coverage_focus"], list)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]
        
        # Test analyze_code structure  
        result = adapter.analyze_code("def g(): pass")
        assert isinstance(result["testability_score"], (int, float))
        assert isinstance(result["complexity_metrics"], dict)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["potential_issues"], list)
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]
        
        # Test refine_content structure
        result = adapter.refine_content("old content", "instructions")
        assert isinstance(result["refined_content"], str)
        assert isinstance(result["changes_made"], str)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["metadata"], dict)
        assert "parsed" in result["metadata"]


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
