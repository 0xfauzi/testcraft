"""Token calculator catalog compliance tests.

This module tests that TokenCalculator respects catalog caps and contexts,
handles unknown models properly, and integrates with catalog limits.

Extracted from the original test_catalog_validation.py as part of the refactoring
to split concerns and use fixtures.
"""

import logging
import pytest
from unittest.mock import patch

from testcraft.adapters.llm.token_calculator import TokenCalculator
from testcraft.config.model_catalog_loader import resolve_model
from tests.conftest import (
    SAFETY_MARGIN_CASES,
    USE_CASE_SCENARIOS,
)

# Provider/model combinations for parametrize (extracted from fixture)
PROVIDER_MODEL_CASES = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet"),
    ("anthropic", "claude-3-7-sonnet"),
]


class TestTokenCalculatorCatalogCompliance:
    """Test TokenCalculator respecting catalog caps and contexts."""

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet"),
        ("anthropic", "claude-3-7-sonnet"),
    ])
    def test_token_calculator_respects_catalog_max_context(self, provider, model):
        """Test that TokenCalculator respects catalog max_context limits."""
        entry = resolve_model(provider, model)
        if not entry:
            pytest.skip(f"Model {provider}/{model} not found in catalog")
        
        calc = TokenCalculator(provider=provider, model=model)
        
        # TokenCalculator should respect the catalog limits
        assert calc.limits.max_context == entry.limits.max_context
        
        # Calculate tokens with reasonable input should be properly bounded
        large_input = int(entry.limits.max_context * 0.7)  # 70% of context
        max_tokens = calc.calculate_max_tokens(
            use_case="test_generation", 
            input_length=large_input,
            safety_margin=0.8
        )
        
        # Should not exceed available context window (with reasonable safety margin)
        max_allowed_context = int(entry.limits.max_context * 0.8)
        assert large_input + max_tokens <= max_allowed_context

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet"),
        ("openai", "gpt-4o-mini"),
    ])
    def test_token_calculator_respects_catalog_max_output(self, provider, model):
        """Test that TokenCalculator respects catalog default_max_output limits."""
        entry = resolve_model(provider, model)
        if not entry:
            pytest.skip(f"Model {provider}/{model} not found in catalog")
            
        calc = TokenCalculator(provider=provider, model=model)
        
        # Standard calculation should not exceed catalog limits
        max_tokens = calc.calculate_max_tokens(use_case="test_generation")
        assert max_tokens <= entry.limits.default_max_output
        
        # Even with high safety margin, should not exceed catalog max
        max_tokens_high_safety = calc.calculate_max_tokens(
            use_case="test_generation", 
            safety_margin=1.0  # No safety margin
        )
        assert max_tokens_high_safety <= entry.limits.default_max_output

    def test_token_calculator_thinking_budgets_only_when_supported(self):
        """Test that thinking budgets are only applied when model supports them."""
        # Test with a model that doesn't support thinking (OpenAI GPT models)
        openai_calc = TokenCalculator(provider="openai", model="gpt-4o")
        openai_entry = resolve_model("openai", "gpt-4o")
        
        if openai_entry and (not openai_entry.limits.max_thinking or openai_entry.limits.max_thinking <= 0):
            thinking_tokens = openai_calc.calculate_thinking_tokens(use_case="test_generation")
            assert thinking_tokens is None  # Should not support thinking
            assert not openai_calc.supports_thinking_mode()
        
        # Test with a model that supports thinking (if available in catalog)
        anthropic_calc = TokenCalculator(provider="anthropic", model="claude-sonnet-4")
        anthropic_entry = resolve_model("anthropic", "claude-sonnet-4")
        
        if anthropic_entry and anthropic_entry.limits.max_thinking and anthropic_entry.limits.max_thinking > 0:
            thinking_tokens = anthropic_calc.calculate_thinking_tokens(use_case="test_generation")
            assert thinking_tokens is not None
            assert thinking_tokens > 0
            assert thinking_tokens <= anthropic_entry.limits.max_thinking
            assert anthropic_calc.supports_thinking_mode()

    def test_token_calculator_unknown_model_safe_defaults(self):
        """Test that unknown models fall back to safe defaults with proper warnings."""
        # Capture logging to verify warning
        with patch('testcraft.adapters.llm.token_calculator.logger') as mock_logger:
            calc = TokenCalculator(provider="unknown-provider", model="unknown-model")
            
            # Should have logged a warning about unknown model
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Unknown model" in warning_call
            assert "safe defaults" in warning_call
            
            # Should have conservative defaults
            assert calc.limits.max_context == 200000
            assert calc.limits.max_output == 32000
            assert calc.limits.recommended_output == 25000

    def test_token_calculator_extended_features_respect_catalog_limits(self):
        """Test that extended features still respect catalog limits as baseline."""
        # Test with extended features enabled
        calc_extended = TokenCalculator(
            provider="anthropic", 
            model="claude-sonnet-4",
            enable_extended_context=True,
            enable_extended_output=True
        )
        
        entry = resolve_model("anthropic", "claude-sonnet-4")
        if entry:
            # Even with extended features, should not go below catalog minimums
            assert calc_extended.limits.max_context >= entry.limits.max_context
            # Extended output might be higher, but should be reasonable
            assert calc_extended.limits.max_output >= entry.limits.default_max_output

    @pytest.mark.parametrize("safety_margin", SAFETY_MARGIN_CASES)
    def test_token_calculator_safety_margins_enforced(self, safety_margin):
        """Test that safety margins are properly enforced and don't exceed catalog limits."""
        calc = TokenCalculator(provider="openai", model="gpt-4o")
        entry = resolve_model("openai", "gpt-4o")
        
        if not entry:
            pytest.skip("OpenAI gpt-4o not found in catalog")
            
        max_tokens = calc.calculate_max_tokens(
            use_case="test_generation", 
            safety_margin=safety_margin
        )
        
        # Should never exceed the safety-adjusted catalog limit
        expected_max = int(entry.limits.default_max_output * safety_margin)
        assert max_tokens <= expected_max

    @pytest.mark.parametrize("use_case", USE_CASE_SCENARIOS)
    def test_token_calculator_use_case_multipliers_within_bounds(self, use_case):
        """Test that use case multipliers don't cause violations of catalog limits."""
        calc = TokenCalculator(provider="anthropic", model="claude-3-5-sonnet")
        entry = resolve_model("anthropic", "claude-3-5-sonnet")
        
        if not entry:
            pytest.skip("Claude 3.5 Sonnet not found in catalog")
            
        max_tokens = calc.calculate_max_tokens(use_case=use_case)
        
        # Even with multipliers, should not exceed catalog limits
        assert max_tokens <= entry.limits.default_max_output
        assert max_tokens > 0  # Should still be positive

    def test_token_calculator_context_ceiling_enforcement(self):
        """Test that context ceiling is enforced based on catalog max_context."""
        calc = TokenCalculator(provider="openai", model="gpt-4o")
        entry = resolve_model("openai", "gpt-4o")
        
        if not entry:
            pytest.skip("OpenAI gpt-4o not found in catalog")
            
        # Test with input that approaches context limit
        safety_margin = 0.8
        safe_context = int(entry.limits.max_context * safety_margin)
        
        # Input near the limit
        large_input = safe_context - 1000
        max_tokens = calc.calculate_max_tokens(
            use_case="test_generation",
            input_length=large_input,
            safety_margin=safety_margin
        )
        
        # Should leave space for output while respecting context ceiling
        assert max_tokens <= (safe_context - large_input)
        assert max_tokens > 0

    def test_token_calculator_minimum_viable_output_enforced(self):
        """Test that minimum viable output is enforced even under extreme constraints."""
        calc = TokenCalculator(provider="openai", model="gpt-4o")
        
        # Test with input that would leave no room for output
        max_tokens = calc.calculate_max_tokens(
            use_case="test_generation",
            input_length=999999,  # Extremely large input
            safety_margin=0.8
        )
        
        # Should still provide minimum viable output
        assert max_tokens >= 100  # Minimum as defined in TokenCalculator

    def test_token_calculator_reasoning_model_detection_from_catalog(self):
        """Test that reasoning model capabilities are detected from catalog flags."""
        # Test o1-mini which should be marked as reasoning capable in catalog
        o1_calc = TokenCalculator(provider="openai", model="o1-mini")
        o1_entry = resolve_model("openai", "o1-mini")
        
        if o1_entry and o1_entry.flags:
            expected_reasoning = o1_entry.flags.reasoning_capable or False
            assert o1_calc.is_reasoning_model() == expected_reasoning
            
            if expected_reasoning:
                assert o1_calc.has_reasoning_capabilities() is True

    def test_token_calculator_get_model_info_reflects_catalog(self):
        """Test that get_model_info returns information consistent with catalog."""
        calc = TokenCalculator(provider="anthropic", model="claude-3-5-sonnet")
        entry = resolve_model("anthropic", "claude-3-5-sonnet")
        
        if not entry:
            pytest.skip("Claude 3.5 Sonnet not found in catalog")
            
        model_info = calc.get_model_info()
        
        # Verify consistency with catalog
        assert model_info["provider"] == entry.provider
        assert model_info["model"] == entry.model_id
        assert model_info["max_context_tokens"] == entry.limits.max_context
        assert model_info["max_output_tokens"] == entry.limits.default_max_output
        
        # Verify calculated fields are reasonable
        assert model_info["supports_large_context"] == (entry.limits.max_context > 32000)
        assert model_info["supports_large_output"] == (entry.limits.default_max_output > 8000)

    def test_token_calculator_supported_models_from_catalog(self):
        """Test that get_supported_models returns models from catalog."""
        from testcraft.config.model_catalog_loader import get_providers, get_models
        
        # Test for each provider
        providers = get_providers()
        
        for provider in providers[:3]:  # Test first 3 providers to keep test fast
            models = TokenCalculator.get_supported_models(provider)
            catalog_models = get_models(provider)
            
            # Should return the same models as the catalog
            assert set(models) == set(catalog_models)

    def test_token_calculator_all_providers_from_catalog(self):
        """Test that get_all_providers returns providers from catalog."""
        from testcraft.config.model_catalog_loader import get_providers
        
        calc_providers = TokenCalculator.get_all_providers()
        catalog_providers = get_providers()
        
        # Should return the same providers as the catalog
        assert set(calc_providers) == set(catalog_providers)


class TestTokenCalculatorIntegration:
    """Test token calculator integration with existing test patterns."""

    def test_integration_with_existing_test_patterns(self):
        """Test that catalog validation integrates with existing test patterns."""
        # Test existing token calculator patterns still work
        calc = TokenCalculator(provider="anthropic", model="claude-3-7-sonnet")
        thinking = calc.calculate_thinking_tokens(use_case="code_analysis", complexity_level="complex")
        
        # Should work as before, but now with catalog backing
        if calc.supports_thinking_mode():
            assert thinking is None or thinking > 0
        
        # Test that get_model_info still works
        info = calc.get_model_info()
        assert "provider" in info
        assert "model" in info
        assert "max_context_tokens" in info

    @pytest.mark.parametrize("provider,model", PROVIDER_MODEL_CASES)
    def test_typical_workflow_token_budgets_stay_within_catalog_bounds(self, provider, model):
        """Test that typical workflow token budgets stay within catalog bounds."""
        entry = resolve_model(provider, model)
        if not entry:
            pytest.skip(f"Model {provider}/{model} not found in catalog")
            
        calc = TokenCalculator(provider=provider, model=model)
        
        # Test different workflow scenarios
        workflows = [
            ("test_generation", "Large test file generation"),
            ("code_analysis", "Complex code analysis"),
            ("refinement", "Test refinement and improvement"),
        ]
        
        for use_case, description in workflows:
            max_tokens = calc.calculate_max_tokens(use_case)
            
            # Should always respect catalog limits
            assert max_tokens <= entry.limits.default_max_output, f"Failed for {provider}/{model} {use_case}"
            assert max_tokens > 0, f"No tokens allocated for {provider}/{model} {use_case}"
            
            # Test thinking tokens if supported
            if calc.supports_thinking_mode():
                thinking_tokens = calc.calculate_thinking_tokens(use_case)
                if thinking_tokens is not None:
                    assert thinking_tokens <= entry.limits.max_thinking, f"Thinking tokens exceeded for {provider}/{model}"

    def test_error_handling_with_catalog_validation(self):
        """Test error handling when catalog validation fails."""
        # Test with nonexistent model
        unknown_calc = TokenCalculator(provider="unknown", model="unknown")
        
        # Should fall back to safe defaults without crashing
        assert unknown_calc.limits.max_context > 0
        assert unknown_calc.limits.max_output > 0
        
        # Should still be able to calculate tokens
        tokens = unknown_calc.calculate_max_tokens("test_generation")
        assert tokens > 0
