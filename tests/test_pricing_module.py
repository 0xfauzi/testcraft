"""Tests for the centralized pricing module."""

import pytest
from unittest.mock import Mock, patch

from testcraft.adapters.llm.pricing import (
    get_pricing,
    get_pricing_per_1k,
    calculate_cost,
    PricingError,
    _infer_provider,
    _extract_token_counts,
    _get_fallback_pricing,
)


class TestGetPricing:
    """Test get_pricing function."""

    def test_get_pricing_with_catalog_data(self):
        """Test getting pricing from model catalog."""
        # Test with a model that should be in the catalog
        pricing = get_pricing("claude-3-7-sonnet", "anthropic")
        
        assert "input" in pricing
        assert "output" in pricing
        assert isinstance(pricing["input"], (int, float))
        assert isinstance(pricing["output"], (int, float))
        assert pricing["input"] > 0
        assert pricing["output"] > 0

    def test_get_pricing_with_alias(self):
        """Test getting pricing using model alias."""
        # Test with an alias that should resolve
        pricing = get_pricing("claude-3.7-sonnet", "anthropic")
        
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_inferred_provider(self):
        """Test getting pricing with inferred provider."""
        pricing = get_pricing("claude-3-7-sonnet")  # No provider specified
        
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_model_not_found(self):
        """Test error when model not found."""
        with pytest.raises(PricingError, match="Model 'nonexistent-model' not found"):
            get_pricing("nonexistent-model", "openai")

    def test_get_pricing_no_pricing_data(self):
        """Test error when model has no pricing data."""
        # Mock a model entry without pricing
        mock_entry = Mock()
        mock_entry.pricing = None
        
        with patch("testcraft.adapters.llm.pricing.resolve_model", return_value=mock_entry):
            with pytest.raises(PricingError, match="Pricing not available"):
                get_pricing("test-model", "test-provider")


class TestGetPricingPer1k:
    """Test get_pricing_per_1k function."""

    def test_get_pricing_per_1k_conversion(self):
        """Test conversion from per-million to per-1k pricing."""
        # Mock get_pricing to return known values
        with patch("testcraft.adapters.llm.pricing.get_pricing") as mock_get_pricing:
            mock_get_pricing.return_value = {"input": 3000.0, "output": 15000.0}  # Per million
            
            pricing = get_pricing_per_1k("test-model", "test-provider")
            
            assert pricing["input"] == 3.0  # 3000 / 1000
            assert pricing["output"] == 15.0  # 15000 / 1000


class TestCalculateCost:
    """Test calculate_cost function."""

    def test_calculate_cost_with_object_usage(self):
        """Test cost calculation with object-style usage."""
        # Mock usage object
        usage = Mock()
        usage.prompt_tokens = 1000
        usage.completion_tokens = 500
        
        with patch("testcraft.adapters.llm.pricing.get_pricing_per_1k") as mock_pricing:
            mock_pricing.return_value = {"input": 0.003, "output": 0.015}
            
            cost = calculate_cost(usage, "test-model", "test-provider")
            
            # Expected: (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
            assert cost == pytest.approx(0.0105, rel=1e-6)

    def test_calculate_cost_with_dict_usage(self):
        """Test cost calculation with dict-style usage."""
        usage = {
            "input_tokens": 2000,
            "output_tokens": 1000,
        }
        
        with patch("testcraft.adapters.llm.pricing.get_pricing_per_1k") as mock_pricing:
            mock_pricing.return_value = {"input": 0.001, "output": 0.005}
            
            cost = calculate_cost(usage, "test-model", "test-provider")
            
            # Expected: (2000/1000 * 0.001) + (1000/1000 * 0.005) = 0.002 + 0.005 = 0.007
            assert cost == pytest.approx(0.007, rel=1e-6)

    def test_calculate_cost_with_fallback_pricing(self):
        """Test cost calculation falls back to hardcoded pricing when catalog fails."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        
        # Mock get_pricing_per_1k to raise PricingError
        with patch("testcraft.adapters.llm.pricing.get_pricing_per_1k", side_effect=PricingError("Test error")):
            cost = calculate_cost(usage, "gpt-4o-mini", "openai")
            
            # Should use fallback pricing without raising error
            assert isinstance(cost, float)
            assert cost > 0

    def test_calculate_cost_anthropic_usage_format(self):
        """Test cost calculation with Anthropic-style usage format."""
        # Create a simple object with the required attributes
        class MockUsage:
            def __init__(self):
                self.input_tokens = 1500
                self.output_tokens = 750
        
        usage = MockUsage()
        
        with patch("testcraft.adapters.llm.pricing.get_pricing_per_1k") as mock_pricing:
            mock_pricing.return_value = {"input": 0.003, "output": 0.015}
            
            cost = calculate_cost(usage, "claude-3-sonnet", "anthropic")
            
            # Expected: (1500/1000 * 0.003) + (750/1000 * 0.015) = 0.0045 + 0.01125 = 0.01575
            assert cost == pytest.approx(0.01575, rel=1e-6)


class TestInferProvider:
    """Test _infer_provider function."""

    def test_infer_openai_models(self):
        """Test inferring OpenAI provider."""
        assert _infer_provider("gpt-4") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o4-mini") == "openai"
        assert _infer_provider("gpt-3.5-turbo") == "openai"

    def test_infer_anthropic_models(self):
        """Test inferring Anthropic provider."""
        assert _infer_provider("claude-3-sonnet") == "anthropic"
        assert _infer_provider("claude-opus-4") == "anthropic"
        assert _infer_provider("sonnet-4") == "anthropic"

    def test_infer_azure_models(self):
        """Test inferring Azure provider."""
        assert _infer_provider("azure-gpt-4") == "azure-openai"

    def test_infer_bedrock_models(self):
        """Test inferring Bedrock provider."""
        assert _infer_provider("bedrock-claude") == "bedrock"

    def test_infer_unknown_model_defaults_to_openai(self):
        """Test that unknown models default to OpenAI."""
        assert _infer_provider("unknown-model") == "openai"


class TestExtractTokenCounts:
    """Test _extract_token_counts function."""

    def test_extract_openai_style_tokens(self):
        """Test extracting OpenAI-style token counts."""
        usage = Mock()
        usage.prompt_tokens = 1000
        usage.completion_tokens = 500
        
        input_tokens, output_tokens = _extract_token_counts(usage)
        
        assert input_tokens == 1000
        assert output_tokens == 500

    def test_extract_anthropic_style_tokens(self):
        """Test extracting Anthropic-style token counts."""
        # Create a simple object with the required attributes
        class MockUsage:
            def __init__(self):
                self.input_tokens = 1500
                self.output_tokens = 750
        
        usage = MockUsage()
        
        input_tokens, output_tokens = _extract_token_counts(usage)
        
        assert input_tokens == 1500
        assert output_tokens == 750

    def test_extract_dict_style_tokens(self):
        """Test extracting token counts from dictionary."""
        usage = {
            "prompt_tokens": 2000,
            "completion_tokens": 1000,
        }
        
        input_tokens, output_tokens = _extract_token_counts(usage)
        
        assert input_tokens == 2000
        assert output_tokens == 1000

    def test_extract_dict_anthropic_style_tokens(self):
        """Test extracting Anthropic-style token counts from dictionary."""
        usage = {
            "input_tokens": 2500,
            "output_tokens": 1250,
        }
        
        input_tokens, output_tokens = _extract_token_counts(usage)
        
        assert input_tokens == 2500
        assert output_tokens == 1250

    def test_extract_tokens_missing_data(self):
        """Test extracting tokens with missing data defaults to zero."""
        usage = {"prompt_tokens": 1000}  # Missing completion_tokens
        
        input_tokens, output_tokens = _extract_token_counts(usage)
        
        assert input_tokens == 1000
        assert output_tokens == 0

    def test_extract_tokens_invalid_format(self):
        """Test error with invalid usage format."""
        with pytest.raises(PricingError, match="Cannot extract token counts"):
            _extract_token_counts("invalid")

    def test_extract_tokens_invalid_values(self):
        """Test error with invalid token values."""
        usage = {"prompt_tokens": "invalid", "completion_tokens": 500}
        
        with pytest.raises(PricingError, match="Invalid token count format"):
            _extract_token_counts(usage)


class TestGetFallbackPricing:
    """Test _get_fallback_pricing function."""

    def test_fallback_openai_exact_match(self):
        """Test fallback pricing for exact OpenAI model match."""
        pricing = _get_fallback_pricing("gpt-4", "openai")
        
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03

    def test_fallback_anthropic_exact_match(self):
        """Test fallback pricing for exact Anthropic model match."""
        pricing = _get_fallback_pricing("claude-3-opus", "anthropic")
        
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03

    def test_fallback_partial_match(self):
        """Test fallback pricing with partial model name match."""
        pricing = _get_fallback_pricing("gpt-4-custom", "openai")
        
        # Should use conservative fallback rates
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03

    def test_fallback_no_match_openai_default(self):
        """Test fallback pricing when no match found for OpenAI."""
        pricing = _get_fallback_pricing("unknown-model", "openai")
        
        # Should use conservative fallback rates
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03

    def test_fallback_no_match_anthropic_default(self):
        """Test fallback pricing when no match found for Anthropic."""
        pricing = _get_fallback_pricing("unknown-model", "anthropic")
        
        # Should use conservative fallback rates
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03

    def test_fallback_inferred_provider(self):
        """Test fallback pricing with inferred provider."""
        pricing = _get_fallback_pricing("gpt-4", None)  # No provider specified
        
        # Should use conservative fallback rates
        assert pricing["input"] == 0.01
        assert pricing["output"] == 0.03


class TestIntegration:
    """Integration tests for the pricing module."""

    def test_end_to_end_openai_calculation(self):
        """Test end-to-end cost calculation for OpenAI."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        }
        
        # This should work with real catalog data or fallback
        cost = calculate_cost(usage, "gpt-4o-mini", "openai")
        
        assert isinstance(cost, float)
        assert cost > 0
        # Should be very small cost for mini model
        assert cost < 0.01

    def test_end_to_end_anthropic_calculation(self):
        """Test end-to-end cost calculation for Anthropic."""
        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
        }
        
        # This should work with real catalog data or fallback
        cost = calculate_cost(usage, "claude-3-7-sonnet", "anthropic")
        
        assert isinstance(cost, float)
        assert cost > 0

    def test_pricing_consistency_across_formats(self):
        """Test that different usage formats give same results."""
        # Object-style usage
        usage_obj = Mock()
        usage_obj.prompt_tokens = 1000
        usage_obj.completion_tokens = 500
        
        # Dict-style usage
        usage_dict = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        }
        
        cost_obj = calculate_cost(usage_obj, "gpt-4o-mini", "openai")
        cost_dict = calculate_cost(usage_dict, "gpt-4o-mini", "openai")
        
        assert cost_obj == pytest.approx(cost_dict, rel=1e-6)
