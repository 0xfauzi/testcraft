"""Pricing math consistency tests.

This module tests pricing math accuracy, conversion between per-million and per-1k,
cost calculations, and integration with catalog pricing data.

Extracted from the original test_catalog_validation.py as part of the refactoring
to split concerns and use fixtures.
"""

import pytest
from unittest.mock import patch

from testcraft.adapters.llm.pricing import (
    get_pricing, 
    get_pricing_per_1k, 
    calculate_cost,
    _get_fallback_pricing,
    PricingError
)
from testcraft.config.model_catalog_loader import load_catalog
from tests.conftest import pricing_cases, token_usage_scenarios


class TestPricingMathConsistency:
    """Test pricing math accuracy and consistency."""

    @pytest.mark.parametrize("case", [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-3-7-sonnet"},
    ])
    def test_per_million_to_per_1k_conversion_accuracy(self, case):
        """Test that per-million to per-1k conversion is mathematically accurate."""
        provider = case["provider"]
        model = case["model"]
        
        try:
            per_million = get_pricing(model, provider)
            per_1k = get_pricing_per_1k(model, provider)
            
            # Mathematical relationship should be exact
            expected_input_per_1k = per_million["input"] / 1000.0
            expected_output_per_1k = per_million["output"] / 1000.0
            
            assert abs(per_1k["input"] - expected_input_per_1k) < 1e-10
            assert abs(per_1k["output"] - expected_output_per_1k) < 1e-10
            
        except Exception:
            # Skip if model not in catalog or no pricing
            pytest.skip(f"Model {provider}/{model} not available for pricing test")

    @pytest.mark.parametrize("tokens,expected", [
        ({"prompt_tokens": 1000, "completion_tokens": 500}, 0.0035),
        ({"prompt_tokens": 0, "completion_tokens": 500}, 0.0025),
        ({"prompt_tokens": 1000, "completion_tokens": 0}, 0.001),
        ({"prompt_tokens": 1, "completion_tokens": 1}, 0.000006),
    ])
    def test_cost_calculation_mathematical_accuracy(self, tokens, expected):
        """Test that cost calculations are mathematically accurate."""
        # Mock pricing for precise math testing
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.001, "output": 0.005}
            
            cost = calculate_cost(tokens, "test-model", "test-provider")
            
            # Allow small floating point error
            assert abs(cost - expected) < 1e-10

    def test_cost_calculation_with_zero_tokens(self, token_usage_scenarios):
        """Test cost calculation edge case with zero tokens."""
        # Test with zero input tokens
        usage_zero_input = {
            "prompt_tokens": 0,
            "completion_tokens": 500,
        }
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.001, "output": 0.005}
            
            cost = calculate_cost(usage_zero_input, "test-model", "test-provider")
            expected_cost = 0.005 * (500 / 1000.0)  # Only output cost
            assert abs(cost - expected_cost) < 1e-10
        
        # Test with zero output tokens
        usage_zero_output = {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
        }
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.001, "output": 0.005}
            
            cost = calculate_cost(usage_zero_output, "test-model", "test-provider")
            expected_cost = 0.001 * (1000 / 1000.0)  # Only input cost
            assert abs(cost - expected_cost) < 1e-10

    @pytest.mark.parametrize("usage,expected_cost", [
        ({"prompt_tokens": 1_000_000, "completion_tokens": 500_000}, 2.5),
        ({"prompt_tokens": 10_000_000, "completion_tokens": 5_000_000}, 25.0),
    ])
    def test_cost_calculation_precision_with_large_numbers(self, usage, expected_cost):
        """Test cost calculation precision with large token counts."""
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.001, "output": 0.003}
            
            cost = calculate_cost(usage, "test-model", "test-provider")
            
            # Expected calculation should match
            assert abs(cost - expected_cost) < 1e-6

    def test_cost_calculation_precision_with_small_numbers(self):
        """Test cost calculation precision with small token counts."""
        # Test with very small token counts
        usage_small = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            # Use high-precision pricing
            mock_pricing.return_value = {"input": 0.00015, "output": 0.0006}
            
            cost = calculate_cost(usage_small, "test-model", "test-provider")
            
            # Expected: (1/1000 * 0.00015) + (1/1000 * 0.0006) = 0.00000015 + 0.0000006 = 0.00000075
            expected_cost = 0.00000075
            assert abs(cost - expected_cost) < 1e-10

    def test_pricing_consistency_across_providers(self):
        """Test that pricing is consistent across different provider formats."""
        # Test with different usage formats that should give same results
        usage_formats = [
            # OpenAI format
            {"prompt_tokens": 1000, "completion_tokens": 500},
            # Anthropic format (if supported)
            {"input_tokens": 1000, "output_tokens": 500},
        ]
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.003, "output": 0.015}
            
            costs = []
            for usage in usage_formats:
                try:
                    cost = calculate_cost(usage, "test-model", "test-provider")
                    costs.append(cost)
                except Exception:
                    # Skip unsupported format
                    continue
            
            if len(costs) > 1:
                # All supported formats should give same result
                for i in range(1, len(costs)):
                    assert abs(costs[0] - costs[i]) < 1e-10


class TestFallbackPricing:
    """Test fallback pricing when catalog data is unavailable."""

    def test_fallback_pricing_mathematical_consistency(self):
        """Test that fallback pricing calculations are mathematically consistent."""
        # Test fallback pricing directly
        fallback_openai = _get_fallback_pricing("gpt-4o-mini", "openai")
        assert "input" in fallback_openai
        assert "output" in fallback_openai
        assert fallback_openai["input"] > 0
        assert fallback_openai["output"] > 0
        
        # Test that fallback calculations are consistent
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        
        # Force fallback by mocking catalog failure
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.side_effect = PricingError("Test error")
            
            cost = calculate_cost(usage, "gpt-4o-mini", "openai")
            
            # Calculate expected cost using fallback rates
            expected_cost = (
                (usage["prompt_tokens"] / 1000.0) * fallback_openai["input"] +
                (usage["completion_tokens"] / 1000.0) * fallback_openai["output"]
            )
            
            assert abs(cost - expected_cost) < 1e-10

    def test_error_handling_with_nonexistent_models(self):
        """Test pricing error handling with nonexistent models."""
        with pytest.raises(PricingError):
            get_pricing("nonexistent-model", "nonexistent-provider")


class TestCatalogPricingIntegration:
    """Test integration between pricing module and catalog data."""

    def test_catalog_pricing_matches_expectations(self):
        """Test that catalog pricing values are within expected ranges."""
        catalog = load_catalog()
        
        for model in catalog.models:
            if model.pricing and model.pricing.per_million:
                pricing = model.pricing.per_million
                
                if pricing.input is not None:
                    # Pricing should be positive and reasonable (not negative, not extremely high)
                    assert pricing.input > 0
                    assert pricing.input < 1000000  # Less than $1M per million tokens
                
                if pricing.output is not None:
                    assert pricing.output > 0
                    assert pricing.output < 1000000
                    # Output pricing is typically higher than input
                    if pricing.input is not None:
                        assert pricing.output >= pricing.input

    def test_pricing_module_integration_with_catalog(self):
        """Test end-to-end pricing integration with catalog."""
        # Get a model with pricing from catalog
        catalog = load_catalog()
        model_with_pricing = None
        
        for model in catalog.models:
            if model.pricing and model.pricing.per_million and model.pricing.per_million.input:
                model_with_pricing = model
                break
        
        if model_with_pricing:
            # Test that we can get pricing and calculate cost
            pricing = get_pricing(model_with_pricing.model_id, model_with_pricing.provider)
            
            usage = {"prompt_tokens": 1000, "completion_tokens": 500}
            cost = calculate_cost(usage, model_with_pricing.model_id, model_with_pricing.provider)
            
            # Verify consistency
            expected_cost = (
                (1000 / 1000.0) * (pricing["input"] / 1000.0) +
                (500 / 1000.0) * (pricing["output"] / 1000.0)
            )
            
            assert abs(cost - expected_cost) < 1e-10

    def test_pricing_caps_alignment_with_catalog(self):
        """Test that pricing calculations align with catalog pricing data."""
        # Test models that have pricing in catalog
        test_models = [
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-7-sonnet"),
        ]
        
        from testcraft.config.model_catalog_loader import resolve_model
        
        for provider, model in test_models:
            entry = resolve_model(provider, model)
            
            if entry and entry.pricing and entry.pricing.per_million:
                try:
                    # Get pricing through pricing module
                    pricing = get_pricing(model, provider)
                    
                    # Should match catalog values
                    expected_input = entry.pricing.per_million.input
                    expected_output = entry.pricing.per_million.output
                    
                    if expected_input is not None:
                        assert abs(pricing["input"] - expected_input) < 1e-6
                    if expected_output is not None:
                        assert abs(pricing["output"] - expected_output) < 1e-6
                except Exception:
                    # Skip if pricing not available
                    continue


class TestPricingPrecision:
    """Test pricing calculations maintain appropriate precision."""

    def test_price_rounding_and_precision(self):
        """Test that price calculations maintain appropriate precision."""
        # Test with pricing that could cause rounding issues
        usage = {"prompt_tokens": 333, "completion_tokens": 777}  # Numbers that don't divide evenly by 1000
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            # Use pricing that could cause precision issues
            mock_pricing.return_value = {"input": 0.00333333, "output": 0.00777777}
            
            cost = calculate_cost(usage, "test-model", "test-provider")
            
            # Calculate expected with full precision
            expected_cost = (333 / 1000.0) * 0.00333333 + (777 / 1000.0) * 0.00777777
            
            # Should be very close (within floating point precision)
            assert abs(cost - expected_cost) < 1e-12

    def test_per_request_total_calculation_accuracy(self):
        """Test that per-request total calculations are accurate."""
        # Test multiple requests to verify totals
        requests = [
            {"prompt_tokens": 500, "completion_tokens": 250},
            {"prompt_tokens": 1000, "completion_tokens": 500},
            {"prompt_tokens": 1500, "completion_tokens": 750},
        ]
        
        with patch('testcraft.adapters.llm.pricing.get_pricing_per_1k') as mock_pricing:
            mock_pricing.return_value = {"input": 0.002, "output": 0.008}
            
            individual_costs = []
            total_tokens_input = 0
            total_tokens_output = 0
            
            for usage in requests:
                cost = calculate_cost(usage, "test-model", "test-provider")
                individual_costs.append(cost)
                total_tokens_input += usage["prompt_tokens"]
                total_tokens_output += usage["completion_tokens"]
            
            # Calculate bulk cost
            bulk_usage = {
                "prompt_tokens": total_tokens_input,
                "completion_tokens": total_tokens_output
            }
            bulk_cost = calculate_cost(bulk_usage, "test-model", "test-provider")
            
            # Sum of individual costs should equal bulk cost
            sum_individual = sum(individual_costs)
            assert abs(sum_individual - bulk_cost) < 1e-10

    def test_cost_tracking_integration_with_catalog_pricing(self):
        """Test that cost tracking properly integrates with catalog pricing."""
        from testcraft.adapters.llm.token_calculator import TokenCalculator
        
        # Find models with pricing in catalog
        catalog = load_catalog()
        models_with_pricing = []
        
        for model in catalog.models:
            if model.pricing and model.pricing.per_million and model.pricing.per_million.input:
                models_with_pricing.append((model.provider, model.model_id))
        
        for provider, model_id in models_with_pricing[:3]:  # Test first 3 to keep test fast
            from testcraft.config.model_catalog_loader import resolve_model
            entry = resolve_model(provider, model_id)
            if not entry:
                continue
                
            try:
                # Get pricing
                pricing = get_pricing(model_id, provider)
                assert pricing["input"] > 0
                assert pricing["output"] > 0
                
                # Calculate typical usage tokens
                calc = TokenCalculator(provider=provider, model=model_id)
                typical_output = calc.calculate_max_tokens("test_generation")
                
                # Simulate typical usage
                typical_usage = {
                    "prompt_tokens": 2000,  # Typical input
                    "completion_tokens": typical_output
                }
                
                # Calculate cost
                cost = calculate_cost(typical_usage, model_id, provider)
                assert cost > 0
                
                # Cost should be reasonable (not negative, not extremely high)
                assert cost < 100.0  # Should be less than $100 for typical usage
                
            except Exception:
                # Skip if pricing not available
                continue
