"""Centralized pricing module for LLM adapters.

This module provides a unified API for retrieving model pricing information
from the model catalog and calculating API costs per request.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config.model_catalog_loader import resolve_model

logger = logging.getLogger(__name__)


class PricingError(Exception):
    """Pricing calculation errors."""
    pass


def get_pricing(model: str, provider: str | None = None) -> dict[str, float]:
    """Get pricing information for a model.
    
    Args:
        model: Model identifier or alias
        provider: Provider name (if None, will try to infer from common patterns)
        
    Returns:
        Dictionary with 'input' and 'output' pricing per million tokens
        
    Raises:
        PricingError: If model not found or pricing not available
    """
    # Try to infer provider if not specified
    if provider is None:
        provider = _infer_provider(model)
    
    # Resolve model from catalog
    entry = resolve_model(provider, model)
    if entry is None:
        raise PricingError(f"Model '{model}' not found for provider '{provider}'")
    
    # Extract pricing information
    if entry.pricing is None or entry.pricing.per_million is None:
        raise PricingError(f"Pricing not available for model '{model}' (provider: {provider})")
    
    pricing = entry.pricing.per_million
    if pricing.input is None or pricing.output is None:
        raise PricingError(f"Incomplete pricing data for model '{model}' (provider: {provider})")
    
    return {
        "input": pricing.input,
        "output": pricing.output,
    }


def get_pricing_per_1k(model: str, provider: str | None = None) -> dict[str, float]:
    """Get pricing information per 1,000 tokens (legacy format).
    
    Args:
        model: Model identifier or alias
        provider: Provider name (if None, will try to infer from common patterns)
        
    Returns:
        Dictionary with 'input' and 'output' pricing per 1,000 tokens
        
    Raises:
        PricingError: If model not found or pricing not available
    """
    per_million = get_pricing(model, provider)
    return {
        "input": per_million["input"] / 1000.0,
        "output": per_million["output"] / 1000.0,
    }


def calculate_cost(
    usage: dict[str, Any] | Any,
    model: str,
    provider: str | None = None,
) -> float:
    """Calculate API cost based on token usage and model pricing.
    
    Args:
        usage: Usage object or dict with token counts
        model: Model identifier or alias
        provider: Provider name (if None, will try to infer from common patterns)
        
    Returns:
        Calculated cost in USD
        
    Raises:
        PricingError: If model not found, pricing not available, or usage format invalid
    """
    # Get pricing per 1,000 tokens for calculation
    try:
        pricing = get_pricing_per_1k(model, provider)
    except PricingError:
        # Fallback to hardcoded pricing for backward compatibility
        logger.warning(f"Using fallback pricing for model '{model}' (provider: {provider})")
        pricing = _get_fallback_pricing(model, provider)
    
    # Extract token counts from usage (support multiple formats)
    input_tokens, output_tokens = _extract_token_counts(usage)
    
    # Calculate cost
    input_cost = (input_tokens / 1000.0) * pricing["input"]
    output_cost = (output_tokens / 1000.0) * pricing["output"]
    total_cost = input_cost + output_cost
    
    logger.debug(
        f"Cost calculation for {model}: input={input_cost:.6f}, output={output_cost:.6f}, total={total_cost:.6f}"
    )
    
    return total_cost


def _infer_provider(model: str) -> str:
    """Infer provider from model name patterns."""
    model_lower = model.lower()
    
    # Azure OpenAI patterns (check first since it contains "gpt")
    if "azure" in model_lower:
        return "azure-openai"
    
    # Bedrock patterns (check before anthropic since it might contain "claude")
    if "bedrock" in model_lower:
        return "bedrock"
    
    # OpenAI patterns
    if any(pattern in model_lower for pattern in ["gpt", "o4", "o3", "davinci", "curie", "babbage", "ada"]):
        return "openai"
    
    # Anthropic patterns
    if any(pattern in model_lower for pattern in ["claude", "sonnet", "opus", "haiku"]):
        return "anthropic"
    
    # Default to OpenAI for unknown patterns
    logger.warning(f"Could not infer provider for model '{model}', defaulting to 'openai'")
    return "openai"


def _extract_token_counts(usage: dict[str, Any] | Any) -> tuple[int, int]:
    """Extract input and output token counts from usage object or dict.
    
    Returns:
        Tuple of (input_tokens, output_tokens)
        
    Raises:
        PricingError: If token counts cannot be extracted
    """
    input_tokens = 0
    output_tokens = 0
    
    # Try object-style access first
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
    elif hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
    # Try dict-style access
    elif isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    else:
        raise PricingError(f"Cannot extract token counts from usage object: {type(usage)}")
    
    # Ensure we have valid integers
    try:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
    except (ValueError, TypeError) as e:
        raise PricingError(f"Invalid token count format: {e}") from e
    
    return input_tokens, output_tokens


def _get_fallback_pricing(model: str, provider: str | None) -> dict[str, float]:
    """Get safe fallback pricing when model catalog lookup fails.
    
    MIGRATION NOTE: Hardcoded pricing tables removed per Task 31.10.
    All pricing should come from the model catalog. This fallback provides
    only conservative defaults when the catalog is unavailable.
    """
    logger.warning(
        f"Model catalog unavailable for '{model}' (provider: {provider}). "
        f"Using conservative fallback pricing. Please ensure model catalog is properly configured."
    )
    
    # Conservative fallback - use moderate pricing estimates (per 1,000 tokens)
    # These are intentionally conservative to avoid unexpected costs
    return {
        "input": 0.01,   # $10 per million input tokens (0.01 per 1k)
        "output": 0.03   # $30 per million output tokens (0.03 per 1k)
    }
