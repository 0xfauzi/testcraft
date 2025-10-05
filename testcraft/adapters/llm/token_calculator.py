"""Smart token calculation for LLM adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from ...config.model_catalog import (
    ModelLimits,
)
from ...config.model_catalog import (
    get_all_models_for_provider as catalog_get_all_models_for_provider,
)
from ...config.model_catalog import (
    get_all_providers as catalog_get_all_providers,
)
from ...config.model_catalog import (
    get_flags as catalog_get_flags,
)
from ...config.model_catalog import (
    get_limits as catalog_get_limits,
)
from ...config.model_catalog import (
    normalize_model_id as catalog_normalize,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenLimits:
    """Token limits for a specific model."""

    max_context: int
    max_output: int
    recommended_output: int  # Conservative recommendation
    max_thinking: int | None = None  # Maximum thinking/reasoning tokens (if supported)
    recommended_thinking: int | None = None  # Conservative thinking tokens


class TokenCalculator:
    """Calculates optimal token allocation for different providers and use cases."""

    # Use case specific multipliers for recommended output
    USE_CASE_MULTIPLIERS: dict[str, float] = {
        "test_generation": 1.5,  # Test generation needs more tokens
        "code_analysis": 0.7,  # Analysis can be more concise
        "refinement": 1.2,  # Refinement needs moderate tokens
    }

    def __init__(self, provider: str, model: str) -> None:
        """Initialize token calculator for a specific provider and model.

        Args:
            provider: Provider name (openai, anthropic, azure-openai, bedrock)
            model: Model identifier
        """
        # Normalize to canonical provider/model for lookup in catalog
        canonical_provider, canonical_model = catalog_normalize(provider, model)
        self.provider = canonical_provider
        self.model = canonical_model
        try:
            cat_limits: ModelLimits = catalog_get_limits(self.provider, self.model)
            self.flags = catalog_get_flags(self.provider, self.model)
        except ValueError as e:  # Unknown model
            raise ValueError(
                f"Unknown model {provider}/{model}: add it to the catalog"
            ) from e

        # Adapt catalog limits to internal structure expected by methods
        self.limits = TokenLimits(
            max_context=cat_limits.max_context,
            max_output=cat_limits.default_max_output,
            recommended_output=cat_limits.default_max_output,
            max_thinking=cat_limits.max_thinking,
            recommended_thinking=(
                (cat_limits.max_thinking // 2) if cat_limits.max_thinking else None
            ),
        )

    def _get_model_limits(self) -> TokenLimits:
        """Deprecated: limits are fetched from the catalog in __init__."""
        return self.limits

    def calculate_max_tokens(
        self,
        use_case: Literal["test_generation", "code_analysis", "refinement"],
        input_length: int | None = None,
        safety_margin: float = 0.8,  # Use 80% of max to be safe
    ) -> int:
        """Calculate optimal max_tokens for a specific use case.

        Args:
            use_case: The type of operation being performed
            input_length: Estimated input length in tokens (optional)
            safety_margin: Safety factor to apply to limits (0.0-1.0)

        Returns:
            Recommended max_tokens value
        """
        # Start with base recommendation
        base_tokens = self.limits.recommended_output

        # Apply use case multiplier
        multiplier = self.USE_CASE_MULTIPLIERS.get(use_case, 1.0)
        adjusted_tokens = int(base_tokens * multiplier)

        # Apply safety margin
        safe_tokens = int(adjusted_tokens * safety_margin)

        # Ensure we don't exceed provider limits
        max_safe = int(self.limits.max_output * safety_margin)
        final_tokens = min(safe_tokens, max_safe)

        # If we have input length, ensure total doesn't exceed context window
        if input_length is not None:
            max_context_safe = int(self.limits.max_context * safety_margin)
            available_for_output = max_context_safe - input_length
            if available_for_output > 0:
                final_tokens = min(final_tokens, available_for_output)
            else:
                logger.warning(
                    f"Input length {input_length} exceeds safe context window"
                )
                final_tokens = min(final_tokens, 512)  # Minimal fallback

        # Ensure minimum viable output
        final_tokens = max(final_tokens, 100)

        logger.debug(
            f"Calculated max_tokens: {final_tokens} for {use_case} on {self.provider}/{self.model}"
        )
        return final_tokens

    def calculate_thinking_tokens(
        self,
        use_case: Literal["test_generation", "code_analysis", "refinement"],
        complexity_level: Literal["simple", "moderate", "complex"] = "moderate",
        safety_margin: float = 0.8,
    ) -> int | None:
        """Calculate optimal thinking tokens for extended reasoning mode.

        Args:
            use_case: The type of operation being performed
            complexity_level: Complexity of the task (affects thinking budget)
            safety_margin: Safety factor to apply to limits (0.0-1.0)

        Returns:
            Recommended thinking tokens, or None if model doesn't support thinking
        """
        # Check support at the very beginning and return None immediately if not supported
        if (self.limits.max_thinking is None) or (
            not getattr(self, "flags", None) or not self.flags.supports_thinking
        ):
            return None  # Model doesn't support thinking mode

        # Base thinking tokens from model limits
        base_thinking = self.limits.recommended_thinking or (
            self.limits.max_thinking // 2
        )

        # Complexity multipliers
        complexity_multipliers = {"simple": 0.5, "moderate": 1.0, "complex": 1.8}

        # Use case affects thinking needs differently than output
        thinking_use_case_multipliers = {
            "test_generation": 1.3,  # More thinking for comprehensive test planning
            "code_analysis": 1.6,  # Deep analysis needs extensive reasoning
            "refinement": 1.0,  # Moderate thinking for improvements
        }

        # Calculate thinking tokens
        complexity_factor = complexity_multipliers.get(complexity_level, 1.0)
        use_case_factor = thinking_use_case_multipliers.get(use_case, 1.0)

        adjusted_thinking = int(base_thinking * complexity_factor * use_case_factor)

        # Apply safety margin and ensure within limits
        safe_thinking = int(adjusted_thinking * safety_margin)
        max_safe_thinking = int(self.limits.max_thinking * safety_margin)
        final_thinking = min(safe_thinking, max_safe_thinking)

        # Ensure minimum viable thinking without exceeding the safe cap
        min_viable = min(1000, max_safe_thinking)
        final_thinking = max(final_thinking, min_viable)

        logger.debug(
            f"Calculated thinking tokens: {final_thinking} for {use_case} ({complexity_level}) on {self.provider}/{self.model}"
        )
        return final_thinking

    def supports_thinking_mode(self) -> bool:
        """Check if the model supports extended thinking mode (Claude-style configurable thinking)."""
        return (
            (self.limits.max_thinking is not None)
            and getattr(self, "flags", None)
            and self.flags.supports_thinking
        )

    def is_reasoning_model(self) -> bool:
        """Check if the model has built-in reasoning capabilities (OpenAI o-series style)."""
        # Prefer catalog flag when available
        try:
            return bool(getattr(self, "flags", None) and self.flags.reasoning)
        except Exception:
            # Fallback heuristic
            openai_reasoning_models = ["o4-mini", "o3", "o4"]
            return self.provider in ["openai", "azure-openai"] and any(
                reasoning_model in self.model
                for reasoning_model in openai_reasoning_models
            )

    def has_reasoning_capabilities(self) -> bool:
        """Check if the model has any form of reasoning capabilities."""
        return self.supports_thinking_mode() or self.is_reasoning_model()

    def estimate_input_tokens(self, text: str) -> int:
        """Rough estimate of token count for input text.

        Uses a simple heuristic: ~4 characters per token on average.
        This is approximate and actual tokenization may vary.

        Args:
            text: Input text to estimate

        Returns:
            Estimated token count
        """
        # Simple heuristic: 4 chars per token on average
        # This is approximate - real tokenizers are more complex
        return max(1, len(text) // 4)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model's capabilities.

        Returns:
            Dictionary with model limits and capabilities
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "max_context_tokens": self.limits.max_context,
            "max_output_tokens": self.limits.max_output,
            "recommended_output_tokens": self.limits.recommended_output,
            "max_thinking_tokens": self.limits.max_thinking,
            "recommended_thinking_tokens": self.limits.recommended_thinking,
            "supports_thinking_mode": self.supports_thinking_mode(),  # Claude-style configurable thinking
            "is_reasoning_model": self.is_reasoning_model(),  # OpenAI o-series built-in reasoning
            "has_reasoning_capabilities": self.has_reasoning_capabilities(),  # Any reasoning
            "supports_large_context": self.limits.max_context > 32000,
            "supports_large_output": self.limits.max_output > 8000,
        }

    @classmethod
    def get_supported_models(cls, provider: str) -> list[str]:
        """Get list of supported models for a provider.

        Args:
            provider: Provider name

        Returns:
            List of supported model identifiers
        """
        return catalog_get_all_models_for_provider(provider)

    @classmethod
    def get_all_providers(cls) -> list[str]:
        """Get list of all supported providers.

        Returns:
            List of provider names
        """
        return catalog_get_all_providers()
