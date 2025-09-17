"""Smart token calculation for LLM adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from ...config.model_catalog_loader import resolve_model, get_models, get_providers

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
    """Calculates optimal token allocation for different providers and use cases.

    Refactored to load model limits from the TOML catalog via
    `testcraft.config.model_catalog_loader` and to avoid hardcoded limits.
    """

    # Use case specific multipliers for recommended output
    USE_CASE_MULTIPLIERS: dict[str, float] = {
        "test_generation": 1.5,  # Test generation needs more tokens
        "code_analysis": 0.7,  # Analysis can be more concise
        "refinement": 1.2,  # Refinement needs moderate tokens
    }

    def __init__(self, provider: str, model: str, enable_extended_context: bool = False, enable_extended_output: bool = False):
        """Initialize token calculator for a specific provider and model.

        Args:
            provider: Provider name (openai, anthropic, azure-openai, bedrock)
            model: Model identifier
            enable_extended_context: Whether to use extended context limits beyond documented defaults
            enable_extended_output: Whether to use extended output limits beyond documented defaults
        """
        self.provider = provider
        self.model = model
        self.enable_extended_context = enable_extended_context
        self.enable_extended_output = enable_extended_output
        self.limits = self._get_model_limits()

    def _get_model_limits(self) -> TokenLimits:
        """Get token limits for the current model using the catalog.

        Unknown models fall back to conservative defaults and emit a warning.
        """
        entry = resolve_model(self.provider, self.model)
        if entry is None:
            logger.warning(
                f"Unknown model {self.provider}/{self.model}, using safe defaults"
            )
            if self.provider in ["anthropic", "bedrock"]:
                return TokenLimits(
                    max_context=200000,
                    max_output=32000,
                    recommended_output=25000,
                    max_thinking=32000,
                    recommended_thinking=16000,
                )
            return TokenLimits(
                max_context=200000, max_output=32000, recommended_output=25000
            )

        # Map catalog limits to TokenLimits; recommended_output derived from default_max_output
        limits = entry.limits
        max_thinking = limits.max_thinking if limits.max_thinking and limits.max_thinking > 0 else None
        recommended_thinking = (
            max_thinking // 2 if max_thinking and max_thinking > 0 else None
        )
        
        # Apply beta feature logic for extended limits
        max_context = int(limits.max_context)
        max_output = int(limits.default_max_output)
        recommended_output = int(limits.default_max_output)
        
        if self.enable_extended_context:
            # Allow extended context - for now, keep the catalog limit as is
            # In the future, this could be increased beyond catalog limits
            logger.debug(f"Extended context enabled for {self.provider}/{self.model}: {max_context}")
            
        if self.enable_extended_output:
            # Allow extended output - use a higher multiplier for recommended output
            # But still respect the documented maximum from the catalog
            extended_output = int(limits.default_max_output * 2.0)  # 2x the default
            max_output = min(extended_output, 128000)  # Cap at 128k for safety
            recommended_output = max_output
            logger.debug(f"Extended output enabled for {self.provider}/{self.model}: {max_output}")
        
        return TokenLimits(
            max_context=max_context,
            max_output=max_output,
            recommended_output=recommended_output,
            max_thinking=(int(max_thinking) if max_thinking is not None else None),
            recommended_thinking=(
                int(recommended_thinking) if recommended_thinking is not None else None
            ),
        )

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
            # Enforce context ceiling: input + output must fit within safety-adjusted context
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
        if self.limits.max_thinking is None:
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

        # Ensure minimum viable thinking
        final_thinking = max(final_thinking, 1000) if final_thinking > 0 else None

        logger.debug(
            f"Calculated thinking tokens: {final_thinking} for {use_case} ({complexity_level}) on {self.provider}/{self.model}"
        )
        return final_thinking

    def supports_thinking_mode(self) -> bool:
        """Check if the model supports extended thinking mode (Claude-style configurable thinking)."""
        return self.limits.max_thinking is not None

    def is_reasoning_model(self) -> bool:
        """Check if the model has built-in reasoning capabilities (OpenAI o1-series style)."""
        # Check catalog entry for reasoning capabilities
        entry = resolve_model(self.provider, self.model)
        if entry and entry.flags:
            return bool(entry.flags.reasoning_capable)
        
        # Fallback for unknown models - check for o1-series patterns
        openai_reasoning_models = ["o1-mini", "o1-preview", "o1"]
        return self.provider in ["openai", "azure-openai"] and any(
            reasoning_model in self.model.lower() for reasoning_model in openai_reasoning_models
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
        try:
            return get_models(provider)
        except Exception:
            return []

    @classmethod
    def get_all_providers(cls) -> list[str]:
        """Get list of all supported providers.

        Returns:
            List of provider names
        """
        try:
            return get_providers()
        except Exception:
            return []
