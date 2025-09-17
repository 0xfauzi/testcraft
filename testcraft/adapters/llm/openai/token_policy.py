"""OpenAI token policy and budget management."""

from __future__ import annotations

import logging
from typing import Any, Literal

from ....config.model_catalog_loader import resolve_model

logger = logging.getLogger(__name__)


class OpenAITokenPolicy:
    """Manages token policies, budgets, and model-specific limits for OpenAI."""

    def __init__(self, model: str):
        """Initialize token policy for the given model.

        Args:
            model: OpenAI model name
        """
        self.model = model

    def requires_completion_tokens_param(self) -> bool:
        """Check if the model requires max_completion_tokens instead of max_tokens.

        Uses the model catalog to determine if this is a reasoning model that
        requires max_completion_tokens parameter.

        Returns:
            True if model requires max_completion_tokens, False for max_tokens
        """
        # Check catalog entry for reasoning capabilities
        catalog_entry = resolve_model("openai", self.model)
        if catalog_entry and catalog_entry.flags:
            return bool(catalog_entry.flags.reasoning_capable)

        # Fallback: Unknown models default to standard parameters for safety
        logger.warning(f"Model '{self.model}' not found in catalog, defaulting to max_tokens parameter")
        return False

    def is_o_series_reasoning_model(self) -> bool:
        """Detect OpenAI reasoning models that should use Responses API.

        Uses the model catalog to determine reasoning capability.

        Returns:
            True if the model is a reasoning model (o1-series), else False.
        """
        # Check catalog entry for reasoning capabilities
        catalog_entry = resolve_model("openai", self.model)
        if catalog_entry and catalog_entry.flags:
            return bool(catalog_entry.flags.reasoning_capable)

        # Fallback: Unknown models default to non-reasoning for safety
        logger.debug(f"Model '{self.model}' not found in catalog, defaulting to non-reasoning mode")
        return False

    def supports_temperature_adjustment(self) -> bool:
        """Check if the model supports custom temperature values.

        OpenAI's reasoning models (o1-series) only support the default
        temperature of 1.0 and don't allow custom temperature values.

        Returns:
            True if model supports temperature adjustment, False if only default
        """
        # Reasoning models that only support default temperature (1.0)
        # Use the same logic as is_o_series_reasoning_model for consistency
        return not self.is_o_series_reasoning_model()

    def enforce_catalog_limits(self, requested_tokens: int, use_case: str = "general") -> int:
        """Enforce catalog-defined token limits for the model.

        Args:
            requested_tokens: Number of tokens requested
            use_case: Use case context for logging

        Returns:
            Token count clamped to catalog limits
        """
        catalog_entry = resolve_model("openai", self.model)
        if catalog_entry and catalog_entry.limits:
            # For reasoning models, enforce the documented max output limit
            max_allowed = catalog_entry.limits.default_max_output
            if requested_tokens > max_allowed:
                logger.warning(
                    f"Requested {requested_tokens} tokens for {use_case} but model '{self.model}' "
                    f"limit is {max_allowed} tokens, clamping to catalog limit"
                )
                return max_allowed
        else:
            logger.debug(f"No catalog limits found for model '{self.model}', using requested value")

        return requested_tokens

    def estimate_complexity(
        self, code_content: str
    ) -> Literal["simple", "moderate", "complex"]:
        """Estimate code complexity for thinking token calculation.

        Args:
            code_content: Code to analyze

        Returns:
            Complexity level estimate
        """
        lines = code_content.split("\n")
        line_count = len([line for line in lines if line.strip()])

        # Count potential complexity indicators
        complexity_indicators = [
            "class ",
            "def ",
            "async ",
            "await ",
            "try:",
            "except:",
            "finally:",
            "with ",
            "for ",
            "while ",
            "if ",
            "elif ",
            "lambda",
            "yield",
            "import ",
            "from ",
            "@",
            "raise ",
            "assert ",
        ]

        indicator_count = sum(
            1
            for line in lines
            for indicator in complexity_indicators
            if indicator in line
        )

        # Simple heuristic based on lines and complexity indicators
        if line_count < 50 and indicator_count < 10:
            return "simple"
        elif line_count < 200 and indicator_count < 30:
            return "moderate"
        else:
            return "complex"

    def prepare_request_params(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare request parameters based on model capabilities.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting
            **kwargs: Additional parameters

        Returns:
            Dictionary of prepared request parameters
        """
        params = dict(kwargs)

        # Handle token limits
        if max_tokens is not None:
            enforced_tokens = self.enforce_catalog_limits(max_tokens)
            if self.requires_completion_tokens_param():
                params["max_completion_tokens"] = enforced_tokens
            else:
                params["max_tokens"] = enforced_tokens

        # Handle temperature
        if temperature is not None and self.supports_temperature_adjustment():
            params["temperature"] = temperature
        elif temperature is not None:
            logger.debug(
                f"Skipping temperature parameter for reasoning model {self.model} (uses default temperature)"
            )

        return params
