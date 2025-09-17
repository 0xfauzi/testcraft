"""Provider capabilities abstraction for LLM adapters.

Maps TokenCalculator model info into a structured capabilities object that
adapters and routers can expose without leaking provider-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass

from .token_calculator import TokenCalculator


@dataclass
class ProviderCapabilities:
    supports_thinking_mode: bool
    is_reasoning_model: bool
    has_reasoning_capabilities: bool
    max_context_tokens: int
    max_output_tokens: int
    recommended_output_tokens: int
    max_thinking_tokens: int | None
    recommended_thinking_tokens: int | None


def get_capabilities(
    provider: str, model: str, token_calculator: TokenCalculator | None = None
) -> ProviderCapabilities:
    """Return provider/model capabilities based on TokenCalculator info."""
    tc = token_calculator or TokenCalculator(provider=provider, model=model)
    info = tc.get_model_info()
    return ProviderCapabilities(
        supports_thinking_mode=bool(info.get("supports_thinking_mode")),
        is_reasoning_model=bool(info.get("is_reasoning_model")),
        has_reasoning_capabilities=bool(info.get("has_reasoning_capabilities")),
        max_context_tokens=int(info.get("max_context_tokens", 0)),
        max_output_tokens=int(info.get("max_output_tokens", 0)),
        recommended_output_tokens=int(info.get("recommended_output_tokens", 0)),
        max_thinking_tokens=info.get("max_thinking_tokens"),
        recommended_thinking_tokens=info.get("recommended_thinking_tokens"),
    )


