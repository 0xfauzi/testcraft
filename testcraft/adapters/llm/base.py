"""Base adapter for LLM providers with shared helpers.

This module introduces a reusable base class that centralizes common
capabilities for provider adapters:
- Credential management
- Prompt sourcing via PromptRegistry
- Token budgeting via TokenCalculator (including thinking budgets)
- Refinement response parsing and normalization
- Metadata unification
- Cost tracking hooks
- Optional artifact logging hooks (no-op by default)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ...config.credentials import CredentialManager
from ...ports.cost_port import CostPort
from ...prompts.registry import PromptRegistry
from .common import normalize_refinement_response, parse_json_response
from .token_calculator import TokenCalculator


logger = logging.getLogger(__name__)


class BaseLLMAdapter:
    """Shared foundations for concrete LLM provider adapters.

    Concrete adapters should inherit from this class to obtain consistent,
    centralized behavior while remaining free to implement provider-specific
    request/response handling.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str | None = None,
        prompt_registry: PromptRegistry | None = None,
        cost_port: CostPort | None = None,
        credential_manager: CredentialManager | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or ""
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.cost_port = cost_port
        self.credential_manager = credential_manager or CredentialManager()
        self.token_calculator = TokenCalculator(provider=self.provider, model=self.model)

    # -----------------------------
    # Budgeting and prompts helpers
    # -----------------------------
    def _calc_budgets(
        self,
        use_case: Literal["test_generation", "code_analysis", "refinement"],
        input_text: str | None = None,
        complexity: Literal["simple", "moderate", "complex"] = "moderate",
    ) -> dict[str, int | None]:
        """Calculate request budgets (max/output tokens and thinking tokens).

        Returns a dict with keys: "max_tokens", "thinking_tokens", "input_tokens".
        """
        input_tokens = (
            self.token_calculator.estimate_input_tokens(input_text) if input_text else None
        )
        max_tokens = self.token_calculator.calculate_max_tokens(
            use_case=use_case, input_length=input_tokens
        )
        thinking_tokens = self.token_calculator.calculate_thinking_tokens(
            use_case=use_case, complexity_level=complexity
        )
        return {
            "max_tokens": int(max_tokens),
            "thinking_tokens": int(thinking_tokens) if thinking_tokens else None,
            "input_tokens": int(input_tokens) if input_tokens else None,
        }

    def _prompts(
        self,
        prompt_key: Literal[
            "llm_test_generation",
            "llm_code_analysis",
            "llm_content_refinement",
            "llm_test_planning_v1",
        ],
        *,
        code_content: str | None = None,
        additional_context: dict[str, Any] | None = None,
        system_prompt_override: str | None = None,
        **template_kwargs: Any,
    ) -> tuple[str, str]:
        """Fetch system/user prompts from the registry with optional override.

        template_kwargs are forwarded as formatting variables to the registry
        for prompt templates that accept extra parameters (e.g., test_framework,
        analysis_type).
        """
        system_prompt = system_prompt_override or self.prompt_registry.get_system_prompt(
            prompt_type=prompt_key, **template_kwargs
        )
        user_prompt = self.prompt_registry.get_user_prompt(
            prompt_type=prompt_key,
            code_content=code_content or "",
            additional_context=additional_context or {},
            **template_kwargs,
        )
        return system_prompt, user_prompt

    # -----------------------------------------
    # Parsing, normalization, and metadata utils
    # -----------------------------------------
    def _parse_and_normalize_refinement(
        self, text: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Parse LLM refinement output and normalize schema.

        Returns tuple of (normalized_data_or_none, parsed_info_metadata).
        """
        parsed = parse_json_response(text)
        if parsed.success:
            norm = normalize_refinement_response(parsed.data or {})
            return (
                norm.data if norm.is_valid else None,
                {
                    "parsed": True,
                    "repaired": bool(norm.repaired),
                    "repair_type": norm.repair_type,
                    "error": norm.error,
                },
            )
        return None, {"parsed": False, "error": parsed.error}

    def _unify_metadata(
        self,
        *,
        provider: str,
        model: str,
        usage: dict[str, Any] | None = None,
        parsed_info: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        raw_provider_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create normalized metadata payload across providers."""
        usage = usage or {}
        tokens = {
            "prompt": usage.get("prompt_tokens"),
            "input": usage.get("input_tokens"),
            "completion": usage.get("completion_tokens"),
            "output": usage.get("output_tokens"),
            "total": usage.get("total_tokens"),
        }
        meta: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "parsed": (parsed_info or {}).get("parsed"),
            "repaired": (parsed_info or {}).get("repaired"),
            "repair_type": (parsed_info or {}).get("repair_type"),
            "raw_provider_fields": raw_provider_fields or {},
        }
        if extra:
            meta.update(extra)
        return meta

    # ----------------
    # Cost tracking API
    # ----------------
    def _track_cost(
        self,
        operation: str,
        *,
        tokens_used: int | None = None,
        cost: float | None = None,
        **usage_like: Any,
    ) -> dict[str, Any] | None:
        """Send cost/usage info to the configured CostPort (if any)."""
        if not self.cost_port:
            return None
        payload: dict[str, Any] = {"tokens_used": tokens_used, "cost": cost}
        payload.update({k: v for k, v in usage_like.items() if v is not None})
        try:
            return self.cost_port.track_usage("llm", operation, payload)
        except Exception as e:  # Non-fatal
            logger.warning("Cost tracking failed: %s", e)
            return None

    # --------------------
    # Artifact hook methods
    # --------------------
    def _artifact_request(self, operation: str, payload: dict[str, Any]) -> None:  # pragma: no cover - optional
        """Optional hook for storing verbose request artifacts (no-op by default)."""
        return

    def _artifact_response(self, operation: str, response: dict[str, Any]) -> None:  # pragma: no cover - optional
        """Optional hook for storing verbose response artifacts (no-op by default)."""
        return


