from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ...config.credentials import CredentialError, CredentialManager
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from .azure import AzureOpenAIAdapter, AzureOpenAIError
from .claude import ClaudeAdapter, ClaudeError
from .openai import OpenAIAdapter, OpenAIError

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    pass

logger = logging.getLogger(__name__)


class _NoOpLLM(LLMPort):
    """Fallback adapter that avoids network calls when providers are unavailable."""

    def __init__(self, default_provider: str = "openai") -> None:
        self.default_provider = default_provider

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "tests": "",
            "coverage_focus": [],
            "confidence": 0.0,
            "metadata": {"provider": self.default_provider, "noop": True},
        }

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "testability_score": 0.0,
            "complexity_metrics": {},
            "recommendations": [],
            "potential_issues": [],
        }

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "refined_content": original_content,
            "changes_made": [],
            "confidence": 0.0,
        }

    def generate_test_plan(
        self, code_content: str, context: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "test_plan": [],
            "test_coverage_areas": [],
            "test_priorities": [],
            "estimated_complexity": 0.0,
            "confidence": 0.0,
        }


class LLMRouter(LLMPort):
    """
    Router that selects concrete LLM adapters based on configuration.

    Falls back to a no-op adapter when credentials are missing or initialization
    fails, ensuring CLI and tests remain functional offline.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        cost_port: Any | None = None,
        prompt_registry: PromptRegistry | None = None,
        credential_manager: CredentialManager | None = None,
    ):
        raw_config: Mapping[str, Any] = config or {}
        if hasattr(raw_config, "model_dump"):
            raw_config = raw_config.model_dump()  # type: ignore[assignment]

        self._config = dict(raw_config)
        self.cost_port = cost_port
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.default_provider = self._config.get("default_provider", "openai")
        self._adapter_cache: dict[str, LLMPort] = {}
        self._provider_state: dict[str, dict[str, Any]] = {}

        # Credential manager is shared across providers so that cached secrets are reused.
        self._credential_manager = credential_manager or CredentialManager(
            config_overrides=self._config
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LLMRouter:
        return cls(config=config)

    # ------------------------------------------------------------------
    # Public API (LLMPort implementation)
    # ------------------------------------------------------------------
    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        provider = kwargs.pop("provider", None)
        adapter = self._get_adapter(provider)
        return adapter.generate_tests(code_content, context, test_framework, **kwargs)

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        provider = kwargs.pop("provider", None)
        adapter = self._get_adapter(provider)
        return adapter.analyze_code(code_content, analysis_type, **kwargs)

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        provider = kwargs.pop("provider", None)
        adapter = self._get_adapter(provider)
        return adapter.refine_content(
            original_content,
            refinement_instructions,
            system_prompt=system_prompt,
            **kwargs,
        )

    def generate_test_plan(
        self, code_content: str, context: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        provider = kwargs.pop("provider", None)
        adapter = self._get_adapter(provider)
        return adapter.generate_test_plan(code_content, context, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_adapter(self, provider: str | None) -> LLMPort:
        name = (provider or self.default_provider or "openai").lower()
        if name not in self._adapter_cache:
            self._adapter_cache[name] = self._build_adapter(name)
        return self._adapter_cache[name]

    def _build_adapter(self, provider: str) -> LLMPort:
        builder = {
            "openai": self._build_openai_adapter,
            "anthropic": self._build_anthropic_adapter,
            "azure-openai": self._build_azure_adapter,
            "bedrock": self._build_bedrock_adapter,
            "noop": lambda: _NoOpLLM(default_provider="noop"),
        }.get(provider)

        if builder is None:
            logger.warning(
                "Unknown LLM provider '%s'; falling back to noop adapter", provider
            )
            adapter = _NoOpLLM(default_provider=provider)
            self._record_adapter_state(provider, adapter, reason="unknown_provider")
            return adapter

        reason: str | None = None

        try:
            adapter = builder()
            logger.info("Initialized LLM adapter for provider '%s'", provider)
            self._record_adapter_state(provider, adapter)
            return adapter
        except (CredentialError, OpenAIError, ClaudeError, AzureOpenAIError) as exc:
            logger.warning(
                "Provider '%s' unavailable (%s); using noop adapter", provider, exc
            )
            reason = str(exc)
            adapter = _NoOpLLM(default_provider=provider)
            self._record_adapter_state(provider, adapter, reason=reason)
            return adapter
        except Exception as exc:  # pragma: no cover - defensive logging
            reason = str(exc)
            if provider == "bedrock":
                try:
                    from .bedrock import BedrockError
                except Exception:  # pragma: no cover - import failure
                    BedrockError = RuntimeError  # type: ignore[assignment]

                if isinstance(exc, BedrockError):
                    logger.warning(
                        "Provider '%s' unavailable (%s); using noop adapter",
                        provider,
                        exc,
                    )
                else:
                    logger.exception(
                        "Failed to initialize provider '%s'; falling back to noop",
                        provider,
                    )
            else:
                logger.exception(
                    "Failed to initialize provider '%s'; falling back to noop",
                    provider,
                )

        adapter = _NoOpLLM(default_provider=provider)
        self._record_adapter_state(provider, adapter, reason=reason)
        return adapter

    def _record_adapter_state(
        self, provider: str, adapter: LLMPort, *, reason: str | None = None
    ) -> None:
        provider_key = provider.lower()
        status = "noop" if isinstance(adapter, _NoOpLLM) else "ready"
        self._provider_state[provider_key] = {
            "provider": provider_key,
            "status": status,
            "reason": reason,
            "adapter_type": adapter.__class__.__name__,
        }

    def provider_status(self, provider: str | None = None) -> dict[str, Any]:
        name = (provider or self.default_provider or "openai").lower()
        info = self._provider_state.get(name)
        if not info:
            return {"provider": name, "status": "unknown", "reason": None}
        return dict(info)

    def provider_states(self) -> dict[str, dict[str, Any]]:
        return {name: dict(state) for name, state in self._provider_state.items()}

    def ensure_adapter(self, provider: str | None = None) -> None:
        """Ensure the requested provider adapter is initialized."""
        self._get_adapter(provider)

    def _build_openai_adapter(self) -> LLMPort:
        return OpenAIAdapter(
            model=self._config.get("openai_model", "gpt-4.1"),
            timeout=self._config.get("openai_timeout", 60.0),
            max_tokens=self._config.get("openai_max_tokens"),
            temperature=self._config.get("temperature", 0.1),
            max_retries=self._config.get("max_retries", 3),
            base_url=self._config.get("openai_base_url"),
            credential_manager=self._credential_manager,
            prompt_registry=self.prompt_registry,
            cost_port=self.cost_port,
            beta=self._config.get("beta", {}),
        )

    def _build_anthropic_adapter(self) -> LLMPort:
        return ClaudeAdapter(
            model=self._config.get("anthropic_model", "claude-sonnet-4"),
            timeout=self._config.get("anthropic_timeout", 60.0),
            max_tokens=self._config.get("anthropic_max_tokens"),
            temperature=self._config.get("temperature", 0.1),
            max_retries=self._config.get("max_retries", 3),
            credential_manager=self._credential_manager,
            prompt_registry=self.prompt_registry,
            cost_port=self.cost_port,
            beta=self._config.get("beta", {}),
        )

    def _build_azure_adapter(self) -> LLMPort:
        return AzureOpenAIAdapter(
            deployment=self._config.get("azure_openai_deployment", "gpt-4.1"),
            api_version=self._config.get(
                "azure_openai_api_version", "2024-02-15-preview"
            ),
            timeout=self._config.get("azure_openai_timeout", 60.0),
            max_tokens=self._config.get("openai_max_tokens"),
            temperature=self._config.get("temperature", 0.1),
            max_retries=self._config.get("max_retries", 3),
            credential_manager=self._credential_manager,
            prompt_registry=self.prompt_registry,
            cost_port=self.cost_port,
            beta=self._config.get("beta", {}),
        )

    def _build_bedrock_adapter(self) -> LLMPort:
        try:
            from .bedrock import BedrockAdapter
        except Exception as import_exc:
            raise RuntimeError(
                "Bedrock adapter dependencies could not be imported"
            ) from import_exc

        return BedrockAdapter(
            model_id=self._config.get(
                "bedrock_model_id", "anthropic.claude-3-7-sonnet-v1:0"
            ),
            region_name=self._config.get("aws_region"),
            timeout=self._config.get("bedrock_timeout", 60.0),
            max_tokens=self._config.get("anthropic_max_tokens", 4000),
            temperature=self._config.get("temperature", 0.1),
            max_retries=self._config.get("max_retries", 3),
            credential_manager=self._credential_manager,
            prompt_registry=self.prompt_registry,
            cost_port=self.cost_port,
            beta=self._config.get("beta", {}),
        )
