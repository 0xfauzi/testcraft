from __future__ import annotations

import logging
from typing import Any

from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class LLMRouter(LLMPort):
    """Router for multiple LLM providers using user configuration."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cost_port: CostPort | None = None,
        prompt_registry: PromptRegistry | None = None,
    ):
        """Initialize LLM Router with configuration, optional cost tracking, and prompt registry."""
        self.config = config or {}
        self.cost_port = cost_port
        self.prompt_registry = prompt_registry or PromptRegistry()
        self._adapters: dict[str, LLMPort] = {}
        # Get the default provider from config, fallback to openai
        self.default_provider = self.config.get("default_provider", "openai")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMRouter:
        """Create LLMRouter from configuration."""
        return cls(config)

    def _get_provider_config(self, provider: str) -> dict[str, Any]:
        """Extract provider-specific config."""
        provider_config = {}

        if provider == "openai":
            provider_config = {
                "model": self.config.get("openai_model", "gpt-4.1"),
                "max_tokens": self.config.get("openai_max_tokens", 12000),
                "timeout": self.config.get("openai_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "base_url": self.config.get("openai_base_url"),
                "beta": self.config.get("beta", {}),
            }
        elif provider == "anthropic":
            provider_config = {
                "model": self.config.get("anthropic_model", "claude-sonnet-4"),
                "max_tokens": self.config.get("anthropic_max_tokens", 100000),
                "timeout": self.config.get("anthropic_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "beta": self.config.get("beta", {}),
            }
        elif provider == "azure-openai":
            provider_config = {
                "deployment": self.config.get(
                    "azure_openai_deployment", "claude-sonnet-4"
                ),
                "api_version": self.config.get(
                    "azure_openai_api_version", "2024-02-15-preview"
                ),
                "timeout": self.config.get("azure_openai_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "max_tokens": self.config.get("azure_openai_max_tokens", 4000),
                "beta": self.config.get("beta", {}),
            }
        elif provider == "bedrock":
            provider_config = {
                "model_id": self.config.get(
                    "bedrock_model_id", "anthropic.claude-3-7-sonnet-v1:0"
                ),
                "region": self.config.get("aws_region", "us-east-1"),
                "timeout": self.config.get("bedrock_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "beta": self.config.get("beta", {}),
            }

        return provider_config

    def _get_adapter(self, provider: str) -> LLMPort:
        """Get or create adapter for a specific provider."""
        if provider not in self._adapters:
            provider_config = self._get_provider_config(provider)

            if provider == "openai":
                from .openai import OpenAIAdapter

                self._adapters[provider] = OpenAIAdapter(
                    cost_port=self.cost_port,
                    prompt_registry=self.prompt_registry,
                    **provider_config,
                )
            elif provider == "anthropic":
                from .claude import ClaudeAdapter

                self._adapters[provider] = ClaudeAdapter(
                    cost_port=self.cost_port,
                    prompt_registry=self.prompt_registry,
                    **provider_config,
                )
            elif provider == "azure-openai":
                from .azure import AzureOpenAIAdapter

                self._adapters[provider] = AzureOpenAIAdapter(
                    cost_port=self.cost_port,
                    prompt_registry=self.prompt_registry,
                    **provider_config,
                )
            elif provider == "bedrock":
                from .bedrock import BedrockAdapter

                self._adapters[provider] = BedrockAdapter(
                    cost_port=self.cost_port,
                    prompt_registry=self.prompt_registry,
                    **provider_config,
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._adapters[provider]

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate tests using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.generate_tests(code_content, context, test_framework, **kwargs)

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze code using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.analyze_code(code_content, analysis_type, **kwargs)

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refine existing content based on specific instructions."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.refine_content(
            original_content,
            refinement_instructions,
            system_prompt=system_prompt,
            **kwargs,
        )

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a comprehensive test plan for the provided code content."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.generate_test_plan(code_content, context, **kwargs)
