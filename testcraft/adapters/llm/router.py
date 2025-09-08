from __future__ import annotations

from typing import Any

from ...ports.llm_port import LLMPort


class LLMRouter(LLMPort):
    """Router for multiple LLM providers using user configuration."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize LLM Router with configuration."""
        self.config = config or {}
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
                "model": self.config.get("openai_model", "o4-mini"),
                "max_tokens": self.config.get("openai_max_tokens", 12000),
                "timeout": self.config.get("openai_timeout", 60.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "base_url": self.config.get("openai_base_url"),
            }
        elif provider == "anthropic":
            provider_config = {
                "model": self.config.get("anthropic_model", "claude-3-7-sonnet"),
                "max_tokens": self.config.get("anthropic_max_tokens", 100000),
                "timeout": self.config.get("anthropic_timeout", 60.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
            }
        elif provider == "azure-openai":
            provider_config = {
                "deployment_name": self.config.get(
                    "azure_openai_deployment", "o4-mini"
                ),
                "api_version": self.config.get(
                    "azure_openai_api_version", "2024-02-15-preview"
                ),
                "timeout": self.config.get("azure_openai_timeout", 60.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
            }
        elif provider == "bedrock":
            provider_config = {
                "model_id": self.config.get(
                    "bedrock_model_id", "anthropic.claude-3-7-sonnet-v1:0"
                ),
                "region": self.config.get("aws_region", "us-east-1"),
                "timeout": self.config.get("bedrock_timeout", 60.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
            }

        return provider_config

    def _get_adapter(self, provider: str) -> LLMPort:
        """Get or create adapter for a specific provider."""
        if provider not in self._adapters:
            provider_config = self._get_provider_config(provider)

            if provider == "openai":
                from .openai import OpenAIAdapter

                self._adapters[provider] = OpenAIAdapter(**provider_config)
            elif provider == "anthropic":
                from .claude import ClaudeAdapter

                self._adapters[provider] = ClaudeAdapter(**provider_config)
            elif provider == "azure-openai":
                from .azure import AzureAdapter

                self._adapters[provider] = AzureAdapter(**provider_config)
            elif provider == "bedrock":
                from .bedrock import BedrockAdapter

                self._adapters[provider] = BedrockAdapter(**provider_config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._adapters[provider]

    async def generate_tests(
        self, code: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate tests using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return await adapter.generate_tests(code, context)

    async def analyze_code(self, code: str, focus_areas: list) -> dict[str, Any]:
        """Analyze code using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return await adapter.analyze_code(code, focus_areas)

    async def refine_tests(self, tests: str, feedback: str) -> str:
        """Refine tests using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return await adapter.refine_tests(tests, feedback)
