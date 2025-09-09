from __future__ import annotations

from typing import Any

from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort


class LLMRouter(LLMPort):
    """Router for multiple LLM providers using user configuration."""

    def __init__(self, config: dict[str, Any] | None = None, cost_port: CostPort | None = None):
        """Initialize LLM Router with configuration and optional cost tracking."""
        self.config = config or {}
        self.cost_port = cost_port
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
                "deployment": self.config.get(
                    "azure_openai_deployment", "gpt-4o-mini"
                ),
                "api_version": self.config.get(
                    "azure_openai_api_version", "2024-02-15-preview"
                ),
                "timeout": self.config.get("azure_openai_timeout", 60.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "max_tokens": self.config.get("azure_openai_max_tokens", 4000),
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

                self._adapters[provider] = OpenAIAdapter(cost_port=self.cost_port, **provider_config)
            elif provider == "anthropic":
                from .claude import ClaudeAdapter

                self._adapters[provider] = ClaudeAdapter(cost_port=self.cost_port, **provider_config)
            elif provider == "azure-openai":
                from .azure import AzureOpenAIAdapter

                self._adapters[provider] = AzureOpenAIAdapter(cost_port=self.cost_port, **provider_config)
            elif provider == "bedrock":
                from .bedrock import BedrockAdapter

                self._adapters[provider] = BedrockAdapter(cost_port=self.cost_port, **provider_config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._adapters[provider]

    async def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate tests using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.generate_tests(code_content, context, test_framework, **kwargs)

    async def analyze_code(self, code: str, focus_areas: list) -> dict[str, Any]:
        """Analyze code using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.analyze_code(code, focus_areas)

    async def refine_tests(self, tests: str, feedback: str) -> str:
        """Refine tests using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        # Note: The actual adapters have refine_content method, not refine_tests
        # This might need to be updated based on the actual adapter interface
        if hasattr(adapter, 'refine_content'):
            result = adapter.refine_content(tests, feedback)
            return result.get('refined_content', tests)
        else:
            # Fallback for adapters that don't support refinement
            return tests
