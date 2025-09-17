from __future__ import annotations

import logging
from typing import Any

from ...ports.cost_port import CostPort
from ...ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class LLMRouter(LLMPort):
    """Router for multiple LLM providers using user configuration."""

    def __init__(
        self, config: dict[str, Any] | None = None, cost_port: CostPort | None = None
    ):
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
        """Extract provider-specific config (simplified, typed per adapter)."""
        provider_config = {}

        if provider == "openai":
            provider_config = {
                "model": self.config.get("openai_model", "gpt-4.1"),
                "max_tokens": self.config.get("openai_max_tokens", 12000),
                "timeout": self.config.get("openai_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "base_url": self.config.get("openai_base_url"),
            }
        elif provider == "anthropic":
            provider_config = {
                "model": self.config.get("anthropic_model", "claude-sonnet-4"),
                "max_tokens": self.config.get("anthropic_max_tokens", 100000),
                "timeout": self.config.get("anthropic_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                # Beta features configuration
                "enable_extended_context": self.config.get("anthropic_enable_extended_context", False),
                "enable_extended_output": self.config.get("anthropic_enable_extended_output", False),
            }
        elif provider == "azure-openai":
            provider_config = {
                # FIXED: Azure OpenAI should use official OpenAI models
                "deployment": self.config.get("azure_openai_deployment", "gpt-4.1"),
                "api_version": self.config.get(
                    "azure_openai_api_version", "2024-02-15-preview"
                ),
                "timeout": self.config.get("azure_openai_timeout", 180.0),
                "temperature": self.config.get("temperature", 0.1),
                "max_retries": self.config.get("max_retries", 3),
                "max_tokens": self.config.get("azure_openai_max_tokens", 4000),
            }
        elif provider == "bedrock":
            provider_config = {
                # UPDATED: Use official Claude Sonnet 4 on Bedrock
                "model_id": self.config.get(
                    "bedrock_model_id", "anthropic.claude-sonnet-4-20250514-v1:0"
                ),
                # Adapter expects region_name keyword
                "region_name": self.config.get("aws_region", "us-east-1"),
                "timeout": self.config.get("bedrock_timeout", 180.0),
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

                self._adapters[provider] = OpenAIAdapter(
                    cost_port=self.cost_port, **provider_config
                )
            elif provider == "anthropic":
                from .claude import ClaudeAdapter

                self._adapters[provider] = ClaudeAdapter(
                    cost_port=self.cost_port, **provider_config
                )
            elif provider == "azure-openai":
                from .azure import AzureOpenAIAdapter

                self._adapters[provider] = AzureOpenAIAdapter(
                    cost_port=self.cost_port, **provider_config
                )
            elif provider == "bedrock":
                from .bedrock import BedrockAdapter

                self._adapters[provider] = BedrockAdapter(
                    cost_port=self.cost_port, **provider_config
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
        # Surface generation preferences to underlying adapters via kwargs
        return adapter.generate_tests(
            code_content,
            context,
            test_framework,
            **kwargs,
        )

    def analyze_code(self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any) -> dict[str, Any]:
        """Analyze code using configured provider (LLMPort parity)."""
        adapter = self._get_adapter(self.default_provider)
        return adapter.analyze_code(code_content, analysis_type, **kwargs)

    # Removed deprecated refine_tests. Use refine_content instead.
    
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
        if hasattr(adapter, "refine_content"):
            return adapter.refine_content(
                original_content,
                refinement_instructions,
                system_prompt=system_prompt,
                **kwargs,
            )
        else:
            # Fallback for adapters that don't support refinement
            logger.warning(
                f"Provider {self.default_provider} adapter does not support refine_content method"
            )
            return {
                "refined_content": original_content,
                "changes_made": "No refinement available",
                "confidence": 0.0,
                "metadata": {"error": "refine_content not supported"}
            }

    def generate_test_plan(
        self,
        code_content: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate test plan using configured provider."""
        adapter = self._get_adapter(self.default_provider)
        if hasattr(adapter, "generate_test_plan"):
            return adapter.generate_test_plan(code_content, context, **kwargs)
        else:
            # Fallback: use refine_content with planning prompt from registry
            logger.warning(
                f"Provider {self.default_provider} adapter does not support generate_test_plan method, using fallback"
            )
            
            if hasattr(adapter, "refine_content"):
                # Use proper prompts from registry
                from ...prompts.registry import PromptRegistry
                registry = PromptRegistry()
                
                try:
                    system_prompt = registry.get_system_prompt("llm_test_planning_v1")
                    user_prompt = registry.get_user_prompt(
                        "llm_test_planning_v1",
                        code_content=code_content,
                        additional_context={"context": context} if context else {}
                    )
                    
                    # Combine prompts for refine_content
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    result = adapter.refine_content(
                        original_content=code_content,
                        refinement_instructions=combined_prompt,
                        **kwargs
                    )
                    refined = result.get("refined_content", "")
                    
                    # Try to parse as JSON, fallback to text
                    try:
                        import json
                        parsed = json.loads(refined)
                        return {
                            "plan_summary": parsed.get("plan_summary", "Generated via fallback method"),
                            "detailed_plan": parsed.get("detailed_plan", refined),
                            "confidence": parsed.get("confidence", result.get("confidence", 0.5)),
                            "scenarios": parsed.get("scenarios", []),
                            "mocks": parsed.get("mocks", ""),
                            "fixtures": parsed.get("fixtures", ""),
                            "notes": "Generated using refine_content fallback with registry prompts"
                        }
                    except json.JSONDecodeError:
                        return {
                            "plan_summary": "Generated via fallback method",
                            "detailed_plan": refined,
                            "confidence": result.get("confidence", 0.5),
                            "notes": "Generated using refine_content fallback"
                        }
                except Exception as e:
                    logger.warning(f"Registry fallback failed: {e}")
                    return {
                        "plan_summary": "Fallback generation failed",
                        "detailed_plan": f"Error using registry prompts: {str(e)}",
                        "confidence": 0.0,
                        "notes": "Registry fallback error"
                    }
            else:
                return {
                    "plan_summary": "Test plan generation not available",
                    "detailed_plan": "Provider does not support test planning",
                    "confidence": 0.0,
                    "notes": "No planning capability available"
                }

    def get_capabilities(self) -> dict[str, Any]:
        """Delegate to adapter capabilities via token calculator and model info."""
        try:
            adapter = self._get_adapter(self.default_provider)
            if hasattr(adapter, "token_calculator") and hasattr(adapter, "provider"):
                from .capabilities import get_capabilities
                caps = get_capabilities(adapter.provider, getattr(adapter, "model", ""), getattr(adapter, "token_calculator", None))
                return {
                    "supports_thinking_mode": caps.supports_thinking_mode,
                    "is_reasoning_model": caps.is_reasoning_model,
                    "has_reasoning_capabilities": caps.has_reasoning_capabilities,
                    "max_context_tokens": caps.max_context_tokens,
                    "max_output_tokens": caps.max_output_tokens,
                    "recommended_output_tokens": caps.recommended_output_tokens,
                    "max_thinking_tokens": caps.max_thinking_tokens,
                    "recommended_thinking_tokens": caps.recommended_thinking_tokens,
                }
        except Exception:
            pass
        return {}