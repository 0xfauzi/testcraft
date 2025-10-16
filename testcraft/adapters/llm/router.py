from __future__ import annotations

from typing import Any

from ...ports.llm_port import LLMPort


class _NoOpLLM(LLMPort):
    """No-op LLM adapter used to avoid network calls in tests."""

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
            "metadata": {"noop": True},
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
    """Minimal LLM router that returns a no-op adapter for tests."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cost_port: Any | None = None,
        prompt_registry: Any | None = None,
    ):
        self.config = config or {}
        self.cost_port = cost_port
        self.prompt_registry = prompt_registry
        self.default_provider = self.config.get("default_provider", "openai")
        self._adapter: LLMPort = _NoOpLLM(default_provider=self.default_provider)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMRouter:
        return cls(config)

    def _get_adapter(self) -> LLMPort:
        return self._adapter

    def generate_tests(
        self,
        code_content: str,
        context: str | None = None,
        test_framework: str = "pytest",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._get_adapter().generate_tests(
            code_content, context, test_framework, **kwargs
        )

    def analyze_code(
        self, code_content: str, analysis_type: str = "comprehensive", **kwargs: Any
    ) -> dict[str, Any]:
        return self._get_adapter().analyze_code(code_content, analysis_type, **kwargs)

    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._get_adapter().refine_content(
            original_content,
            refinement_instructions,
            system_prompt=system_prompt,
            **kwargs,
        )

    def generate_test_plan(
        self, code_content: str, context: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._get_adapter().generate_test_plan(code_content, context, **kwargs)
