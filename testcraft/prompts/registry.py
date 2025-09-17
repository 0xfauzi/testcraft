"""
Prompt templates and registry with versioning.

This module provides a versioned prompt registry that supplies:
- System prompts for test generation and refinement
- User prompt templates that safely embed code and context
- JSON Schemas for structured LLM outputs (generation and refinement)
- Light anti-injection guidance and safe template rendering

The registry is designed to align with the `PromptPort` protocol while
remaining framework-agnostic. It does not import adapters or external
dependencies.
"""

from __future__ import annotations

from typing import Any

# Import from modular components
from .renderer import PromptError, render_template
from .sanitization import sanitize_text, sanitize_code
from .schemas import SchemaDefinition, get_schema_definition
from .templates.v1 import system, user, evaluation


class PromptRegistry:
    """
    Versioned registry for system/user prompts and JSON schemas.

    Notes on anti-injection:
    - System prompts include explicit constraints and guardrails
    - User prompts wrap dynamic content between SAFE delimiters
    - Light sanitization is applied to user-provided inputs
    """

    SUPPORTED_VERSIONS = {"v1"}

    def __init__(self, version: str = "v1") -> None:
        if version not in self.SUPPORTED_VERSIONS:
            raise PromptError(f"Unsupported prompt version: {version}")
        self.version = version

        # Initialize system prompt store - delegate to template modules
        self._system_prompts: dict[str, dict[str, str]] = {
            "v1": {
                "generation": system.system_prompt_generation_v1(),
                "refinement": system.system_prompt_refinement_v1(),
                "llm_test_generation": system.system_prompt_llm_test_generation_v1(),
                "llm_code_analysis": system.system_prompt_llm_code_analysis_v1(),
                "llm_content_refinement": system.system_prompt_llm_content_refinement_v1(),
                "llm_manual_fix_suggestions": system.system_prompt_llm_manual_fix_suggestions_v1(),
                "llm_test_planning": system.system_prompt_llm_test_planning_v1(),
                "llm_test_planning_v1": system.system_prompt_llm_test_planning_v1(),
                "llm_judge": system.system_prompt_llm_judge_v1(),
                "pairwise_comparison": system.system_prompt_pairwise_comparison_v1(),
                "rubric_evaluation": system.system_prompt_rubric_evaluation_v1(),
                "statistical_analysis": system.system_prompt_statistical_analysis_v1(),
                "bias_mitigation": system.system_prompt_bias_mitigation_v1(),
            }
        }

        # Initialize user prompt store - delegate to template modules
        self._user_prompts: dict[str, dict[str, str]] = {
            "v1": {
                "generation": user.user_prompt_generation_v1(),
                "refinement": user.user_prompt_refinement_v1(),
                "llm_test_generation": user.user_prompt_llm_test_generation_v1(),
                "llm_code_analysis": user.user_prompt_llm_code_analysis_v1(),
                "llm_content_refinement": user.user_prompt_llm_content_refinement_v1(),
                "llm_manual_fix_suggestions": user.user_prompt_llm_manual_fix_suggestions_v1(),
                "llm_test_planning": user.user_prompt_llm_test_planning_v1(),
                "llm_test_planning_v1": user.user_prompt_llm_test_planning_v1(),
                "llm_judge": user.user_prompt_llm_judge_v1(),
                "pairwise_comparison": user.user_prompt_pairwise_comparison_v1(),
                "rubric_evaluation": user.user_prompt_rubric_evaluation_v1(),
                "statistical_analysis": user.user_prompt_statistical_analysis_v1(),
                "bias_mitigation": user.user_prompt_bias_mitigation_v1(),
            }
        }

    def get_system_prompt(
        self,
        prompt_type: str,
        **kwargs: Any,
    ) -> str:
        """Get system prompt by type with optional customization."""
        template = self._lookup(self._system_prompts, prompt_type)
        return render_template(template, kwargs) if kwargs else template

    def get_user_prompt(
        self,
        prompt_type: str,
        code_content: str = "",
        additional_context: str = "",
        **kwargs: Any,
    ) -> str:
        """Get user prompt with embedded content."""
        template = self._lookup(self._user_prompts, prompt_type)
        
        # Apply sanitization to dynamic content
        safe_code = sanitize_code(code_content)
        safe_context = sanitize_text(additional_context)
        
        # Prepare template values
        template_values = {
            "version": self.version,
            "code_content": safe_code,
            "additional_context": safe_context,
            **kwargs,
        }
        
        return render_template(template, template_values)

    def get_schema(
        self,
        schema_type: str, 
        language: str = "python"
    ) -> SchemaDefinition:
        """Get schema definition for structured output."""
        return get_schema_definition(schema_type, language, self.version)

    def customize_prompt(
        self,
        base_template: str, 
        customizations: dict[str, Any]
    ) -> str:
        """Apply customizations to a base template."""
        return render_template(base_template, customizations)

    def get_prompt(self, category: str, prompt_type: str, **kwargs: Any) -> str | None:
        """
        Get prompt by category and type.

        Args:
            category: 'system' or 'user'
            prompt_type: Type of prompt to retrieve
            **kwargs: Template customization values

        Returns:
            Rendered prompt string, or None if not found
        """
        try:
            if category == "system":
                return self.get_system_prompt(prompt_type, **kwargs)
            elif category == "user":
                return self.get_user_prompt(prompt_type, **kwargs)
            else:
                return None
        except (KeyError, PromptError):
            return None

    def validate_prompt(
        self,
        prompt: str,
        max_length: int = 50000,
        check_injection: bool = True,
    ) -> bool:
        """
        Validate a prompt for length and basic injection patterns.
        
        Args:
            prompt: Prompt string to validate
            max_length: Maximum allowed length
            check_injection: Whether to check for injection patterns
            
        Returns:
            True if prompt passes validation, False otherwise
        """
        if not prompt or not prompt.strip():
            return False
            
        if len(prompt) > max_length:
            return False

        if check_injection:
            # Check for common injection patterns
            suspicious_patterns = [
                "ignore previous instructions",
                "disregard the above",
                "override system prompt",
                "pretend you are",
                "roleplay as",
            ]
            lower_prompt = prompt.lower()
            if any(pattern in lower_prompt for pattern in suspicious_patterns):
                return False

        return True

    def _schema_for(self, schema_type: str, language: str) -> SchemaDefinition:
        """Legacy method for backward compatibility."""
        return self.get_schema(schema_type, language)

    # Helper methods
    def _lookup(self, store: dict[str, dict[str, str]], prompt_type: str) -> str:
        """Look up a prompt template from the store."""
        try:
            return store[self.version][prompt_type]
        except KeyError as exc:
            raise PromptError(
                f"Template not found for version={self.version}, type={prompt_type}"
            ) from exc


# Re-export components for backward compatibility
__all__ = [
    "PromptRegistry",
    "PromptError", 
    "SchemaDefinition",
    "sanitize_text",
    "sanitize_code",
]
