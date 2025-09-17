"""
LLM-as-judge evaluation functions.

This module provides LLM-based evaluation of test quality using structured
prompts and rubric-driven scoring following 2025 best practices.
"""

import logging
from pathlib import Path
from typing import Any

from ...ports.evaluation_port import LLMJudgeResult
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class LLMJudge:
    """Handles LLM-as-judge evaluation of test quality."""

    def __init__(
        self,
        llm_adapter: LLMPort,
        prompt_registry: PromptRegistry | None = None,
    ):
        """
        Initialize LLM judge.

        Args:
            llm_adapter: LLM adapter for judge evaluations
            prompt_registry: Optional prompt registry for evaluation prompts
        """
        self.llm_adapter = llm_adapter
        self.prompt_registry = prompt_registry or PromptRegistry()

    def evaluate_with_llm_judge(
        self,
        test_content: str,
        source_file: str,
        rubric_dimensions: list[str] | None = None,
        prompt_version: str | None = None,
        **kwargs: Any,
    ) -> LLMJudgeResult:
        """
        Evaluate test quality using LLM-as-judge with rubric-driven scoring.

        This uses structured prompts to get both numeric scores and rationales
        for each evaluation dimension, following 2025 best practices.
        """
        logger.debug(f"Running LLM judge evaluation for {source_file}")

        try:
            # Use provided dimensions or defaults
            dimensions = rubric_dimensions or [
                "correctness",
                "coverage",
                "clarity",
                "safety",
            ]

            # Get enhanced evaluation prompt from registry (2025 best practices)
            prompt_version = prompt_version or "llm_judge_v1"

            # Use new rubric-driven evaluation prompts
            system_prompt = self.prompt_registry.get_prompt("system", prompt_version)
            user_prompt = self.prompt_registry.get_prompt("user", prompt_version)

            if not system_prompt or not user_prompt:
                # Fallback to built-in prompts
                evaluation_prompt = self._get_default_evaluation_prompt()
                prompt_version = "builtin_default"
            else:
                # Use advanced prompts with context formatting
                evaluation_prompt = system_prompt

            # Read source code for context
            try:
                source_content = Path(source_file).read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read source file {source_file}: {e}")
                source_content = "# Source file not available"

            # Use enhanced prompt structure for 2025 best practices
            if system_prompt and user_prompt:
                # Format user prompt with context
                formatted_user_prompt = user_prompt.format(
                    version="v1",
                    dimensions=", ".join(dimensions),
                    source_content=source_content,
                    test_content=test_content,
                    additional_context=kwargs.get(
                        "additional_context", "No additional context provided"
                    ),
                )

                # Call LLM with system/user prompt structure
                llm_response = self.llm_adapter.analyze_code(
                    formatted_user_prompt,
                    analysis_type="evaluation",
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=system_prompt,
                    **kwargs,
                )
            else:
                # Fallback to old structure
                context = self._build_evaluation_context(
                    test_content, source_content, source_file, dimensions
                )

                formatted_prompt = evaluation_prompt.format(
                    test_content=test_content,
                    source_content=source_content,
                    source_file=source_file,
                    dimensions=", ".join(dimensions),
                    context=context,
                )

                llm_response = self.llm_adapter.analyze_code(
                    formatted_prompt,
                    analysis_type="evaluation",
                    max_tokens=2000,
                    temperature=0.1,
                    **kwargs,
                )

            # Parse structured response (will be moved to parsers module)
            scores, rationales = self._parse_llm_evaluation_response(
                llm_response, dimensions
            )

            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0

            result = LLMJudgeResult(
                scores=scores,
                rationales=rationales,
                overall_score=overall_score,
                prompt_version=prompt_version,
                confidence=llm_response.get("confidence", 0.8),
            )

            logger.debug(
                f"LLM judge evaluation completed with overall score: {overall_score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return LLMJudgeResult.empty()

    def _build_evaluation_context(
        self,
        test_content: str,
        source_content: str,
        source_file: str,
        dimensions: list[str],
    ) -> str:
        """Build context string for LLM evaluation."""
        context_parts = [
            f"Source file: {source_file}",
            f"Evaluation dimensions: {', '.join(dimensions)}",
            f"Test content length: {len(test_content)} characters",
            f"Source content length: {len(source_content)} characters",
        ]

        return "\n".join(context_parts)

    def _get_default_evaluation_prompt(self) -> str:
        """Get default LLM evaluation prompt."""
        return """
You are an expert Python test reviewer. Evaluate the following generated test code on these dimensions:

{dimensions}

Rate each dimension from 1-5 and provide a brief rationale:
- Correctness (1-5): Does the test correctly validate the intended behavior?
- Coverage (1-5): Does the test increase code coverage, especially for edge cases?
- Clarity (1-5): Is the test code readable and maintainable?
- Safety (1-5): Does the test avoid modifying source files or introducing side effects?

Source Code:
```python
{source_content}
```

Generated Test:
```python
{test_content}
```

Respond in JSON format:
{{
    "scores": {{
        "correctness": <score>,
        "coverage": <score>,
        "clarity": <score>,
        "safety": <score>
    }},
    "rationales": {{
        "correctness": "<rationale>",
        "coverage": "<rationale>",
        "clarity": "<rationale>",
        "safety": "<rationale>"
    }}
}}
"""

    def _parse_llm_evaluation_response(
        self, llm_response: dict[str, Any], dimensions: list[str]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Parse LLM evaluation response into scores and rationales."""
        try:
            # Try to extract JSON from response
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                data = response_text
            else:
                # Try to parse JSON from text
                import re
                import json

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            scores = {}
            rationales = {}

            for dimension in dimensions:
                scores[dimension] = float(data.get("scores", {}).get(dimension, 3.0))
                rationales[dimension] = data.get("rationales", {}).get(
                    dimension, "No rationale provided"
                )

            return scores, rationales

        except Exception as e:
            logger.warning(f"Failed to parse LLM evaluation response: {e}")
            # Return default scores
            return dict.fromkeys(dimensions, 3.0), dict.fromkeys(
                dimensions, "Parse error"
            )
