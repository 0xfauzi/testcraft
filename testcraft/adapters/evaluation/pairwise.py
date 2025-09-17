"""
Pairwise comparison evaluation functions.

This module provides side-by-side comparison of test variants using LLM evaluation
following 2025 A/B testing best practices with statistical confidence estimation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ...ports.evaluation_port import ComparisonMode
from ...ports.llm_port import LLMPort
from ...prompts.registry import PromptRegistry
from ..io.artifact_store import ArtifactStoreAdapter, ArtifactType

logger = logging.getLogger(__name__)


class PairwiseComparator:
    """Handles pairwise comparison of test variants."""

    def __init__(
        self,
        llm_adapter: LLMPort,
        artifact_store: ArtifactStoreAdapter,
        prompt_registry: PromptRegistry | None = None,
    ):
        """
        Initialize pairwise comparator.

        Args:
            llm_adapter: LLM adapter for comparisons
            artifact_store: Artifact storage adapter
            prompt_registry: Optional prompt registry for comparison prompts
        """
        self.llm_adapter = llm_adapter
        self.artifact_store = artifact_store
        self.prompt_registry = prompt_registry or PromptRegistry()

    def evaluate_pairwise(
        self,
        test_a: str,
        test_b: str,
        source_file: str,
        comparison_mode: ComparisonMode = "a_vs_b",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Compare two test variants using pairwise LLM evaluation.

        This implements side-by-side comparison following 2025 A/B testing
        best practices with statistical confidence estimation.
        """
        logger.debug(f"Running pairwise evaluation for {source_file}")

        try:
            # Use enhanced pairwise comparison prompts (2025 best practices)
            system_prompt = self.prompt_registry.get_prompt(
                "system", "pairwise_comparison_v1"
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", "pairwise_comparison_v1"
            )

            # Read source content for context
            try:
                source_content = Path(source_file).read_text(encoding="utf-8")
            except Exception:
                source_content = "# Source file not available"

            if system_prompt and user_prompt:
                # Use advanced prompt structure
                formatted_user_prompt = user_prompt.format(
                    version="v1",
                    comparison_mode=comparison_mode,
                    source_content=source_content,
                    test_a=test_a,
                    test_b=test_b,
                    evaluation_context=kwargs.get(
                        "evaluation_context", "Standard pairwise comparison"
                    ),
                )

                llm_response = self.llm_adapter.analyze_code(
                    formatted_user_prompt,
                    analysis_type="pairwise_comparison",
                    temperature=0.1,
                    system_prompt=system_prompt,
                    **kwargs,
                )
            else:
                # Fallback to old structure
                comparison_prompt = self._get_default_pairwise_prompt()
                formatted_prompt = comparison_prompt.format(
                    test_a=test_a,
                    test_b=test_b,
                    source_content=source_content,
                    source_file=source_file,
                    comparison_mode=comparison_mode,
                )

                llm_response = self.llm_adapter.analyze_code(
                    formatted_prompt,
                    analysis_type="pairwise_comparison",
                    temperature=0.1,
                    **kwargs,
                )

            # Parse comparison result (will be moved to parsers module)
            comparison_result = self._parse_pairwise_response(llm_response)

            # Add metadata
            comparison_result.update(
                {
                    "source_file": source_file,
                    "comparison_mode": comparison_mode,
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_a_length": len(test_a),
                    "test_b_length": len(test_b),
                }
            )

            # Store comparison artifact
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                comparison_result,
                tags=["pairwise", "comparison", Path(source_file).stem],
                description=f"Pairwise comparison for {source_file}",
            )

            logger.info(
                f"Pairwise evaluation completed: winner = {comparison_result.get('winner', 'unknown')}"
            )
            return comparison_result

        except Exception as e:
            logger.error(f"Pairwise evaluation failed: {e}")
            raise

    def _get_default_pairwise_prompt(self) -> str:
        """Get default pairwise comparison prompt."""
        return """
Compare these two test implementations for the same source code.
Determine which test is better overall and explain your reasoning.

Source Code:
```python
{source_content}
```

Test A:
```python
{test_a}
```

Test B:
```python
{test_b}
```

Consider: correctness, coverage, clarity, maintainability, and safety.

Respond in JSON format:
{{
    "winner": "a|b|tie",
    "confidence": <0.0-1.0>,
    "reasoning": "<detailed explanation>",
    "scores": {{
        "test_a": <1-5>,
        "test_b": <1-5>
    }}
}}
"""

    def _parse_pairwise_response(self, llm_response: dict[str, Any]) -> dict[str, Any]:
        """Parse pairwise comparison response."""
        try:
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                data = response_text
            else:
                import re
                import json

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            return {
                "winner": data.get("winner", "tie"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", "No reasoning provided"),
                "scores": data.get("scores", {"test_a": 3.0, "test_b": 3.0}),
            }

        except Exception as e:
            logger.warning(f"Failed to parse pairwise response: {e}")
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reasoning": f"Parse error: {e}",
                "scores": {"test_a": 3.0, "test_b": 3.0},
            }
