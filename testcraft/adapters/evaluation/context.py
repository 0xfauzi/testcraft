"""
Context building functions for evaluation operations.

This module provides utilities for formatting evaluation data and building
context strings for various evaluation scenarios including statistical analysis
and bias detection.
"""

import json
import logging
from typing import Any

from ...ports.evaluation_port import EvaluationResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Handles building context for various evaluation operations."""

    @staticmethod
    def build_evaluation_context(
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

    @staticmethod
    def format_evaluation_data_for_analysis(
        evaluation_data: list[dict[str, Any]]
    ) -> str:
        """Format evaluation data for statistical analysis prompts."""
        try:
            formatted_data = []
            for i, data in enumerate(evaluation_data):
                formatted_entry = {
                    "entry_id": i,
                    "variant_id": data.get("variant_id", "unknown"),
                    "overall_score": data.get("overall_score", 0),
                    "dimension_scores": data.get("scores", {}),
                    "acceptance_passed": data.get("acceptance_passed", False),
                }
                formatted_data.append(formatted_entry)

            return json.dumps(formatted_data, indent=2)
        except Exception as e:
            logger.warning(f"Failed to format evaluation data: {e}")
            return "[]"

    @staticmethod
    def format_evaluation_history_for_bias_analysis(
        evaluation_history: list[EvaluationResult]
    ) -> str:
        """Format evaluation history for bias analysis prompts."""
        try:
            formatted_history = []
            for result in evaluation_history:
                entry = {
                    "test_id": result.test_id,
                    "timestamp": result.timestamp,
                    "acceptance_passed": result.acceptance.all_checks_pass,
                    "test_length": len(result.test_content),
                    "source_file": result.source_file,
                }

                if result.llm_judge:
                    entry.update(
                        {
                            "llm_scores": result.llm_judge.scores,
                            "overall_score": result.llm_judge.overall_score,
                            "prompt_version": result.llm_judge.prompt_version,
                        }
                    )

                formatted_history.append(entry)

            return json.dumps(formatted_history, indent=2)
        except Exception as e:
            logger.warning(f"Failed to format evaluation history: {e}")
            return "[]"

    @staticmethod
    def build_analysis_parameters(
        confidence_level: float,
        analysis_type: str,
        sample_size: int,
        requested_tests: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build analysis parameters for statistical analysis."""
        return {
            "confidence_level": confidence_level,
            "analysis_type": analysis_type,
            "sample_size": sample_size,
            "requested_tests": requested_tests or ["t_test", "bootstrap", "effect_size"],
        }

    @staticmethod
    def build_bias_analysis_scope(
        bias_types: list[str] | None = None,
        evaluation_count: int = 0,
        include_consistency_metrics: bool = True,
        include_drift_detection: bool = True,
    ) -> dict[str, Any]:
        """Build analysis scope for bias detection."""
        return {
            "bias_types_to_check": bias_types
            or [
                "length_bias",
                "complexity_bias",
                "style_bias",
                "framework_bias",
                "anchoring_bias",
                "order_bias",
                "confirmation_bias",
            ],
            "evaluation_period": f"{evaluation_count} evaluations",
            "consistency_metrics": include_consistency_metrics,
            "drift_detection": include_drift_detection,
        }
