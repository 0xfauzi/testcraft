"""
A/B testing functions for evaluation pipelines.

This module provides utilities for A/B testing workflows including test generation
with variants, summary statistics calculation, and recommendation generation.
"""

import logging
from pathlib import Path
from typing import Any

from ...ports.evaluation_port import EvaluationResult

logger = logging.getLogger(__name__)


class ABTestingManager:
    """Handles A/B testing workflows and analysis."""

    @staticmethod
    def generate_test_with_variant(
        variant: dict[str, str], test_case: dict[str, str]
    ) -> str:
        """Generate test content using a specific prompt variant."""
        # This is a simplified implementation - you would integrate with your test generation system
        try:
            source_file = test_case.get("source_file", "example.py")
            test_case.get("source_content", "# No source content available")

            # Use LLM to generate test with the specific variant
            prompt_content = variant.get(
                "prompt", "Generate a comprehensive test for the given code."
            )

            # Simple test generation (you would use your actual generation pipeline here)
            generated_test = f"""
import pytest
from pathlib import Path

def test_{Path(source_file).stem}():
    '''Generated test using variant: {variant.get("id", "unknown")}'''
    # This is a placeholder test generated for evaluation
    # Variant prompt: {prompt_content[:100]}...
    assert True  # Placeholder assertion
"""
            return generated_test
        except Exception as e:
            logger.warning(f"Test generation with variant failed: {e}")
            return "# Test generation failed\ndef test_placeholder(): assert True"

    @staticmethod
    def calculate_variant_summary_stats(
        variant_results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Calculate summary statistics for a prompt variant."""
        try:
            acceptance_rate = sum(
                1 for r in variant_results if r.acceptance.all_checks_pass
            ) / len(variant_results)

            llm_scores = [
                r.llm_judge.overall_score for r in variant_results if r.llm_judge
            ]
            mean_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0

            return {
                "total_tests": len(variant_results),
                "acceptance_rate": acceptance_rate,
                "mean_llm_score": mean_score,
                "successful_evaluations": len(llm_scores),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate variant summary stats: {e}")
            return {"error": f"Calculation failed: {e}"}

    @staticmethod
    def calculate_pairwise_summary(
        comparison_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate summary for pairwise comparisons."""
        try:
            if not comparison_results:
                return {"error": "No comparison results"}

            winners = [result.get("winner", "tie") for result in comparison_results]
            winner_counts = {
                "a": winners.count("a"),
                "b": winners.count("b"),
                "tie": winners.count("tie"),
            }

            confidences = [
                result.get("confidence", 0.5) for result in comparison_results
            ]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "total_comparisons": len(comparison_results),
                "winner_distribution": winner_counts,
                "mean_confidence": mean_confidence,
                "clear_winner": (
                    "a"
                    if winner_counts["a"] > winner_counts["b"] + winner_counts["tie"]
                    else (
                        "b"
                        if winner_counts["b"]
                        > winner_counts["a"] + winner_counts["tie"]
                        else "tie"
                    )
                ),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate pairwise summary: {e}")
            return {"error": f"Calculation failed: {e}"}

    @staticmethod
    def generate_ab_testing_recommendations(
        pipeline_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on A/B testing results."""
        recommendations = []

        try:
            # Analyze variant performance
            variant_evals = pipeline_results.get("variant_evaluations", [])
            if variant_evals:
                best_variant = max(
                    variant_evals,
                    key=lambda v: v.get("summary_stats", {}).get("mean_llm_score", 0),
                )
                recommendations.append(
                    f"Consider using variant '{best_variant.get('variant_id', 'unknown')}' based on highest mean LLM score"
                )

            # Statistical analysis recommendations
            stat_analysis = pipeline_results.get("statistical_analysis")
            if stat_analysis and not stat_analysis.get("error"):
                significance = stat_analysis.get("significance_assessment", "unknown")
                if significance == "not_significant":
                    recommendations.append(
                        "Differences between variants are not statistically significant - consider larger sample size or different evaluation criteria"
                    )
                elif significance in ["significant", "highly_significant"]:
                    recommendations.append(
                        "Statistically significant differences detected - results are reliable for decision making"
                    )

            # Bias analysis recommendations
            bias_analysis = pipeline_results.get("bias_analysis")
            if bias_analysis and not bias_analysis.get("error"):
                fairness_score = bias_analysis.get("fairness_score", 0.5)
                if fairness_score < 0.7:
                    recommendations.append(
                        "Potential evaluation bias detected - review evaluation methodology and consider bias mitigation strategies"
                    )

            # General recommendations
            metadata = pipeline_results.get("metadata", {})
            if metadata.get("test_case_count", 0) < 20:
                recommendations.append(
                    "Consider increasing test case count for more robust evaluation"
                )

            if not recommendations:
                recommendations.append(
                    "A/B testing pipeline completed successfully - review detailed results for insights"
                )

        except Exception as e:
            logger.warning(f"Failed to generate A/B testing recommendations: {e}")
            recommendations.append(
                "Review A/B testing results manually due to recommendation generation error"
            )

        return recommendations
