"""
Analytics functions for evaluation trend analysis.

This module provides utilities for analyzing evaluation trends, quality distributions,
and generating recommendations based on historical evaluation data.
"""

import logging
from typing import Any

from ...ports.evaluation_port import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationAnalytics:
    """Handles analytics and trend analysis for evaluation results."""

    @staticmethod
    def generate_recommendations(
        stats: dict[str, Any], file_results: list[EvaluationResult]
    ) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if stats["acceptance_pass_rate"] < 0.5:
            recommendations.append(
                "Consider improving test generation prompts - low acceptance rate"
            )

        if stats["average_llm_score"] < 3.0:
            recommendations.append("Generated tests may need quality improvements")

        if not stats["coverage_improvements"]:
            recommendations.append(
                "Tests may not be improving code coverage significantly"
            )

        return recommendations

    @staticmethod
    def analyze_acceptance_trends(results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze trends in acceptance check results."""
        syntax_passes = sum(1 for r in results if r.acceptance.syntactically_valid)
        import_passes = sum(1 for r in results if r.acceptance.imports_successfully)
        pytest_passes = sum(1 for r in results if r.acceptance.pytest_passes)

        return {
            "syntax_pass_rate": syntax_passes / len(results),
            "import_pass_rate": import_passes / len(results),
            "pytest_pass_rate": pytest_passes / len(results),
        }

    @staticmethod
    def analyze_llm_trends(results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze trends in LLM judge scores."""
        llm_results = [r.llm_judge for r in results if r.llm_judge]

        if not llm_results:
            return {"no_llm_data": True}

        avg_scores = {}
        for dimension in ["correctness", "coverage", "clarity", "safety"]:
            scores = [r.scores.get(dimension, 0) for r in llm_results if r.scores]
            avg_scores[dimension] = sum(scores) / len(scores) if scores else 0

        overall_scores = [r.overall_score for r in llm_results]

        return {
            "average_dimension_scores": avg_scores,
            "average_overall_score": sum(overall_scores) / len(overall_scores),
            "score_trend": (
                "improving"
                if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[0]
                else "stable"
            ),
        }

    @staticmethod
    def analyze_coverage_trends(results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze trends in coverage improvements."""
        improvements = [
            r.acceptance.coverage_improvement
            for r in results
            if r.acceptance.coverage_improvement is not None
        ]

        if not improvements:
            return {"no_coverage_data": True}

        return {
            "average_improvement": sum(improvements) / len(improvements),
            "positive_improvements": sum(1 for i in improvements if i > 0),
            "improvement_rate": sum(1 for i in improvements if i > 0)
            / len(improvements),
        }

    @staticmethod
    def analyze_quality_distribution(results: list[EvaluationResult]) -> dict[str, Any]:
        """Analyze the distribution of test quality."""
        quality_buckets = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

        for result in results:
            if result.llm_judge and result.llm_judge.overall_score > 0:
                score = result.llm_judge.overall_score
                if score >= 4.5:
                    quality_buckets["excellent"] += 1
                elif score >= 3.5:
                    quality_buckets["good"] += 1
                elif score >= 2.5:
                    quality_buckets["fair"] += 1
                else:
                    quality_buckets["poor"] += 1

        return quality_buckets

    @staticmethod
    def generate_trend_recommendations(trends: dict[str, Any]) -> list[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []

        acceptance = trends.get("acceptance_trends", {})
        if acceptance.get("syntax_pass_rate", 0) < 0.9:
            recommendations.append("Improve prompt templates to reduce syntax errors")

        llm_trends = trends.get("llm_judge_trends", {})
        if not llm_trends.get("no_llm_data"):
            avg_score = llm_trends.get("average_overall_score", 0)
            if avg_score < 3.0:
                recommendations.append(
                    "Consider refining test generation approach for higher quality"
                )

        coverage_trends = trends.get("coverage_trends", {})
        if not coverage_trends.get("no_coverage_data"):
            improvement_rate = coverage_trends.get("improvement_rate", 0)
            if improvement_rate < 0.7:
                recommendations.append(
                    "Focus on generating tests that improve code coverage"
                )

        return recommendations
