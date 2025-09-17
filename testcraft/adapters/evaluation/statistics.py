"""
Statistical analysis functions for evaluation data.

This module provides statistical analysis utilities including fallback analysis,
result validation, and bias metrics computation for evaluation results.
"""

import logging
from datetime import datetime
from typing import Any

from ...ports.evaluation_port import EvaluationResult

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Handles statistical analysis of evaluation data."""

    @staticmethod
    def fallback_statistical_analysis(
        evaluation_data: list[dict[str, Any]], confidence_level: float
    ) -> dict[str, Any]:
        """Provide basic statistical analysis when LLM-based analysis is unavailable."""
        try:
            if not evaluation_data:
                return {"error": "No evaluation data provided"}

            # Simple statistical calculations
            scores = [item.get("overall_score", 0) for item in evaluation_data]
            mean_score = sum(scores) / len(scores) if scores else 0

            # Basic statistical assessment
            sample_size = len(evaluation_data)
            is_adequate = sample_size >= 30

            return {
                "statistical_test": "descriptive",
                "p_value": 0.5,  # Neutral
                "confidence_interval": {
                    "lower": max(0, mean_score - 0.5),
                    "upper": min(5, mean_score + 0.5),
                    "confidence_level": confidence_level,
                },
                "effect_size": {"cohens_d": 0.0, "interpretation": "unknown"},
                "significance_assessment": "not_assessed",
                "sample_adequacy": {
                    "current_sample_size": sample_size,
                    "recommended_minimum": 30,
                    "power_achieved": 0.8 if is_adequate else 0.5,
                },
                "reliability_metrics": {
                    "evaluation_consistency": 0.7,
                    "potential_bias_detected": False,
                },
                "interpretation": f"Basic descriptive analysis of {sample_size} evaluations with mean score {mean_score:.2f}",
                "recommendations": [
                    "Increase sample size for statistical testing",
                    "Use LLM-based analysis for detailed insights",
                ],
                "fallback_analysis": True,
            }
        except Exception as e:
            return {"error": f"Fallback analysis failed: {e}"}

    @staticmethod
    def fallback_bias_analysis(evaluation_history: list[EvaluationResult]) -> dict[str, Any]:
        """Provide basic bias analysis when LLM-based analysis is unavailable."""
        try:
            if not evaluation_history:
                return {"error": "No evaluation history provided"}

            # Basic bias indicators
            test_lengths = [len(result.test_content) for result in evaluation_history]
            length_variance = (
                max(test_lengths) - min(test_lengths) if test_lengths else 0
            )

            overall_scores = [
                result.llm_judge.overall_score
                for result in evaluation_history
                if result.llm_judge
            ]
            score_variance = (
                (max(overall_scores) - min(overall_scores)) if overall_scores else 0
            )

            # Basic fairness assessment
            fairness_score = (
                0.8 if length_variance < 1000 and score_variance < 2.0 else 0.6
            )

            return {
                "bias_analysis": {
                    "detected_biases": (
                        ["length_bias"] if length_variance > 1000 else []
                    ),
                    "bias_severity": (
                        {"length_bias": "moderate"} if length_variance > 1000 else {}
                    ),
                    "confidence": 0.6,
                },
                "evaluation_consistency": {
                    "consistency_score": max(0.5, 1.0 - (score_variance / 5.0)),
                    "variance_analysis": f"Score variance: {score_variance:.2f}, Length variance: {length_variance}",
                    "drift_detected": score_variance > 2.0,
                },
                "calibration_assessment": {
                    "calibration_score": 0.7,
                    "systematic_errors": (
                        ["High variance detected"] if score_variance > 2.0 else []
                    ),
                    "improvement_needed": score_variance > 1.5,
                },
                "mitigation_recommendations": {
                    "immediate_actions": ["Monitor evaluation consistency"],
                    "process_improvements": ["Implement evaluation guidelines"],
                    "monitoring_suggestions": ["Track variance metrics"],
                },
                "fairness_score": fairness_score,
                "summary": f"Basic bias analysis of {len(evaluation_history)} evaluations - fairness score: {fairness_score:.2f}",
                "fallback_analysis": True,
            }
        except Exception as e:
            return {"error": f"Fallback bias analysis failed: {e}"}

    @staticmethod
    def validate_statistical_results(
        analysis_result: dict[str, Any], evaluation_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Add computational validation to statistical analysis results."""
        try:
            # Basic validation and augmentation
            sample_size = len(evaluation_data)
            analysis_result["validation"] = {
                "sample_size_confirmed": sample_size,
                "computational_checks": "basic_validation_applied",
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Validate confidence intervals make sense
            if "confidence_interval" in analysis_result:
                ci = analysis_result["confidence_interval"]
                if ci["lower"] > ci["upper"]:
                    logger.warning("Invalid confidence interval detected, correcting")
                    ci["lower"], ci["upper"] = ci["upper"], ci["lower"]

            return analysis_result
        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
            return analysis_result

    @staticmethod
    def compute_bias_metrics(
        bias_result: dict[str, Any], evaluation_history: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Add computational bias metrics to LLM bias analysis."""
        try:
            # Basic computational metrics
            test_lengths = [len(result.test_content) for result in evaluation_history]
            scores = [
                result.llm_judge.overall_score
                for result in evaluation_history
                if result.llm_judge
            ]

            bias_result["computational_metrics"] = {
                "length_statistics": {
                    "mean": (
                        sum(test_lengths) / len(test_lengths) if test_lengths else 0
                    ),
                    "min": min(test_lengths) if test_lengths else 0,
                    "max": max(test_lengths) if test_lengths else 0,
                    "variance": (
                        length_variance
                        if (length_variance := (max(test_lengths) - min(test_lengths)))
                        else 0
                    ),
                },
                "score_statistics": {
                    "mean": sum(scores) / len(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "variance": (max(scores) - min(scores)) if scores else 0,
                },
                "evaluation_count": len(evaluation_history),
            }

            return bias_result
        except Exception as e:
            logger.warning(f"Bias metrics computation failed: {e}")
            return bias_result
