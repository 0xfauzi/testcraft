"""
Response parsing functions for evaluation results.

This module provides parsing utilities for LLM responses in evaluation contexts,
including structured JSON extraction and fallback handling.
"""

import json
import logging
import re
from typing import Any

from ...ports.evaluation_port import EvaluationResult

logger = logging.getLogger(__name__)


class ResponseParser:
    """Handles parsing of various LLM evaluation responses."""

    @staticmethod
    def parse_llm_evaluation_response(
        llm_response: dict[str, Any], dimensions: list[str]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Parse LLM evaluation response into scores and rationales."""
        try:
            # Try to extract JSON from response
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                data = response_text
            else:
                # Try to parse JSON from text
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

    @staticmethod
    def parse_pairwise_response(llm_response: dict[str, Any]) -> dict[str, Any]:
        """Parse pairwise comparison response."""
        try:
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                data = response_text
            else:
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

    @staticmethod
    def parse_statistical_analysis_response(
        llm_response: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse LLM statistical analysis response."""
        try:
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                return response_text

            # Try to parse JSON from text response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Return fallback structure
            return {
                "statistical_test": "fallback",
                "p_value": 0.5,
                "confidence_interval": {
                    "lower": -0.5,
                    "upper": 0.5,
                    "confidence_level": 0.95,
                },
                "effect_size": {"cohens_d": 0.0, "interpretation": "negligible"},
                "significance_assessment": "not_significant",
                "sample_adequacy": {
                    "current_sample_size": len([]),
                    "recommended_minimum": 30,
                    "power_achieved": 0.5,
                },
                "reliability_metrics": {
                    "evaluation_consistency": 0.5,
                    "potential_bias_detected": False,
                },
                "interpretation": "Fallback analysis - original parsing failed",
                "recommendations": [
                    "Increase sample size",
                    "Validate evaluation methodology",
                ],
            }
        except Exception as e:
            logger.warning(f"Failed to parse statistical analysis response: {e}")
            return {"error": f"Parsing failed: {e}", "fallback_used": True}

    @staticmethod
    def parse_bias_analysis_response(llm_response: dict[str, Any]) -> dict[str, Any]:
        """Parse LLM bias analysis response."""
        try:
            response_text = llm_response.get("analysis", "")
            if isinstance(response_text, dict):
                return response_text

            # Try to parse JSON from text response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Return fallback structure
            return {
                "bias_analysis": {
                    "detected_biases": ["unknown_bias"],
                    "bias_severity": {"unknown_bias": "low"},
                    "confidence": 0.5,
                },
                "evaluation_consistency": {
                    "consistency_score": 0.7,
                    "variance_analysis": "Fallback analysis - unable to perform detailed assessment",
                    "drift_detected": False,
                },
                "calibration_assessment": {
                    "calibration_score": 0.7,
                    "systematic_errors": [
                        "Unable to detect systematic errors in fallback mode"
                    ],
                    "improvement_needed": True,
                },
                "mitigation_recommendations": {
                    "immediate_actions": ["Review evaluation methodology"],
                    "process_improvements": ["Implement structured rubrics"],
                    "monitoring_suggestions": ["Track evaluation consistency"],
                },
                "fairness_score": 0.7,
                "summary": "Fallback bias analysis - detailed assessment unavailable due to parsing error",
            }
        except Exception as e:
            logger.warning(f"Failed to parse bias analysis response: {e}")
            return {"error": f"Parsing failed: {e}", "fallback_used": True}

    @staticmethod
    def dict_to_evaluation_result(result_dict: dict[str, Any]) -> EvaluationResult:
        """Convert dictionary back to EvaluationResult object."""
        # Simplified conversion - you might want to implement a proper deserialization
        try:
            from ...ports.evaluation_port import AcceptanceResult, LLMJudgeResult

            acceptance_data = result_dict.get("acceptance", {})
            acceptance = AcceptanceResult(
                syntactically_valid=acceptance_data.get("syntactically_valid", False),
                imports_successfully=acceptance_data.get("imports_successfully", False),
                pytest_passes=acceptance_data.get("pytest_passes", False),
                coverage_improvement=acceptance_data.get("coverage_improvement"),
                error_messages=acceptance_data.get("error_messages", []),
            )

            llm_judge_data = result_dict.get("llm_judge")
            llm_judge = None
            if llm_judge_data:
                llm_judge = LLMJudgeResult(
                    scores=llm_judge_data.get("scores", {}),
                    rationales=llm_judge_data.get("rationales", {}),
                    overall_score=llm_judge_data.get("overall_score", 0),
                    prompt_version=llm_judge_data.get("prompt_version", "unknown"),
                )

            return EvaluationResult(
                test_id=result_dict.get("test_id", "unknown"),
                source_file=result_dict.get("source_file", "unknown"),
                test_content=result_dict.get("test_content", ""),
                acceptance=acceptance,
                llm_judge=llm_judge,
                metadata=result_dict.get("metadata", {}),
                timestamp=result_dict.get("timestamp"),
            )
        except Exception as e:
            logger.warning(f"Failed to convert dict to EvaluationResult: {e}")
            # Return minimal result
            from ...ports.evaluation_port import AcceptanceResult

            return EvaluationResult(
                test_id="conversion_failed",
                source_file="unknown",
                test_content="",
                acceptance=AcceptanceResult(False, False, False),
            )
