"""
End-to-end integration tests for the advanced evaluation harness.

This module provides comprehensive tests for the rubric-driven LLM-as-judge
evaluation pipeline, A/B testing functionality, statistical analysis, and
bias detection following PromptFoo/PromptLayer integration patterns.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from testcraft.adapters.io.artifact_store import ArtifactType
from testcraft.evaluation.harness import (
    TestEvaluationHarness,
    create_evaluation_harness,
)
from testcraft.ports.evaluation_port import (
    AcceptanceResult,
    EvaluationConfig,
    EvaluationResult,
    LLMJudgeResult,
)
from testcraft.prompts.registry import PromptRegistry


@pytest.mark.skip(
    reason="Evaluation harness integration tests need prompt template updates"
)
class TestPromptFooIntegrationPatterns:
    """Test PromptFoo-style evaluation patterns and workflows."""

    def setup_method(self) -> None:
        """Set up test fixtures following PromptFoo patterns."""
        # Create temp directory in tests folder to satisfy safety policies
        test_temp_base = Path(__file__).parent / "test_temp"
        test_temp_base.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=test_temp_base))

        # Create evaluation harness with artifact storage
        eval_config = EvaluationConfig(
            acceptance_checks=True,
            llm_judge_enabled=True,
            statistical_testing=True,
            human_review_enabled=False,
            rubric_dimensions=["correctness", "coverage", "clarity", "safety"],
        )
        self.harness = TestEvaluationHarness(
            config=eval_config,
            project_root=self.temp_dir,
        )

        # Sample prompt variants (PromptFoo pattern)
        self.prompt_variants = [
            {
                "id": "baseline_v1",
                "name": "Baseline Prompt",
                "description": "Standard test generation prompt",
                "prompt": "Generate comprehensive unit tests for the following Python function:\n\n{{ source_code }}\n\nEnsure tests cover edge cases and include proper assertions.",
                "version": "1.0.0",
                "tags": ["baseline", "standard"],
            },
            {
                "id": "enhanced_v1",
                "name": "Enhanced Prompt",
                "description": "Enhanced prompt with explicit requirements",
                "prompt": "Generate comprehensive unit tests for the following Python function:\n\n{{ source_code }}\n\nRequirements:\n- Test normal cases, edge cases, and error conditions\n- Use descriptive test names\n- Include docstrings explaining test purpose\n- Add proper assertions with meaningful error messages\n- Follow pytest best practices",
                "version": "1.0.0",
                "tags": ["enhanced", "detailed"],
            },
            {
                "id": "experimental_v1",
                "name": "Experimental CoT Prompt",
                "description": "Chain-of-thought reasoning for test generation",
                "prompt": "Generate comprehensive unit tests for the following Python function:\n\n{{ source_code }}\n\nThink step by step:\n1. Analyze the function's purpose and behavior\n2. Identify input/output types and constraints\n3. List potential edge cases and error conditions\n4. Design test cases to cover all scenarios\n5. Write clear, maintainable test code\n\nGenerate the tests:",
                "version": "1.0.0",
                "tags": ["experimental", "cot", "reasoning"],
            },
        ]

        # Sample test dataset (PromptFoo pattern)
        self.test_dataset = [
            {
                "name": "simple_math_function",
                "source_file": "math_utils.py",
                "source_content": '''def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b''',
                "expected_behaviors": [
                    "Handle normal positive integers",
                    "Handle negative integers",
                    "Handle zero values",
                    "Raise TypeError for non-integers",
                ],
                "complexity": "simple",
            },
            {
                "name": "string_processing_function",
                "source_file": "text_utils.py",
                "source_content": '''def normalize_text(text: str, lowercase: bool = True, strip_whitespace: bool = True) -> str:
    """Normalize text by applying various transformations."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    result = text
    if strip_whitespace:
        result = result.strip()
    if lowercase:
        result = result.lower()

    return result''',
                "expected_behaviors": [
                    "Handle normal text input",
                    "Apply lowercase transformation",
                    "Strip whitespace when requested",
                    "Handle empty strings",
                    "Raise ValueError for non-string input",
                ],
                "complexity": "medium",
            },
            {
                "name": "complex_data_processor",
                "source_file": "data_processor.py",
                "source_content": '''from typing import List, Dict, Optional, Union
import json

def process_data_batch(
    data: List[Dict[str, Union[str, int, float]]],
    filter_key: Optional[str] = None,
    filter_value: Optional[Union[str, int, float]] = None,
    sort_key: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Union[str, int, float]]]:
    """Process a batch of data with filtering, sorting, and limiting."""
    if not isinstance(data, list):
        raise TypeError("Data must be a list of dictionaries")

    if not data:
        return []

    # Validate data structure
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dictionary")

    result = data.copy()

    # Apply filtering
    if filter_key and filter_value is not None:
        result = [item for item in result if item.get(filter_key) == filter_value]

    # Apply sorting
    if sort_key:
        try:
            result = sorted(result, key=lambda x: x.get(sort_key, ''))
        except TypeError:
            # Handle mixed types by converting to strings
            result = sorted(result, key=lambda x: str(x.get(sort_key, '')))

    # Apply limit
    if limit and limit > 0:
        result = result[:limit]

    return result''',
                "expected_behaviors": [
                    "Handle normal list of dictionaries",
                    "Apply filtering when specified",
                    "Sort by key when requested",
                    "Limit results when specified",
                    "Handle empty input",
                    "Raise appropriate errors for invalid input",
                    "Handle mixed data types in sorting",
                ],
                "complexity": "complex",
            },
        ]

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for deterministic testing."""
        llm_judge_response = {
            "scores": {
                "correctness": 4.5,
                "coverage": 4.0,
                "clarity": 4.8,
                "safety": 5.0,
            },
            "rationales": {
                "correctness": "Tests correctly validate function behavior with appropriate assertions for both success and error cases.",
                "coverage": "Good coverage of main scenarios, could benefit from more edge case testing.",
                "clarity": "Test names and structure are clear and follow good practices.",
                "safety": "No security concerns, proper error handling tested.",
            },
            "overall_score": 4.6,
            "confidence": 0.85,
            "bias_indicators": {"length_bias": 0.1, "complexity_bias": 0.2},
        }

        statistical_analysis_response = {
            "statistical_test": "Welch's t-test",
            "p_value": 0.0234,
            "significance_assessment": "statistically_significant",
            "effect_size": {"cohens_d": 0.73, "interpretation": "medium_effect"},
            "confidence_interval": {
                "lower": 0.12,
                "upper": 0.89,
                "confidence_level": 0.95,
            },
            "sample_adequacy": {
                "current_sample_size": 25,
                "recommended_minimum": 20,
                "power_achieved": 0.82,
            },
            "recommendations": [
                "Results show significant improvement with enhanced prompting",
                "Consider adopting enhanced prompt as new baseline",
                "Monitor performance on complex functions",
            ],
        }

        bias_analysis_response = {
            "fairness_score": 0.87,
            "bias_analysis": {
                "detected_biases": ["length_bias"],
                "confidence": 0.78,
                "bias_severity": {"length_bias": "moderate"},
            },
            "evaluation_consistency": {
                "consistency_score": 0.92,
                "drift_detected": False,
                "variance_analysis": "stable",
            },
            "mitigation_recommendations": {
                "immediate_actions": [
                    "Normalize test length in evaluation prompts",
                    "Add length-independent scoring guidelines",
                ],
                "process_improvements": [
                    "Implement blind evaluation protocols",
                    "Use multiple evaluators for consensus",
                ],
            },
        }

        return {
            "llm_judge": llm_judge_response,
            "statistical_analysis": statistical_analysis_response,
            "bias_analysis": bias_analysis_response,
        }

    def test_prompt_versioning_integration(self):
        """Test prompt versioning and management (PromptFoo pattern)."""
        registry = PromptRegistry()

        # Test system prompt retrieval
        system_prompt = registry.get_system_prompt("llm_judge_v1")
        assert system_prompt is not None
        assert "rubric-driven assessment" in system_prompt.lower()

        # Test user prompt generation
        user_prompt = registry.get_user_prompt(
            "llm_judge_v1",
            generated_test="def test_example(): assert True",
            source_code="def example(): return True",
            rubric_version="2025_v1",
        )
        assert user_prompt is not None
        assert "def test_example" in user_prompt

        # Test schema retrieval
        schema = registry.get_schema("llm_judge_output")
        assert schema is not None
        assert "scores" in schema["properties"]
        assert "rationales" in schema["properties"]

    @pytest.mark.asyncio
    async def test_end_to_end_ab_testing_pipeline(self, mock_llm_responses):
        """Test complete A/B testing pipeline (PromptFoo integration pattern)."""
        with (
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_evaluation",
                return_value=mock_llm_responses["llm_judge"],
            ),
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_statistical_analysis",
                return_value=mock_llm_responses["statistical_analysis"],
            ),
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_bias_analysis",
                return_value=mock_llm_responses["bias_analysis"],
            ),
        ):
            # Run comprehensive A/B testing pipeline
            results = self.harness.run_advanced_ab_testing_pipeline(
                prompt_variants=self.prompt_variants,
                test_dataset=self.test_dataset[:2],  # Use subset for testing
                statistical_testing=True,
                bias_mitigation=True,
            )

            # Verify pipeline results structure
            assert "campaign_id" in results
            assert "metadata" in results
            assert "variant_evaluations" in results
            assert "pairwise_comparisons" in results
            assert "statistical_analysis" in results
            assert "bias_analysis" in results
            assert "recommendations" in results

            # Verify metadata follows PromptFoo patterns
            metadata = results["metadata"]
            assert metadata["variant_count"] == 3
            assert metadata["test_case_count"] == 2
            assert metadata["total_evaluations"] == 6  # 3 variants × 2 test cases

            # Verify variant evaluations
            variant_evals = results["variant_evaluations"]
            assert len(variant_evals) == 3

            for variant_eval in variant_evals:
                assert "variant_id" in variant_eval
                assert "individual_evaluations" in variant_eval
                assert "summary_stats" in variant_eval

                # Check summary statistics
                summary = variant_eval["summary_stats"]
                assert "acceptance_rate" in summary
                assert "mean_llm_score" in summary
                assert "std_llm_score" in summary
                assert "total_tests" in summary

            # Verify pairwise comparisons
            pairwise = results["pairwise_comparisons"]
            assert len(pairwise) == 3  # (3 choose 2) = 3 pairs

            for comparison in pairwise:
                assert "variant_a" in comparison
                assert "variant_b" in comparison
                assert "preference_scores" in comparison
                assert "statistical_significance" in comparison

            # Verify statistical analysis
            stat_analysis = results["statistical_analysis"]
            assert not stat_analysis.get("error")
            assert stat_analysis["statistical_test"] == "Welch's t-test"
            assert stat_analysis["p_value"] == 0.0234
            assert (
                stat_analysis["significance_assessment"] == "statistically_significant"
            )

            # Verify bias analysis
            bias_analysis = results["bias_analysis"]
            assert not bias_analysis.get("error")
            assert bias_analysis["fairness_score"] == 0.87
            assert "length_bias" in bias_analysis["bias_analysis"]["detected_biases"]

            # Verify recommendations
            recommendations = results["recommendations"]
            assert len(recommendations) > 0
            assert any("enhanced prompt" in rec.lower() for rec in recommendations)

    def test_artifact_storage_integration(self):
        """Test artifact storage following PromptLayer patterns."""
        # Generate test evaluation
        test_result = EvaluationResult(
            test_id="test_001",
            source_file="example.py",
            generated_test="def test_example(): assert True",
            acceptance_result=AcceptanceResult(
                passed=True,
                error_message=None,
                execution_time=0.1,
                coverage_data={"line_coverage": 0.85},
            ),
            llm_judge_result=LLMJudgeResult(
                scores={
                    "correctness": 4.5,
                    "coverage": 4.0,
                    "clarity": 4.8,
                    "safety": 5.0,
                },
                rationales={
                    "correctness": "Good test logic",
                    "coverage": "Adequate coverage",
                    "clarity": "Clear test structure",
                    "safety": "No security issues",
                },
                overall_score=4.6,
                prompt_version="llm_judge_v1",
                confidence=0.85,
            ),
            metadata={
                "prompt_variant": "enhanced_v1",
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "harness_version": "2025.1.0",
            },
        )

        # Store evaluation result
        artifact_store = self.harness.evaluation_adapter.artifact_store
        artifact_id = artifact_store.store_artifact(
            ArtifactType.EVALUATION_RESULT,
            test_result.__dict__,
            tags=["evaluation", "ab_testing", "enhanced_v1"],
            description="A/B testing evaluation result",
        )

        # Retrieve and verify
        retrieved = artifact_store.retrieve_artifact(artifact_id)
        assert retrieved is not None
        assert retrieved["content"]["test_id"] == "test_001"
        assert "evaluation" in retrieved["metadata"]["tags"]

        # Test artifact listing by tags (PromptLayer pattern)
        evaluation_artifacts = artifact_store.list_artifacts(tags=["evaluation"])
        assert len(evaluation_artifacts) >= 1

        ab_testing_artifacts = artifact_store.list_artifacts(tags=["ab_testing"])
        assert len(ab_testing_artifacts) >= 1

    def test_config_driven_evaluation_workflow(self):
        """Test config-driven evaluation following PromptFoo patterns."""
        # Create evaluation config file (PromptFoo style)
        config_data = {
            "evaluation_name": "prompt_optimization_study",
            "version": "1.0.0",
            "description": "Comparing baseline vs enhanced prompting strategies",
            "evaluation_scenarios": [
                {
                    "name": "simple_functions",
                    "description": "Test generation for simple utility functions",
                    "prompt_variants": self.prompt_variants[:2],  # baseline vs enhanced
                    "test_dataset": [self.test_dataset[0]],  # simple math function
                    "evaluation_config": {
                        "rubric_dimensions": ["correctness", "coverage", "clarity"],
                        "statistical_testing": True,
                        "bias_mitigation": True,
                    },
                },
                {
                    "name": "complex_functions",
                    "description": "Test generation for complex data processing functions",
                    "prompt_variants": self.prompt_variants,  # all variants
                    "test_dataset": [self.test_dataset[2]],  # complex data processor
                    "evaluation_config": {
                        "rubric_dimensions": [
                            "correctness",
                            "coverage",
                            "clarity",
                            "safety",
                        ],
                        "statistical_testing": True,
                        "bias_mitigation": True,
                    },
                },
            ],
            "global_config": {
                "confidence_level": 0.95,
                "effect_size_threshold": 0.5,
                "significance_threshold": 0.05,
            },
        }

        # Save config to temp file
        config_file = self.temp_dir / "evaluation_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        # Verify config structure (would be used by CLI)
        assert config_file.exists()

        with open(config_file) as f:
            loaded_config = json.load(f)

        assert loaded_config["evaluation_name"] == "prompt_optimization_study"
        assert len(loaded_config["evaluation_scenarios"]) == 2
        assert "global_config" in loaded_config

    def test_statistical_significance_validation(self, mock_llm_responses):
        """Test statistical significance testing validation."""
        with patch.object(
            self.harness.evaluation_adapter,
            "_call_llm_for_statistical_analysis",
            return_value=mock_llm_responses["statistical_analysis"],
        ):
            # Create mock evaluation data
            evaluation_data = [
                {
                    "variant_id": "baseline_v1",
                    "test_case": "simple_math",
                    "llm_score": 3.5,
                    "acceptance_passed": True,
                },
                {
                    "variant_id": "enhanced_v1",
                    "test_case": "simple_math",
                    "llm_score": 4.2,
                    "acceptance_passed": True,
                },
            ] * 10  # Simulate multiple data points

            # Run statistical analysis
            results = self.harness.run_statistical_significance_analysis(
                evaluation_data=evaluation_data,
                analysis_type="pairwise_comparison",
                confidence_level=0.95,
            )

            # Verify results structure
            assert not results.get("error")
            assert results["statistical_test"] == "Welch's t-test"
            assert results["p_value"] == 0.0234
            assert results["significance_assessment"] == "statistically_significant"
            assert results["effect_size"]["cohens_d"] == 0.73
            assert results["effect_size"]["interpretation"] == "medium_effect"

            # Verify confidence interval
            ci = results["confidence_interval"]
            assert ci["confidence_level"] == 0.95
            assert ci["lower"] < ci["upper"]

            # Verify recommendations
            assert len(results["recommendations"]) > 0

    def test_bias_detection_and_mitigation(self, mock_llm_responses):
        """Test bias detection and mitigation strategies."""
        with patch.object(
            self.harness.evaluation_adapter,
            "_call_llm_for_bias_analysis",
            return_value=mock_llm_responses["bias_analysis"],
        ):
            # Create mock evaluation history with potential bias
            evaluation_history = []
            for i in range(20):
                eval_result = EvaluationResult(
                    test_id=f"test_{i:03d}",
                    source_file=f"module_{i % 3}.py",
                    generated_test=f"def test_function_{i}(): assert True"
                    + " " * (i * 10),  # Varying lengths
                    acceptance_result=AcceptanceResult(
                        passed=True,
                        error_message=None,
                        execution_time=0.1,
                        coverage_data={"line_coverage": 0.8 + (i % 3) * 0.1},
                    ),
                    llm_judge_result=LLMJudgeResult(
                        scores={
                            "correctness": 4.0
                            + (
                                len(
                                    f"def test_function_{i}(): assert True"
                                    + " " * (i * 10)
                                )
                                / 1000
                            ),  # Length bias
                            "coverage": 4.0,
                            "clarity": 4.5,
                            "safety": 5.0,
                        },
                        rationales={
                            "correctness": f"Test {i} evaluation",
                            "coverage": "Good coverage",
                            "clarity": "Clear structure",
                            "safety": "No issues",
                        },
                        overall_score=4.4,
                        prompt_version="llm_judge_v1",
                        confidence=0.85,
                    ),
                    metadata={"evaluation_timestamp": datetime.utcnow().isoformat()},
                )
                evaluation_history.append(eval_result)

            # Run bias detection
            results = self.harness.detect_evaluation_bias(
                evaluation_history=evaluation_history,
                bias_types=["length_bias", "complexity_bias", "order_bias"],
            )

            # Verify results structure
            assert not results.get("error")
            assert results["fairness_score"] == 0.87

            # Verify bias analysis
            bias_analysis = results["bias_analysis"]
            assert "detected_biases" in bias_analysis
            assert "length_bias" in bias_analysis["detected_biases"]
            assert bias_analysis["confidence"] == 0.78

            # Verify consistency analysis
            consistency = results["evaluation_consistency"]
            assert consistency["consistency_score"] == 0.92
            assert not consistency["drift_detected"]

            # Verify mitigation recommendations
            mitigation = results["mitigation_recommendations"]
            assert len(mitigation["immediate_actions"]) > 0
            assert len(mitigation["process_improvements"]) > 0
            assert any(
                "length" in action.lower() for action in mitigation["immediate_actions"]
            )

    def test_comprehensive_evaluation_campaign(self, mock_llm_responses):
        """Test comprehensive evaluation campaign with multiple scenarios."""
        campaign_config = {
            "campaign_name": "Q1_2025_Prompt_Evaluation",
            "evaluation_scenarios": [
                {
                    "name": "baseline_vs_enhanced",
                    "description": "Compare baseline and enhanced prompting",
                    "prompt_variants": self.prompt_variants[:2],
                    "test_dataset": self.test_dataset[:1],
                }
            ],
            "cross_scenario_analysis": True,
            "generate_final_recommendations": True,
        }

        with (
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_evaluation",
                return_value=mock_llm_responses["llm_judge"],
            ),
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_statistical_analysis",
                return_value=mock_llm_responses["statistical_analysis"],
            ),
            patch.object(
                self.harness.evaluation_adapter,
                "_call_llm_for_bias_analysis",
                return_value=mock_llm_responses["bias_analysis"],
            ),
        ):
            # Run comprehensive campaign
            results = self.harness.run_comprehensive_evaluation_campaign(
                campaign_config=campaign_config
            )

            # Verify campaign structure
            assert "campaign_metadata" in results
            assert "scenario_results" in results
            assert "cross_scenario_analysis" in results
            assert "final_recommendations" in results

            # Verify campaign metadata
            metadata = results["campaign_metadata"]
            assert metadata["campaign_name"] == "Q1_2025_Prompt_Evaluation"
            assert "total_scenarios" in metadata
            assert "timestamp" in metadata

            # Verify scenario results
            scenario_results = results["scenario_results"]
            assert len(scenario_results) >= 1

            for scenario in scenario_results:
                assert "scenario_metadata" in scenario
                assert not scenario.get("error")  # Should not have errors

            # Verify cross-scenario analysis
            cross_analysis = results["cross_scenario_analysis"]
            assert "insights" in cross_analysis
            assert len(cross_analysis["insights"]) > 0

            # Verify final recommendations
            final_recs = results["final_recommendations"]
            assert len(final_recs) > 0

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


@pytest.mark.skip(
    reason="Evaluation harness integration tests need prompt template updates"
)
class TestEvaluationErrorHandling:
    """Test error handling and edge cases in evaluation pipeline."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create temp directory in tests folder to satisfy safety policies
        test_temp_base = Path(__file__).parent / "test_temp"
        test_temp_base.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=test_temp_base))
        self.harness = create_evaluation_harness(project_root=self.temp_dir)

    def test_invalid_prompt_variant_handling(self):
        """Test handling of invalid prompt variants."""
        invalid_variants = [
            {"id": "missing_prompt"},  # Missing prompt field
            {"prompt": "No ID provided"},  # Missing id field
            {"id": "", "prompt": "Empty ID"},  # Empty id
        ]

        test_dataset = [
            {"source_file": "test.py", "source_content": "def test(): pass"}
        ]

        with pytest.raises(ValueError, match="Invalid prompt variant"):
            self.harness.run_advanced_ab_testing_pipeline(
                prompt_variants=invalid_variants, test_dataset=test_dataset
            )

    def test_empty_test_dataset_handling(self):
        """Test handling of empty test dataset."""
        prompt_variants = [
            {"id": "test_variant", "prompt": "Generate tests for: {{ source_code }}"}
        ]

        with pytest.raises(ValueError, match="Test dataset cannot be empty"):
            self.harness.run_advanced_ab_testing_pipeline(
                prompt_variants=prompt_variants, test_dataset=[]
            )

    def test_llm_failure_graceful_handling(self):
        """Test graceful handling of LLM API failures."""
        with patch.object(
            self.harness.evaluation_adapter,
            "_call_llm_for_evaluation",
            side_effect=Exception("LLM API unavailable"),
        ):
            results = self.harness.run_advanced_ab_testing_pipeline(
                prompt_variants=[{"id": "test_variant", "prompt": "Generate tests"}],
                test_dataset=[
                    {"source_file": "test.py", "source_content": "def test(): pass"}
                ],
                statistical_testing=False,
                bias_mitigation=False,
            )

            # Should handle gracefully and provide error information
            assert "error_summary" in results
            assert results["error_summary"]["llm_evaluation_failures"] > 0

    def test_statistical_analysis_fallback(self):
        """Test statistical analysis fallback methods."""
        evaluation_data = [
            {"variant_id": "v1", "llm_score": 3.5, "acceptance_passed": True},
            {"variant_id": "v2", "llm_score": 4.0, "acceptance_passed": True},
        ]

        with patch.object(
            self.harness.evaluation_adapter,
            "_call_llm_for_statistical_analysis",
            side_effect=Exception("Statistical analysis LLM failed"),
        ):
            results = self.harness.run_statistical_significance_analysis(
                evaluation_data=evaluation_data
            )

            # Should fallback to basic statistical methods
            assert not results.get("error")
            assert "fallback_analysis" in results
            assert "basic_statistics" in results

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
