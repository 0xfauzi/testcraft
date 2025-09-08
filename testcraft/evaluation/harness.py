"""
Testcraft Evaluation Harness - Main entry point for test evaluation.

This module provides a convenient interface for evaluating generated tests
using automated acceptance checks, LLM-as-judge evaluation, and A/B testing
capabilities. It integrates all the necessary adapters following clean
architecture principles.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from testcraft.adapters.coverage.main_adapter import TestcraftCoverageAdapter
from testcraft.adapters.evaluation import TestcraftEvaluationAdapter
from testcraft.adapters.evaluation.main_adapter import EvaluationError
from testcraft.adapters.io.artifact_store import ArtifactStoreAdapter, ArtifactType
from testcraft.adapters.io.state_json import StateJsonAdapter
from testcraft.adapters.llm.router import LLMRouter
from testcraft.config import TestCraftConfig
from testcraft.ports.evaluation_port import EvaluationConfig, EvaluationResult
from testcraft.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class TestEvaluationHarness:
    """
    High-level interface for the testcraft evaluation system.

    This harness coordinates all the necessary adapters and provides
    convenient methods for different evaluation scenarios. It handles
    dependency injection and configuration management.
    """

    def __init__(
        self,
        config: TestCraftConfig | None = None,
        project_root: Path | None = None,
        **adapter_kwargs: Any,
    ):
        """
        Initialize the evaluation harness with required dependencies.

        Args:
            config: Testcraft configuration object
            project_root: Project root directory
            **adapter_kwargs: Additional arguments for adapter initialization
        """
        self.project_root = project_root or Path.cwd()
        self.config = config

        # Initialize core adapters
        self._initialize_adapters(**adapter_kwargs)

        # Initialize main evaluation adapter
        self.evaluator = TestcraftEvaluationAdapter(
            coverage_adapter=self.coverage_adapter,
            llm_adapter=self.llm_adapter,
            state_adapter=self.state_adapter,
            artifact_store=self.artifact_store,
            prompt_registry=self.prompt_registry,
            project_root=self.project_root,
        )

        logger.info(
            f"TestEvaluationHarness initialized for project: {self.project_root}"
        )

    def _initialize_adapters(self, **kwargs: Any) -> None:
        """Initialize all required adapters with dependency injection."""
        # Coverage adapter
        self.coverage_adapter = TestcraftCoverageAdapter()

        # State management adapter
        self.state_adapter = StateJsonAdapter(
            project_root=self.project_root,
            state_file=".testcraft_evaluation_state.json",
        )

        # Artifact storage
        artifacts_path = self.project_root / ".testcraft" / "evaluation_artifacts"
        self.artifact_store = ArtifactStoreAdapter(base_path=artifacts_path)

        # LLM adapter (using router for flexibility)
        if self.config and hasattr(self.config, "llm"):
            self.llm_adapter = LLMRouter.from_config(self.config.llm)
        else:
            # Default to Claude adapter
            from testcraft.adapters.llm.claude import ClaudeAdapter

            self.llm_adapter = ClaudeAdapter()

        # Prompt registry
        self.prompt_registry = PromptRegistry()

    def evaluate_single_test(
        self,
        test_content: str,
        source_file: str,
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single generated test comprehensively.

        Args:
            test_content: Generated test code to evaluate
            source_file: Source file the test was generated for
            config: Evaluation configuration
            **kwargs: Additional evaluation parameters

        Returns:
            Complete evaluation result with acceptance checks and LLM scores
        """
        logger.info(f"Evaluating single test for {source_file}")

        eval_config = config or EvaluationConfig()

        return self.evaluator.evaluate_single(
            test_content=test_content,
            source_file=source_file,
            config=eval_config,
            **kwargs,
        )

    def compare_test_variants(
        self,
        test_a: str,
        test_b: str,
        source_file: str,
        comparison_mode: str = "a_vs_b",
    ) -> dict[str, Any]:
        """
        Compare two test variants using pairwise evaluation.

        Args:
            test_a: First test variant
            test_b: Second test variant
            source_file: Source file for context
            comparison_mode: Type of comparison ('a_vs_b', 'best_of_n', 'ranking')

        Returns:
            Comparison result with winner, confidence, and reasoning
        """
        logger.info(f"Comparing test variants for {source_file}")

        return self.evaluator.evaluate_pairwise(
            test_a=test_a,
            test_b=test_b,
            source_file=source_file,
            comparison_mode=comparison_mode,
        )

    def evaluate_test_batch(
        self,
        test_variants: list[dict[str, str]],
        source_files: list[str],
        config: EvaluationConfig | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple test variants efficiently in batch mode.

        Args:
            test_variants: List of test variant dictionaries
            source_files: Corresponding source files for each test
            config: Evaluation configuration

        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating {len(test_variants)} test variants in batch")

        eval_config = config or EvaluationConfig()

        return self.evaluator.evaluate_batch(
            test_variants=test_variants, source_files=source_files, config=eval_config
        )

    def run_golden_repository_evaluation(
        self,
        golden_repo_path: Path,
        test_generator: Callable[[str], str],
        config: EvaluationConfig | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive evaluation against a golden repository.

        Args:
            golden_repo_path: Path to golden repository
            test_generator: Function that generates tests for source files
            config: Evaluation configuration

        Returns:
            Golden repository evaluation results with regression detection
        """
        logger.info(f"Running golden repository evaluation: {golden_repo_path}")

        eval_config = config or EvaluationConfig(
            golden_repos_path=golden_repo_path,
            acceptance_checks=True,
            llm_judge_enabled=True,
        )

        return self.evaluator.run_golden_repo_evaluation(
            golden_repo_path=golden_repo_path,
            test_generator_func=test_generator,
            evaluation_config=eval_config,
        )

    def analyze_evaluation_history(
        self, time_window_days: int | None = 30
    ) -> dict[str, Any]:
        """
        Analyze historical evaluation results to identify trends.

        Args:
            time_window_days: Time window for trend analysis

        Returns:
            Trend analysis results with recommendations
        """
        logger.info("Analyzing evaluation history trends")

        # Get evaluation history from state
        evaluation_history_data = self.state_adapter.get_all_state("evaluation_history")

        # Convert to EvaluationResult objects
        evaluation_history = []
        for eval_data in evaluation_history_data.values():
            if isinstance(eval_data, dict) and "test_id" in eval_data:
                # This is a simplified conversion - you'd implement proper deserialization
                pass

        if not evaluation_history:
            logger.warning("No evaluation history found")
            return {"error": "No evaluation history available"}

        return self.evaluator.analyze_evaluation_trends(
            evaluation_history=evaluation_history, time_window_days=time_window_days
        )

    def get_evaluation_statistics(self) -> dict[str, Any]:
        """
        Get current evaluation statistics and storage information.

        Returns:
            Dictionary with evaluation statistics and recommendations
        """
        logger.info("Gathering evaluation statistics")

        stats = {
            "artifact_storage": self.artifact_store.get_storage_stats(),
            "state_summary": self.state_adapter.get_all_state(),
            "harness_config": {
                "project_root": str(self.project_root),
                "coverage_available": self.coverage_adapter.is_pytest_available(),
                "llm_adapter_type": type(self.llm_adapter).__name__,
            },
        }

        return stats

    def cleanup_artifacts(self, apply_policy: bool = True) -> dict[str, int]:
        """
        Clean up old evaluation artifacts based on configured policies.

        Args:
            apply_policy: Whether to apply the full cleanup policy

        Returns:
            Cleanup statistics
        """
        logger.info("Cleaning up evaluation artifacts")

        if apply_policy:
            return self.artifact_store.apply_cleanup_policy()
        else:
            return self.artifact_store.cleanup_expired()

    # =============================
    # NEW METHODS FOR SUBTASK 20.2: Advanced A/B Testing & Statistical Analysis
    # =============================

    def run_advanced_ab_testing_pipeline(
        self,
        prompt_variants: list[dict[str, str]],
        test_dataset: list[dict[str, str]],
        statistical_testing: bool = True,
        bias_mitigation: bool = True,
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run comprehensive A/B testing pipeline with statistical significance testing
        and bias mitigation following 2025 best practices.

        This is the main entry point for advanced evaluation campaigns that compare
        multiple prompt variants systematically.

        Args:
            prompt_variants: List of prompt variant dictionaries with 'id' and 'prompt' keys
            test_dataset: List of test case dictionaries with 'source_file' and 'source_content' keys
            statistical_testing: Whether to run statistical significance analysis
            bias_mitigation: Whether to run bias detection analysis
            config: Optional evaluation configuration
            **kwargs: Additional parameters

        Returns:
            Comprehensive A/B testing results with recommendations
        """
        logger.info(
            f"Running advanced A/B testing pipeline via harness: {len(prompt_variants)} variants"
        )

        config or EvaluationConfig(
            acceptance_checks=True,
            llm_judge_enabled=True,
            statistical_testing=statistical_testing,
        )

        return self.evaluator.run_advanced_ab_testing_pipeline(
            prompt_variants=prompt_variants,
            test_dataset=test_dataset,
            statistical_testing=statistical_testing,
            bias_mitigation=bias_mitigation,
            **kwargs,
        )

    def run_statistical_significance_analysis(
        self,
        evaluation_data: list[dict[str, Any]],
        analysis_type: str = "pairwise_comparison",
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run statistical significance analysis on evaluation data.

        Args:
            evaluation_data: List of evaluation data dictionaries
            analysis_type: Type of analysis ("pairwise_comparison", "ab_testing", etc.)
            confidence_level: Statistical confidence level (0.8-0.99)
            **kwargs: Additional analysis parameters

        Returns:
            Statistical analysis results with significance assessment
        """
        logger.info(f"Running statistical analysis via harness: {analysis_type}")

        return self.evaluator.run_statistical_significance_analysis(
            evaluation_data=evaluation_data,
            analysis_type=analysis_type,
            confidence_level=confidence_level,
            **kwargs,
        )

    def detect_evaluation_bias(
        self,
        evaluation_history: list[EvaluationResult] | None = None,
        bias_types: list[str] | None = None,
        time_window_days: int | None = 30,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Detect and analyze evaluation bias patterns.

        Args:
            evaluation_history: Optional list of evaluation results to analyze
            bias_types: Optional list of specific bias types to check
            time_window_days: Time window for historical analysis
            **kwargs: Additional bias detection parameters

        Returns:
            Bias analysis results with mitigation recommendations
        """
        logger.info("Running bias detection analysis via harness")

        # If no evaluation history provided, get recent evaluations from state
        if evaluation_history is None:
            evaluation_history = self._get_recent_evaluation_history(time_window_days)

        if not evaluation_history:
            logger.warning("No evaluation history available for bias analysis")
            return {"error": "No evaluation history available"}

        return self.evaluator.detect_evaluation_bias(
            evaluation_history=evaluation_history, bias_types=bias_types, **kwargs
        )

    def run_comprehensive_evaluation_campaign(
        self, campaign_config: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Run a comprehensive evaluation campaign with multiple test scenarios.

        This orchestrates multiple evaluation runs with different configurations,
        statistical analysis, bias detection, and generates actionable recommendations.

        Args:
            campaign_config: Configuration dictionary containing:
                - prompt_variants: List of prompt variants to test
                - test_datasets: List of test datasets to use
                - evaluation_scenarios: List of evaluation scenarios
                - analysis_options: Statistical and bias analysis options
            **kwargs: Additional campaign parameters

        Returns:
            Comprehensive campaign results with cross-scenario analysis
        """
        logger.info("Running comprehensive evaluation campaign")

        try:
            campaign_results = {
                "campaign_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "config": campaign_config,
                    "harness_version": "advanced_v1",
                },
                "scenario_results": [],
                "cross_scenario_analysis": None,
                "final_recommendations": [],
            }

            # Run evaluation scenarios
            scenarios = campaign_config.get("evaluation_scenarios", [])
            for i, scenario in enumerate(scenarios):
                logger.info(
                    f"Running evaluation scenario {i + 1}/{len(scenarios)}: {scenario.get('name', f'scenario_{i}')}"
                )

                scenario_result = self.run_advanced_ab_testing_pipeline(
                    prompt_variants=scenario.get("prompt_variants", []),
                    test_dataset=scenario.get("test_dataset", []),
                    statistical_testing=scenario.get("statistical_testing", True),
                    bias_mitigation=scenario.get("bias_mitigation", True),
                    **scenario.get("additional_params", {}),
                )

                scenario_result["scenario_metadata"] = {
                    "name": scenario.get("name", f"scenario_{i}"),
                    "description": scenario.get("description", ""),
                    "scenario_index": i,
                }

                campaign_results["scenario_results"].append(scenario_result)

            # Cross-scenario analysis
            if len(campaign_results["scenario_results"]) > 1:
                campaign_results["cross_scenario_analysis"] = (
                    self._analyze_cross_scenario_results(
                        campaign_results["scenario_results"]
                    )
                )

            # Generate final recommendations
            campaign_results["final_recommendations"] = (
                self._generate_campaign_recommendations(campaign_results)
            )

            # Store campaign results
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                campaign_results,
                tags=["campaign", "comprehensive", "evaluation"],
                description=f"Comprehensive evaluation campaign: {len(scenarios)} scenarios",
            )

            logger.info("Comprehensive evaluation campaign completed successfully")
            return campaign_results

        except Exception as e:
            logger.error(f"Comprehensive evaluation campaign failed: {e}")
            raise EvaluationError(f"Failed to run evaluation campaign: {e}") from e

    # Private helper methods for new functionality

    def _get_recent_evaluation_history(self, days: int) -> list[EvaluationResult]:
        """Get recent evaluation history from state adapter."""
        try:
            # This is a simplified implementation - you might want to implement proper querying
            all_state = self.state_adapter.get_all_state()
            evaluation_results = []

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            for _key, value in all_state.items():
                if (
                    isinstance(value, dict)
                    and "test_id" in value
                    and "timestamp" in value
                ):
                    try:
                        eval_timestamp = datetime.fromisoformat(value["timestamp"])
                        if eval_timestamp >= cutoff_date:
                            # Convert dict back to EvaluationResult
                            evaluation_results.append(
                                self.evaluator._dict_to_evaluation_result(value)
                            )
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid evaluation record: {e}")
                        continue

            logger.info(
                f"Retrieved {len(evaluation_results)} evaluations from the last {days} days"
            )
            return evaluation_results

        except Exception as e:
            logger.warning(f"Failed to retrieve evaluation history: {e}")
            return []

    def _analyze_cross_scenario_results(
        self, scenario_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze results across multiple evaluation scenarios."""
        try:
            cross_analysis = {
                "scenario_count": len(scenario_results),
                "consistency_analysis": {},
                "performance_comparison": {},
                "bias_patterns": {},
                "statistical_reliability": {},
            }

            # Analyze consistency across scenarios
            variant_performances = {}
            for scenario in scenario_results:
                for variant_eval in scenario.get("variant_evaluations", []):
                    variant_id = variant_eval.get("variant_id")
                    stats = variant_eval.get("summary_stats", {})

                    if variant_id not in variant_performances:
                        variant_performances[variant_id] = []
                    variant_performances[variant_id].append(
                        stats.get("mean_llm_score", 0)
                    )

            # Calculate consistency metrics
            for variant_id, scores in variant_performances.items():
                if len(scores) > 1:
                    mean_score = sum(scores) / len(scores)
                    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                    cross_analysis["consistency_analysis"][variant_id] = {
                        "mean_performance": mean_score,
                        "performance_variance": variance,
                        "consistency_rating": (
                            "high"
                            if variance < 0.5
                            else "medium"
                            if variance < 1.0
                            else "low"
                        ),
                    }

            # Overall cross-scenario insights
            cross_analysis["insights"] = self._generate_cross_scenario_insights(
                cross_analysis
            )

            return cross_analysis

        except Exception as e:
            logger.warning(f"Cross-scenario analysis failed: {e}")
            return {"error": f"Cross-scenario analysis failed: {e}"}

    def _generate_cross_scenario_insights(
        self, cross_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate insights from cross-scenario analysis."""
        insights = []

        try:
            consistency_data = cross_analysis.get("consistency_analysis", {})

            # Find most consistent variant
            most_consistent = min(
                consistency_data.items(),
                key=lambda x: x[1].get("performance_variance", float("inf")),
                default=(None, None),
            )

            if most_consistent[0]:
                insights.append(
                    f"Most consistent variant across scenarios: {most_consistent[0]}"
                )

            # Find highest performing variant
            best_performer = max(
                consistency_data.items(),
                key=lambda x: x[1].get("mean_performance", 0),
                default=(None, None),
            )

            if best_performer[0]:
                insights.append(
                    f"Highest average performance: {best_performer[0]} (score: {best_performer[1].get('mean_performance', 0):.2f})"
                )

            # General insights
            high_consistency_count = sum(
                1
                for data in consistency_data.values()
                if data.get("consistency_rating") == "high"
            )

            if high_consistency_count > len(consistency_data) * 0.5:
                insights.append(
                    "Good overall consistency detected across variants and scenarios"
                )
            else:
                insights.append(
                    "Variable performance detected - consider scenario-specific optimization"
                )

        except Exception as e:
            logger.warning(f"Failed to generate cross-scenario insights: {e}")
            insights.append("Cross-scenario analysis completed with limited insights")

        return insights

    def _generate_campaign_recommendations(
        self, campaign_results: dict[str, Any]
    ) -> list[str]:
        """Generate final recommendations for the evaluation campaign."""
        recommendations = []

        try:
            scenario_results = campaign_results.get("scenario_results", [])
            cross_analysis = campaign_results.get("cross_scenario_analysis", {})

            # Analyze overall campaign success
            successful_scenarios = sum(
                1 for scenario in scenario_results if not scenario.get("error")
            )

            if successful_scenarios == len(scenario_results):
                recommendations.append(
                    "All evaluation scenarios completed successfully"
                )
            else:
                recommendations.append(
                    f"{successful_scenarios}/{len(scenario_results)} scenarios completed successfully - review failed scenarios"
                )

            # Cross-scenario recommendations
            if cross_analysis.get("insights"):
                recommendations.extend(cross_analysis["insights"])

            # Statistical reliability assessment
            significant_results = sum(
                1
                for scenario in scenario_results
                if scenario.get("statistical_analysis", {}).get(
                    "significance_assessment"
                )
                in ["significant", "highly_significant"]
            )

            if significant_results > 0:
                recommendations.append(
                    f"{significant_results} scenarios show statistically significant results - findings are reliable"
                )
            else:
                recommendations.append(
                    "No statistically significant differences detected - consider larger sample sizes or different evaluation criteria"
                )

            # Bias assessment
            bias_issues = sum(
                1
                for scenario in scenario_results
                if scenario.get("bias_analysis", {}).get("fairness_score", 1.0) < 0.7
            )

            if bias_issues > 0:
                recommendations.append(
                    f"Potential bias detected in {bias_issues} scenarios - review evaluation methodology"
                )

            # General recommendations
            recommendations.append(
                "Review detailed scenario results for specific optimization opportunities"
            )

        except Exception as e:
            logger.warning(f"Failed to generate campaign recommendations: {e}")
            recommendations.append(
                "Campaign completed - review results manually due to recommendation generation error"
            )

        return recommendations


# Convenience factory functions


def create_evaluation_harness(
    project_root: Path | None = None, config_file: str | None = None
) -> TestEvaluationHarness:
    """
    Create a test evaluation harness with default configuration.

    Args:
        project_root: Project root directory
        config_file: Optional path to testcraft config file

    Returns:
        Initialized TestEvaluationHarness
    """
    project_root = project_root or Path.cwd()

    # Load configuration if provided
    config = None
    if config_file:
        from testcraft.config.loader import load_config

        config = load_config(project_root / config_file)

    return TestEvaluationHarness(config=config, project_root=project_root)


def quick_evaluate(
    test_content: str, source_file: str, project_root: Path | None = None
) -> EvaluationResult:
    """
    Quick evaluation of a single test with default settings.

    Args:
        test_content: Test code to evaluate
        source_file: Source file path
        project_root: Optional project root

    Returns:
        Evaluation result
    """
    harness = create_evaluation_harness(project_root)
    return harness.evaluate_single_test(test_content, source_file)


def quick_compare(
    test_a: str, test_b: str, source_file: str, project_root: Path | None = None
) -> dict[str, Any]:
    """
    Quick comparison of two test variants.

    Args:
        test_a: First test variant
        test_b: Second test variant
        source_file: Source file path
        project_root: Optional project root

    Returns:
        Comparison result
    """
    harness = create_evaluation_harness(project_root)
    return harness.compare_test_variants(test_a, test_b, source_file)


if __name__ == "__main__":
    # Example usage
    harness = create_evaluation_harness()
    print("Testcraft Evaluation Harness initialized successfully!")

    stats = harness.get_evaluation_statistics()
    print(f"Artifact storage: {stats['artifact_storage']['total_artifacts']} artifacts")
    print(f"Coverage available: {stats['harness_config']['coverage_available']}")
