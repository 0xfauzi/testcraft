"""
Main evaluation adapter implementing the EvaluationPort interface.

This module provides the primary evaluation adapter that coordinates
automated acceptance checks, LLM-as-judge evaluation, and A/B testing
capabilities following clean architecture principles.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...ports.coverage_port import CoveragePort
from ...ports.evaluation_port import (AcceptanceResult, ComparisonMode,
                                      EvaluationConfig, EvaluationPort,
                                      EvaluationResult, LLMJudgeResult)
from ...ports.llm_port import LLMPort
from ...ports.state_port import StatePort
from ...prompts.registry import PromptRegistry
from ..io.artifact_store import ArtifactStoreAdapter
from .ab_testing import ABTestingManager
from .acceptance import AcceptanceChecker
from .analytics import EvaluationAnalytics
from .artifacts import ArtifactManager
from .context import ContextBuilder
from .llm_judge import LLMJudge
from .pairwise import PairwiseComparator
from .parsers import ResponseParser
from .safety import SafetyValidator
from .statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised when evaluation operations fail."""

    pass


class TestcraftEvaluationAdapter(EvaluationPort):
    """
    Main evaluation adapter implementing comprehensive test evaluation.

    This adapter orchestrates automated acceptance checks, LLM-as-judge
    evaluation, and A/B testing capabilities. It follows clean architecture
    principles by depending on injected ports rather than concrete adapters.
    """

    def __init__(
        self,
        coverage_adapter: CoveragePort,
        llm_adapter: LLMPort,
        state_adapter: StatePort,
        artifact_store: ArtifactStoreAdapter | None = None,
        prompt_registry: PromptRegistry | None = None,
        project_root: Path | None = None,
        safety_enabled: bool = True,
    ):
        """
        Initialize the evaluation adapter with required dependencies.

        Args:
            coverage_adapter: Coverage measurement adapter
            llm_adapter: LLM adapter for judge evaluations
            state_adapter: State management adapter
            artifact_store: Optional artifact storage adapter
            prompt_registry: Optional prompt registry for evaluation prompts
            project_root: Project root for safety validation
            safety_enabled: Whether to enforce safety policies
        """
        self.coverage_adapter = coverage_adapter
        self.llm_adapter = llm_adapter
        self.state_adapter = state_adapter
        self.artifact_store = artifact_store or ArtifactStoreAdapter()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.project_root = project_root or Path.cwd()
        self.safety_enabled = safety_enabled

        # Initialize evaluation state tracking
        self._evaluation_cache: dict[str, EvaluationResult] = {}

        # Initialize component modules
        self._acceptance_checker = AcceptanceChecker(
            coverage_adapter, self.project_root, safety_enabled
        )
        self._llm_judge = LLMJudge(llm_adapter, self.prompt_registry)
        self._pairwise_comparator = PairwiseComparator(
            llm_adapter, self.artifact_store, self.prompt_registry
        )
        self._artifact_manager = ArtifactManager(self.artifact_store)
        self._safety_validator = SafetyValidator(self.project_root, safety_enabled)

        logger.info("TestcraftEvaluationAdapter initialized successfully")

    def run_acceptance_checks(
        self,
        test_content: str,
        source_file: str,
        baseline_coverage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """
        Run comprehensive automated acceptance checks on test content.

        This performs syntactic validation, import checking, pytest execution,
        and optional coverage improvement measurement.
        """
        logger.debug(f"Running acceptance checks for {source_file}")

        try:
            result = self._acceptance_checker.run_acceptance_checks(
                test_content, source_file, baseline_coverage, **kwargs
            )
            logger.debug(f"Acceptance checks completed: {result.all_checks_pass}")
            return result

        except Exception as e:
            logger.error(f"Acceptance checks failed: {e}")
            raise EvaluationError(f"Failed to run acceptance checks: {e}") from e

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
            result = self._llm_judge.evaluate_with_llm_judge(
                test_content, source_file, rubric_dimensions, prompt_version, **kwargs
            )
            logger.debug(
                f"LLM judge evaluation completed with overall score: {result.overall_score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return LLMJudgeResult.empty()

    def evaluate_single(
        self,
        test_content: str,
        source_file: str,
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Perform complete evaluation of a single test with both automated and LLM checks.
        """
        logger.debug(f"Running single evaluation for {source_file}")

        try:
            config = config or EvaluationConfig()
            test_id = kwargs.get(
                "test_id", f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            # Run acceptance checks
            acceptance_result = self.run_acceptance_checks(
                test_content, source_file, **kwargs
            )

            # Run LLM judge evaluation if enabled and acceptance checks pass
            llm_judge_result = None
            if config.llm_judge_enabled and acceptance_result.all_checks_pass:
                llm_judge_result = self.evaluate_with_llm_judge(
                    test_content, source_file, config.rubric_dimensions, **kwargs
                )

            # Create complete evaluation result
            result = EvaluationResult(
                test_id=test_id,
                source_file=source_file,
                test_content=test_content,
                acceptance=acceptance_result,
                llm_judge=llm_judge_result,
                metadata=kwargs.get("metadata", {}),
            )

            # Store result in cache and artifacts
            self._evaluation_cache[test_id] = result
            self._artifact_manager.store_evaluation_artifact(result)

            logger.info(f"Single evaluation completed for {source_file}")
            return result

        except Exception as e:
            logger.error(f"Single evaluation failed: {e}")
            raise EvaluationError(f"Failed to evaluate test: {e}") from e

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
            result = self._pairwise_comparator.evaluate_pairwise(
                test_a, test_b, source_file, comparison_mode, **kwargs
            )
            logger.info(
                f"Pairwise evaluation completed: winner = {result.get('winner', 'unknown')}"
            )
            return result

        except Exception as e:
            logger.error(f"Pairwise evaluation failed: {e}")
            raise EvaluationError(f"Failed to run pairwise evaluation: {e}") from e

    def evaluate_batch(
        self,
        test_variants: list[dict[str, str]],
        source_files: list[str],
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple test variants efficiently in batch mode.
        """
        logger.info(f"Running batch evaluation for {len(test_variants)} variants")

        try:
            config = config or EvaluationConfig()
            results = []

            if len(test_variants) != len(source_files):
                raise ValueError(
                    "Number of test variants must match number of source files"
                )

            for i, (variant, source_file) in enumerate(
                zip(test_variants, source_files, strict=False)
            ):
                try:
                    test_content = variant.get("content", "")
                    test_id = variant.get(
                        "id", f"batch_{i}_{datetime.utcnow().strftime('%H%M%S')}"
                    )

                    # Evaluate single variant
                    result = self.evaluate_single(
                        test_content, source_file, config, test_id=test_id, **kwargs
                    )

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to evaluate variant {i}: {e}")
                    # Continue with other variants

            # Store batch results
            self._artifact_manager.store_batch_summary(results)

            logger.info(
                f"Batch evaluation completed: {len(results)}/{len(test_variants)} successful"
            )
            return results

        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise EvaluationError(f"Failed to run batch evaluation: {e}") from e

    def run_golden_repo_evaluation(
        self,
        golden_repo_path: Path,
        test_generator_func: Callable,
        evaluation_config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run comprehensive evaluation against a golden repository.

        This discovers Python files, generates tests, and evaluates them
        to detect regressions and measure overall test generation quality.
        """
        logger.info(f"Running golden repo evaluation on {golden_repo_path}")

        try:
            if not golden_repo_path.exists():
                raise ValueError(f"Golden repo path does not exist: {golden_repo_path}")

            config = evaluation_config or EvaluationConfig()

            # Discover Python source files
            source_files = list(golden_repo_path.rglob("*.py"))
            source_files = [f for f in source_files if not self._safety_validator.should_skip_file(f)]

            if not source_files:
                raise ValueError("No valid Python files found in golden repository")

            logger.info(f"Found {len(source_files)} source files to evaluate")

            # Generate and evaluate tests for each file
            file_results = []
            overall_stats = {
                "total_files": len(source_files),
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "acceptance_pass_rate": 0.0,
                "average_llm_score": 0.0,
                "coverage_improvements": [],
                "regressions_detected": [],
            }

            for source_file in source_files:
                try:
                    # Generate test using provided function
                    test_content = test_generator_func(str(source_file))

                    if not test_content:
                        logger.warning(f"No test generated for {source_file}")
                        continue

                    # Evaluate generated test
                    evaluation_result = self.evaluate_single(
                        test_content, str(source_file), config, **kwargs
                    )

                    file_results.append(evaluation_result)
                    overall_stats["successful_evaluations"] += 1

                    # Update statistics
                    if evaluation_result.acceptance.all_checks_pass:
                        overall_stats["acceptance_pass_rate"] += 1

                    if evaluation_result.llm_judge:
                        overall_stats[
                            "average_llm_score"
                        ] += evaluation_result.llm_judge.overall_score

                    if evaluation_result.acceptance.coverage_improvement:
                        overall_stats["coverage_improvements"].append(
                            evaluation_result.acceptance.coverage_improvement
                        )

                except Exception as e:
                    logger.error(f"Failed to evaluate {source_file}: {e}")
                    overall_stats["failed_evaluations"] += 1

            # Calculate final statistics
            if overall_stats["successful_evaluations"] > 0:
                overall_stats["acceptance_pass_rate"] /= overall_stats[
                    "successful_evaluations"
                ]
                overall_stats["average_llm_score"] /= overall_stats[
                    "successful_evaluations"
                ]

            # Detect regressions (placeholder for sophisticated regression detection)
            regression_detected = overall_stats["acceptance_pass_rate"] < 0.7

            golden_repo_result = {
                "overall_results": overall_stats,
                "file_results": [result.to_dict() for result in file_results],
                "regression_detected": regression_detected,
                "recommendations": EvaluationAnalytics.generate_recommendations(
                    overall_stats, file_results
                ),
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "golden_repo_path": str(golden_repo_path),
                "config": {
                    "acceptance_checks": config.acceptance_checks,
                    "llm_judge_enabled": config.llm_judge_enabled,
                    "rubric_dimensions": config.rubric_dimensions,
                },
            }

            # Store golden repo evaluation results
            self._artifact_manager.store_golden_repo_results(
                golden_repo_result, golden_repo_path
            )

            logger.info(
                f"Golden repo evaluation completed: {overall_stats['successful_evaluations']} files evaluated"
            )
            return golden_repo_result

        except Exception as e:
            logger.error(f"Golden repo evaluation failed: {e}")
            raise EvaluationError(f"Failed to run golden repo evaluation: {e}") from e

    def analyze_evaluation_trends(
        self,
        evaluation_history: list[EvaluationResult],
        time_window_days: int | None = 30,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze trends in evaluation results to identify patterns and improvements.
        """
        logger.debug(
            f"Analyzing evaluation trends for {len(evaluation_history)} results"
        )

        try:
            if not evaluation_history:
                return {"error": "No evaluation history provided"}

            # Filter by time window if specified
            if time_window_days:
                cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
                evaluation_history = [
                    result
                    for result in evaluation_history
                    if datetime.fromisoformat(result.timestamp) >= cutoff_date
                ]

            if not evaluation_history:
                return {"error": "No results in specified time window"}

            # Analyze trends using analytics module
            trends = {
                "time_window_days": time_window_days,
                "total_evaluations": len(evaluation_history),
                "acceptance_trends": EvaluationAnalytics.analyze_acceptance_trends(
                    evaluation_history
                ),
                "llm_judge_trends": EvaluationAnalytics.analyze_llm_trends(evaluation_history),
                "coverage_trends": EvaluationAnalytics.analyze_coverage_trends(evaluation_history),
                "quality_distribution": EvaluationAnalytics.analyze_quality_distribution(
                    evaluation_history
                ),
                "recommendations": [],
            }

            # Generate recommendations based on trends
            trends["recommendations"] = EvaluationAnalytics.generate_trend_recommendations(trends)

            logger.info("Evaluation trend analysis completed")
            return trends

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise EvaluationError(f"Failed to analyze evaluation trends: {e}") from e

    # =============================
    # NEW METHODS FOR SUBTASK 20.2: Enhanced Statistical Testing & Bias Mitigation
    # =============================

    def run_statistical_significance_analysis(
        self,
        evaluation_data: list[dict[str, Any]],
        analysis_type: str = "pairwise_comparison",
        confidence_level: float = 0.95,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run statistical significance analysis on evaluation data following 2025 best practices.

        This implements t-tests, bootstrap sampling, and effect size calculations
        to assess the statistical reliability of evaluation comparisons.
        """
        logger.debug(f"Running statistical significance analysis: {analysis_type}")

        try:
            # Get statistical analysis prompt
            system_prompt = self.prompt_registry.get_prompt(
                "system", "statistical_analysis_v1"
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", "statistical_analysis_v1"
            )

            if not system_prompt or not user_prompt:
                logger.warning("Statistical analysis prompts not available")
                return StatisticalAnalyzer.fallback_statistical_analysis(
                    evaluation_data, confidence_level
                )

            # Format evaluation data for analysis
            formatted_data = ContextBuilder.format_evaluation_data_for_analysis(evaluation_data)

            # Prepare analysis context
            analysis_parameters = ContextBuilder.build_analysis_parameters(
                confidence_level, analysis_type, len(evaluation_data)
            )

            # Format user prompt
            formatted_user_prompt = user_prompt.format(
                version="v1",
                evaluation_data=formatted_data,
                comparison_context=f"Statistical analysis of {analysis_type} evaluation data",
                analysis_parameters=json.dumps(analysis_parameters, indent=2),
            )

            # Call LLM for statistical analysis
            llm_response = self.llm_adapter.analyze_code(
                formatted_user_prompt,
                analysis_type="statistical_analysis",
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=1500,
                **kwargs,
            )

            # Parse statistical analysis response
            analysis_result = ResponseParser.parse_statistical_analysis_response(llm_response)

            # Add computational validation where possible
            analysis_result = StatisticalAnalyzer.validate_statistical_results(
                analysis_result, evaluation_data
            )

            # Store analysis artifact
            self._artifact_manager.store_statistical_analysis(
                analysis_result, analysis_type
            )

            logger.info(
                f"Statistical analysis completed: {analysis_result.get('significance_assessment', 'unknown')}"
            )
            return analysis_result

        except Exception as e:
            logger.error(f"Statistical significance analysis failed: {e}")
            return {
                "error": f"Statistical analysis failed: {e}",
                "fallback_analysis": StatisticalAnalyzer.fallback_statistical_analysis(
                    evaluation_data, confidence_level
                ),
            }

    def detect_evaluation_bias(
        self,
        evaluation_history: list[EvaluationResult],
        bias_types: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Detect and analyze evaluation bias patterns following 2025 best practices.

        This implements bias detection for length bias, complexity bias, style bias,
        anchoring effects, and other systematic evaluation issues.
        """
        logger.debug(
            f"Running bias detection analysis on {len(evaluation_history)} evaluations"
        )

        try:
            # Get bias mitigation prompt
            system_prompt = self.prompt_registry.get_prompt(
                "system", "bias_mitigation_v1"
            )
            user_prompt = self.prompt_registry.get_prompt("user", "bias_mitigation_v1")

            if not system_prompt or not user_prompt:
                logger.warning("Bias mitigation prompts not available")
                return StatisticalAnalyzer.fallback_bias_analysis(evaluation_history)

            # Prepare evaluation history data
            history_data = ContextBuilder.format_evaluation_history_for_bias_analysis(
                evaluation_history
            )

            # Define analysis scope
            analysis_scope = ContextBuilder.build_bias_analysis_scope(
                bias_types, len(evaluation_history)
            )

            # Format user prompt
            formatted_user_prompt = user_prompt.format(
                version="v1",
                evaluation_history=history_data,
                evaluation_context="Comprehensive bias detection and mitigation analysis",
                analysis_scope=json.dumps(analysis_scope, indent=2),
            )

            # Call LLM for bias analysis
            llm_response = self.llm_adapter.analyze_code(
                formatted_user_prompt,
                analysis_type="bias_detection",
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=2000,
                **kwargs,
            )

            # Parse bias analysis response
            bias_result = ResponseParser.parse_bias_analysis_response(llm_response)

            # Add computational bias metrics
            bias_result = StatisticalAnalyzer.compute_bias_metrics(bias_result, evaluation_history)

            # Store bias analysis artifact
            self._artifact_manager.store_bias_analysis(bias_result, len(evaluation_history))

            logger.info(
                f"Bias analysis completed: fairness score = {bias_result.get('fairness_score', 'unknown')}"
            )
            return bias_result

        except Exception as e:
            logger.error(f"Bias detection analysis failed: {e}")
            return {
                "error": f"Bias detection failed: {e}",
                "fallback_analysis": StatisticalAnalyzer.fallback_bias_analysis(evaluation_history),
            }

    def run_advanced_ab_testing_pipeline(
        self,
        prompt_variants: list[dict[str, str]],
        test_dataset: list[dict[str, str]],
        statistical_testing: bool = True,
        bias_mitigation: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run comprehensive A/B testing pipeline with statistical significance testing
        and bias mitigation following 2025 best practices.

        This implements the full PromptFoo/PromptLayer-style evaluation workflow
        with side-by-side comparison, statistical analysis, and bias detection.
        """
        logger.info(
            f"Running advanced A/B testing pipeline: {len(prompt_variants)} variants, {len(test_dataset)} test cases"
        )

        try:
            pipeline_results = {
                "metadata": {
                    "variant_count": len(prompt_variants),
                    "test_case_count": len(test_dataset),
                    "statistical_testing_enabled": statistical_testing,
                    "bias_mitigation_enabled": bias_mitigation,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "variant_evaluations": [],
                "pairwise_comparisons": [],
                "statistical_analysis": None,
                "bias_analysis": None,
                "recommendations": [],
            }

            # Step 1: Run individual evaluations for each variant
            logger.info("Step 1: Running individual variant evaluations")
            for i, variant in enumerate(prompt_variants):
                variant_id = variant.get("id", f"variant_{i}")
                variant_results = []

                for j, test_case in enumerate(test_dataset):
                    # Generate test using this prompt variant
                    test_content = ABTestingManager.generate_test_with_variant(variant, test_case)

                    # Evaluate the generated test
                    evaluation_result = self.evaluate_single(
                        test_content=test_content,
                        source_file=test_case.get("source_file", f"test_case_{j}.py"),
                        config=EvaluationConfig(
                            llm_judge_enabled=True, statistical_testing=True
                        ),
                        test_id=f"{variant_id}_case_{j}",
                        variant_id=variant_id,
                        **kwargs,
                    )

                    variant_results.append(evaluation_result)

                pipeline_results["variant_evaluations"].append(
                    {
                        "variant_id": variant_id,
                        "variant_metadata": variant,
                        "results": [r.to_dict() for r in variant_results],
                        "summary_stats": ABTestingManager.calculate_variant_summary_stats(
                            variant_results
                        ),
                    }
                )

            # Step 2: Run pairwise comparisons between variants
            logger.info("Step 2: Running pairwise comparisons")
            for i, variant_a in enumerate(prompt_variants):
                for j, variant_b in enumerate(prompt_variants):
                    if i >= j:  # Avoid duplicate comparisons
                        continue

                    comparison_results = []
                    for k, test_case in enumerate(test_dataset):
                        # Generate tests with both variants
                        test_a = ABTestingManager.generate_test_with_variant(variant_a, test_case)
                        test_b = ABTestingManager.generate_test_with_variant(variant_b, test_case)

                        # Run pairwise comparison
                        comparison_result = self.evaluate_pairwise(
                            test_a=test_a,
                            test_b=test_b,
                            source_file=test_case.get(
                                "source_file", f"test_case_{k}.py"
                            ),
                            comparison_mode="a_vs_b",
                            evaluation_context=f"A/B testing: {variant_a.get('id', f'variant_{i}')} vs {variant_b.get('id', f'variant_{j}')}",
                            **kwargs,
                        )

                        comparison_results.append(comparison_result)

                    pipeline_results["pairwise_comparisons"].append(
                        {
                            "variant_a_id": variant_a.get("id", f"variant_{i}"),
                            "variant_b_id": variant_b.get("id", f"variant_{j}"),
                            "comparisons": comparison_results,
                            "summary": ABTestingManager.calculate_pairwise_summary(
                                comparison_results
                            ),
                        }
                    )

            # Step 3: Statistical significance analysis
            if statistical_testing:
                logger.info("Step 3: Running statistical significance analysis")
                evaluation_data = []
                for variant_eval in pipeline_results["variant_evaluations"]:
                    for result in variant_eval["results"]:
                        evaluation_data.append(
                            {
                                "variant_id": variant_eval["variant_id"],
                                "overall_score": result.get("llm_judge", {}).get(
                                    "overall_score", 0
                                ),
                                "scores": result.get("llm_judge", {}).get("scores", {}),
                                "acceptance_passed": result.get("acceptance", {}).get(
                                    "all_checks_pass", False
                                ),
                            }
                        )

                pipeline_results[
                    "statistical_analysis"
                ] = self.run_statistical_significance_analysis(
                    evaluation_data=evaluation_data,
                    analysis_type="ab_testing_pipeline",
                    **kwargs,
                )

            # Step 4: Bias detection and mitigation
            if bias_mitigation:
                logger.info("Step 4: Running bias detection analysis")
                evaluation_history = []
                for variant_eval in pipeline_results["variant_evaluations"]:
                    for result_dict in variant_eval["results"]:
                        # Convert dict back to EvaluationResult for bias analysis
                        evaluation_history.append(
                            ResponseParser.dict_to_evaluation_result(result_dict)
                        )

                pipeline_results["bias_analysis"] = self.detect_evaluation_bias(
                    evaluation_history=evaluation_history, **kwargs
                )

            # Step 5: Generate recommendations
            logger.info("Step 5: Generating recommendations")
            pipeline_results[
                "recommendations"
            ] = ABTestingManager.generate_ab_testing_recommendations(pipeline_results)

            # Store comprehensive A/B testing results
            self._artifact_manager.store_ab_testing_results(
                pipeline_results, len(prompt_variants)
            )

            logger.info("Advanced A/B testing pipeline completed successfully")
            return pipeline_results

        except Exception as e:
            logger.error(f"A/B testing pipeline failed: {e}")
            raise EvaluationError(f"Failed to run A/B testing pipeline: {e}") from e
