"""
Evaluation Port interface definition.

This module defines the interface for test evaluation operations,
including automated acceptance checks, LLM-as-judge evaluation,
and A/B testing for prompt variants.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from typing_extensions import Protocol


@dataclass
class EvaluationConfig:
    """Configuration for evaluation operations."""

    golden_repos_path: Path | None = None
    acceptance_checks: bool = True
    llm_judge_enabled: bool = True
    rubric_dimensions: list[str] = None
    statistical_testing: bool = True
    human_review_enabled: bool = False

    def __post_init__(self) -> None:
        if self.rubric_dimensions is None:
            self.rubric_dimensions = ["correctness", "coverage", "clarity", "safety"]


@dataclass
class AcceptanceResult:
    """Result of automated acceptance checks."""

    syntactically_valid: bool
    imports_successfully: bool
    pytest_passes: bool
    coverage_improvement: float | None = None
    error_messages: list[str] = None

    def __post_init__(self) -> None:
        if self.error_messages is None:
            self.error_messages = []

    @property
    def all_checks_pass(self) -> bool:
        """Check if all acceptance criteria are met."""
        return (
            self.syntactically_valid
            and self.imports_successfully
            and self.pytest_passes
        )


@dataclass
class LLMJudgeResult:
    """Result of LLM-as-judge evaluation."""

    scores: dict[str, float]  # dimension -> score (1-5)
    rationales: dict[str, str]  # dimension -> explanation
    overall_score: float
    prompt_version: str
    confidence: float = 0.0

    @classmethod
    def empty(cls) -> "LLMJudgeResult":
        """Create empty result for failed evaluations."""
        return cls(
            scores={}, rationales={}, overall_score=0.0, prompt_version="unknown"
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result for a test."""

    test_id: str
    source_file: str
    test_content: str
    acceptance: AcceptanceResult
    llm_judge: LLMJudgeResult | None = None
    metadata: dict[str, Any] = None
    timestamp: str = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            from datetime import datetime

            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_id": self.test_id,
            "source_file": self.source_file,
            "test_content": self.test_content,
            "acceptance": {
                "syntactically_valid": self.acceptance.syntactically_valid,
                "imports_successfully": self.acceptance.imports_successfully,
                "pytest_passes": self.acceptance.pytest_passes,
                "coverage_improvement": self.acceptance.coverage_improvement,
                "error_messages": self.acceptance.error_messages,
                "all_checks_pass": self.acceptance.all_checks_pass,
            },
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        if self.llm_judge:
            result["llm_judge"] = {
                "scores": self.llm_judge.scores,
                "rationales": self.llm_judge.rationales,
                "overall_score": self.llm_judge.overall_score,
                "prompt_version": self.llm_judge.prompt_version,
                "confidence": self.llm_judge.confidence,
            }

        return result


EvaluationMode = Literal["single", "pairwise", "batch"]
ComparisonMode = Literal["a_vs_b", "best_of_n", "ranking"]


class EvaluationPort(Protocol):
    """
    Interface for test evaluation operations.

    This protocol defines the contract for evaluating generated tests,
    including automated acceptance checks, LLM-as-judge evaluation,
    and A/B testing capabilities.
    """

    def run_acceptance_checks(
        self,
        test_content: str,
        source_file: str,
        baseline_coverage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AcceptanceResult:
        """
        Run automated acceptance checks on generated test content.

        Args:
            test_content: Generated test code to validate
            source_file: Source file the test was generated for
            baseline_coverage: Optional baseline coverage data for comparison
            **kwargs: Additional check parameters

        Returns:
            AcceptanceResult with validation results

        Raises:
            EvaluationError: If acceptance checking fails
        """
        ...

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

        Args:
            test_content: Test content to evaluate
            source_file: Source file for context
            rubric_dimensions: Specific dimensions to evaluate
            prompt_version: Specific evaluation prompt version to use
            **kwargs: Additional evaluation parameters

        Returns:
            LLMJudgeResult with scores, rationales, and metadata

        Raises:
            EvaluationError: If LLM evaluation fails
        """
        ...

    def evaluate_single(
        self,
        test_content: str,
        source_file: str,
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Perform complete evaluation of a single generated test.

        Args:
            test_content: Test content to evaluate
            source_file: Source file the test was generated for
            config: Evaluation configuration
            **kwargs: Additional evaluation parameters

        Returns:
            Complete EvaluationResult

        Raises:
            EvaluationError: If evaluation fails
        """
        ...

    def evaluate_pairwise(
        self,
        test_a: str,
        test_b: str,
        source_file: str,
        comparison_mode: ComparisonMode = "a_vs_b",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Compare two test variants using pairwise evaluation.

        Args:
            test_a: First test variant
            test_b: Second test variant
            source_file: Source file for context
            comparison_mode: Type of comparison to perform
            **kwargs: Additional comparison parameters

        Returns:
            Dictionary containing:
                - 'winner': 'a', 'b', or 'tie'
                - 'confidence': Confidence in the decision (0.0-1.0)
                - 'reasoning': Explanation of the decision
                - 'scores': Individual scores for each test
                - 'metadata': Additional comparison metadata

        Raises:
            EvaluationError: If pairwise evaluation fails
        """
        ...

    def evaluate_batch(
        self,
        test_variants: list[dict[str, str]],
        source_files: list[str],
        config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple test variants in batch mode for efficiency.

        Args:
            test_variants: List of test variant dictionaries with 'content' and 'id' keys
            source_files: Corresponding source files for each test
            config: Evaluation configuration
            **kwargs: Additional batch parameters

        Returns:
            List of EvaluationResult objects

        Raises:
            EvaluationError: If batch evaluation fails
        """
        ...

    def run_golden_repo_evaluation(
        self,
        golden_repo_path: Path,
        test_generator_func: callable,
        evaluation_config: EvaluationConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run evaluation against a golden repository for regression testing.

        Args:
            golden_repo_path: Path to golden repository
            test_generator_func: Function that generates tests for source files
            evaluation_config: Configuration for the evaluation run
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - 'overall_results': Summary statistics
                - 'file_results': Per-file evaluation results
                - 'regression_detected': Boolean indicating regressions
                - 'recommendations': List of improvement suggestions

        Raises:
            EvaluationError: If golden repo evaluation fails
        """
        ...

    def analyze_evaluation_trends(
        self,
        evaluation_history: list[EvaluationResult],
        time_window_days: int | None = 30,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze trends in evaluation results over time.

        Args:
            evaluation_history: Historical evaluation results
            time_window_days: Time window for trend analysis
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing trend analysis results

        Raises:
            EvaluationError: If trend analysis fails
        """
        ...
