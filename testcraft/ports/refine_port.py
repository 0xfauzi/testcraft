"""
Refine Port interface definition.

This module defines the interface for test refinement operations,
including existing test improvement and optimization.
"""

from pathlib import Path
from typing import Any

from typing_extensions import Protocol

from ..domain.models import RefineOutcome


class RefinePort(Protocol):
    """
    Interface for test refinement operations.

    This protocol defines the contract for refining existing tests,
    including improvement suggestions and optimization recommendations.
    """

    def refine(
        self,
        test_files: list[str | Path],
        source_files: list[str | Path] | None = None,
        refinement_goals: list[str] | None = None,
        **kwargs: Any,
    ) -> RefineOutcome:
        """
        Refine existing test files to improve quality and coverage.

        Args:
            test_files: List of test file paths to refine
            source_files: Optional list of corresponding source file paths
            refinement_goals: Optional list of specific refinement goals
            **kwargs: Additional refinement parameters

        Returns:
            RefineOutcome object containing:
                - updated_files: List of files that were updated
                - rationale: Explanation of changes made
                - plan: Detailed plan for the refinement

        Raises:
            RefineError: If refinement fails
        """
        ...

    def analyze_test_quality(
        self,
        test_file: str | Path,
        source_file: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze the quality of an existing test file.

        Args:
            test_file: Path to the test file to analyze
            source_file: Optional path to the source file being tested
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing:
                - 'quality_score': Overall quality score (0.0 to 1.0)
                - 'coverage_score': Test coverage score
                - 'maintainability_score': Maintainability score
                - 'issues': List of quality issues found
                - 'recommendations': List of improvement recommendations

        Raises:
            RefineError: If analysis fails
        """
        ...

    def suggest_improvements(
        self,
        test_file: str | Path,
        improvement_type: str = "comprehensive",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Suggest specific improvements for a test file.

        Args:
            test_file: Path to the test file to improve
            improvement_type: Type of improvements to suggest
            **kwargs: Additional suggestion parameters

        Returns:
            Dictionary containing:
                - 'suggestions': List of improvement suggestions
                - 'priority': Priority levels for each suggestion
                - 'estimated_effort': Estimated effort for each improvement
                - 'expected_benefit': Expected benefit of each improvement

        Raises:
            RefineError: If suggestion generation fails
        """
        ...

    def optimize_test_structure(
        self,
        test_file: str | Path,
        optimization_goals: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize the structure and organization of a test file.

        Args:
            test_file: Path to the test file to optimize
            optimization_goals: Optional specific optimization goals
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary containing:
                - 'optimized_structure': Suggested optimized structure
                - 'changes_needed': List of changes required
                - 'benefits': Expected benefits of optimization
                - 'migration_plan': Plan for migrating to optimized structure

        Raises:
            RefineError: If optimization fails
        """
        ...

    def refine_from_failures(
        self,
        test_file: str | Path,
        failure_output: str,
        source_context: dict[str, Any] | None = None,
        max_iterations: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Refine a test file based on pytest failure output.

        Args:
            test_file: Path to the test file that failed
            failure_output: Raw pytest failure output (stdout/stderr)
            source_context: Optional source code context for fixing
            max_iterations: Maximum number of refinement attempts
            **kwargs: Additional refinement parameters

        Returns:
            Dictionary containing:
                - 'success': Whether refinement was successful
                - 'refined_content': Updated test file content if successful
                - 'iterations_used': Number of refinement iterations performed
                - 'final_status': Final pytest run status
                - 'error': Error message if refinement failed

        Raises:
            RefineError: If refinement fails permanently
        """
        ...

    def manual_fix_suggestions(
        self,
        test_file: str | Path,
        failure_output: str,
        source_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate targeted, manual fix suggestions when automated refinement is exhausted.

        Args:
            test_file: Path to the failing test file
            failure_output: Formatted pytest failure output (stdout/stderr)
            source_context: Optional source code context related to the test
            **kwargs: Additional hint parameters

        Returns:
            Dictionary containing:
                - 'manual_suggestions': Step-by-step, actionable instructions
                - 'root_cause': Brief explanation of why the test is failing
                - 'active_import_path': Runtime import path for accurate patch targets
                - 'preflight_suggestions': Canonicalization checks/hints
                - 'llm_confidence': Optional confidence score from the model
                - 'improvement_areas': Suggested focus areas (if available)
        """
        ...

    def enhance_test_coverage(
        self,
        test_file: str | Path,
        source_file: str | Path,
        coverage_gaps: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Enhance test coverage for specific areas of the source code.

        Args:
            test_file: Path to the test file to enhance
            source_file: Path to the source file being tested
            coverage_gaps: Optional list of line numbers with poor coverage
            **kwargs: Additional enhancement parameters

        Returns:
            Dictionary containing:
                - 'new_tests': Suggested new test cases
                - 'coverage_improvement': Expected coverage improvement
                - 'test_additions': Specific test additions needed
                - 'coverage_analysis': Analysis of current vs. improved coverage

        Raises:
            RefineError: If coverage enhancement fails
        """
        ...
