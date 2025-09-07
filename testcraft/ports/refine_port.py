"""
Refine Port interface definition.

This module defines the interface for test refinement operations,
including existing test improvement and optimization.
"""

from typing import Dict, Any, Optional, List, Union
from typing_extensions import Protocol
from pathlib import Path
from ..domain.models import RefineOutcome


class RefinePort(Protocol):
    """
    Interface for test refinement operations.
    
    This protocol defines the contract for refining existing tests,
    including improvement suggestions and optimization recommendations.
    """
    
    def refine(
        self,
        test_files: List[Union[str, Path]],
        source_files: Optional[List[Union[str, Path]]] = None,
        refinement_goals: Optional[List[str]] = None,
        **kwargs: Any
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
        test_file: Union[str, Path],
        source_file: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
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
        test_file: Union[str, Path],
        improvement_type: str = "comprehensive",
        **kwargs: Any
    ) -> Dict[str, Any]:
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
        test_file: Union[str, Path],
        optimization_goals: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
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
    
    def enhance_test_coverage(
        self,
        test_file: Union[str, Path],
        source_file: Union[str, Path],
        coverage_gaps: Optional[List[int]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
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
