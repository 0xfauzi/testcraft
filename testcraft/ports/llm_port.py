"""
LLM Port interface definition.

This module defines the interface for Large Language Model operations,
including test generation and analysis capabilities.
"""

from typing import Dict, Any, Optional, List
from typing_extensions import Protocol


class LLMPort(Protocol):
    """
    Interface for Large Language Model operations.
    
    This protocol defines the contract for LLM interactions, including
    test generation, code analysis, and content refinement.
    """
    
    def generate_tests(
        self,
        code_content: str,
        context: Optional[str] = None,
        test_framework: str = "pytest",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate test cases for the provided code content.
        
        Args:
            code_content: The source code to generate tests for
            context: Optional context information about the code
            test_framework: The testing framework to use (default: pytest)
            **kwargs: Additional parameters for test generation
            
        Returns:
            Dictionary containing:
                - 'tests': Generated test code as string
                - 'coverage_focus': List of areas to focus testing on
                - 'confidence': Confidence score (0.0 to 1.0)
                - 'metadata': Additional generation metadata
                
        Raises:
            LLMError: If test generation fails
        """
        ...
    
    def analyze_code(
        self,
        code_content: str,
        analysis_type: str = "comprehensive",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Analyze code for testability, complexity, and potential issues.
        
        Args:
            code_content: The source code to analyze
            analysis_type: Type of analysis to perform
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing:
                - 'testability_score': Score indicating how testable the code is
                - 'complexity_metrics': Various complexity measurements
                - 'recommendations': List of improvement suggestions
                - 'potential_issues': List of identified problems
                
        Raises:
            LLMError: If analysis fails
        """
        ...
    
    def refine_content(
        self,
        original_content: str,
        refinement_instructions: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Refine existing content based on specific instructions.
        
        Args:
            original_content: The content to refine
            refinement_instructions: Specific instructions for refinement
            **kwargs: Additional refinement parameters
            
        Returns:
            Dictionary containing:
                - 'refined_content': The improved content
                - 'changes_made': Description of changes applied
                - 'confidence': Confidence in the refinement (0.0 to 1.0)
                
        Raises:
            LLMError: If refinement fails
        """
        ...
