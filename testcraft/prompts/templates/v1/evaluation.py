"""
Evaluation prompt templates for v1 prompts.

This module contains evaluation-specific prompt templates for LLM judges,
pairwise comparisons, rubric evaluations, and bias mitigation.
"""

from __future__ import annotations

# This file is intentionally simple since most evaluation templates 
# are already covered in system.py and user.py. Additional evaluation-specific
# templates can be added here as needed.

def evaluation_rubric_template() -> str:
    """Template for evaluation rubrics."""
    return """
    Evaluation Rubric:
    - Correctness (1-5): How accurately do tests validate expected behavior?
    - Coverage (1-5): How comprehensively do tests cover important code paths?
    - Clarity (1-5): How readable and maintainable are the tests?
    - Safety (1-5): How well isolated are tests from side effects?
    - Maintainability (1-5): How easy will tests be to maintain over time?
    """


def pairwise_comparison_template() -> str:
    """Template for pairwise comparison instructions."""
    return """
    Compare the two test implementations objectively:
    1. Score each test on the evaluation dimensions (1-5 scale)
    2. Identify key differences between the implementations
    3. Determine which test is superior and why
    4. Assess your confidence in the comparison result
    5. Provide statistical notes on the reliability of the assessment
    """


def bias_detection_template() -> str:
    """Template for bias detection guidelines."""
    return """
    Bias Detection Guidelines:
    - Look for systematic patterns in evaluation scores
    - Check for consistency across different test samples
    - Identify potential length bias, complexity bias, or style preferences
    - Assess evaluation drift over time
    - Consider potential evaluator biases or systematic errors
    """
