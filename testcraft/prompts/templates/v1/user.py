"""
User prompt templates for v1 prompts.

This module contains all user prompts used to present tasks and context
to LLMs for test generation, refinement, and other tasks.
"""

from __future__ import annotations


# Define safe delimiters for prompt injection protection
SAFE_BEGIN = "BEGIN_SAFE_PROMPT"
SAFE_END = "END_SAFE_PROMPT"


def user_prompt_generation_v1() -> str:
    """User prompt template for test generation tasks."""
    return (
        f"{SAFE_BEGIN}\n"
        "VERSION: {version}\n"
        "TASK: Generate comprehensive Python tests following the systematic 5-step process.\n\n"
        "CODE TO ANALYZE AND TEST:\n"
        "```python\n{code_content}\n```\n\n"
        "ADDITIONAL_CONTEXT_JSON (Enriched Context Guide):\n"
        "The context below contains enriched information to help generate smarter tests:\n"
        "• Contract info: Function signatures, docstrings, exceptions - validate these in tests\n"
        "• Dependencies: Environment variables, HTTP clients, fixtures - mock appropriately  \n"
        "• Error paths: Known exceptions - test error handling scenarios\n"
        "• Project settings: Test patterns and configuration - follow existing conventions\n"
        "• Side effects: Network/file operations - mock to avoid side effects\n\n"
        "{additional_context}\n\n"
        "INSTRUCTIONS:\n"
        "1. ANALYZE the code thoroughly, examining structure, dependencies, and execution paths\n"
        "2. EXPLAIN your analysis, focusing on key components and critical execution paths\n"
        "3. CREATE a comprehensive testing plan covering happy paths, edge cases, and error conditions\n"
        "4. IMPLEMENT the tests following your plan and project conventions\n"
        "5. VALIDATE that your tests follow the plan and cover all identified scenarios\n\n"
        f"{SAFE_END}\n"
        "Return ONLY the JSON object with your complete 5-step analysis and generated tests."
    )


def user_prompt_refinement_v1() -> str:
    """User prompt template for test refinement tasks."""
    return (
        f"{SAFE_BEGIN}\n"
        "VERSION: {version}\n"
        "TASK: Refine existing tests using the systematic 5-step refinement process.\n\n"
        "REFINEMENT_CONTEXT_JSON:\n{additional_context}\n\n"
        "OPTIONAL_SOURCE_CODE_REFERENCE:\n"
        "```python\n{code_content}\n```\n\n"
        "INSTRUCTIONS:\n"
        "1. ANALYZE the issues and existing test problems thoroughly\n"
        "2. EXPLAIN your analysis of what's wrong and why refinement is needed\n"
        "3. DEVELOP a targeted refinement strategy to address specific issues\n"
        "4. IMPLEMENT the refinements following your strategy\n"
        "5. VALIDATE that your refinements solve the original problems effectively\n\n"
        f"{SAFE_END}\n"
        "Return ONLY the JSON object with your complete 5-step refinement analysis and results."
    )


def user_prompt_llm_test_generation_v1() -> str:
    """User prompt template for LLM-based test generation."""
    return (
        f"{SAFE_BEGIN}\n"
        "CODE TO TEST:\n```python\n{code_content}\n```\n\n"
        "REPOSITORY_CONTEXT:\n"
        "Code snippets, patterns, dependencies, and project structure:\n"
        "{repository_context}\n\n"
        "ENHANCED_CONTEXT:\n"
        "Module paths, planning details, and test scenarios:\n"
        "{enhanced_context}\n\n"
        "GENERATION_PREFERENCES:\n"
        "Settings for test generation behavior:\n"
        "{generation_preferences}\n\n"
        "Apply the 5-step systematic process to generate comprehensive tests:\n"
        "1-2. Deep analysis and explanation of the code\n"
        "3. Comprehensive test strategy planning\n"
        "4. Systematic test implementation\n"
        "5. Quality validation and coverage verification\n\n"
        "Focus on:\n"
        "- Happy path scenarios with typical usage\n"
        "- Edge cases and boundary conditions\n"
        "- Error handling and exception scenarios\n"
        "- Integration points and external dependencies\n\n"
        f"{SAFE_END}\n"
        "Return your analysis and tests as JSON matching the expected schema."
    )


def user_prompt_llm_code_analysis_v1() -> str:
    """User prompt template for LLM-based code analysis."""
    return (
        f"{SAFE_BEGIN}\n"
        "CODE FOR ANALYSIS:\n```python\n{code}\n```\n\n"
        "CONTEXT:\n{context}\n\n"
        "Please analyze this code for:\n"
        "- Testability (score 0-10)\n"
        "- Complexity metrics\n"
        "- Quality issues and code smells\n"
        "- Specific improvement recommendations\n"
        "- Dependencies and architecture assessment\n\n"
        f"{SAFE_END}\n"
        "Return analysis as JSON matching the expected schema."
    )


def user_prompt_llm_content_refinement_v1() -> str:
    """User prompt template for LLM-based content refinement."""
    return (
        f"{SAFE_BEGIN}\n"
        "CONTENT TO REFINE:\n{content}\n\n"
        "CONTEXT:\n{context}\n\n"
        "REFINEMENT GOALS:\n{goals}\n\n"
        "Please refine this content focusing on:\n"
        "- Code clarity and readability\n"
        "- Test effectiveness and coverage\n"
        "- Error handling improvements\n"
        "- Performance optimizations\n"
        "- Maintainability enhancements\n\n"
        "Preserve existing functionality while making targeted improvements.\n\n"
        f"{SAFE_END}\n"
        "Return refinements as JSON matching the expected schema."
    )


def user_prompt_llm_manual_fix_suggestions_v1() -> str:
    """User prompt template for manual fix suggestions."""
    return (
        f"{SAFE_BEGIN}\n"
        "FAILURE INFORMATION:\n{failures}\n\n"
        "CODE CONTEXT:\n```python\n{code}\n```\n\n"
        "ERROR DETAILS:\n{error_details}\n\n"
        "Please analyze the failures and provide specific, actionable suggestions to fix:\n"
        "- Root cause analysis of the issues\n"
        "- Targeted suggestions addressing specific problems\n"
        "- Code examples or patterns to follow\n"
        "- Prevention strategies for similar issues\n\n"
        f"{SAFE_END}\n"
        "Return fix suggestions as JSON matching the expected schema."
    )


def user_prompt_llm_test_planning_v1() -> str:
    """User prompt template for test planning."""
    return (
        f"{SAFE_BEGIN}\n"
        "CODE TO PLAN TESTS FOR:\n```python\n{code}\n```\n\n"
        "CONTEXT:\n{context}\n\n"
        "REQUIREMENTS:\n{requirements}\n\n"
        "Create a comprehensive test plan that covers:\n"
        "- Test scenarios for all important code paths\n"
        "- Mocking strategy for external dependencies\n"
        "- Fixture and test data requirements\n"
        "- Edge cases and error condition handling\n"
        "- Integration testing considerations\n\n"
        f"{SAFE_END}\n"
        "Return test plan as JSON matching the expected schema."
    )


def user_prompt_llm_judge_v1() -> str:
    """User prompt template for LLM judge evaluation."""
    return (
        f"{SAFE_BEGIN}\n"
        "TEST CODE TO EVALUATE:\n```python\n{test_code}\n```\n\n"
        "ORIGINAL CODE CONTEXT:\n```python\n{original_code}\n```\n\n"
        "EVALUATION CRITERIA:\n{criteria}\n\n"
        "Please evaluate this test code across these dimensions:\n"
        "- Correctness: Does it validate the right behavior?\n"
        "- Coverage: Are important paths and cases covered?\n"
        "- Clarity: Is it readable and well-structured?\n"
        "- Safety: Is it properly isolated without side effects?\n"
        "- Maintainability: Will it be easy to maintain over time?\n\n"
        "Provide scores (1-5) and rationales for each dimension.\n\n"
        f"{SAFE_END}\n"
        "Return evaluation as JSON matching the expected schema."
    )


def user_prompt_pairwise_comparison_v1() -> str:
    """User prompt template for pairwise comparison."""
    return (
        f"{SAFE_BEGIN}\n"
        "TEST A:\n```python\n{test_a}\n```\n\n"
        "TEST B:\n```python\n{test_b}\n```\n\n"
        "ORIGINAL CODE:\n```python\n{original_code}\n```\n\n"
        "COMPARISON CRITERIA:\n{criteria}\n\n"
        "Compare these two test implementations objectively:\n"
        "- Score each test on quality dimensions (1-5)\n"
        "- Identify key differences and trade-offs\n"
        "- Determine which test is better overall\n"
        "- Assess your confidence in the comparison\n"
        "- Provide statistical notes on result reliability\n\n"
        f"{SAFE_END}\n"
        "Return comparison as JSON matching the expected schema."
    )


def user_prompt_rubric_evaluation_v1() -> str:
    """User prompt template for rubric-based evaluation."""
    return (
        f"{SAFE_BEGIN}\n"
        "TEST CODE TO EVALUATE:\n```python\n{test_code}\n```\n\n"
        "ORIGINAL CODE:\n```python\n{original_code}\n```\n\n"
        "RUBRIC DIMENSIONS:\n{rubric_dimensions}\n\n"
        "EVALUATION CONTEXT:\n{context}\n\n"
        "Evaluate the test code using the provided rubric:\n"
        "- Score each rubric dimension (1-5) with clear rationale\n"
        "- Calculate overall score and quality tier\n"
        "- Identify specific strengths and weaknesses\n"
        "- Provide actionable improvement recommendations\n"
        "- Assess your confidence in the evaluation\n\n"
        f"{SAFE_END}\n"
        "Return evaluation as JSON matching the expected schema."
    )


def user_prompt_statistical_analysis_v1() -> str:
    """User prompt template for statistical analysis."""
    return (
        f"{SAFE_BEGIN}\n"
        "EVALUATION DATA:\n{evaluation_data}\n\n"
        "STATISTICAL CONTEXT:\n{context}\n\n"
        "ANALYSIS REQUIREMENTS:\n{requirements}\n\n"
        "Perform statistical analysis of the evaluation results:\n"
        "- Choose appropriate statistical test for the data\n"
        "- Calculate p-values, confidence intervals, effect sizes\n"
        "- Assess sample adequacy and statistical power\n"
        "- Evaluate result reliability and detect potential bias\n"
        "- Provide clear interpretation and recommendations\n\n"
        f"{SAFE_END}\n"
        "Return analysis as JSON matching the expected schema."
    )


def user_prompt_bias_mitigation_v1() -> str:
    """User prompt template for bias detection and mitigation."""
    return (
        f"{SAFE_BEGIN}\n"
        "EVALUATION HISTORY:\n{evaluation_history}\n\n"
        "BIAS ANALYSIS CONTEXT:\n{context}\n\n"
        "FAIRNESS REQUIREMENTS:\n{requirements}\n\n"
        "Analyze the evaluation process for bias and fairness:\n"
        "- Detect systematic biases in evaluation patterns\n"
        "- Assess consistency across different samples/evaluators\n"
        "- Evaluate calibration of the assessment process\n"
        "- Identify potential drift in evaluation standards\n"
        "- Recommend immediate and long-term mitigation strategies\n\n"
        f"{SAFE_END}\n"
        "Return bias analysis as JSON matching the expected schema."
    )
