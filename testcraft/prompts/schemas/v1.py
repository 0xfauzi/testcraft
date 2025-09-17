"""
JSON Schema definitions for structured LLM outputs (v1).

This module contains all schema definitions for generation, refinement, and 
evaluation outputs, along with their examples and validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaDefinition:
    """Container for a JSON schema with examples, validation rules, and metadata."""
    schema: dict[str, Any]
    examples: dict[str, Any]
    validation_rules: dict[str, Any]
    metadata: dict[str, Any]


def get_schema_definition(schema_type: str, language: str, version: str) -> SchemaDefinition:
    """
    Get schema definition for the specified type and language.
    
    Args:
        schema_type: Type of schema (e.g., 'generation_output', 'refinement_output')
        language: Programming language for schema context
        version: Schema version (currently only 'v1' supported)
        
    Returns:
        SchemaDefinition with schema, examples, validation rules, and metadata
        
    Raises:
        ValueError: If schema_type is not supported
    """
    if schema_type == "generation_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the test file to create",
                    "pattern": r"^tests/.+\.py$",
                },
                "content": {
                    "type": "string",
                    "description": "Complete test file content",
                    "minLength": 1,
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "file_path": "tests/test_example.py",
                "content": "import pytest\n\ndef test_something():\n    assert True\n",
            }
        }
        validation_rules = {
            "max_content_bytes": 200_000,
            "deny_outside_tests_dir": True,
        }
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "generation_output_enhanced":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["file_path", "content", "analysis", "test_strategy", "validation"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the test file to create",
                    "pattern": r"^tests/.+\.py$",
                },
                "content": {
                    "type": "string",
                    "description": "Complete test file content (Step 4 implementation)",
                    "minLength": 1,
                },
                "analysis": {
                    "type": "string",
                    "description": "Step 1-2: Thorough code analysis and explanation of key components",
                    "minLength": 50,
                },
                "test_strategy": {
                    "type": "string",
                    "description": "Step 3: Comprehensive testing plan covering all scenarios",
                    "minLength": 50,
                },
                "validation": {
                    "type": "string",
                    "description": "Step 5: Quality review confirming tests follow the plan",
                    "minLength": 30,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the generated tests",
                },
                "coverage_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas covered by tests",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "file_path": "tests/test_example.py",
                "content": "import pytest\n\ndef test_something():\n    assert True\n",
                "analysis": "The code contains a UserAPI class with authentication methods that require HTTP mocking...",
                "test_strategy": "Test plan: 1) Happy path authentication, 2) Invalid credentials, 3) Network errors...",
                "validation": "Generated tests cover all planned scenarios with proper mocking and assertions",
                "confidence": 0.85,
                "coverage_areas": ["authentication", "error_handling", "edge_cases"]
            }
        }
        validation_rules = {
            "max_content_bytes": 300_000,
            "deny_outside_tests_dir": True,
        }
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "refinement_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["updated_files", "rationale"],
            "properties": {
                "updated_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "List of updated test file paths",
                },
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Why the changes were made",
                },
                "plan": {
                    "type": ["string", "null"],
                    "description": "Optional follow-up plan",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "updated_files": ["tests/test_example.py"],
                "rationale": "Increase branch coverage for edge cases",
                "plan": "Add paramized tests for invalid inputs",
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "refinement_output_enhanced":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["updated_files", "rationale", "issue_analysis", "refinement_strategy", "validation"],
            "properties": {
                "updated_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "List of updated test file paths",
                },
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Why the changes were made",
                },
                "plan": {
                    "type": ["string", "null"],
                    "description": "Optional follow-up plan",
                },
                "issue_analysis": {
                    "type": "string",
                    "description": "Step 1-2: Analysis of issues and explanation of problems",
                    "minLength": 50,
                },
                "refinement_strategy": {
                    "type": "string",
                    "description": "Step 3: Targeted strategy to address specific issues",
                    "minLength": 50,
                },
                "validation": {
                    "type": "string",
                    "description": "Step 5: Verification that refinements solve original problems",
                    "minLength": 30,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the refinement quality",
                },
                "improvement_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas that were improved",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "updated_files": ["tests/test_example.py"],
                "rationale": "Fixed failing tests and added missing edge case coverage",
                "plan": "Consider adding integration tests for end-to-end scenarios",
                "issue_analysis": "Tests were failing due to outdated mocking patterns and missing error path coverage...",
                "refinement_strategy": "Update mocks to use current API, add parametrized tests for edge cases...",
                "validation": "All original test failures are resolved and coverage gaps are filled",
                "confidence": 0.90,
                "improvement_areas": ["error_handling", "edge_cases", "mocking"]
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "llm_test_generation_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "analysis",
                "test_strategy",
                "tests",
                "validation",
                "coverage_focus",
                "confidence",
                "reasoning",
            ],
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Step 1-2: Deep code analysis and component understanding",
                    "minLength": 50,
                },
                "test_strategy": {
                    "type": "string",
                    "description": "Step 3: Comprehensive test strategy and approach",
                    "minLength": 50,
                },
                "tests": {
                    "type": "string",
                    "description": "Step 4: Generated test code implementation",
                    "minLength": 1,
                },
                "validation": {
                    "type": "string",
                    "description": "Step 5: Quality review and coverage verification",
                    "minLength": 30,
                },
                "coverage_focus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas that tests focus on covering",
                    "minItems": 1,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score in generated tests",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Summary of systematic approach and key decisions",
                    "minLength": 1,
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "analysis": "The code contains a UserAPI class with authentication methods that require HTTP mocking and error handling",
                "test_strategy": "Test plan: 1) Happy path authentication with valid credentials, 2) Invalid credentials scenarios, 3) Network error conditions",
                "tests": "import pytest\nfrom unittest.mock import patch\n\ndef test_authenticate_success():\n    assert True",
                "validation": "Generated tests cover all planned scenarios with proper mocking and assertions as specified in the strategy",
                "coverage_focus": ["authentication", "error_handling", "edge_cases"],
                "confidence": 0.85,
                "reasoning": "Systematic 5-step approach ensured comprehensive coverage of authentication flows and error conditions",
            }
        }
        validation_rules = {"max_content_bytes": 500_000}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "llm_code_analysis_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "testability_score",
                "complexity_metrics",
                "recommendations",
                "potential_issues",
                "analysis_summary",
            ],
            "properties": {
                "testability_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "description": "Testability score from 0-10",
                },
                "complexity_metrics": {
                    "type": "object",
                    "description": "Code complexity measurements",
                    "additionalProperties": {"type": ["number", "string"]},
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific recommendations for improvement",
                    "minItems": 0,
                },
                "potential_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified code issues or smells",
                    "minItems": 0,
                },
                "analysis_summary": {
                    "type": "string",
                    "description": "Overall summary of the analysis",
                    "minLength": 1,
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "testability_score": 7.5,
                "complexity_metrics": {
                    "cyclomatic_complexity": 3,
                    "function_count": 5,
                },
                "recommendations": [
                    "Add input validation",
                    "Extract complex logic",
                ],
                "potential_issues": ["Missing error handling"],
                "analysis_summary": "Code is generally well-structured but needs better error handling",
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "llm_content_refinement_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "refined_content",
                "changes_made",
                "confidence",
                "improvement_areas",
            ],
            "properties": {
                "refined_content": {
                    "type": "string",
                    "description": "Improved/refined content",
                    "minLength": 1,
                },
                "changes_made": {
                    "type": "string",
                    "description": "Summary of changes applied",
                    "minLength": 1,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in refinement quality",
                },
                "improvement_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas that were improved",
                    "minItems": 0,
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "refined_content": "# Improved code here",
                "changes_made": "Added error handling and improved readability",
                "confidence": 0.9,
                "improvement_areas": [
                    "error_handling",
                    "readability",
                    "performance",
                ],
            }
        }
        validation_rules = {"max_content_bytes": 500_000}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "llm_test_plan_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "plan_summary",
                "detailed_plan"
            ],
            "properties": {
                "plan_summary": {
                    "type": "string",
                    "description": "Brief 1-3 sentence overview of the testing approach",
                    "minLength": 10,
                },
                "detailed_plan": {
                    "type": "string", 
                    "description": "Comprehensive test implementation plan with scenarios, mocking, fixtures",
                    "minLength": 50,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the plan quality",
                },
                "scenarios": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of test scenarios to cover",
                    "minItems": 0,
                },
                "mocks": {
                    "type": "string",
                    "description": "Mocking strategy for external dependencies",
                },
                "fixtures": {
                    "type": "string",
                    "description": "Test data and fixture requirements",
                },
                "data_matrix": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Test data requirements and variations",
                    "minItems": 0,
                },
                "edge_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Edge cases and boundary conditions to test",
                    "minItems": 0,
                },
                "error_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Error conditions and exception scenarios",
                    "minItems": 0,
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "External dependencies that need consideration",
                    "minItems": 0,
                },
                "notes": {
                    "type": "string",
                    "description": "Additional considerations, risks, or implementation tips",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "plan_summary": "Test authentication service with happy path, invalid credentials, and network failures",
                "detailed_plan": "1. Mock HTTP client for external auth API. 2. Test valid login with success response. 3. Test invalid credentials with 401 response. 4. Test network timeout scenarios with connection errors.",
                "confidence": 0.85,
                "scenarios": ["valid_login", "invalid_credentials", "network_timeout"],
                "mocks": "Mock requests.Session for HTTP calls, mock time.sleep for retries",
                "fixtures": "User credentials fixture, API response fixtures for success/error cases",
                "data_matrix": ["valid_user_data", "invalid_passwords", "malformed_requests"],
                "edge_cases": ["empty_username", "null_password", "unicode_credentials"],
                "error_paths": ["connection_timeout", "invalid_json_response", "500_server_error"],
                "dependencies": ["external_auth_api", "database_connection", "redis_session_store"],
                "notes": "Consider rate limiting in tests, use deterministic timeouts"
            }
        }
        validation_rules = {"max_content_bytes": 100_000}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    # Evaluation-specific schemas
    if schema_type == "llm_judge_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": [
                "scores",
                "rationales",
                "overall_assessment",
                "confidence",
                "improvement_suggestions",
            ],
            "properties": {
                "scores": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_]+$": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 5.0,
                        }
                    },
                    "additionalProperties": False,
                },
                "rationales": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_]+$": {"type": "string", "minLength": 10}
                    },
                    "additionalProperties": False,
                },
                "overall_assessment": {
                    "type": "string",
                    "minLength": 20,
                    "description": "Holistic summary of test quality",
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "improvement_suggestions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 5},
                    "minItems": 0,
                    "maxItems": 10,
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "scores": {
                    "correctness": 4.5,
                    "coverage": 3.5,
                    "clarity": 4.0,
                    "safety": 5.0,
                },
                "rationales": {
                    "correctness": "Tests correctly verify expected behavior with proper assertions",
                    "coverage": "Most critical paths covered, missing some edge cases",
                    "clarity": "Well-structured tests with descriptive names",
                    "safety": "No unsafe patterns or anti-patterns detected",
                },
                "overall_assessment": "Good test quality with room for improvement in edge case coverage",
                "confidence": 0.85,
                "improvement_suggestions": [
                    "Add tests for error conditions",
                    "Include boundary value testing",
                ],
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "pairwise_comparison_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["winner", "confidence", "reasoning", "scores"],
            "properties": {
                "winner": {
                    "type": "string",
                    "enum": ["a", "b", "tie"],
                    "description": "Which test performs better overall",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the comparison result",
                },
                "reasoning": {
                    "type": "string",
                    "minLength": 20,
                    "description": "Detailed explanation of comparison decision",
                },
                "scores": {
                    "type": "object",
                    "required": ["test_a", "test_b"],
                    "properties": {
                        "test_a": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 5.0,
                        },
                        "test_b": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 5.0,
                        },
                    },
                    "additionalProperties": False,
                },
                "dimension_scores": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_]+$": {
                            "type": "object",
                            "properties": {
                                "test_a": {"type": "number", "minimum": 1.0, "maximum": 5.0},
                                "test_b": {"type": "number", "minimum": 1.0, "maximum": 5.0},
                            },
                        }
                    },
                    "description": "Optional breakdown by evaluation dimensions",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "winner": "a",
                "confidence": 0.85,
                "reasoning": "Test A provides better coverage of edge cases and clearer assertion messages",
                "scores": {"test_a": 4.2, "test_b": 3.8},
                "dimension_scores": {
                    "correctness": {"test_a": 4.5, "test_b": 4.0},
                    "coverage": {"test_a": 4.0, "test_b": 3.5},
                    "clarity": {"test_a": 4.0, "test_b": 4.0}
                }
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "rubric_evaluation_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["dimension_scores", "dimension_rationales", "overall_score", "quality_tier", "improvement_suggestions"],
            "properties": {
                "dimension_scores": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_]+$": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 5.0,
                        }
                    },
                    "additionalProperties": False,
                    "description": "Scores for each rubric dimension (1-5 scale)",
                },
                "dimension_rationales": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z_]+$": {"type": "string", "minLength": 10}
                    },
                    "additionalProperties": False,
                    "description": "Explanation for each dimension score",
                },
                "overall_score": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "description": "Calculated overall quality score",
                },
                "quality_tier": {
                    "type": "string",
                    "enum": ["excellent", "good", "fair", "poor"],
                    "description": "Quality tier based on overall score",
                },
                "improvement_suggestions": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 10},
                    "minItems": 0,
                    "maxItems": 10,
                    "description": "Actionable recommendations for improvement",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the evaluation quality",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "dimension_scores": {
                    "correctness": 4.0,
                    "coverage": 3.5,
                    "clarity": 4.2,
                    "safety": 4.8,
                    "maintainability": 3.8
                },
                "dimension_rationales": {
                    "correctness": "Tests verify expected behavior with appropriate assertions",
                    "coverage": "Good coverage of main paths, missing some edge cases",
                    "clarity": "Well-structured with descriptive test names",
                    "safety": "Excellent isolation, no side effects detected",
                    "maintainability": "Generally maintainable with room for improvement"
                },
                "overall_score": 4.1,
                "quality_tier": "good",
                "improvement_suggestions": [
                    "Add tests for boundary conditions",
                    "Include error path validation"
                ],
                "confidence": 0.88
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "statistical_analysis_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["statistical_test", "p_value", "confidence_interval", "effect_size", "significance_assessment", "sample_adequacy"],
            "properties": {
                "statistical_test": {
                    "type": "string",
                    "description": "Name of the statistical test applied",
                },
                "p_value": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Statistical significance p-value",
                },
                "confidence_interval": {
                    "type": "object",
                    "required": ["lower", "upper", "confidence_level"],
                    "properties": {
                        "lower": {"type": "number", "description": "Lower bound"},
                        "upper": {"type": "number", "description": "Upper bound"},
                        "confidence_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "additionalProperties": False,
                },
                "effect_size": {
                    "type": "object",
                    "required": ["cohens_d", "interpretation"],
                    "properties": {
                        "cohens_d": {"type": "number", "description": "Cohen's d effect size"},
                        "interpretation": {
                            "type": "string",
                            "enum": ["negligible", "small", "medium", "large", "unknown"],
                        },
                    },
                    "additionalProperties": False,
                },
                "significance_assessment": {
                    "type": "string",
                    "enum": ["significant", "not_significant", "not_assessed"],
                    "description": "Overall significance determination",
                },
                "sample_adequacy": {
                    "type": "object",
                    "required": ["current_sample_size", "recommended_minimum", "power_achieved"],
                    "properties": {
                        "current_sample_size": {"type": "integer", "minimum": 0},
                        "recommended_minimum": {"type": "integer", "minimum": 1},
                        "power_achieved": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "adequacy_assessment": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "interpretation": {
                    "type": "string",
                    "minLength": 20,
                    "description": "Clear interpretation of findings",
                },
                "reliability_notes": {
                    "type": "string",
                    "description": "Notes on result reliability and potential limitations",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "statistical_test": "t-test",
                "p_value": 0.0234,
                "confidence_interval": {
                    "lower": 3.2,
                    "upper": 4.8,
                    "confidence_level": 0.95
                },
                "effect_size": {
                    "cohens_d": 0.67,
                    "interpretation": "medium"
                },
                "significance_assessment": "significant",
                "sample_adequacy": {
                    "current_sample_size": 45,
                    "recommended_minimum": 30,
                    "power_achieved": 0.82,
                    "adequacy_assessment": "adequate"
                },
                "interpretation": "Results show statistically significant improvement with medium effect size",
                "reliability_notes": "Sample size adequate, assumptions met, results generalizable"
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "bias_mitigation_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["fairness_score", "bias_detection_results", "mitigation_recommendations", "bias_assessment"],
            "properties": {
                "fairness_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall fairness score (0=biased, 1=fair)",
                },
                "bias_detection_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["bias_type", "severity", "description"],
                        "properties": {
                            "bias_type": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "description": {"type": "string", "minLength": 10},
                            "affected_samples": {"type": "integer", "minimum": 0},
                        },
                    },
                    "description": "Specific biases detected in evaluation process",
                },
                "mitigation_recommendations": {
                    "type": "object",
                    "required": ["immediate_actions", "process_improvements", "monitoring_strategies"],
                    "properties": {
                        "immediate_actions": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 10},
                        },
                        "process_improvements": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 10},
                        },
                        "monitoring_strategies": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 10},
                        },
                    },
                    "additionalProperties": False,
                },
                "bias_assessment": {
                    "type": "string",
                    "minLength": 30,
                    "description": "Comprehensive summary of bias analysis",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in bias assessment",
                },
                "evaluation_consistency": {
                    "type": "object",
                    "properties": {
                        "inter_evaluator_agreement": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "temporal_stability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "consistency_notes": {"type": "string"},
                    },
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "fairness_score": 0.78,
                "bias_detection_results": [
                    {
                        "bias_type": "length_bias",
                        "severity": "medium",
                        "description": "Tendency to score longer tests higher regardless of quality",
                        "affected_samples": 12
                    }
                ],
                "mitigation_recommendations": {
                    "immediate_actions": [
                        "Review and re-score affected evaluations",
                        "Implement length-blind evaluation protocols"
                    ],
                    "process_improvements": [
                        "Add evaluation guidelines emphasizing quality over length",
                        "Implement bias detection monitoring"
                    ],
                    "monitoring_strategies": [
                        "Regular bias analysis on new evaluations",
                        "Cross-evaluator consistency checks"
                    ]
                },
                "bias_assessment": "Moderate length bias detected affecting 27% of evaluations. Systematic pattern suggests evaluator training needed.",
                "confidence": 0.85,
                "evaluation_consistency": {
                    "inter_evaluator_agreement": 0.73,
                    "temporal_stability": 0.81,
                    "consistency_notes": "Good temporal stability, moderate inter-evaluator agreement"
                }
            }
        }
        validation_rules = {}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    if schema_type == "manual_fix_suggestions_output":
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["manual_suggestions", "root_cause"],
            "properties": {
                "manual_suggestions": {
                    "type": "string",
                    "minLength": 10,
                    "description": "Specific actionable suggestions to fix the issues",
                },
                "root_cause": {
                    "type": "string",
                    "minLength": 10,
                    "description": "Analysis of the underlying cause of the problems",
                },
                "llm_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the suggested fixes",
                },
                "improvement_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas that need improvement",
                    "minItems": 0,
                },
                "code_examples": {
                    "type": "string",
                    "description": "Optional code examples demonstrating suggested fixes",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Priority level of the fixes",
                },
                "estimated_effort": {
                    "type": "string",
                    "enum": ["trivial", "minor", "moderate", "major"],
                    "description": "Estimated effort to implement the fixes",
                },
            },
            "additionalProperties": False,
        }
        examples = {
            "valid": {
                "manual_suggestions": "Update the mock setup to use 'patch.object' instead of 'patch' and ensure the mocked method returns the expected data structure",
                "root_cause": "Test failure caused by incorrect mocking pattern - the original method signature changed but mock wasn't updated",
                "llm_confidence": 0.92,
                "improvement_areas": ["mocking_patterns", "test_maintenance"],
                "code_examples": "@patch.object(APIClient, 'fetch_data', return_value={'status': 'success'})",
                "priority": "medium",
                "estimated_effort": "minor"
            }
        }
        validation_rules = {"max_content_bytes": 100_000}
        metadata = {"language": language, "version": version}
        return SchemaDefinition(schema, examples, validation_rules, metadata)

    raise ValueError(f"Unsupported schema type: {schema_type}")
