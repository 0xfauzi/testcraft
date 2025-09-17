"""Test refinement configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class RefineConfig(BaseModel):
    """Configuration for AI-powered test refinement."""

    enable: bool = Field(default=False, description="Enable AI-powered test refinement")

    max_retries: int = Field(
        default=2, ge=0, le=10, description="Maximum refinement attempts"
    )

    backoff_base_sec: float = Field(
        default=1.0, ge=0.1, description="Base delay between refinement attempts"
    )

    backoff_max_sec: float = Field(
        default=8.0, ge=1.0, description="Maximum delay between attempts"
    )

    stop_on_no_change: bool = Field(
        default=True, description="Stop if LLM returns no changes"
    )

    max_total_minutes: float = Field(
        default=5.0, ge=0.1, description="Maximum total time for refinement"
    )

    # Immediate refinement configuration
    immediate_refinement: bool = Field(
        default=True, description="Enable immediate write-and-refine per file"
    )

    max_refine_workers: int = Field(
        default=2, ge=1, le=8, description="Limit concurrent pytest/refine workers"
    )

    keep_failed_writes: bool = Field(
        default=False, description="Keep test files that fail to write or have syntax errors"
    )

    refine_on_first_failure_only: bool = Field(
        default=True, description="Stop refinement at first pytest failure within a file"
    )

    refinement_backoff_sec: float = Field(
        default=0.2, ge=0.0, le=5.0, description="Backoff between refinement iterations"
    )

    # Strict refinement policies (new)
    strict_assertion_preservation: bool = Field(
        default=True, 
        description="Prevent refinement from weakening test assertions to pass (detects production bugs)"
    )
    
    fail_on_xfail_markers: bool = Field(
        default=True,
        description="Treat tests marked with xfail as refinement failures (prevents masking bugs)"
    )
    
    allow_xfail_on_suspected_bugs: bool = Field(
        default=False,
        description="Allow adding xfail markers when production bugs are suspected (teams can opt-in)"
    )
    
    report_suspected_prod_bugs: bool = Field(
        default=True,
        description="Generate detailed reports when refinement suspects production bugs"
    )

    # Refinement guardrails configuration
    refinement_guardrails: dict[str, Any] = Field(
        default_factory=lambda: {
            "reject_empty": True,
            "reject_literal_none": True, 
            "reject_identical": True,
            "validate_syntax": True,
            "format_on_refine": True,
        },
        description="Safety guardrails for refinement operations"
    )

    pytest_args_for_refinement: list[str] = Field(
        default=["-vv", "--tb=short", "-x"],
        description="Pytest arguments specifically for refinement runs"
    )

    # Content validation and equivalence checking
    allow_ast_equivalence_check: bool = Field(
        default=True,
        description="Enable AST-based semantic equivalence checking to detect meaningful vs cosmetic changes"
    )
    
    treat_cosmetic_as_no_change: bool = Field(
        default=True,
        description="Treat cosmetic-only changes (whitespace, formatting) as no change to avoid unnecessary iterations"
    )
    
    max_diff_hunks: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of diff hunks to include in logs and reports for readability"
    )
    
    # Failed test annotation configuration
    annotate_failed_tests: bool = Field(
        default=True,
        description="Annotate test files with fix instructions when refinement fails"
    )
    
    annotation_placement: Literal["top", "bottom"] = Field(
        default="top",
        description="Where to place failure annotations in test files"
    )
    
    annotation_include_failure_excerpt: bool = Field(
        default=True,
        description="Include trimmed failure output in annotations"
    )
    
    annotation_max_failure_chars: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Maximum characters of failure output to include in annotations"
    )
    
    annotation_style: Literal["docstring", "hash"] = Field(
        default="docstring",
        description="Style of annotation: docstring (triple quotes) or hash (comment lines)"
    )
    
    include_llm_fix_instructions: bool = Field(
        default=True,
        description="Include LLM fix instructions in failure annotations"
    )
    
    # Import path resolution and targeting
    prefer_runtime_import_paths: bool = Field(
        default=True,
        description="Prefer import paths extracted from error traces over source tree aliases for better mocking"
    )
    
    # Timeout and hang prevention
    enable_timeout_detection: bool = Field(
        default=True,
        description="Enable timeout detection and classification for hanging tests"
    )
    
    timeout_threshold_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Threshold for classifying test executions as hanging/timing out"
    )
    
    # Schema validation and repair
    enable_schema_repair: bool = Field(
        default=True,
        description="Enable single-shot LLM schema repair for malformed refinement outputs"
    )
    
    schema_repair_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for schema repair prompts (lower = more deterministic)"
    )
    
    # Preflight analysis
    enable_preflight_analysis: bool = Field(
        default=True,
        description="Enable preflight canonicalization analysis to provide proactive suggestions to LLM"
    )
