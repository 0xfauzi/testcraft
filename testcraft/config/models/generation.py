"""Test generation configuration models."""

from typing import Literal

from pydantic import BaseModel, Field

from .prompt_budget import (
    ContextBudgetConfig,
    ContextCategoriesConfig,
    ContextEnrichmentConfig,
    PromptBudgetConfig,
)
from .refine import RefineConfig


class MergeConfig(BaseModel):
    """DEPRECATED: Test merging configuration (AST merge is always used)."""
    
    strategy: Literal["append", "ast-merge"] = Field(default="ast-merge", description="DEPRECATED: AST merge always used")
    dry_run: bool = Field(default=False, description="DEPRECATED: No longer used")
    formatter: str = Field(default="none", description="DEPRECATED: No longer used")


class PostGenerationTestRunnerConfig(BaseModel):
    """DEPRECATED: Post-generation test runner configuration (no longer used)."""
    
    enable: bool = Field(default=False, description="DEPRECATED: No longer used")
    args: list[str] = Field(default_factory=list, description="DEPRECATED: No longer used")
    cwd: str | None = Field(default=None, description="DEPRECATED: No longer used")
    junit_xml: bool = Field(default=True, description="DEPRECATED: No longer used")


class TestGenerationConfig(BaseModel):
    """Configuration for test generation behavior."""

    # Core generation settings
    batch_size: int = Field(
        default=5, ge=1, le=50,
        description="Number of files to process in parallel"
    )

    coverage_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Coverage threshold for reporting"
    )

    disable_ruff_format: bool = Field(
        default=False,
        description="Disable Ruff formatting if it causes issues"
    )

    immediate_refinement: bool = Field(
        default=True,
        description="Enable immediate write-and-refine per file"
    )

    enable_refinement: bool = Field(
        default=True,
        description="Enable AI-powered test refinement"
    )

    max_refinement_iterations: int = Field(
        default=3, ge=1, le=10,
        description="Maximum refinement attempts"
    )

    max_refine_workers: int = Field(
        default=2, ge=1, le=8,
        description="Limit concurrent pytest/refine workers"
    )

    keep_failed_writes: bool = Field(
        default=False,
        description="Keep test files that fail to write or have syntax errors"
    )

    refinement_backoff_sec: float = Field(
        default=0.2, ge=0.0, le=5.0,
        description="Backoff between refinement iterations"
    )

    test_framework: Literal["pytest", "unittest"] = Field(
        default="pytest",
        description="Testing framework to use"
    )

    # Budget configurations
    prompt_budgets: PromptBudgetConfig = Field(
        default_factory=PromptBudgetConfig,
        description="Prompt budget limits"
    )

    context_budgets: ContextBudgetConfig = Field(
        default_factory=ContextBudgetConfig,
        description="Context budget limits"
    )

    context_categories: ContextCategoriesConfig = Field(
        default_factory=ContextCategoriesConfig,
        description="Context categories to include"
    )

    # Context enrichment configuration (moved under generation)
    context_enrichment: ContextEnrichmentConfig = Field(
        default_factory=ContextEnrichmentConfig,
        description="Context enrichment configuration",
    )

    # Legacy generation settings (keeping for backwards compatibility)
    include_docstrings: bool | Literal["minimal"] = Field(
        default=True, description="Include docstrings in test methods"
    )

    generate_fixtures: bool = Field(
        default=True, description="Generate pytest fixtures for common setup"
    )

    parametrize_similar_tests: bool = Field(
        default=True, description="Use @pytest.mark.parametrize for similar tests"
    )

    max_test_methods_per_class: int = Field(
        default=20, ge=0, description="Maximum test methods per class (0 for unlimited)"
    )

    always_analyze_new_files: bool = Field(
        default=False, description="Always analyze new files even if they have tests"
    )

    # Refinement configuration
    refine: RefineConfig = Field(
        default_factory=RefineConfig, description="Test refinement configuration"
    )

    # Deprecated - will be removed in future versions
    test_runner: PostGenerationTestRunnerConfig = Field(
        default_factory=PostGenerationTestRunnerConfig,
        description="Post-generation test runner configuration (deprecated)",
    )

    merge: MergeConfig = Field(
        default_factory=MergeConfig, description="Test merging configuration (deprecated)"
    )
