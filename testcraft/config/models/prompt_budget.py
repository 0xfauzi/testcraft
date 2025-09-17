"""Prompt budget and context configuration models."""

from pydantic import BaseModel, Field


class PromptBudgetConfig(BaseModel):
    """Configuration for prompt budget limits."""

    per_item_chars: int = Field(
        default=1500, ge=100, le=5000,
        description="Character limit per context item"
    )

    total_chars: int = Field(
        default=10000, ge=1000, le=50000,
        description="Total character limit for context"
    )

    section_caps: dict[str, int] = Field(
        default_factory=lambda: {
            "snippets": 10,
            "neighbors": 5,
            "test_exemplars": 5,
            "contracts": 8,
            "deps_config_fixtures": 2,
            "coverage_hints": 3,
            "callgraph": 3,
            "error_paths": 3,
            "usage_examples": 5,
            "pytest_settings": 1,
            "side_effects": 1,
            "path_constraints": 3,
        },
        description="Maximum items per context section"
    )


class DirectoryTreeBudgetConfig(BaseModel):
    """Configuration for directory tree traversal limits."""

    max_depth: int = Field(
        default=4, ge=1, le=10,
        description="Maximum directory depth to traverse"
    )

    max_entries_per_dir: int = Field(
        default=200, ge=10, le=1000,
        description="Maximum files/directories per directory"
    )

    include_py_only: bool = Field(
        default=True,
        description="Only include .py files and directories"
    )


class ContextBudgetConfig(BaseModel):
    """Configuration for context budget limits."""

    directory_tree: DirectoryTreeBudgetConfig = Field(
        default_factory=DirectoryTreeBudgetConfig,
        description="Directory tree traversal limits"
    )


class ContextCategoriesConfig(BaseModel):
    """Configuration for context categories to include."""

    snippets: bool = Field(default=True, description="Include code snippets")
    neighbors: bool = Field(default=True, description="Include neighboring files")
    test_exemplars: bool = Field(default=True, description="Include test examples")
    contracts: bool = Field(default=True, description="Include interface contracts")
    deps_config_fixtures: bool = Field(default=True, description="Include dependencies and fixtures")
    coverage_hints: bool = Field(default=True, description="Include coverage hints")
    callgraph: bool = Field(default=True, description="Include call graph analysis")
    error_paths: bool = Field(default=True, description="Include error paths")
    usage_examples: bool = Field(default=True, description="Include usage examples")
    pytest_settings: bool = Field(default=True, description="Include pytest settings")
    side_effects: bool = Field(default=True, description="Include side effects detection")
    path_constraints: bool = Field(default=True, description="Include path constraints")


class ContextEnrichmentConfig(BaseModel):
    """Configuration for context enrichment during test generation."""

    enable_env_detection: bool = Field(
        default=True, description="Enable environment variable usage detection"
    )

    enable_db_boundary_detection: bool = Field(
        default=True, description="Enable database client boundary detection"
    )

    enable_http_boundary_detection: bool = Field(
        default=True, description="Enable HTTP client boundary detection"
    )

    enable_comprehensive_fixtures: bool = Field(
        default=True, description="Enable comprehensive pytest fixture discovery"
    )

    enable_side_effect_detection: bool = Field(
        default=True, description="Enable side-effect boundary detection"
    )

    max_env_vars: int = Field(
        default=20, ge=1, le=100, description="Maximum environment variables to include"
    )

    max_fixtures: int = Field(
        default=15, ge=1, le=50, description="Maximum fixtures to include"
    )
