"""Deprecated configuration models for backwards compatibility."""

from typing import Literal

from pydantic import BaseModel, Field


class TestStyleConfig(BaseModel):
    """DEPRECATED: Test generation style configuration (moved to generation.test_framework)."""
    
    framework: Literal["pytest", "unittest"] = Field(
        default="pytest", description="DEPRECATED: Use generation.test_framework"
    )
    assertion_style: Literal["pytest", "unittest", "auto"] = Field(
        default="pytest", description="DEPRECATED: No longer used"
    )
    mock_library: Literal["unittest.mock", "pytest-mock", "auto"] = Field(
        default="unittest.mock", description="DEPRECATED: No longer used"
    )


class CoverageConfig(BaseModel):
    """DEPRECATED: Test coverage configuration (no runtime implementation)."""
    
    minimum_line_coverage: float = Field(default=80.0, description="DEPRECATED: No longer used")
    minimum_branch_coverage: float = Field(default=70.0, description="DEPRECATED: No longer used")
    regenerate_if_below: float = Field(default=60.0, description="DEPRECATED: No longer used")
    pytest_args: list[str] = Field(default_factory=list, description="DEPRECATED: No longer used")
    junit_xml: bool = Field(default=True, description="DEPRECATED: No longer used")


class QualityConfig(BaseModel):
    """DEPRECATED: Test quality analysis configuration (no runtime implementation)."""
    
    enable_quality_analysis: bool = Field(default=True, description="DEPRECATED: No longer used")
    enable_mutation_testing: bool = Field(default=True, description="DEPRECATED: No longer used")
    minimum_quality_score: float = Field(default=75.0, description="DEPRECATED: No longer used")
    minimum_mutation_score: float = Field(default=80.0, description="DEPRECATED: No longer used")
    max_mutants_per_file: int = Field(default=50, description="DEPRECATED: No longer used")
    mutation_timeout: int = Field(default=30, description="DEPRECATED: No longer used")
    display_detailed_results: bool = Field(default=True, description="DEPRECATED: No longer used")
    enable_pattern_analysis: bool = Field(default=True, description="DEPRECATED: No longer used")
