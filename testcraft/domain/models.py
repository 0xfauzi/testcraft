"""
Domain models for the testcraft system.

This module contains the core domain models using Pydantic for validation
and serialization. These models represent the fundamental entities in the
test generation and analysis system.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class TestElementType(str, Enum):
    """Enumeration of test element types."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


class TestElement(BaseModel):
    """
    Represents a single element that can be tested.

    This model captures information about code elements that need test coverage,
    including their location, type, and documentation.
    """

    name: str = Field(..., description="Name of the test element")
    type: TestElementType = Field(
        ..., description="Type of the element (function, class, method, module)"
    )
    line_range: tuple[int, int] = Field(
        ..., description="Start and end line numbers (inclusive)"
    )
    docstring: str | None = Field(
        None, description="Documentation string for the element"
    )

    @validator("line_range")
    def validate_line_range(cls, v: Any) -> Any:
        """Validate that line range is valid (start <= end)."""
        start, end = v
        if start > end:
            raise ValueError("Start line must be less than or equal to end line")
        if start < 1:
            raise ValueError("Line numbers must be positive")
        return v

    class Config:
        """Pydantic configuration for TestElement."""

        frozen = True  # Make immutable
        use_enum_values = True


class CoverageResult(BaseModel):
    """
    Represents code coverage information for a file or module.

    This model captures coverage metrics including line coverage, branch coverage,
    and specific lines that are missing coverage.
    """

    line_coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Line coverage percentage (0.0 to 1.0)"
    )
    branch_coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Branch coverage percentage (0.0 to 1.0)"
    )
    missing_lines: list[int] = Field(
        default_factory=list, description="List of line numbers with no coverage"
    )

    @validator("missing_lines")
    def validate_missing_lines(cls, v: Any) -> Any:
        """Validate that missing lines are positive integers."""
        for line in v:
            if line < 1:
                raise ValueError("Line numbers must be positive")
        return sorted(set(v))  # Remove duplicates and sort

    class Config:
        """Pydantic configuration for CoverageResult."""

        frozen = True  # Make immutable


class GenerationResult(BaseModel):
    """
    Represents the result of test generation for a specific file.

    This model captures whether test generation was successful, the generated
    content, and any error messages if generation failed.
    """

    file_path: str = Field(..., description="Path to the generated test file")
    content: str | None = Field(None, description="Generated test content")
    success: bool = Field(..., description="Whether test generation was successful")
    error_message: str | None = Field(
        None, description="Error message if generation failed"
    )

    @validator("error_message")
    def validate_error_message(cls, v: Any, values: Any) -> Any:
        """Validate that error message is provided when success is False."""
        if not values.get("success", True) and not v:
            raise ValueError("Error message must be provided when success is False")
        return v

    class Config:
        """Pydantic configuration for GenerationResult."""

        frozen = True  # Make immutable


class TestGenerationPlan(BaseModel):
    """
    Represents a plan for generating tests for specific elements.

    This model captures the elements to be tested, existing test information,
    and coverage metrics before test generation.
    """

    elements_to_test: list[TestElement] = Field(
        ..., description="List of elements that need tests"
    )
    existing_tests: list[str] = Field(
        default_factory=list, description="Paths to existing test files"
    )
    coverage_before: CoverageResult | None = Field(
        None, description="Coverage metrics before test generation"
    )

    @validator("elements_to_test")
    def validate_elements_not_empty(cls, v: Any) -> Any:
        """Validate that at least one element is provided."""
        if not v:
            raise ValueError("At least one element must be provided for testing")
        return v

    class Config:
        """Pydantic configuration for TestGenerationPlan."""

        frozen = True  # Make immutable


class RefineOutcome(BaseModel):
    """
    Represents the outcome of refining existing tests.

    This model captures information about files that were updated during
    test refinement, the rationale for changes, and the refinement plan.
    """

    updated_files: list[str] = Field(
        ..., description="Paths to files that were updated"
    )
    rationale: str = Field(..., description="Explanation of why changes were made")
    plan: str | None = Field(None, description="Detailed plan for the refinement")

    @validator("updated_files")
    def validate_updated_files_not_empty(cls, v: Any) -> Any:
        """Validate that at least one file was updated."""
        if not v:
            raise ValueError("At least one file must be updated")
        return v

    class Config:
        """Pydantic configuration for RefineOutcome."""

        frozen = True  # Make immutable


class AnalysisReport(BaseModel):
    """
    Represents the analysis report for files that need test processing.

    This model captures information about files that require test generation
    or refinement, reasons for processing, and existing test presence.
    """

    files_to_process: list[str] = Field(
        ..., description="Paths to files that need test processing"
    )
    reasons: dict[str, str] = Field(
        ..., description="Mapping of file paths to reasons for processing"
    )
    existing_test_presence: dict[str, bool] = Field(
        ..., description="Mapping of file paths to existing test presence"
    )

    @validator("files_to_process")
    def validate_files_not_empty(cls, v: Any) -> Any:
        """Validate that files_to_process is a valid list."""
        # Allow empty lists for empty projects - this is a valid scenario
        if v is None:
            raise ValueError("files_to_process cannot be None")
        return v

    @validator("reasons")
    def validate_reasons_match_files(cls, v: Any, values: Any) -> Any:
        """Validate that reasons are provided for all files."""
        files = values.get("files_to_process", [])
        for file_path in files:
            if file_path not in v:
                raise ValueError(f"Reason must be provided for file: {file_path}")
        return v

    @validator("existing_test_presence")
    def validate_test_presence_match_files(cls, v: Any, values: Any) -> Any:
        """Validate that test presence info is provided for all files."""
        files = values.get("files_to_process", [])
        for file_path in files:
            if file_path not in v:
                raise ValueError(
                    f"Test presence info must be provided for file: {file_path}"
                )
        return v

    class Config:
        """Pydantic configuration for AnalysisReport."""

        frozen = True  # Make immutable
