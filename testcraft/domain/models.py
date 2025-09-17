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
    def validate_line_range(cls, v):
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
    def validate_missing_lines(cls, v):
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
    def validate_error_message(cls, v, values):
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
    def validate_elements_not_empty(cls, v):
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
    def validate_updated_files_not_empty(cls, v):
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
    def validate_files_not_empty(cls, v):
        """Validate that files_to_process is a valid list."""
        # Allow empty lists for empty projects - this is a valid scenario
        if v is None:
            raise ValueError("files_to_process cannot be None")
        return v

    @validator("reasons")
    def validate_reasons_match_files(cls, v, values):
        """Validate that reasons are provided for all files."""
        files = values.get("files_to_process", [])
        for file_path in files:
            if file_path not in v:
                raise ValueError(f"Reason must be provided for file: {file_path}")
        return v

    @validator("existing_test_presence")
    def validate_test_presence_match_files(cls, v, values):
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


class PlannableElementKey(str):
    """
    Deterministic key for uniquely identifying a testable element.
    
    Format: "{abs_source_path}::{element.type}::{element.name}::{line_start}-{line_end}"
    """
    
    @classmethod
    def from_element(cls, source_path: str, element: TestElement) -> "PlannableElementKey":
        """Create a key from a source file path and test element."""
        line_start, line_end = element.line_range
        element_type = element.type.value if hasattr(element.type, 'value') else str(element.type)
        return cls(f"{source_path}::{element_type}::{element.name}::{line_start}-{line_end}")
    
    @property
    def source_path(self) -> str:
        """Extract source path from the key."""
        return self.split("::")[0]
    
    @property
    def element_type(self) -> str:
        """Extract element type from the key."""
        return self.split("::")[1]
    
    @property
    def element_name(self) -> str:
        """Extract element name from the key."""
        return self.split("::")[2]
    
    @property
    def line_range(self) -> tuple[int, int]:
        """Extract line range from the key."""
        range_str = self.split("::")[3]
        start, end = range_str.split("-")
        return int(start), int(end)


class TestElementPlan(BaseModel):
    """
    Represents a test plan for a single element.
    
    Contains the element information, eligibility reasoning, and detailed
    planning information generated by the LLM.
    """
    
    element: TestElement = Field(..., description="The test element this plan covers")
    eligibility_reason: str = Field(..., description="Why this element is eligible for testing")
    plan_summary: str = Field(..., description="1-3 sentence summary of the test plan")
    detailed_plan: str = Field(..., description="Concrete test implementation plan")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the plan (0.0 to 1.0)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags categorizing this plan")
    
    @validator("plan_summary")
    def validate_plan_summary_not_empty(cls, v):
        """Validate that plan summary is not empty."""
        if not v.strip():
            raise ValueError("Plan summary cannot be empty")
        return v.strip()
    
    @validator("detailed_plan")
    def validate_detailed_plan_not_empty(cls, v):
        """Validate that detailed plan is not empty."""
        if not v.strip():
            raise ValueError("Detailed plan cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration for TestElementPlan."""
        
        frozen = True  # Make immutable


class PlanningSession(BaseModel):
    """
    Represents a complete planning session with multiple test elements.
    
    Contains all planning artifacts, user selections, and session metadata.
    """
    
    session_id: str = Field(..., description="Unique identifier for this planning session")
    project_path: str = Field(..., description="Path to the project being planned")
    created_at: float = Field(..., description="Unix timestamp when session was created")
    items: list[TestElementPlan] = Field(..., description="All test element plans in this session")
    selected_keys: list[str] = Field(
        default_factory=list, 
        description="Keys of elements selected for generation"
    )
    stats: dict[str, Any] = Field(
        default_factory=dict, 
        description="Session statistics and metadata"
    )
    
    @validator("session_id")
    def validate_session_id_not_empty(cls, v):
        """Validate that session ID is not empty."""
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()
    
    @validator("project_path")
    def validate_project_path_not_empty(cls, v):
        """Validate that project path is not empty."""
        if not v.strip():
            raise ValueError("Project path cannot be empty")
        return v.strip()
    
    @validator("created_at")
    def validate_created_at_positive(cls, v):
        """Validate that created_at is a positive timestamp."""
        if v <= 0:
            raise ValueError("Created timestamp must be positive")
        return v
    
    def get_element_plan(self, key: str) -> TestElementPlan | None:
        """Get element plan by key."""
        for item in self.items:
            element_key = PlannableElementKey.from_element("", item.element)
            if element_key.endswith(key) or element_key == key:
                return item
        return None
    
    def get_selected_plans(self) -> list[TestElementPlan]:
        """Get all plans for selected elements."""
        return [item for item in self.items 
                if any(key in PlannableElementKey.from_element("", item.element) 
                      for key in self.selected_keys)]
    
    class Config:
        """Pydantic configuration for PlanningSession."""
        
        frozen = True  # Make immutable
