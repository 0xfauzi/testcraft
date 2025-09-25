"""
Domain models for the testcraft system.

This module contains the core domain models using Pydantic for validation
and serialization. These models represent the fundamental entities in the
test generation and analysis system.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, validator


class TestCraftError(Exception):
    """Base exception for TestCraft domain errors."""

    pass


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


# Context Assembly Models (Repository-Aware Test Generation)


class Target(BaseModel):
    """Target information for test generation."""

    module_file: str = Field(..., description="Path to the target module file")
    object: str = Field(..., description="Target object (Class.method, function, etc.)")

    class Config:
        """Pydantic configuration for Target."""

        frozen = True


class ImportMap(BaseModel):
    """Import mapping information for a target file."""

 class ImportMap(BaseModel):
     target_import: str = Field(..., description="Canonical import statement to use")
     sys_path_roots: list[str] = Field(
         ..., description="List of sys.path root directories"
     )
     needs_bootstrap: bool = Field(
         ..., description="Whether bootstrap conftest.py is needed"
     )
     bootstrap_conftest: str = Field(
         ..., description="Bootstrap conftest.py content (empty if not needed)"
     )

     @model_validator(mode="after")
     def _validate_bootstrap(self) -> "ImportMap":
         needs = self.needs_bootstrap
         content = (self.bootstrap_conftest or "").strip()
         if needs and not content:
             raise ValueError("bootstrap_conftest must be non-empty when needs_bootstrap is True")
         if not needs and content:
             raise ValueError("bootstrap_conftest must be empty when needs_bootstrap is False")
         return self

    class Config:
        """Pydantic configuration for ImportMap."""

        frozen = True


class Focal(BaseModel):
    """Focal code information for the target."""

    source: str = Field(..., description="Source code of the focal element")
    signature: str = Field(..., description="Function/method signature")
    docstring: str | None = Field(None, description="Docstring if available")

    class Config:
        """Pydantic configuration for Focal."""

        frozen = True


class ResolvedDef(BaseModel):
    """Resolved definition for a symbol the LLM may need to call."""

    name: str = Field(..., description="Symbol name")
    kind: Literal["class", "func", "const", "enum", "fixture"] = Field(
        ..., description="Kind of symbol"
    )
    signature: str = Field(..., description="Signature or declaration")
    doc: str | None = Field(None, description="Documentation if available")
    body: str = Field(..., description="Implementation body or 'omitted' placeholder")

    class Config:
        """Pydantic configuration for ResolvedDef."""

        frozen = True


class RankedMethod(BaseModel):
    """Ranked method information for property context."""

    qualname: str = Field(..., description="Qualified name of the method")
    level: Literal["intra", "repo"] = Field(
        ..., description="Analysis level (intra-class or repo-wide)"
    )
    relation: Literal["complete", "G", "W", "T"] = Field(
        ..., description="GIVEN/WHEN/THEN relation type"
    )

    class Config:
        """Pydantic configuration for RankedMethod."""

        frozen = True


class GwtSnippets(BaseModel):
    """GIVEN/WHEN/THEN code snippets for property-based context."""

    given: list[str] = Field(default_factory=list, description="GIVEN snippets")
    when: list[str] = Field(default_factory=list, description="WHEN snippets")
    then: list[str] = Field(default_factory=list, description="THEN snippets")

    class Config:
        """Pydantic configuration for GwtSnippets."""

        frozen = True


class TestBundle(BaseModel):
    """Test bundle with associated fixtures, mocks, and assertions."""

    test_name: str = Field(..., description="Name of the test")
    imports: list[str] = Field(default_factory=list, description="Required imports")
    fixtures: list[str] = Field(default_factory=list, description="Fixtures used")
    mocks: list[str] = Field(default_factory=list, description="Mocks used")
    assertions: list[str] = Field(
        default_factory=list, description="Assertion patterns"
    )

    class Config:
        """Pydantic configuration for TestBundle."""

        frozen = True


class PropertyContext(BaseModel):
    """Property-based context for APT-style test generation."""

    ranked_methods: list[RankedMethod] = Field(
        default_factory=list, description="Ranked methods with property relations"
    )
    gwt_snippets: GwtSnippets = Field(
        default_factory=GwtSnippets, description="GIVEN/WHEN/THEN code snippets"
    )
    test_bundles: list[TestBundle] = Field(
        default_factory=list, description="Related test bundles"
    )

    class Config:
        """Pydantic configuration for PropertyContext."""

        frozen = True


class DeterminismConfig(BaseModel):
    """Determinism configuration for tests."""

    seed: int = Field(default=1337, description="Random seed for determinism")
    tz: str = Field(default="UTC", description="Timezone for time operations")
    freeze_time: bool = Field(
        default=True, description="Whether to freeze time in tests"
    )

    class Config:
        """Pydantic configuration for DeterminismConfig."""

        frozen = True


class IOPolicy(BaseModel):
    """I/O policy configuration for test safety."""

    network: Literal["forbidden", "mocked"] = Field(
        default="forbidden", description="Network access policy"
    )
    fs: Literal["forbidden", "tmp_path_only", "mocked"] = Field(
        default="tmp_path_only", description="Filesystem access policy"
    )

    class Config:
        """Pydantic configuration for IOPolicy."""

        frozen = True


class Conventions(BaseModel):
    """Test conventions and constraints."""

    test_style: str = Field(default="pytest", description="Test framework style")
    allowed_libs: list[str] = Field(
        default_factory=lambda: ["pytest", "hypothesis"],
        description="Allowed testing libraries",
    )
    determinism: DeterminismConfig = Field(
        default_factory=DeterminismConfig, description="Determinism configuration"
    )
    io_policy: IOPolicy = Field(
        default_factory=IOPolicy, description="I/O policy for test safety"
    )

    class Config:
        """Pydantic configuration for Conventions."""

        frozen = True


class Budget(BaseModel):
    """Token budget configuration for LLM operations."""

    max_input_tokens: int = Field(
        default=60000, description="Maximum input tokens for LLM context"
    )

    @validator("max_input_tokens")
    def validate_positive_tokens(cls, v: Any) -> Any:
        """Validate that token count is positive."""
        if v <= 0:
            raise ValueError("Token count must be positive")
        return v

    class Config:
        """Pydantic configuration for Budget."""

        frozen = True


class ContextPack(BaseModel):
    """
    Complete context package for repository-aware test generation.

    This model represents the exact schema specified in the context assembly
    specification, containing all information needed for the LLM to generate
    repository-aware, compilable tests.
    """

    target: Target = Field(..., description="Target information")
    import_map: ImportMap = Field(..., description="Import mapping and bootstrap")
    focal: Focal = Field(..., description="Focal code information")
    resolved_defs: list[ResolvedDef] = Field(
        default_factory=list,
        description="On-demand resolved symbol definitions",
    )
    property_context: PropertyContext = Field(
        default_factory=PropertyContext,
        description="Property-based context (APT-style)",
    )
    conventions: Conventions = Field(
        default_factory=Conventions, description="Test conventions and policies"
    )
    budget: Budget = Field(
        default_factory=Budget, description="Token budget configuration"
    )

    class Config:
        """Pydantic configuration for ContextPack."""

        frozen = True
