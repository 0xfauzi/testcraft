"""
Domain models for the testcraft system.

This module contains the core domain models using Pydantic for validation
and serialization. These models represent the fundamental entities in the
test generation and analysis system.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    @field_validator("line_range")
    def validate_line_range(cls, v: Any) -> Any:
        """Validate that line range is valid (start <= end)."""
        start, end = v
        if start > end:
            raise ValueError("Start line must be less than or equal to end line")
        if start < 1:
            raise ValueError("Line numbers must be positive")
        return v

    model_config = ConfigDict(frozen=True, use_enum_values=True)


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

    @field_validator("missing_lines")
    def validate_missing_lines(cls, v: Any) -> Any:
        """Validate that missing lines are positive integers."""
        for line in v:
            if line < 1:
                raise ValueError("Line numbers must be positive")
        return sorted(set(v))  # Remove duplicates and sort

    model_config = ConfigDict(frozen=True)


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

    @model_validator(mode="after")
    def ensure_error_message_when_unsuccessful(self) -> "GenerationResult":
        """Validate that error message is provided when success is False."""
        if not self.success and not self.error_message:
            raise ValueError("Error message must be provided when success is False")
        return self

    model_config = ConfigDict(frozen=True)


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
    file_path: Path = Field(..., description="Source file path covered by this plan")
    project_root: Path | None = Field(
        None, description="Project root associated with this plan"
    )
    module_path: str | None = Field(
        None, description="Canonical import path for the source file"
    )
    test_output_path: Path | None = Field(
        None, description="Planned output path for generated tests"
    )

    @field_validator("elements_to_test")
    def validate_elements_not_empty(cls, v: Any) -> Any:
        """Validate that at least one element is provided."""
        if not v:
            raise ValueError("At least one element must be provided for testing")
        return v

    model_config = ConfigDict(frozen=True)


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

    @field_validator("updated_files")
    def validate_updated_files_not_empty(cls, v: Any) -> Any:
        """Validate that at least one file was updated."""
        if not v:
            raise ValueError("At least one file must be updated")
        return v

    model_config = ConfigDict(frozen=True)


class PlanningRequest(BaseModel):
    """
    Request for plan generation.

    This model captures the information needed to request a test generation plan,
    including the target file, object, and optional prompt customizations.
    """

    target_file: Path = Field(..., description="Path to the target file")
    target_object: str = Field(
        ..., description="Target object (e.g., 'Class.method' or 'function')"
    )
    project_root: Path | None = Field(None, description="Project root directory")
    prompt_customization: str | None = Field(
        None, description="Custom prompt modifications"
    )

    model_config = ConfigDict(frozen=True)


class PlanOption(BaseModel):
    """
    Individual plan alternative from LLM.

    This model represents a single plan generated by the LLM, with unique
    identification via hash and tracking of the model used.
    """

    plan_id: str = Field(..., description="SHA-256 hash of plan content")
    plan_content: dict[str, Any] = Field(
        ..., description="Plan content from LLMOrchestrator"
    )
    model_used: str = Field(..., description="LLM model used for generation")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")

    model_config = ConfigDict(frozen=True)


class PlanningResult(BaseModel):
    """
    Result of planning with acceptance/edit history.

    This model captures the complete planning workflow result, including
    the original request, generated plan, acceptance status, and full
    edit history for audit purposes.
    """

    request: PlanningRequest = Field(..., description="Original planning request")
    plan_option: PlanOption = Field(..., description="Generated plan option")
    accepted: bool = Field(default=False, description="Whether plan was accepted")
    rejected: bool = Field(default=False, description="Whether plan was rejected")
    edit_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Edit history with timestamps"
    )
    accepted_at: datetime | None = Field(None, description="Acceptance timestamp (UTC)")

    @model_validator(mode="after")
    def validate_acceptance_timestamp(self) -> "PlanningResult":
        """Validate acceptance timestamp presence when accepted is True."""
        if self.accepted and self.accepted_at is None:
            raise ValueError("accepted_at must be provided when accepted is True")
        return self

    model_config = ConfigDict(frozen=True)


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

    @field_validator("files_to_process")
    def validate_files_not_empty(cls, v: Any) -> Any:
        """Validate that files_to_process is a valid list."""
        # Allow empty lists for empty projects - this is a valid scenario
        if v is None:
            raise ValueError("files_to_process cannot be None")
        return v

    @model_validator(mode="after")
    def validate_dependency_mappings(self) -> "AnalysisReport":
        """Validate that reasons and test presence map to every file."""
        files = self.files_to_process or []
        for file_path in files:
            if file_path not in self.reasons:
                raise ValueError(f"Reason must be provided for file: {file_path}")
            if file_path not in self.existing_test_presence:
                raise ValueError(
                    f"Test presence info must be provided for file: {file_path}"
                )
        return self

    model_config = ConfigDict(frozen=True)


# Context Assembly Models (Repository-Aware Test Generation)


class Target(BaseModel):
    """Target information for test generation."""

    module_file: str = Field(..., description="Path to the target module file")
    object: str = Field(..., description="Target object (Class.method, function, etc.)")
    module_path: str | None = Field(
        None, description="Canonical module import path for the target"
    )
    relative_path: str | None = Field(
        None, description="Path to the target relative to the project root"
    )
    class_name: str | None = Field(
        None, description="Class name when target refers to a method"
    )
    method_name: str | None = Field(
        None, description="Method name when target refers to a method"
    )
    is_method: bool = Field(
        default=False, description="Whether the target refers to a method"
    )
    exists_in_source: bool = Field(
        default=False, description="Whether the target was located in source code"
    )

    model_config = ConfigDict(frozen=True)


class ImportMap(BaseModel):
    """Import mapping information for a target file."""

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
            raise ValueError(
                "bootstrap_conftest must be non-empty when needs_bootstrap is True"
            )
        if not needs and content:
            raise ValueError(
                "bootstrap_conftest must be empty when needs_bootstrap is False"
            )
        return self

    model_config = ConfigDict(frozen=True)


class Focal(BaseModel):
    """Focal code information for the target."""

    source: str = Field(..., description="Source code of the focal element")
    signature: str = Field(..., description="Function/method signature")
    docstring: str | None = Field(None, description="Docstring if available")
    is_placeholder: bool = Field(
        default=False,
        description="True when the focal source is a placeholder rather than real code",
    )
    placeholder_reason: str | None = Field(
        default=None,
        description="Explanation for placeholder focal source, when applicable",
    )

    model_config = ConfigDict(frozen=True)


class ResolvedDef(BaseModel):
    """Resolved definition for a symbol the LLM may need to call."""

    name: str = Field(..., description="Symbol name")
    kind: Literal["class", "func", "const", "enum", "fixture"] = Field(
        ..., description="Kind of symbol"
    )
    signature: str = Field(..., description="Signature or declaration")
    doc: str | None = Field(None, description="Documentation if available")
    body: str = Field(..., description="Implementation body or 'omitted' placeholder")

    model_config = ConfigDict(frozen=True)


class RankedMethod(BaseModel):
    """Ranked method information for property context."""

    qualname: str = Field(..., description="Qualified name of the method")
    level: Literal["intra", "repo"] = Field(
        ..., description="Analysis level (intra-class or repo-wide)"
    )
    relation: Literal["complete", "G", "W", "T"] = Field(
        ..., description="GIVEN/WHEN/THEN relation type"
    )

    model_config = ConfigDict(frozen=True)


class GwtSnippets(BaseModel):
    """GIVEN/WHEN/THEN code snippets for property-based context."""

    given: list[str] = Field(default_factory=list, description="GIVEN snippets")
    when: list[str] = Field(default_factory=list, description="WHEN snippets")
    then: list[str] = Field(default_factory=list, description="THEN snippets")

    model_config = ConfigDict(frozen=True)


class TestBundle(BaseModel):
    """Test bundle with associated fixtures, mocks, and assertions."""

    test_name: str = Field(..., description="Name of the test")
    imports: list[str] = Field(default_factory=list, description="Required imports")
    fixtures: list[str] = Field(default_factory=list, description="Fixtures used")
    mocks: list[str] = Field(default_factory=list, description="Mocks used")
    assertions: list[str] = Field(
        default_factory=list, description="Assertion patterns"
    )

    model_config = ConfigDict(frozen=True)


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

    model_config = ConfigDict(frozen=True)


class DeterminismConfig(BaseModel):
    """Determinism configuration for tests."""

    seed: int = Field(default=1337, description="Random seed for determinism")
    tz: str = Field(default="UTC", description="Timezone for time operations")
    freeze_time: bool = Field(
        default=True, description="Whether to freeze time in tests"
    )

    model_config = ConfigDict(frozen=True)


class IOPolicy(BaseModel):
    """I/O policy configuration for test safety."""

    network: Literal["forbidden", "mocked"] = Field(
        default="forbidden", description="Network access policy"
    )
    fs: Literal["forbidden", "tmp_path_only", "mocked"] = Field(
        default="tmp_path_only", description="Filesystem access policy"
    )

    model_config = ConfigDict(frozen=True)


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

    model_config = ConfigDict(frozen=True)


class Budget(BaseModel):
    """Token budget configuration for LLM operations."""

    max_input_tokens: int = Field(
        default=60000, description="Maximum input tokens for LLM context"
    )

    @field_validator("max_input_tokens")
    def validate_positive_tokens(cls, v: Any) -> Any:
        """Validate that token count is positive."""
        if v <= 0:
            raise ValueError("Token count must be positive")
        return v

    model_config = ConfigDict(frozen=True)


class DependencyMetadata(BaseModel):
    """Metadata about declared project dependencies for prompt and resolver context."""

    external_modules: list[str] = Field(
        default_factory=list,
        description="Normalized module names declared as project dependencies",
    )
    distributions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of declared dependency names to version specifiers",
    )

    model_config = ConfigDict(frozen=True)


class ContextPack(BaseModel):
    """
    Complete context package for repository-aware test generation.

    This model represents the exact schema specified in the context assembly
    specification, containing all information needed for the LLM to generate
    repository-aware, compilable tests.
    """

    target: Target = Field(..., description="Target information")
    import_map: ImportMap | None = Field(
        None, description="Import mapping and bootstrap"
    )
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
    context: str | None = Field(
        None, description="Enriched context string for backward compatibility"
    )
    dependency_metadata: DependencyMetadata = Field(
        default_factory=DependencyMetadata,
        description="Metadata about dependencies available to the project runtime",
    )

    def dict(
        self,
        *,
        include: Any | None = None,
        exclude: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> dict[str, Any]:
        """Convert ContextPack to dictionary for backward compatibility."""
        # Get the base dictionary using model_dump for proper serialization
        result: dict[str, Any] = self.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by key for backward compatibility."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in ContextPack."""
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        """Get item by key for backward compatibility."""
        if hasattr(self, key):
            # Delegate to model_dump() to get dict representation for backward compatibility
            data = self.model_dump()
            return data[key]
        raise KeyError(f"Key '{key}' not found in ContextPack")

    def __len__(self) -> int:
        """Return number of fields in ContextPack."""
        return 9  # target, import_map, focal, resolved_defs, property_context, conventions, budget, context, dependency_metadata

    model_config = ConfigDict(frozen=True)
