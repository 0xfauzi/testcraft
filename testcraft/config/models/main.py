"""Main TestCraft configuration model."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .deprecated import CoverageConfig, QualityConfig, TestStyleConfig
from .discovery import TestPatternConfig
from .environment import EnvironmentConfig
from .evaluation import CostConfig, EvaluationConfig, LLMProviderConfig
from .generation import TestGenerationConfig
from .prompt_budget import ContextEnrichmentConfig
from .telemetry import TelemetryConfig
from .ui import LoggingConfig, PlanningConfig, UIConfig


class TestCraftConfig(BaseModel):
    """Main configuration model for TestCraft."""

    # Test discovery and patterns
    test_patterns: TestPatternConfig = Field(
        default_factory=TestPatternConfig, description="Test file discovery patterns"
    )

    # LLM providers and AI configuration
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="Large Language Model provider configuration",
    )

    # Test generation behavior (expanded with new fields)
    generation: TestGenerationConfig = Field(
        default_factory=TestGenerationConfig,
        description="Test generation behavior configuration",
    )

    # Cost management
    cost_management: CostConfig = Field(
        default_factory=CostConfig, description="Cost management and optimization"
    )

    # Telemetry and observability
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Telemetry and observability configuration",
    )

    # Evaluation harness configuration (optional)
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Test evaluation harness configuration",
    )

    # Environment management
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Environment detection and management",
    )

    # Logging configuration  
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging behavior configuration",
    )

    # UI configuration
    ui: UIConfig = Field(
        default_factory=UIConfig,
        description="User interface behavior configuration",
    )

    # Planning configuration
    planning: PlanningConfig = Field(
        default_factory=PlanningConfig,
        description="Test planning phase configuration",
    )

    # DEPRECATED SECTIONS - Will be removed in future versions
    # These are kept for backwards compatibility but will emit warnings
    style: TestStyleConfig | None = Field(
        default_factory=TestStyleConfig,
        description="DEPRECATED: Test generation style configuration (moved to generation.test_framework)",
    )

    coverage: CoverageConfig | None = Field(
        default_factory=CoverageConfig, 
        description="DEPRECATED: Test coverage configuration (unused by runtime logic)"
    )

    quality: QualityConfig | None = Field(
        default_factory=QualityConfig, 
        description="DEPRECATED: Test quality analysis configuration (no runtime implementation)"
    )

    context_enrichment: ContextEnrichmentConfig | None = Field(
        default=None,
        description="DEPRECATED: Context enrichment configuration (moved under generation.context_enrichment)",
    )

    @field_validator("coverage")
    @classmethod
    def validate_coverage_thresholds(cls, v):
        """Ensure coverage thresholds are logically consistent (deprecated)."""
        if v is None:
            return v
        if v.regenerate_if_below > v.minimum_line_coverage:
            raise ValueError(
                "regenerate_if_below cannot be higher than minimum_line_coverage"
            )
        if v.minimum_branch_coverage > v.minimum_line_coverage:
            raise ValueError(
                "minimum_branch_coverage cannot be higher than minimum_line_coverage"
            )
        return v

    @field_validator("cost_management")
    @classmethod
    def validate_cost_thresholds(cls, v):
        """Ensure cost thresholds are logically consistent."""
        thresholds = v.cost_thresholds
        if thresholds.warning_threshold > thresholds.per_request_limit:
            raise ValueError(
                "warning_threshold cannot be higher than per_request_limit"
            )
        return v

    @field_validator("quality")
    @classmethod
    def validate_quality_scores(cls, v):
        """Ensure quality scores are within valid ranges (deprecated)."""
        if v is None:
            return v
        if v.minimum_quality_score > 100 or v.minimum_quality_score < 0:
            raise ValueError("minimum_quality_score must be between 0 and 100")
        if v.minimum_mutation_score > 100 or v.minimum_mutation_score < 0:
            raise ValueError("minimum_mutation_score must be between 0 and 100")
        return v

    def get_nested_value(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'coverage.minimum_line_coverage')."""
        keys = key.split(".")
        value = self.model_dump()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update_from_dict(self, updates: dict[str, Any]) -> "TestCraftConfig":
        """Update configuration with values from a dictionary."""
        current_dict = self.model_dump()
        updated_dict = self._deep_merge(current_dict, updates)
        return TestCraftConfig(**updated_dict)

    def _deep_merge(self, base: dict, updates: dict) -> dict:
        """Deeply merge updates into base dictionary."""
        result = base.copy()

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )
