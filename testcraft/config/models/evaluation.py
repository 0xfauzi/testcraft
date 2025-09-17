"""Evaluation and LLM provider configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CostThresholdConfig(BaseModel):
    """Configuration for cost thresholds and limits."""

    daily_limit: float = Field(
        default=50.0, ge=0.0, description="Maximum daily cost in USD"
    )

    per_request_limit: float = Field(
        default=2.0, ge=0.0, description="Maximum cost per request in USD"
    )

    warning_threshold: float = Field(
        default=1.0, ge=0.0, description="Warn when request exceeds this cost"
    )


class CostConfig(BaseModel):
    """Configuration for cost management and optimization."""

    max_file_size_kb: int = Field(
        default=50, ge=1, description="Skip files larger than this (KB)"
    )

    max_context_size_chars: int = Field(
        default=100000, ge=1000, description="Limit total context size"
    )

    max_files_per_request: int = Field(
        default=15, ge=1, description="Override batch size for large files"
    )

    use_cheaper_model_threshold_kb: int = Field(
        default=10, ge=1, description="Use cheaper model for files < this size"
    )

    enable_content_compression: bool = Field(
        default=True, description="Remove comments/whitespace in prompts"
    )

    cost_thresholds: CostThresholdConfig = Field(
        default_factory=CostThresholdConfig, description="Cost thresholds and limits"
    )

    skip_trivial_files: bool = Field(
        default=True, description="Skip files with < 5 functions/classes"
    )

    token_usage_logging: bool = Field(
        default=True, description="Log token usage for cost tracking"
    )


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider settings."""

    # OpenAI Configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    openai_model: str = Field(
        default="gpt-4.1", description="OpenAI model to use for test generation"
    )
    openai_base_url: str | None = Field(
        default=None, description="Custom OpenAI API base URL (optional)"
    )
    openai_max_tokens: int = Field(
        default=12000,
        ge=100,
        le=16384,
        description="Maximum tokens for OpenAI requests",
    )
    openai_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for OpenAI requests (seconds)",
    )

    # Anthropic Claude Configuration
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4",
        description="Anthropic model to use for test generation",
    )
    anthropic_max_tokens: int = Field(
        default=100000,
        ge=100,
        le=128000,
        description="Maximum tokens for Anthropic requests",
    )
    anthropic_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Anthropic requests (seconds)",
    )

    # Azure OpenAI Configuration
    azure_openai_api_key: str | None = Field(
        default=None,
        description="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY environment variable)",
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT environment variable)",
    )
    azure_openai_deployment: str = Field(
        default="gpt-4.1", description="Azure OpenAI deployment name (must use official OpenAI models)"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    azure_openai_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Azure OpenAI requests (seconds)",
    )

    # AWS Bedrock Configuration
    aws_region: str | None = Field(
        default=None,
        description="AWS region for Bedrock (or set AWS_REGION environment variable)",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID (or set AWS_ACCESS_KEY_ID environment variable)",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key (or set AWS_SECRET_ACCESS_KEY environment variable)",
    )
    bedrock_model_id: str = Field(
        default="anthropic.claude-sonnet-4-20250514-v1:0", description="Official Claude Sonnet 4 on AWS Bedrock"
    )
    bedrock_timeout: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for Bedrock requests (seconds)",
    )

    # General LLM Settings
    default_provider: Literal["openai", "anthropic", "azure-openai", "bedrock"] = Field(
        default="openai", description="Default LLM provider to use"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retries for LLM requests"
    )
    enable_streaming: bool = Field(
        default=False, description="Enable streaming responses where supported"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (lower = more deterministic)",
    )

    # Beta features configuration
    # Enable extended context/output features per provider (default: False for safety)
    openai_enable_extended_context: bool = Field(
        default=False,
        description="Enable extended context window beyond documented limits for OpenAI models"
    )
    openai_enable_extended_output: bool = Field(
        default=False,
        description="Enable extended output tokens beyond documented defaults for OpenAI models"
    )
    
    anthropic_enable_extended_context: bool = Field(
        default=False,
        description="Enable extended context window beyond documented limits for Anthropic models"
    )
    anthropic_enable_extended_output: bool = Field(
        default=False,
        description="Enable extended output tokens beyond documented defaults for Anthropic models"
    )
    
    azure_openai_enable_extended_context: bool = Field(
        default=False,
        description="Enable extended context window beyond documented limits for Azure OpenAI models"
    )
    azure_openai_enable_extended_output: bool = Field(
        default=False,
        description="Enable extended output tokens beyond documented defaults for Azure OpenAI models"
    )
    
    bedrock_enable_extended_context: bool = Field(
        default=False,
        description="Enable extended context window beyond documented limits for Bedrock models"
    )
    bedrock_enable_extended_output: bool = Field(
        default=False,
        description="Enable extended output tokens beyond documented defaults for Bedrock models"
    )

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "azure_openai_api_key",
        "aws_secret_access_key",
    )
    @classmethod
    def validate_api_keys_not_empty(cls, v):
        """Ensure API keys are not empty strings."""
        if v is not None and v.strip() == "":
            return None
        return v

    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_azure_endpoint(cls, v):
        """Validate Azure endpoint URL format."""
        if v is not None and v.strip() and not v.startswith(("http://", "https://")):
            raise ValueError(
                "Azure OpenAI endpoint must be a valid URL starting with http:// or https://"
            )
        return v


class EvaluationConfig(BaseModel):
    """Configuration for test evaluation harness."""

    enabled: bool = Field(
        default=False, description="Enable evaluation harness functionality"
    )

    golden_repos_path: str | None = Field(
        default=None, description="Path to golden repositories for regression testing"
    )

    # Acceptance checks configuration
    acceptance_checks: bool = Field(
        default=True,
        description="Enable automated acceptance checks (syntax, imports, pytest)",
    )

    # LLM-as-judge configuration
    llm_judge_enabled: bool = Field(
        default=True, description="Enable LLM-as-judge evaluation"
    )

    rubric_dimensions: list[str] = Field(
        default=["correctness", "coverage", "clarity", "safety"],
        description="Evaluation dimensions for LLM-as-judge",
    )

    # A/B testing and statistical analysis
    statistical_testing: bool = Field(
        default=True,
        description="Enable statistical significance testing for A/B comparisons",
    )

    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Statistical confidence level for A/B testing",
    )

    # Human review configuration
    human_review_enabled: bool = Field(
        default=False, description="Enable human-in-the-loop review"
    )

    # Artifact and state management
    artifacts_path: str = Field(
        default=".testcraft/evaluation_artifacts",
        description="Path for storing evaluation artifacts",
    )

    state_file: str = Field(
        default=".testcraft_evaluation_state.json",
        description="File for storing evaluation state",
    )

    # Evaluation timeouts and limits
    evaluation_timeout_seconds: int = Field(
        default=300, ge=10, le=3600, description="Timeout for individual evaluations"
    )

    batch_size: int = Field(
        default=10, ge=1, le=100, description="Batch size for A/B testing"
    )

    # Prompt registry configuration for evaluation
    prompt_version: str | None = Field(
        default=None,
        description="Specific prompt version for LLM-as-judge (None = latest)",
    )
