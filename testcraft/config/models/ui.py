"""UI and logging configuration models."""

from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for logging behavior."""
    
    max_debug_chars: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Maximum characters to display in console debug logs before truncation"
    )
    
    persist_verbose_artifacts: bool = Field(
        default=True,
        description="Save full LLM request/response artifacts to disk when verbose"
    )
    
    suppress_modules: list[str] = Field(
        default=["asyncio", "httpx", "openai", "urllib3", "textual"],
        description="External library modules to suppress debug logs from in non-verbose mode"
    )


class UIConfig(BaseModel):
    """Configuration for user interface behavior."""
    
    default_style: Literal["classic", "minimal"] = Field(
        default="classic",
        description="Default UI style when not explicitly specified"
    )
    
    minimal_warn_counter: bool = Field(
        default=True,
        description="Show warning counter in minimal mode summaries"
    )


class PlanningConfig(BaseModel):
    """Configuration for test planning functionality."""
    
    enabled: bool = Field(
        default=True,
        description="Enable test planning phase before generation"
    )
    
    auto_accept: bool = Field(
        default=False,
        description="Automatically accept all generated plans without user interaction"
    )
    
    max_elements_per_batch: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of elements to plan in a single batch"
    )
    
    plan_prompt_version: str = Field(
        default="v1",
        description="Version of planning prompts to use"
    )
    
    max_plan_tokens: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description="Maximum tokens for planning requests"
    )
    
    include_confidence_scores: bool = Field(
        default=True,
        description="Include confidence scores in planning results"
    )
    
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for accepting plans"
    )
