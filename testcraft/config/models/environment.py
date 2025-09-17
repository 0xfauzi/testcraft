"""Environment configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TestEnvironmentConfig(BaseModel):
    """Configuration for test execution environment."""

    propagate: bool = Field(
        default=True, description="Inherit current environment variables"
    )

    extra: dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )

    append_pythonpath: list[str] = Field(
        default_factory=list, description="Paths to append to PYTHONPATH"
    )


class EnvironmentOverrideConfig(BaseModel):
    """Configuration for environment-specific overrides."""

    poetry: dict[str, Any] = Field(
        default={"use_poetry_run": True, "respect_poetry_venv": True},
        description="Poetry-specific settings",
    )

    pipenv: dict[str, Any] = Field(
        default={"use_pipenv_run": True}, description="Pipenv-specific settings"
    )

    conda: dict[str, Any] = Field(
        default={"activate_environment": True}, description="Conda-specific settings"
    )

    uv: dict[str, Any] = Field(
        default={"use_uv_run": False}, description="UV-specific settings"
    )


class EnvironmentConfig(BaseModel):
    """Configuration for environment detection and management."""

    auto_detect: bool = Field(
        default=True, description="Auto-detect current environment manager"
    )

    preferred_manager: Literal[
        "poetry", "pipenv", "conda", "uv", "venv", "auto"
    ] = Field(default="auto", description="Preferred environment manager")

    respect_virtual_env: bool = Field(
        default=True, description="Always use current virtual env"
    )

    dependency_validation: bool = Field(
        default=True, description="Validate deps before running tests"
    )

    overrides: EnvironmentOverrideConfig = Field(
        default_factory=EnvironmentOverrideConfig,
        description="Environment-specific overrides",
    )
