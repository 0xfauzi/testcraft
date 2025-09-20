"""Secure credential management for TestCraft LLM providers."""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


class LLMCredentials(BaseModel):
    """Secure credential storage for LLM providers."""

    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    # Anthropic Claude
    anthropic_api_key: SecretStr | None = Field(default=None)

    # Azure OpenAI
    azure_openai_api_key: SecretStr | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)

    # AWS Bedrock
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_region: str | None = Field(default=None)

    def get_openai_api_key(self) -> str | None:
        """Get OpenAI API key as plain string."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    def get_anthropic_api_key(self) -> str | None:
        """Get Anthropic API key as plain string."""
        return (
            self.anthropic_api_key.get_secret_value()
            if self.anthropic_api_key
            else None
        )

    def get_azure_openai_api_key(self) -> str | None:
        """Get Azure OpenAI API key as plain string."""
        return (
            self.azure_openai_api_key.get_secret_value()
            if self.azure_openai_api_key
            else None
        )

    def get_aws_access_key_id(self) -> str | None:
        """Get AWS access key ID as plain string."""
        return (
            self.aws_access_key_id.get_secret_value()
            if self.aws_access_key_id
            else None
        )

    def get_aws_secret_access_key(self) -> str | None:
        """Get AWS secret access key as plain string."""
        return (
            self.aws_secret_access_key.get_secret_value()
            if self.aws_secret_access_key
            else None
        )

    def has_openai_credentials(self) -> bool:
        """Check if OpenAI credentials are available."""
        return self.get_openai_api_key() is not None

    def has_anthropic_credentials(self) -> bool:
        """Check if Anthropic credentials are available."""
        return self.get_anthropic_api_key() is not None

    def has_azure_openai_credentials(self) -> bool:
        """Check if Azure OpenAI credentials are available."""
        return (
            self.get_azure_openai_api_key() is not None
            and self.azure_openai_endpoint is not None
        )

    def has_aws_bedrock_credentials(self) -> bool:
        """Check if AWS Bedrock credentials are available."""
        return (
            self.get_aws_access_key_id() is not None
            and self.get_aws_secret_access_key() is not None
            and self.aws_region is not None
        )

    def get_available_providers(self) -> list[str]:
        """Get list of providers with valid credentials."""
        providers = []
        if self.has_openai_credentials():
            providers.append("openai")
        if self.has_anthropic_credentials():
            providers.append("anthropic")
        if self.has_azure_openai_credentials():
            providers.append("azure-openai")
        if self.has_aws_bedrock_credentials():
            providers.append("bedrock")
        return providers


class CredentialError(Exception):
    """Raised when credential loading or validation fails."""

    pass


class CredentialManager:
    """Manages secure loading and validation of LLM provider credentials."""

    # Environment variable mappings
    ENV_MAPPINGS = {
        "openai_api_key": ["OPENAI_API_KEY"],
        "openai_base_url": ["OPENAI_BASE_URL"],
        "anthropic_api_key": ["ANTHROPIC_API_KEY"],
        "azure_openai_api_key": ["AZURE_OPENAI_API_KEY"],
        "azure_openai_endpoint": ["AZURE_OPENAI_ENDPOINT"],
        "aws_access_key_id": ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": ["AWS_SECRET_ACCESS_KEY"],
        "aws_region": ["AWS_REGION", "AWS_DEFAULT_REGION"],
    }

    def __init__(self, config_overrides: dict[str, Any] | None = None) -> None:
        """Initialize credential manager.

        Args:
            config_overrides: Optional configuration overrides from config file
        """
        self.config_overrides = config_overrides or {}
        self._credentials_cache: LLMCredentials | None = None

    def load_credentials(self, reload: bool = False) -> LLMCredentials:
        """Load credentials from environment variables and configuration.

        Args:
            reload: Force reload even if cached

        Returns:
            LLMCredentials object with loaded credentials

        Raises:
            CredentialError: If credential loading fails
        """
        if self._credentials_cache is not None and not reload:
            return self._credentials_cache

        try:
            credential_data = {}

            # Load from environment variables (highest priority)
            for field_name, env_vars in self.ENV_MAPPINGS.items():
                value = self._load_from_env(env_vars)
                if value:
                    credential_data[field_name] = value

            # Apply configuration overrides (lower priority)
            for field_name in self.ENV_MAPPINGS.keys():
                config_key = field_name
                if (
                    config_key in self.config_overrides
                    and field_name not in credential_data
                ):
                    value = self.config_overrides[config_key]
                    if value and str(value).strip():
                        credential_data[field_name] = value

            # Create LLMCredentials with proper SecretStr conversion
            secret_credentials: dict[str, SecretStr | None] = {}
            for key, value in credential_data.items():
                secret_credentials[key] = SecretStr(value) if value else None
            self._credentials_cache = LLMCredentials(**secret_credentials)

            # Log available providers (without exposing secrets)
            available = self._credentials_cache.get_available_providers()
            if available:
                logger.info(
                    f"Loaded credentials for LLM providers: {', '.join(available)}"
                )
            else:
                logger.warning(
                    "No LLM provider credentials found - adapters will fail without API keys"
                )

            return self._credentials_cache

        except Exception as e:
            error_msg = f"Failed to load credentials: {e}"
            logger.error(error_msg)
            raise CredentialError(error_msg) from e

    def _load_from_env(self, env_vars: list[str]) -> str | None:
        """Load value from environment variables (try in order)."""
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value and value.strip():
                return value.strip()
        return None

    def validate_provider_credentials(self, provider: str) -> bool:
        """Validate that credentials are available for a specific provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'azure-openai', 'bedrock')

        Returns:
            True if credentials are valid and available
        """
        credentials = self.load_credentials()

        validation_methods = {
            "openai": credentials.has_openai_credentials,
            "anthropic": credentials.has_anthropic_credentials,
            "azure-openai": credentials.has_azure_openai_credentials,
            "bedrock": credentials.has_aws_bedrock_credentials,
        }

        if provider not in validation_methods:
            raise CredentialError(f"Unknown provider: {provider}")

        return validation_methods[provider]()

    def get_provider_credentials(self, provider: str) -> dict[str, Any]:
        """Get credentials and settings for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary with provider-specific credentials and settings

        Raises:
            CredentialError: If provider is invalid or credentials are missing
        """
        credentials = self.load_credentials()

        if provider == "openai":
            if not credentials.has_openai_credentials():
                raise CredentialError(
                    "OpenAI credentials not found. Set OPENAI_API_KEY environment variable."
                )
            return {
                "api_key": credentials.get_openai_api_key(),
                "base_url": credentials.openai_base_url,
            }

        elif provider == "anthropic":
            if not credentials.has_anthropic_credentials():
                raise CredentialError(
                    "Anthropic credentials not found. Set ANTHROPIC_API_KEY environment variable."
                )
            return {
                "api_key": credentials.get_anthropic_api_key(),
            }

        elif provider == "azure-openai":
            if not credentials.has_azure_openai_credentials():
                raise CredentialError(
                    "Azure OpenAI credentials not found. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
                )
            return {
                "api_key": credentials.get_azure_openai_api_key(),
                "azure_endpoint": credentials.azure_openai_endpoint,
            }

        elif provider == "bedrock":
            if not credentials.has_aws_bedrock_credentials():
                raise CredentialError(
                    "AWS Bedrock credentials not found. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables."
                )
            return {
                "aws_access_key_id": credentials.get_aws_access_key_id(),
                "aws_secret_access_key": credentials.get_aws_secret_access_key(),
                "region_name": credentials.aws_region,
            }

        else:
            raise CredentialError(f"Unknown provider: {provider}")

    def clear_cache(self) -> None:
        """Clear credential cache (useful for testing or reloading)."""
        self._credentials_cache = None
        logger.debug("Credential cache cleared")

    def create_env_template(self, filepath: str | Path | None = None) -> Path:
        """Create a template .env file with all supported environment variables.

        Args:
            filepath: Path for the .env file. Defaults to .env.example

        Returns:
            Path to the created template file
        """
        if filepath is None:
            filepath = Path(".env.example")
        else:
            filepath = Path(filepath)

        template_content = """# TestCraft LLM Provider Configuration
# Copy this file to .env and fill in your API keys
# Never commit .env files with real credentials to version control!

# =============================================================================
# OpenAI Configuration
# =============================================================================
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: Custom endpoint

# =============================================================================
# Anthropic Claude Configuration
# =============================================================================
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# Azure OpenAI Configuration
# =============================================================================
# Get your credentials from Azure Portal > Cognitive Services > OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com

# =============================================================================
# AWS Bedrock Configuration
# =============================================================================
# Use AWS IAM credentials with Bedrock permissions
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1

# =============================================================================
# Security Notes:
# - Never commit real API keys to version control
# - Use environment variables in production environments
# - Rotate API keys regularly
# - Consider using AWS IAM roles instead of access keys when possible
# - Keep API keys secure and don't share them
# =============================================================================
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template_content)

        logger.info(f"Environment template created at {filepath}")
        return filepath


# Convenience function for quick credential loading
def load_credentials(
    config_overrides: dict[str, Any] | None = None,
) -> LLMCredentials:
    """Load LLM provider credentials from all sources.

    Args:
        config_overrides: Optional configuration overrides

    Returns:
        LLMCredentials object with loaded credentials
    """
    manager = CredentialManager(config_overrides)
    return manager.load_credentials()
