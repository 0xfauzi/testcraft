"""
Environment preflight validation for safe execution.

This module performs non-invasive checks to ensure the runtime environment
has the minimum required dependencies and credentials before kicking off
generation. If critical checks fail, callers should abort early with a
clear message and suggestions for remediation.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Any

from ...config.models import TestCraftConfig


class EnvironmentValidator:
    """
    Validate that the environment is ready for generation and refinement.

    The validator checks:
    - Python version compatibility (>= 3.11)
    - Presence of pytest for executing/refining tests
    - LLM provider credentials for the configured default provider
    - Basic SDK importability for the selected LLM provider
    """

    @staticmethod
    def validate_for_generate(
        config: TestCraftConfig,
        require_refinement: bool = True,
        require_coverage_tools: bool = False,
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Python version
        py_ok = sys.version_info >= (3, 11)
        checks.append(
            {
                "name": "python_version",
                "ok": py_ok,
                "details": platform.python_version(),
            }
        )
        if not py_ok:
            errors.append("Python 3.11+ is required to run TestCraft.")
            suggestions.append(
                "Use uv to create/activate a 3.11 environment: 'uv venv --python 3.11 && source .venv/bin/activate'"
            )

        # Pytest availability (always required for refinement, and generally expected)
        pytest_spec = importlib.util.find_spec("pytest")
        pytest_ok = pytest_spec is not None
        checks.append({"name": "pytest_installed", "ok": pytest_ok})
        if require_refinement and not pytest_ok:
            errors.append("pytest is not installed or not importable.")
            suggestions.append(
                "Install pytest with uv: 'uv add --dev pytest pytest-cov'"
            )

        # LLM provider credentials (always required to generate tests)
        provider = (config.llm.default_provider or "openai").lower()

        def _has_value(v: str | None) -> bool:
            return bool(v and str(v).strip())

        provider_ok = True
        if provider == "openai":
            key_ok = _has_value(os.getenv("OPENAI_API_KEY")) or _has_value(
                config.llm.openai_api_key
            )
            sdk_ok = importlib.util.find_spec("openai") is not None
            checks.extend(
                [
                    {"name": "openai_key", "ok": key_ok},
                    {"name": "openai_sdk", "ok": sdk_ok},
                ]
            )
            if not key_ok:
                provider_ok = False
                errors.append(
                    "Missing OpenAI API key (set OPENAI_API_KEY or config.llm.openai_api_key)."
                )
                suggestions.append("Export your key: 'export OPENAI_API_KEY=sk-...'")
            if not sdk_ok:
                provider_ok = False
                suggestions.append("Install SDK with uv: 'uv add openai'")

        elif provider == "anthropic":
            key_ok = _has_value(os.getenv("ANTHROPIC_API_KEY")) or _has_value(
                config.llm.anthropic_api_key
            )
            sdk_ok = importlib.util.find_spec("anthropic") is not None
            checks.extend(
                [
                    {"name": "anthropic_key", "ok": key_ok},
                    {"name": "anthropic_sdk", "ok": sdk_ok},
                ]
            )
            if not key_ok:
                provider_ok = False
                errors.append(
                    "Missing Anthropic API key (set ANTHROPIC_API_KEY or config.llm.anthropic_api_key)."
                )
                suggestions.append("Export your key: 'export ANTHROPIC_API_KEY=...'")
            if not sdk_ok:
                provider_ok = False
                suggestions.append("Install SDK with uv: 'uv add anthropic'")

        elif provider == "azure-openai":
            key_ok = _has_value(os.getenv("AZURE_OPENAI_API_KEY")) or _has_value(
                config.llm.azure_openai_api_key
            )
            endpoint_ok = _has_value(os.getenv("AZURE_OPENAI_ENDPOINT")) or _has_value(
                config.llm.azure_openai_endpoint
            )
            sdk_ok = importlib.util.find_spec("openai") is not None
            checks.extend(
                [
                    {"name": "azure_openai_key", "ok": key_ok},
                    {"name": "azure_openai_endpoint", "ok": endpoint_ok},
                    {"name": "openai_sdk", "ok": sdk_ok},
                ]
            )
            if not key_ok or not endpoint_ok:
                provider_ok = False
                errors.append(
                    "Missing Azure OpenAI credentials (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)."
                )
                suggestions.append(
                    "Export credentials: 'export AZURE_OPENAI_API_KEY=...; export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com'"
                )
            if not sdk_ok:
                provider_ok = False
                suggestions.append("Install SDK with uv: 'uv add openai'")

        elif provider == "bedrock":
            key_ok = _has_value(os.getenv("AWS_ACCESS_KEY_ID")) and _has_value(
                os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            region_ok = _has_value(os.getenv("AWS_REGION")) or _has_value(
                config.llm.aws_region
            )
            sdk_ok = importlib.util.find_spec("langchain_aws") is not None
            checks.extend(
                [
                    {"name": "aws_credentials", "ok": key_ok},
                    {"name": "aws_region", "ok": region_ok},
                    {"name": "langchain_aws_sdk", "ok": sdk_ok},
                ]
            )
            if not key_ok or not region_ok:
                provider_ok = False
                errors.append(
                    "Missing AWS Bedrock credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)."
                )
                suggestions.append(
                    "Export AWS credentials: 'export AWS_ACCESS_KEY_ID=...; export AWS_SECRET_ACCESS_KEY=...; export AWS_REGION=us-east-1'"
                )
            if not sdk_ok:
                provider_ok = False
                suggestions.append("Install SDK with uv: 'uv add langchain-aws'")

        else:
            warnings.append(
                f"Unknown LLM provider '{provider}', assuming handled by router."
            )

        if not provider_ok:
            errors.append(
                "LLM provider not fully configured; cannot generate tests without it."
            )

        # Optional: Coverage toolchain checks (warn-only unless caller decides otherwise)
        if require_coverage_tools:
            cov_ok = importlib.util.find_spec("coverage") is not None
            pytest_cov_ok = importlib.util.find_spec("pytest_cov") is not None
            checks.extend(
                [
                    {"name": "coverage_pkg", "ok": cov_ok},
                    {"name": "pytest_cov_plugin", "ok": pytest_cov_ok},
                ]
            )
            if not cov_ok or not pytest_cov_ok:
                warnings.append(
                    "Coverage plugins not fully available (pytest-cov/coverage). Will fall back to AST estimation."
                )
                suggestions.append(
                    "Install coverage stack with uv: 'uv add --dev pytest-cov coverage'"
                )

        # Optional: formatting toolchain checks (warn-only)
        ruff_ok = importlib.util.find_spec("ruff") is not None
        black_ok = importlib.util.find_spec("black") is not None
        isort_ok = importlib.util.find_spec("isort") is not None
        checks.extend(
            [
                {"name": "ruff", "ok": ruff_ok},
                {"name": "black", "ok": black_ok},
                {"name": "isort", "ok": isort_ok},
            ]
        )
        if not ruff_ok:
            warnings.append(
                "Ruff formatter not available; will fall back to Black if present."
            )
            suggestions.append("Install Ruff with uv: 'uv add --dev ruff'")
        elif not black_ok:
            # Only warn if Ruff is present but Black fallback is missing
            warnings.append(
                "Black fallback not available; formatting will rely on Ruff only."
            )
            suggestions.append("Install Black fallback with uv: 'uv add --dev black'")

        ok = not errors
        message = (
            "Environment preflight failed: " + "; ".join(errors)
            if errors
            else "Environment preflight checks passed."
        )

        return {
            "ok": ok,
            "message": message,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "checks": checks,
        }
