"""Model Catalog: central source for LLM limits, pricing, and flags.

Reads TOML from `model_catalog.toml`, validates with Pydantic, provides
normalized lookup helpers and a small verification utility.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

CATALOG_FILENAME = "model_catalog.toml"


class ModelLimits(BaseModel):
    max_context: int = Field(ge=1)
    default_max_output: int = Field(ge=1)
    max_thinking: int | None = Field(default=None, ge=1)


class PerMillionPricing(BaseModel):
    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


class ModelPricing(BaseModel):
    per_million: PerMillionPricing


class ModelFlags(BaseModel):
    beta: bool = False
    supports_thinking: bool = False
    reasoning: bool = False
    deprecated: bool = False


class ModelSource(BaseModel):
    url: str
    last_verified: str

    @field_validator("last_verified")
    @classmethod
    def _validate_iso_datetime(cls, v: str) -> str:
        # Best-effort ISO 8601 validation for maintainability
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                "last_verified must be an ISO timestamp, e.g., 2025-07-01T00:00:00Z"
            ) from exc
        return v


class ModelMetadata(BaseModel):
    provider: str
    model_id: str
    limits: ModelLimits
    flags: ModelFlags
    pricing: ModelPricing
    source: ModelSource


class ModelCatalogData(BaseModel):
    models: list[ModelMetadata] = Field(default_factory=list)


@dataclass
class _CatalogCache:
    mtime_ns: int
    data: ModelCatalogData


_CACHE: _CatalogCache | None = None


def _catalog_path() -> Path:
    return Path(__file__).with_name(CATALOG_FILENAME)


def _read_catalog_file() -> bytes:
    path = _catalog_path()
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model catalog not found at {path}. Create {CATALOG_FILENAME} with [[models]] entries."
        ) from exc


def load_catalog() -> ModelCatalogData:
    """Load and validate the model catalog with simple mtime-based caching."""
    global _CACHE
    path = _catalog_path()
    mtime_ns = path.stat().st_mtime_ns

    if _CACHE and _CACHE.mtime_ns == mtime_ns:
        return _CACHE.data

    try:
        raw = _read_catalog_file()
        parsed = tomllib.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - pass through detail
        raise ValueError(
            f"Failed to parse TOML at {path}: {exc}. Please check syntax near the indicated line."
        ) from exc

    try:
        data = ModelCatalogData(**parsed)
    except ValidationError as exc:
        raise ValueError(f"Model catalog validation failed: {exc}") from exc

    _CACHE = _CatalogCache(mtime_ns=mtime_ns, data=data)
    return data


def _canonical_key(provider: str, model_id: str) -> str:
    return f"{provider.strip().lower()}::{model_id.strip()}"


def _normalization_map() -> dict[str, tuple[str, str]]:
    """Known alias → (canonical_provider, canonical_model_id).

    - Azure deployments map to OpenAI canonical models
    - Bedrock model IDs map to Anthropic canonical models
    - Dated/variant Anthropic aliases → canonical
    """
    mapping: dict[str, tuple[str, str]] = {
        # Azure → OpenAI
        "azure-openai::gpt-5": ("openai", "gpt-5"),
        "azure-openai::gpt-4.1": ("openai", "gpt-4.1"),
        "azure-openai::o4-mini": ("openai", "o4-mini"),
        "azure-openai::gpt-4o": ("openai", "gpt-4.1"),  # treat 4o under 4.1 caps here
        "azure-openai::gpt-4o-mini": ("openai", "o4-mini"),
        "azure-openai::gpt-4": ("openai", "gpt-4.1"),
        "azure-openai::gpt-3.5-turbo": ("openai", "gpt-4.1"),  # fallback caps
        # Bedrock → Anthropic
        "bedrock::anthropic.claude-3-7-sonnet-v1:0": ("anthropic", "claude-3-7-sonnet"),
        "bedrock::anthropic.claude-sonnet-4-v1:0": ("anthropic", "claude-sonnet-4"),
        "bedrock::anthropic.claude-opus-4-v1:0": ("anthropic", "claude-opus-4"),
        "bedrock::anthropic.claude-3-sonnet-20240229-v1:0": (
            "anthropic",
            "claude-3-7-sonnet",
        ),
        "bedrock::anthropic.claude-3-haiku-20240307-v1:0": (
            "anthropic",
            "claude-3-7-sonnet",
        ),
        # Anthropic dated aliases → canonical
        "anthropic::claude-3-7-sonnet-latest": ("anthropic", "claude-3-7-sonnet"),
        "anthropic::claude-sonnet-4-20250514": ("anthropic", "claude-sonnet-4"),
        "anthropic::claude-opus-4-20250514": ("anthropic", "claude-opus-4"),
    }
    return mapping


def normalize_model_id(provider: str, raw_id: str) -> tuple[str, str]:
    key = _canonical_key(provider, raw_id.lower())
    alias = _normalization_map().get(key)
    if alias:
        return alias

    # Heuristics for Azure deployments
    if provider.lower() == "azure-openai":
        lowered = raw_id.lower()
        if "gpt-5" in lowered:
            return ("openai", "gpt-5")
        if "gpt-4.1" in lowered or "gpt-4o" in lowered or "gpt-4" in lowered:
            return ("openai", "gpt-4.1")
        if "o4-mini" in lowered or "gpt-4o-mini" in lowered:
            return ("openai", "o4-mini")
        return ("openai", raw_id)  # fall back to raw under openai

    # Heuristics for Bedrock Anthropic models
    if provider.lower() == "bedrock":
        lowered = raw_id.lower()
        if "claude-3-7-sonnet" in lowered:
            return ("anthropic", "claude-3-7-sonnet")
        if "claude-sonnet-4" in lowered:
            return ("anthropic", "claude-sonnet-4")
        if "claude-opus-4" in lowered:
            return ("anthropic", "claude-opus-4")
        return ("anthropic", raw_id)

    return (provider.lower(), raw_id)


def _index_by_key(data: ModelCatalogData) -> dict[str, ModelMetadata]:
    idx: dict[str, ModelMetadata] = {}
    for m in data.models:
        idx[_canonical_key(m.provider, m.model_id)] = m
    return idx


def get_metadata(provider: str, model: str) -> ModelMetadata:
    canonical_provider, canonical_model = normalize_model_id(provider, model)
    data = load_catalog()
    idx = _index_by_key(data)
    key = _canonical_key(canonical_provider, canonical_model)
    if key not in idx:
        raise ValueError(
            f"Unknown model: {canonical_provider}/{canonical_model}. Add an entry to {CATALOG_FILENAME}."
        )
    return idx[key]


def get_limits(provider: str, model: str) -> ModelLimits:
    return get_metadata(provider, model).limits


def get_pricing(provider: str, model: str) -> PerMillionPricing:
    return get_metadata(provider, model).pricing.per_million


def get_flags(provider: str, model: str) -> ModelFlags:
    return get_metadata(provider, model).flags


def verify_catalog() -> dict[str, Any]:
    """Lightweight verification: duplicates, missing fields counts."""
    results: dict[str, Any] = {
        "total_models": 0,
        "duplicates": [],
        "providers": {},
    }

    data = load_catalog()
    results["total_models"] = len(data.models)
    seen: set[str] = set()
    provider_counts: dict[str, int] = {}

    for m in data.models:
        key = _canonical_key(m.provider, m.model_id)
        if key in seen:
            results["duplicates"].append(key)
        else:
            seen.add(key)
        provider_counts[m.provider] = provider_counts.get(m.provider, 0) + 1

        # sanity checks
        if m.flags.supports_thinking and m.limits.max_thinking is None:
            results.setdefault("issues", []).append(
                f"{key}: supports_thinking=true but no max_thinking set"
            )

    results["providers"] = provider_counts
    return results
