"""Model Catalog Loader

Loads the TestCraft model catalog TOML exactly once (cached) and provides a
typed API for looking up model limits/flags by provider and model id or alias.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import tomllib
from pydantic import BaseModel, Field, ValidationError


CATALOG_FILENAME = "model_catalog.toml"


class LimitsModel(BaseModel):
    max_context: int = Field(..., ge=1)
    default_max_output: int = Field(..., ge=1)
    max_thinking: int | None = Field(default=None, ge=0)


class FlagsModel(BaseModel):
    vision: bool | None = None
    tool_use: bool | None = None
    structured_outputs: bool | None = None
    reasoning_capable: bool | None = None


class PricingPerMillionModel(BaseModel):
    input: float | None = None
    output: float | None = None


class PricingModel(BaseModel):
    per_million: PricingPerMillionModel | None = None


class BetaModel(BaseModel):
    headers: dict[str, str] = Field(default_factory=dict)


class SourceModel(BaseModel):
    url: str | None = None
    last_verified: str | None = None
    notes: str | None = None


class CatalogEntry(BaseModel):
    provider: str
    model_id: str
    aliases: list[str] = Field(default_factory=list)
    limits: LimitsModel
    flags: FlagsModel | None = None
    beta: BetaModel | None = None
    pricing: PricingModel | None = None
    source: SourceModel | None = None


class ModelCatalog(BaseModel):
    version: str | None = None
    models: list[CatalogEntry] = Field(default_factory=list)

    def find_by_provider(self, provider: str) -> list[CatalogEntry]:
        provider_lc = provider.lower()
        return [m for m in self.models if m.provider.lower() == provider_lc]

    def resolve(self, provider: str, model_id_or_alias: str) -> CatalogEntry | None:
        provider_models = self.find_by_provider(provider)
        needle = model_id_or_alias.lower()
        for entry in provider_models:
            if entry.model_id.lower() == needle:
                return entry
            if any(alias.lower() == needle for alias in entry.aliases):
                return entry
        return None

    def providers(self) -> list[str]:
        seen: set[str] = set()
        for m in self.models:
            seen.add(m.provider)
        return sorted(seen)

    def model_ids(self, provider: str) -> list[str]:
        return sorted(e.model_id for e in self.find_by_provider(provider))


def _catalog_path() -> Path:
    return Path(__file__).parent / CATALOG_FILENAME


@lru_cache(maxsize=1)
def load_catalog() -> ModelCatalog:
    path = _catalog_path()
    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    try:
        # Normalize top-level arrays of tables [[models]] expected as "models"
        # Pydantic will validate structure and types
        return ModelCatalog(**data)
    except ValidationError as e:  # pragma: no cover - exceptional
        # Re-raise with a clearer message
        raise ValueError(f"Invalid model catalog at {path}: {e}") from e


def resolve_model(provider: str, model: str) -> CatalogEntry | None:
    """Resolve a model for a provider by id or alias (case-insensitive)."""
    return load_catalog().resolve(provider, model)


def get_providers() -> list[str]:
    return load_catalog().providers()


def get_models(provider: str) -> list[str]:
    return load_catalog().model_ids(provider)


def iter_models(provider: str | None = None) -> Iterable[CatalogEntry]:  # pragma: no cover - helper
    catalog = load_catalog()
    if provider is None:
        yield from catalog.models
    else:
        yield from catalog.find_by_provider(provider)


