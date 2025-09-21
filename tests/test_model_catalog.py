from __future__ import annotations

from pathlib import Path

from testcraft.config.model_catalog import (
    get_limits,
    get_pricing,
    get_flags,
    load_catalog,
    normalize_model_id,
    verify_catalog,
)


def test_load_catalog_ok() -> None:
    data = load_catalog()
    assert data.models, "Catalog should contain at least one model"


def test_normalize_mappings() -> None:
    prov, model = normalize_model_id("azure-openai", "my-gpt-4o-deployment")
    assert prov == "openai"
    assert model in {"gpt-4.1", "gpt-4.1"}

    prov2, model2 = normalize_model_id(
        "bedrock", "anthropic.claude-3-7-sonnet-v1:0"
    )
    assert prov2 == "anthropic"
    assert model2 == "claude-3-7-sonnet"


def test_verify_catalog_no_duplicates() -> None:
    report = verify_catalog()
    assert report["total_models"] >= 3
    assert not report.get("duplicates")


def test_accessors_work() -> None:
    limits = get_limits("openai", "gpt-4.1")
    assert limits.max_context > 0 and limits.default_max_output > 0

    pricing = get_pricing("openai", "gpt-4.1")
    assert pricing.input >= 0 and pricing.output >= 0  # type: ignore[attr-defined]

    flags = get_flags("openai", "o4-mini")
    assert hasattr(flags, "reasoning")


