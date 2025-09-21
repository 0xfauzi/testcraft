from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from ...config.model_catalog import (
    ModelCatalogData,
    ModelMetadata,
    load_catalog,
    normalize_model_id,
    verify_catalog,
)
import tomllib


def _iter_models(provider: str | None = None) -> List[ModelMetadata]:
    data = load_catalog()
    items = [m for m in data.models if (provider is None or m.provider == provider)]
    # Stable ordering for UX
    return sorted(items, key=lambda m: (m.provider, m.model_id))


def _format_money_per_million(v: float) -> str:
    return f"${v/1000:.3f}k/M" if v >= 1000 else f"${v:.2f}/M"


@click.group("models")
def models_group() -> None:
    """Model catalog utilities."""


@models_group.command("show")
@click.option("--provider", type=click.Choice(["openai", "anthropic"]))
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def models_show(provider: str | None, fmt: str) -> None:
    """Show models in the catalog."""
    items = _iter_models(provider)
    if fmt == "json":
        import json

        click.echo(
            json.dumps(
                [
                    {
                        "provider": m.provider,
                        "model_id": m.model_id,
                        "limits": m.limits.model_dump(),
                        "flags": m.flags.model_dump(),
                        "pricing": m.pricing.model_dump(),
                        "source": m.source.model_dump(),
                    }
                    for m in items
                ],
                indent=2,
            )
        )
        return

    # Table output
    headers = (
        "Provider/Model",
        "Ctx",
        "Out",
        "Think",
        "$/M In",
        "$/M Out",
        "Verified",
        "Source",
    )
    click.echo(" | ".join(headers))
    click.echo("-" * 96)
    for m in items:
        row = [
            f"{m.provider}/{m.model_id}",
            str(m.limits.max_context),
            str(m.limits.default_max_output),
            "yes" if (m.flags.supports_thinking and (m.limits.max_thinking or 0) > 0) else "no",
            _format_money_per_million(m.pricing.per_million.input),
            _format_money_per_million(m.pricing.per_million.output),
            m.source.last_verified,
            m.source.url[:64] + ("…" if len(m.source.url) > 64 else ""),
        ]
        click.echo(" | ".join(row))


@models_group.command("verify")
def models_verify() -> None:
    """Verify catalog for duplicates and basic consistency."""
    report = verify_catalog()
    ok = not report.get("duplicates") and not report.get("issues")
    status = "OK" if ok else "ISSUES"
    click.echo(f"Catalog verification: {status}")
    for k, v in report.items():
        click.echo(f"- {k}: {v}")
    if not ok:
        raise SystemExit(1)


def _load_external_catalog(path: Path) -> ModelCatalogData:
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    return ModelCatalogData(**data)


def _catalog_index(data: ModelCatalogData) -> Dict[Tuple[str, str], ModelMetadata]:
    return {(m.provider, m.model_id): m for m in data.models}


@models_group.command("diff")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path), required=True)
def models_diff(file_path: Path) -> None:
    """Diff current catalog against a previous TOML catalog file."""
    current = load_catalog()
    other = _load_external_catalog(file_path)

    cur = _catalog_index(current)
    oth = _catalog_index(other)

    added = sorted(set(cur.keys()) - set(oth.keys()))
    removed = sorted(set(oth.keys()) - set(cur.keys()))
    changed: List[Tuple[str, str, List[str]]] = []

    for key in sorted(set(cur.keys()) & set(oth.keys())):
        a = cur[key]
        b = oth[key]
        diffs: List[str] = []
        if a.limits.model_dump() != b.limits.model_dump():
            diffs.append("limits")
        if a.flags.model_dump() != b.flags.model_dump():
            diffs.append("flags")
        if a.pricing.model_dump() != b.pricing.model_dump():
            diffs.append("pricing")
        if diffs:
            changed.append((key[0], key[1], diffs))

    def _fmt_key(k: Tuple[str, str]) -> str:
        return f"{k[0]}/{k[1]}"

    click.echo("Added:")
    for k in added:
        click.echo(f"  + {_fmt_key(k)}")
    click.echo("Removed:")
    for k in removed:
        click.echo(f"  - {_fmt_key(k)}")
    click.echo("Changed:")
    for prov, model, fields in changed:
        click.echo(f"  * {prov}/{model}: {', '.join(fields)}")


def add_model_commands(app: click.Group) -> None:
    """Register the models command group with the main app."""
    app.add_command(models_group)


