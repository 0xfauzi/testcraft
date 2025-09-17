from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..enrichment_detectors import EnrichmentDetectors


def get_deps_config_fixtures(parser_port: Any, config: dict[str, Any], source_path: Path | None) -> list[str]:
    items: list[str] = []
    try:
        if source_path is None:
            return items
        enrichment_cfg = config.get("context_enrichment", {})

        # Read/parse source
        try:
            parse_result = parser_port.parse_file(source_path)
            src_text = (
                "\n".join(parse_result.get("source_lines", []))
                if hasattr(parse_result, "source_lines")
                else None
            )
            ast_tree = parse_result.get("ast") if hasattr(parse_result, "ast") else None
            if src_text is None:
                src_text = source_path.read_text(encoding="utf-8")
            if ast_tree is None and src_text:
                ast_tree = ast.parse(src_text)
        except Exception:
            src_text, ast_tree = "", None

        summary = ["# Deps/Config/Fixtures"]
        enrich = EnrichmentDetectors()

        if enrichment_cfg.get("enable_env_detection", True) and src_text:
            env_config_data = enrich.detect_env_config_usage(src_text, ast_tree)
            if env_config_data["env_vars"]:
                summary.append(f"env_vars: {env_config_data['env_vars']}")
            if env_config_data["config_keys"]:
                summary.append(f"config_keys: {env_config_data['config_keys']}")

        if (
            enrichment_cfg.get("enable_db_boundary_detection", True)
            or enrichment_cfg.get("enable_http_boundary_detection", True)
        ) and src_text:
            client_data = enrich.detect_client_boundaries(src_text, ast_tree)
            if enrichment_cfg.get("enable_db_boundary_detection", True) and client_data["database_clients"]:
                summary.append(f"db_clients: {client_data['database_clients']}")
            if enrichment_cfg.get("enable_http_boundary_detection", True) and client_data["http_clients"]:
                summary.append(f"http_clients: {client_data['http_clients']}")

        if enrichment_cfg.get("enable_comprehensive_fixtures", True):
            project_root = source_path.parent
            while project_root.parent != project_root:
                if any((project_root / marker).exists() for marker in ["pyproject.toml", "setup.py", ".git"]):
                    break
                project_root = project_root.parent
            fixture_data = enrich.discover_comprehensive_fixtures(project_root)
            fixture_lines: list[str] = []
            if fixture_data["builtin"]:
                fixture_lines.append(f"builtin: {fixture_data['builtin']}")
            if fixture_data["custom"]:
                custom_with_scope = [f"{name}({scope})" for name, scope in fixture_data["custom"].items()]
                fixture_lines.append(f"custom: {custom_with_scope}")
            if fixture_data["third_party"]:
                fixture_lines.append(f"third_party: {fixture_data['third_party']}")
            if fixture_lines:
                summary.extend(fixture_lines)

        from .pytest_settings import get_pytest_settings
        pytest_settings = get_pytest_settings(source_path)
        if pytest_settings:
            summary.append(f"pytest_settings: {pytest_settings[:5]}")

        if len(summary) > 1:
            items.append("\n".join(summary)[:600])
    except Exception:
        pass
    return items



