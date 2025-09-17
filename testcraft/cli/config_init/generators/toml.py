"""TOML configuration generator with dynamic model-based generation."""

import typing
from typing import Any

from pydantic import BaseModel

from ....config.models import TestCraftConfig


class TOMLGenerator:
    """Generate TOML configuration dynamically from current models."""

    def generate_minimal_config(self) -> dict[str, Any]:
        """Generate minimal configuration content dynamically from current models."""
        # Create default config instance
        config = TestCraftConfig()

        # Extract minimal settings using model data
        minimal_config = {
            "llm": {
                "default_provider": config.llm.default_provider,
                "openai_model": config.llm.openai_model,
                "anthropic_model": config.llm.anthropic_model,
                "temperature": config.llm.temperature,
            },
            "generation": {
                "test_framework": config.generation.test_framework,
                "batch_size": config.generation.batch_size,
                "coverage_threshold": config.generation.coverage_threshold,
                "enable_refinement": config.generation.enable_refinement,
                "refine": {
                    "enable": config.generation.refine.enable,
                    "annotate_failed_tests": config.generation.refine.annotate_failed_tests,
                }
            },
        }

        return minimal_config

    def generate_comprehensive_config(
        self, comment_unused: bool = True, overrides: dict[str, Any] | None = None
    ) -> str:
        """Generate comprehensive TOML configuration dynamically from current models.

        Args:
            comment_unused: When True, emit every option commented out by default.
            overrides: A dot-keyed mapping of config keys to values to activate (uncommented).
        """
        config = TestCraftConfig()
        
        toml_lines = []
        toml_lines.append("# TestCraft Configuration (TOML)")
        toml_lines.append("# Complete configuration with all available options and detailed comments")
        toml_lines.append("# Generated dynamically from current TestCraft models")
        toml_lines.append("#")
        toml_lines.append("# Configuration Schema:")
        toml_lines.append("# - [test_patterns] - File discovery patterns and test discovery settings")
        toml_lines.append("# - [llm] - Large Language Model configuration")
        toml_lines.append("# - [generation] - Test generation behavior (includes budgets, refinement)")
        toml_lines.append("# - [cost_management] - Cost thresholds and optimization")
        toml_lines.append("# - [telemetry] - Observability and telemetry configuration")
        toml_lines.append("# - [evaluation] - Test evaluation harness (optional)")
        toml_lines.append("# - [environment] - Environment detection and management")
        toml_lines.append("# - [logging] - Logging behavior configuration")
        toml_lines.append("# - [ui] - User interface behavior")
        toml_lines.append("# - [planning] - Test planning configuration")
        toml_lines.append("#")
        toml_lines.append("# NOTE: Deprecated sections (style, coverage, quality) are no longer generated")
        toml_lines.append("# Use generation.test_framework instead of style.framework")
        toml_lines.append("")
        
        # Normalize overrides to dot-key map
        overrides = overrides or {}
        flat_overrides = self._flatten_overrides_dict(overrides)
        
        # Generate sections dynamically from the model
        self._add_model_section_to_toml(
            config,
            toml_lines,
            section_prefix="",
            section_title="TestCraft Configuration",
            comment_unused=comment_unused,
            overrides=flat_overrides,
        )
        
        # Add credential information at the end
        toml_lines.extend([
            "",
            "# =============================================================================",
            "# API CREDENTIALS",
            "# =============================================================================",
            "# API credentials are loaded from environment variables for security.",
            "# Set these environment variables in your system or .env file:",
            "#",
            "# OpenAI: OPENAI_API_KEY",
            "# Anthropic: ANTHROPIC_API_KEY",
            "# Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",
            "# AWS Bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION",
            "#",
            "# Custom endpoints (optional):",
            "# AZURE_OPENAI_ENDPOINT, OLLAMA_BASE_URL",
        ])
        
        return "\n".join(toml_lines)

    def _add_model_section_to_toml(
        self, 
        model: BaseModel, 
        toml_lines: list[str], 
        section_prefix: str, 
        section_title: str, 
        comment_unused: bool = True, 
        overrides: dict[str, Any] | None = None
    ) -> None:
        """Add a model section to TOML lines with proper formatting and comments.

        If comment_unused is True, each option is emitted as a commented line unless
        a corresponding override is provided in overrides (dot-key form), in which case
        the option is emitted as an active (uncommented) line with the override value.
        """
        model_fields = model.model_fields
        model_data = model.model_dump()
        overrides = overrides or {}
        
        # Add section header
        if section_prefix:
            toml_lines.extend([
                "",
                "# " + "=" * 77,
                f"# {section_title.upper()}",
                "# " + "=" * 77,
                "",
                f"[{section_prefix}]"
            ])
        
        # Deprecated sections to skip in TOML generation
        deprecated_sections = {'style', 'coverage', 'quality', 'context_enrichment'}
        
        # Partition fields into non-nested and nested to avoid redeclaring parent tables
        non_nested_fields: list[tuple[str, Any, Any, str]] = []
        nested_fields: list[tuple[str, Any, Any, str]] = []

        for field_name, field_info in model_fields.items():
            field_value = model_data.get(field_name)
            field_description = field_info.description or f"Configuration for {field_name}"

            # Skip deprecated sections entirely in TOML generation
            if field_name in deprecated_sections or 'DEPRECATED' in field_description:
                continue

            # Check if this is a nested BaseModel
            annotation = field_info.annotation
            origin = typing.get_origin(annotation)

            if origin is typing.Union:
                args = typing.get_args(annotation)
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            is_nested_model = (
                isinstance(annotation, type)
                and issubclass(annotation, BaseModel)
                and isinstance(field_value, dict)
            )

            if is_nested_model:
                nested_fields.append((field_name, field_info, field_value, field_description))
            else:
                non_nested_fields.append((field_name, field_info, field_value, field_description))

        # If this is the context categories table, add a general consequences note
        if section_prefix.endswith("generation.context_categories"):
            toml_lines.append("# Enabling a category includes that context in prompts; disabling reduces prompt size")
            toml_lines.append("# Consequences: Turning categories off lowers cost/latency but may omit useful signals.")
            toml_lines.append("")

        # First, write all non-nested fields under the current section
        for field_name, _field_info, field_value, field_description in non_nested_fields:
            self._add_field_to_toml(
                toml_lines, section_prefix, field_name, field_value, field_description,
                comment_unused, overrides
            )

        # Then, write nested tables
        for field_name, _field_info, _field_value, field_description in nested_fields:
            nested_section = f"{section_prefix}.{field_name}" if section_prefix else field_name
            nested_model = getattr(model, field_name)
            self._add_model_section_to_toml(
                nested_model,
                toml_lines,
                nested_section,
                field_description,
                comment_unused=comment_unused,
                overrides=overrides,
            )

    def _add_field_to_toml(
        self, 
        toml_lines: list[str],
        section_prefix: str,
        field_name: str,
        field_value: Any,
        field_description: str,
        comment_unused: bool,
        overrides: dict[str, Any]
    ) -> None:
        """Add a single field to TOML with proper formatting."""
        # Special handling for prompt budgets and section caps
        is_prompt_budgets_table = section_prefix.endswith("generation.prompt_budgets")

        # Add field description comment
        toml_lines.append(f"# {field_description}")

        # Add consequences notes for key prompt budget fields
        if is_prompt_budgets_table and field_name == "per_item_chars":
            toml_lines.append("# Consequences: Higher allows richer items but may reduce the number of items that fit.")
            toml_lines.append("# Lower keeps items terse and increases diversity but may truncate useful detail.")
        if is_prompt_budgets_table and field_name == "total_chars":
            toml_lines.append("# Consequences: Higher overall budget improves recall; increases cost and risk of longer prompts.")
            toml_lines.append("# Lower forces tighter selection; safer and cheaper but may omit helpful context.")

        # Render section_caps as a dedicated nested table with per-key explanations
        if (
            is_prompt_budgets_table
            and field_name == "section_caps"
            and isinstance(field_value, dict)
        ):
            self._add_section_caps_table_to_toml(toml_lines, section_prefix, field_value, comment_unused, overrides)
            toml_lines.append("")
            return

        # Determine if this key is overridden (active) or should remain commented
        full_key = f"{section_prefix}.{field_name}" if section_prefix else field_name
        has_override = full_key in overrides
        value_to_write = overrides.get(full_key, field_value)
        comment_prefix = "" if has_override or not comment_unused else "# "

        # Default rendering logic
        if isinstance(value_to_write, str):
            escaped_value = value_to_write.replace('"', '\\"')
            toml_lines.append(f'{comment_prefix}{field_name} = "{escaped_value}"')
        elif isinstance(value_to_write, bool):
            toml_lines.append(f'{comment_prefix}{field_name} = {str(value_to_write).lower()}')
        elif isinstance(value_to_write, (int, float)):
            toml_lines.append(f'{comment_prefix}{field_name} = {value_to_write}')
        elif isinstance(value_to_write, list):
            if not value_to_write:
                toml_lines.append(f'{comment_prefix}{field_name} = []')
            elif all(isinstance(item, str) for item in value_to_write):
                escaped_items = [item.replace('"', '\\"') for item in value_to_write]
                formatted_list = '[' + ', '.join(f'"{item}"' for item in escaped_items) + ']'
                toml_lines.append(f'{comment_prefix}{field_name} = {formatted_list}')
            else:
                formatted_list = '[' + ', '.join(str(item) for item in value_to_write) + ']'
                toml_lines.append(f'{comment_prefix}{field_name} = {formatted_list}')
        elif isinstance(value_to_write, dict):
            if value_to_write:
                dict_items = []
                for k, v in value_to_write.items():
                    if v is None:
                        continue
                    elif isinstance(v, str):
                        escaped_v = v.replace('"', '\\"')
                        dict_items.append(f'"{k}" = "{escaped_v}"')
                    elif isinstance(v, bool):
                        dict_items.append(f'"{k}" = {str(v).lower()}')
                    else:
                        dict_items.append(f'"{k}" = {v}')
                if dict_items:
                    toml_lines.append(f'{comment_prefix}{field_name} = {{ {", ".join(dict_items)} }}')
                else:
                    toml_lines.append(f'{comment_prefix}{field_name} = {{}}')
            else:
                toml_lines.append(f'{comment_prefix}{field_name} = {{}}')
        elif value_to_write is None:
            # Always comment out optional/None fields
            toml_lines.append(f'# {field_name} = null  # Optional field')
        else:
            toml_lines.append(f'{comment_prefix}{field_name} = {value_to_write}')

        toml_lines.append("")

    def _add_section_caps_table_to_toml(
        self, 
        toml_lines: list[str], 
        section_prefix: str, 
        caps: dict[str, Any], 
        comment_unused: bool = True, 
        overrides: dict[str, Any] | None = None
    ) -> None:
        """Emit a dedicated [..section_caps] table with per-key explanations and consequences."""
        overrides = overrides or {}
        # Header and guidance
        toml_lines.append("# Per-section item limits used when assembling LLM context.")
        toml_lines.append("# Increasing a cap shifts budget toward that section; total_chars still applies.")
        toml_lines.append(f"[{section_prefix}.section_caps]")

        # Stable order for readability
        ordered_keys = [
            "snippets", "neighbors", "test_exemplars", "contracts", "deps_config_fixtures",
            "coverage_hints", "callgraph", "error_paths", "usage_examples",
            "pytest_settings", "side_effects", "path_constraints",
        ]

        explanations = {
            "snippets": "Short, symbol-aware code/doc snippets. Higher = more variety; too high can dilute signal.",
            "neighbors": "Import-graph neighbor files (header + slice). Higher adds periphery; increases prompt size.",
            "test_exemplars": "Summaries mined from existing tests. Higher = more examples; may crowd other context.",
            "contracts": "API contract summaries (signature/params/returns/raises/invariants). Higher clarifies spec; costs budget.",
            "deps_config_fixtures": "Env/config keys, DB/HTTP clients, pytest fixtures. Usually 1–2 is enough.",
            "coverage_hints": "Per-file coverage hints (if wired). Keep small; harmless when empty.",
            "callgraph": "Call-graph/import edges + nearby files. Higher adds topology; can add noise.",
            "error_paths": "Exception types from docs/AST. Higher emphasizes error handling; reduces room elsewhere.",
            "usage_examples": "High-quality usage snippets. Higher helps synthesis; too high may overfit to examples.",
            "pytest_settings": "Key pytest ini options. 1 is typically sufficient.",
            "side_effects": "Detected side-effect boundaries (fs/env/network). Higher reveals risks; may add noise.",
            "path_constraints": "Branch/condition summaries. Higher improves logic coverage; can crowd prompt.",
        }

        for key in ordered_keys:
            if key in explanations:
                toml_lines.append(f"# {explanations[key]}")
            value = caps.get(key)
            if value is None:
                continue
            full_key = f"{section_prefix}.section_caps.{key}"
            has_override = full_key in overrides
            value_to_write = overrides.get(full_key, value)
            comment_prefix = "" if has_override or not comment_unused else "# "
            toml_lines.append(f"{comment_prefix}{key} = {value_to_write}")

    def _flatten_overrides_dict(self, data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a nested dict into dot-key mapping for overrides."""
        flat: dict[str, Any] = {}
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_overrides_dict(v, key))
            else:
                flat[key] = v
        return flat

    def format_config_content(self, config_dict: dict[str, Any]) -> str:
        """Format configuration dictionary to TOML format."""
        try:
            import tomli_w
            return tomli_w.dumps(config_dict)
        except ImportError:
            # Fallback to manual TOML generation
            return self._dict_to_simple_toml(config_dict)

    def _dict_to_simple_toml(self, data: dict[str, Any], prefix: str = "") -> str:
        """Convert dict to simple TOML format (fallback)."""
        lines = []

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                lines.append(f"\n[{full_key}]")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        lines.append(f'{sub_key} = "{sub_value}"')
                    else:
                        lines.append(f"{sub_key} = {sub_value}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, list):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f"{key} = {value}")

        return "\n".join(lines)
