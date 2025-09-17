"""YAML configuration generator with dynamic model-based generation."""

from typing import Any

from ....config.models import TestCraftConfig


class YAMLGenerator:
    """Generate YAML configuration dynamically from current models."""

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

    def generate_comprehensive_config(self) -> str:
        """Generate comprehensive YAML configuration dynamically from current models."""
        config = TestCraftConfig()
        config_dict = config.model_dump()
        
        try:
            import yaml
            
            # Add header comments
            header = """# TestCraft Configuration (YAML)
# Complete configuration with all available options and detailed comments
# Generated dynamically from current TestCraft models

"""
            yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)
            return header + yaml_content
            
        except ImportError:
            # Fallback if PyYAML not available
            return self._dict_to_simple_yaml(config_dict)

    def format_config_content(self, config_dict: dict[str, Any]) -> str:
        """Format configuration dictionary to YAML format."""
        try:
            import yaml
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback if PyYAML not available
            return self._dict_to_simple_yaml(config_dict)

    def _dict_to_simple_yaml(self, data: dict[str, Any], indent: int = 0) -> str:
        """Convert dict to simple YAML format (fallback when PyYAML not available)."""
        lines = []
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(self._dict_to_simple_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}:")
                for item in value:
                    if isinstance(item, str):
                        lines.append(f"{indent_str}  - \"{item}\"")
                    else:
                        lines.append(f"{indent_str}  - {item}")
            elif isinstance(value, str):
                lines.append(f"{indent_str}{key}: \"{value}\"")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        
        return "\n".join(lines)
