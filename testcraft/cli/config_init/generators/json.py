"""JSON configuration generator with dynamic model-based generation."""

import json
from typing import Any

from ....config.models import TestCraftConfig


class JSONGenerator:
    """Generate JSON configuration dynamically from current models."""

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
        """Generate comprehensive JSON configuration dynamically from current models."""
        config = TestCraftConfig()
        config_dict = config.model_dump()
        
        # JSON doesn't support comments, so we add them in a special field
        config_dict["_info"] = {
            "description": "TestCraft Configuration (JSON)",
            "note": "Generated dynamically from current TestCraft models",
            "credentials": "Set API keys in environment variables (see documentation)"
        }
        
        return json.dumps(config_dict, indent=2, sort_keys=False)

    def format_config_content(self, config_dict: dict[str, Any]) -> str:
        """Format configuration dictionary to JSON format."""
        return json.dumps(config_dict, indent=2)
