"""
Configuration management for test generation.

Centralizes configuration merging, validation, and defaults for the test
generation workflow. Handles context enrichment mapping and prompt budgets.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GenerationConfig:
    """
    Centralized configuration for test generation with validation and merging.

    Handles the complex configuration merging logic previously scattered
    throughout GenerateUseCase, including context_enrichment to context_categories
    mapping and prompt budget management.
    """

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Get the default configuration with sensible defaults."""
        return {
            "batch_size": 5,  # Number of files to process in parallel
            "enable_context": True,  # Whether to use context retrieval
            "enable_refinement": True,  # Whether to refine failed tests
            "max_refinement_iterations": 3,  # Max refinement attempts
            "coverage_threshold": 0.8,  # Coverage threshold for reporting
            "test_framework": "pytest",  # Default test framework
            "enable_streaming": False,  # Whether to use streaming LLM responses
            # Immediate refinement configuration
            "immediate_refinement": True,  # Enable immediate write-and-refine per file
            "max_refine_workers": 2,  # Limit concurrent pytest/refine to avoid flakiness
            "keep_failed_writes": False,  # Roll back broken test files
            "refine_on_first_failure_only": True,  # Stop at first pytest failure inside file
            "refinement_backoff_sec": 0.2,  # Small backoff between refinement iterations
            "disable_ruff_format": False,  # Disable Ruff formatting if it causes issues
            # Context enrichment feature flags and budgets
            "context_categories": {
                "snippets": True,
                "neighbors": True,
                "test_exemplars": True,
                "contracts": True,
                "deps_config_fixtures": True,
                "coverage_hints": True,
                "callgraph": True,
                "error_paths": True,
                "usage_examples": True,
                "pytest_settings": True,
                "side_effects": True,
                "path_constraints": True,
            },
            # Context enrichment detailed configuration
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_db_boundary_detection": True,
                "enable_http_boundary_detection": True,
                "enable_comprehensive_fixtures": True,
                "enable_side_effect_detection": True,
                "enable_coverage_hints": True,
                "enable_callgraph": True,
                "enable_error_paths": True,
                "enable_usage_examples": True,
            },
            "prompt_budgets": {
                "per_item_chars": 1500,
                "total_chars": 10000,
                "section_caps": {
                    "snippets": 10,
                    "neighbors": 5,
                    "test_exemplars": 5,
                    "contracts": 8,
                    "deps_config_fixtures": 2,
                    "coverage_hints": 3,
                    "callgraph": 3,
                    "error_paths": 3,
                    "usage_examples": 5,
                    "pytest_settings": 1,
                    "side_effects": 1,
                    "path_constraints": 3,
                },
            },
            # Context budgets for resource-intensive operations
            "context_budgets": {
                "directory_tree": {
                    "max_depth": 4,  # Maximum directory depth to traverse
                    "max_entries_per_dir": 200,  # Maximum files/dirs per directory
                    "include_py_only": True,  # Only include .py files and directories
                },
            },
        }

    @classmethod
    def merge_config(cls, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Merge configuration with defaults, handling complex nested merging.

        Args:
            overrides: Optional configuration overrides

        Returns:
            Merged configuration dictionary
        """
        config = cls.get_default_config()

        if not overrides:
            return config

        # Handle special deep merge cases first
        # Deep merge context_categories if provided
        if "context_categories" in overrides and isinstance(
            overrides["context_categories"], dict
        ):
            config["context_categories"].update(overrides["context_categories"])

        # Deep merge prompt_budgets if provided
        if "prompt_budgets" in overrides and isinstance(
            overrides["prompt_budgets"], dict
        ):
            prompt_budgets = overrides["prompt_budgets"]
            # Merge top-level prompt_budgets
            config["prompt_budgets"].update(
                {k: v for k, v in prompt_budgets.items() if k != "section_caps"}
            )
            # Deep merge section_caps
            if "section_caps" in prompt_budgets and isinstance(
                prompt_budgets["section_caps"], dict
            ):
                config["prompt_budgets"]["section_caps"].update(
                    prompt_budgets["section_caps"]
                )

        # Deep merge context_budgets if provided
        if "context_budgets" in overrides and isinstance(
            overrides["context_budgets"], dict
        ):
            context_budgets = overrides["context_budgets"]
            # Merge top-level context_budgets
            config["context_budgets"].update(
                {k: v for k, v in context_budgets.items() if k != "directory_tree"}
            )
            # Deep merge directory_tree
            if "directory_tree" in context_budgets and isinstance(
                context_budgets["directory_tree"], dict
            ):
                config["context_budgets"]["directory_tree"].update(
                    context_budgets["directory_tree"]
                )

        # Only merge keys that are relevant to generation config
        special_keys = {"context_categories", "prompt_budgets", "context_enrichment", "context_budgets"}
        valid_generation_keys = {
            "batch_size", "enable_context", "enable_refinement", "max_refinement_iterations",
            "coverage_threshold", "test_framework", "enable_streaming",
            "immediate_refinement", "max_refine_workers", "keep_failed_writes", 
            "refine_on_first_failure_only", "refinement_backoff_sec", "disable_ruff_format"
        }
        
        for key, value in overrides.items():
            if key not in special_keys:
                if key in config or key in valid_generation_keys:
                    config[key] = value
                # Silently ignore other project config keys (test_patterns, style, etc.)
                # These are valid project config but not relevant to generation

        # Handle context_enrichment mapping to context_categories
        if "context_enrichment" in overrides and isinstance(
            overrides["context_enrichment"], dict
        ):
            enrichment_cfg = overrides["context_enrichment"]
            context_cats = config["context_categories"]

            # Enable deps_config_fixtures if any detection features are enabled
            any_detection_enabled = any(
                [
                    enrichment_cfg.get("enable_env_detection", True),
                    enrichment_cfg.get("enable_db_boundary_detection", True),
                    enrichment_cfg.get("enable_http_boundary_detection", True),
                    enrichment_cfg.get("enable_comprehensive_fixtures", True),
                ]
            )
            context_cats["deps_config_fixtures"] = any_detection_enabled

            # Map side-effect detection directly
            context_cats["side_effects"] = enrichment_cfg.get(
                "enable_side_effect_detection", True
            )

            # Store the full enrichment config for use by detection methods
            config["context_enrichment"] = enrichment_cfg

        return config

    @staticmethod
    def validate_config(config: dict[str, Any]) -> None:
        """
        Validate configuration values and log warnings for invalid settings.

        Args:
            config: Configuration to validate
        """
        # Validate batch_size
        batch_size = config.get("batch_size", 5)
        if not isinstance(batch_size, int) or batch_size < 1:
            logger.warning("Invalid batch_size %s, using default 5", batch_size)
            config["batch_size"] = 5

        # Validate coverage_threshold
        threshold = config.get("coverage_threshold", 0.8)
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            logger.warning(
                "Invalid coverage_threshold %s, using default 0.8", threshold
            )
            config["coverage_threshold"] = 0.8

        # Validate max_refinement_iterations
        max_iters = config.get("max_refinement_iterations", 3)
        if not isinstance(max_iters, int) or max_iters < 1:
            logger.warning(
                "Invalid max_refinement_iterations %s, using default 3", max_iters
            )
            config["max_refinement_iterations"] = 3

        # Validate immediate refinement config
        immediate = config.get("immediate_refinement", True)
        if not isinstance(immediate, bool):
            logger.warning("Invalid immediate_refinement %s, using default True", immediate)
            config["immediate_refinement"] = True

        # Validate max_refine_workers
        max_workers = config.get("max_refine_workers", 2)
        if not isinstance(max_workers, int) or max_workers < 1:
            logger.warning(
                "Invalid max_refine_workers %s, using default 2", max_workers
            )
            config["max_refine_workers"] = 2
        elif max_workers > 8:
            logger.warning(
                "Very large max_refine_workers %s may cause resource issues", max_workers
            )

        # Validate keep_failed_writes
        keep_failed = config.get("keep_failed_writes", False)
        if not isinstance(keep_failed, bool):
            logger.warning("Invalid keep_failed_writes %s, using default False", keep_failed)
            config["keep_failed_writes"] = False

        # Validate refine_on_first_failure_only
        first_failure = config.get("refine_on_first_failure_only", True)
        if not isinstance(first_failure, bool):
            logger.warning("Invalid refine_on_first_failure_only %s, using default True", first_failure)
            config["refine_on_first_failure_only"] = True

        # Validate refinement_backoff_sec
        backoff = config.get("refinement_backoff_sec", 0.2)
        if not isinstance(backoff, (int, float)) or backoff < 0:
            logger.warning(
                "Invalid refinement_backoff_sec %s, using default 0.2", backoff
            )
            config["refinement_backoff_sec"] = 0.2
        elif backoff > 5.0:
            logger.warning(
                "Very large refinement_backoff_sec %s may slow down refinement", backoff
            )

        # Validate disable_ruff_format
        disable_ruff = config.get("disable_ruff_format", False)
        if not isinstance(disable_ruff, bool):
            logger.warning("Invalid disable_ruff_format %s, using default False", disable_ruff)
            config["disable_ruff_format"] = False

        # Validate prompt budgets with comprehensive validation
        prompt_budgets = config.get("prompt_budgets", {})
        if isinstance(prompt_budgets, dict):
            # Validate per_item_chars
            per_item = prompt_budgets.get("per_item_chars")
            if per_item is not None:
                if not isinstance(per_item, int) or per_item < 100:
                    logger.warning(
                        "Invalid per_item_chars %s (must be int >= 100), using default 1500", per_item
                    )
                    prompt_budgets["per_item_chars"] = 1500
                elif per_item > 5000:
                    logger.warning(
                        "Very large per_item_chars %s may cause performance issues", per_item
                    )

            # Validate total_chars
            total_chars = prompt_budgets.get("total_chars")
            if total_chars is not None:
                if not isinstance(total_chars, int) or total_chars < 1000:
                    logger.warning(
                        "Invalid total_chars %s (must be int >= 1000), using default 10000", total_chars
                    )
                    prompt_budgets["total_chars"] = 10000
                elif total_chars > 50000:
                    logger.warning(
                        "Very large total_chars %s may cause performance issues", total_chars
                    )

            # Validate section_caps
            section_caps = prompt_budgets.get("section_caps", {})
            if isinstance(section_caps, dict):
                valid_sections = {
                    "snippets", "neighbors", "test_exemplars", "contracts",
                    "deps_config_fixtures", "coverage_hints", "callgraph",
                    "error_paths", "usage_examples", "pytest_settings",
                    "side_effects", "path_constraints"
                }
                
                # Create lists to track sections to remove to avoid modifying dict during iteration
                sections_to_remove = []
                
                for section, cap in section_caps.items():
                    if section not in valid_sections:
                        logger.warning(
                            "Unknown section_cap '%s', valid sections: %s", 
                            section, sorted(valid_sections)
                        )
                    elif not isinstance(cap, int) or cap < 0:
                        logger.warning(
                            "Invalid section_cap for '%s': %s (must be non-negative int), removing",
                            section, cap
                        )
                        sections_to_remove.append(section)
                    elif cap > 50:
                        logger.warning(
                            "Very large section_cap for '%s': %s may cause performance issues",
                            section, cap
                        )
                
                # Remove invalid sections after iteration
                for section in sections_to_remove:
                    section_caps.pop(section, None)
                        
            # Validate consistency between per_item and total budgets
            final_per_item = prompt_budgets.get("per_item_chars", 1500)
            final_total = prompt_budgets.get("total_chars", 10000)
            if final_per_item * 2 > final_total:
                logger.warning(
                    "per_item_chars (%d) * 2 > total_chars (%d), this may cause truncation issues",
                    final_per_item, final_total
                )

        # Validate context budgets
        context_budgets = config.get("context_budgets", {})
        if isinstance(context_budgets, dict):
            # Validate directory_tree settings
            directory_tree = context_budgets.get("directory_tree", {})
            if isinstance(directory_tree, dict):
                # Validate max_depth
                max_depth = directory_tree.get("max_depth")
                if max_depth is not None:
                    if not isinstance(max_depth, int) or max_depth < 1:
                        logger.warning(
                            "Invalid directory_tree.max_depth %s (must be int >= 1), using default 4", max_depth
                        )
                        directory_tree["max_depth"] = 4
                    elif max_depth > 10:
                        logger.warning(
                            "Very large directory_tree.max_depth %s may cause performance issues", max_depth
                        )

                # Validate max_entries_per_dir
                max_entries = directory_tree.get("max_entries_per_dir")
                if max_entries is not None:
                    if not isinstance(max_entries, int) or max_entries < 10:
                        logger.warning(
                            "Invalid directory_tree.max_entries_per_dir %s (must be int >= 10), using default 200", max_entries
                        )
                        directory_tree["max_entries_per_dir"] = 200
                    elif max_entries > 1000:
                        logger.warning(
                            "Very large directory_tree.max_entries_per_dir %s may cause performance issues", max_entries
                        )

                # Validate include_py_only
                include_py_only = directory_tree.get("include_py_only")
                if include_py_only is not None and not isinstance(include_py_only, bool):
                    logger.warning(
                        "Invalid directory_tree.include_py_only %s (must be boolean), using default True", include_py_only
                    )
                    directory_tree["include_py_only"] = True
