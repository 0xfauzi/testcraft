"""
Tests for GenerationConfig service.

This module contains unit tests for the generation configuration service.
"""

from unittest.mock import patch

import pytest

from testcraft.application.generation.config import GenerationConfig


class TestGenerationConfig:
    """Test cases for GenerationConfig."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = GenerationConfig.get_default_config()

        assert config["batch_size"] == 5
        assert config["enable_context"] is True
        assert config["enable_refinement"] is True
        assert config["max_refinement_iterations"] == 3
        assert config["coverage_threshold"] == 0.8
        assert config["test_framework"] == "pytest"

        # Check context categories
        assert "context_categories" in config
        assert config["context_categories"]["snippets"] is True
        assert config["context_categories"]["neighbors"] is True

        # Check prompt budgets
        assert "prompt_budgets" in config
        assert config["prompt_budgets"]["per_item_chars"] == 1500
        assert config["prompt_budgets"]["total_chars"] == 10000

    def test_merge_config_empty(self):
        """Test config merging with empty overrides."""
        config = GenerationConfig.merge_config(None)
        defaults = GenerationConfig.get_default_config()
        assert config == defaults

        config = GenerationConfig.merge_config({})
        assert config == defaults

    def test_merge_config_with_overrides(self):
        """Test config merging with overrides."""
        overrides = {
            "batch_size": 10,
            "enable_context": False,
            "test_framework": "unittest",
        }

        config = GenerationConfig.merge_config(overrides)

        assert config["batch_size"] == 10
        assert config["enable_context"] is False
        assert config["test_framework"] == "unittest"
        # Unchanged defaults
        assert config["enable_refinement"] is True
        assert config["max_refinement_iterations"] == 3

    def test_merge_context_categories(self):
        """Test deep merging of context categories."""
        overrides = {
            "context_categories": {
                "snippets": False,
                "contracts": True,
            }
        }

        config = GenerationConfig.merge_config(overrides)

        # Should merge deeply
        assert config["context_categories"]["snippets"] is False
        assert config["context_categories"]["contracts"] is True
        # Unchanged defaults should still be there
        assert "neighbors" in config["context_categories"]
        assert config["context_categories"]["neighbors"] is True

    def test_context_enrichment_mapping(self):
        """Test context enrichment to context categories mapping."""
        overrides = {
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_db_boundary_detection": False,
                "enable_side_effect_detection": True,
            }
        }

        config = GenerationConfig.merge_config(overrides)

        assert config["context_categories"]["deps_config_fixtures"] is True
        assert config["context_categories"]["side_effects"] is True
        assert "context_enrichment" in config

    def test_validate_config_valid(self):
        """Test config validation with valid values."""
        config = GenerationConfig.get_default_config()
        # Should not raise
        GenerationConfig.validate_config(config)

    def test_validate_config_fixes_invalid_batch_size(self):
        """Test config validation fixes invalid batch size."""
        config = {"batch_size": -1}

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            mock_warn.assert_called_once()

        assert config["batch_size"] == 5

    def test_validate_config_fixes_invalid_coverage_threshold(self):
        """Test config validation fixes invalid coverage threshold."""
        config = {"coverage_threshold": 1.5}

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            mock_warn.assert_called_once()

        assert config["coverage_threshold"] == 0.8

    def test_validate_prompt_budgets_comprehensive(self):
        """Test comprehensive prompt budget validation."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 50,  # Too small
                "total_chars": 60000,  # Very large (exceeds 50000 threshold)
                "section_caps": {
                    "snippets": 5,
                    "invalid_section": 3,  # Invalid section
                    "contracts": -1,  # Invalid negative value
                    "neighbors": 60,  # Very large value
                }
            }
        }

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            
            # Should have multiple warnings
            assert mock_warn.call_count >= 4
            
            # Check specific warning calls - format the messages properly
            warning_messages = []
            for call in mock_warn.call_args_list:
                if len(call[0]) > 1:
                    # Format the message with its arguments like logger.warning would
                    warning_messages.append(call[0][0] % call[0][1:])
                else:
                    warning_messages.append(call[0][0])
            
            # Should warn about small per_item_chars
            assert any("per_item_chars" in msg and "using default 1500" in msg for msg in warning_messages)
            
            # Should warn about large total_chars
            assert any("Very large total_chars" in msg for msg in warning_messages)
            
            # Should warn about unknown section (includes list of valid sections)
            assert any("Unknown section_cap 'invalid_section', valid sections:" in msg for msg in warning_messages)
            
            # Should warn about invalid negative section cap
            assert any("Invalid section_cap for 'contracts'" in msg for msg in warning_messages)
            
            # Should warn about very large section cap
            assert any("Very large section_cap for 'neighbors'" in msg for msg in warning_messages)

        # Should fix invalid values
        assert config["prompt_budgets"]["per_item_chars"] == 1500

    def test_validate_prompt_budgets_consistency_check(self):
        """Test prompt budget consistency validation."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 3000,  # Large per-item
                "total_chars": 4000,     # Small total (per_item * 2 > total)
            }
        }

        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            GenerationConfig.validate_config(config)
            
            # Should warn about consistency issue
            warning_messages = [str(call) for call in mock_warn.call_args_list]
            assert any("per_item_chars" in msg and "total_chars" in msg and "truncation issues" in msg 
                      for msg in warning_messages)

    def test_validate_prompt_budgets_valid_values(self):
        """Test prompt budget validation with valid values."""
        config = {
            "prompt_budgets": {
                "per_item_chars": 1000,
                "total_chars": 5000,
                "section_caps": {
                    "snippets": 10,
                    "neighbors": 5,
                    "test_exemplars": 3,
                }
            }
        }

        # Should not raise or warn for valid configuration
        GenerationConfig.validate_config(config)
        
        # Values should remain unchanged
        assert config["prompt_budgets"]["per_item_chars"] == 1000
        assert config["prompt_budgets"]["total_chars"] == 5000
