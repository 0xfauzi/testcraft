"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from testcraft.config import ConfigLoader, TestCraftConfig
from testcraft.config.loader import ConfigurationError


class TestTestCraftConfig:
    """Test the main TestCraftConfig model."""

    def test_default_config_creation(self):
        """Test that default configuration can be created."""
        config = TestCraftConfig()

        # Test some default values
        assert config.coverage.minimum_line_coverage == 80.0
        assert config.coverage.minimum_branch_coverage == 70.0
        assert config.style.framework == "pytest"
        assert config.environment.auto_detect is True
        assert config.cost_management.max_file_size_kb == 50
        assert config.security.block_dangerous_patterns is True
        assert config.quality.enable_quality_analysis is True
        assert config.prompt_engineering.use_2025_guidelines is True

    def test_coverage_validation_success(self):
        """Test successful coverage validation."""
        config_data = {
            "coverage": {
                "minimum_line_coverage": 80.0,
                "minimum_branch_coverage": 70.0,
                "regenerate_if_below": 60.0,
            }
        }
        config = TestCraftConfig(**config_data)
        assert config.coverage.minimum_line_coverage == 80.0

    def test_coverage_validation_failure(self):
        """Test coverage validation failures."""
        # regenerate_if_below higher than minimum_line_coverage
        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(
                coverage={"minimum_line_coverage": 70.0, "regenerate_if_below": 80.0}
            )
        assert "regenerate_if_below cannot be higher than minimum_line_coverage" in str(
            exc_info.value
        )

        # minimum_branch_coverage higher than minimum_line_coverage
        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(
                coverage={
                    "minimum_line_coverage": 60.0,
                    "minimum_branch_coverage": 80.0,
                }
            )
        assert (
            "minimum_branch_coverage cannot be higher than minimum_line_coverage"
            in str(exc_info.value)
        )

    def test_cost_validation_failure(self):
        """Test cost validation failures."""
        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(
                cost_management={
                    "cost_thresholds": {
                        "per_request_limit": 1.0,
                        "warning_threshold": 2.0,
                    }
                }
            )
        assert "warning_threshold cannot be higher than per_request_limit" in str(
            exc_info.value
        )

    def test_quality_score_validation(self):
        """Test quality score validation."""
        # Valid scores
        config = TestCraftConfig(
            quality={"minimum_quality_score": 75.0, "minimum_mutation_score": 80.0}
        )
        assert config.quality.minimum_quality_score == 75.0

        # Invalid scores
        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(quality={"minimum_quality_score": 150.0})
        assert "less than or equal to 100" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(quality={"minimum_mutation_score": -10.0})
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = TestCraftConfig()

        # Valid nested keys
        assert config.get_nested_value("coverage.minimum_line_coverage") == 80.0
        assert config.get_nested_value("style.framework") == "pytest"
        assert config.get_nested_value("environment.auto_detect") is True

        # Invalid keys should return default
        assert config.get_nested_value("invalid.key") is None
        assert config.get_nested_value("invalid.key", "default") == "default"

    def test_update_from_dict(self):
        """Test updating configuration from dictionary."""
        config = TestCraftConfig()
        original_coverage = config.coverage.minimum_line_coverage

        updates = {
            "coverage": {"minimum_line_coverage": 90.0},
            "style": {"framework": "unittest"},
        }

        updated_config = config.update_from_dict(updates)

        # Original config should be unchanged
        assert config.coverage.minimum_line_coverage == original_coverage

        # Updated config should have new values
        assert updated_config.coverage.minimum_line_coverage == 90.0
        assert updated_config.style.framework == "unittest"

        # Other values should remain unchanged
        assert updated_config.coverage.minimum_branch_coverage == 70.0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            TestCraftConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestConfigLoader:
    """Test the ConfigLoader class."""

    def test_init_with_config_file(self):
        """Test initialization with specific config file."""
        loader = ConfigLoader("custom.yml")
        assert loader.config_file == Path("custom.yml")

    def test_init_without_config_file(self):
        """Test initialization without config file."""
        loader = ConfigLoader()
        assert loader.config_file is None

    def test_load_config_no_file(self):
        """Test loading config when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            loader = ConfigLoader()
            config = loader.load_config()

            # Should return default configuration
            assert isinstance(config, TestCraftConfig)
            assert config.coverage.minimum_line_coverage == 80.0

    def test_load_config_with_yaml_file(self):
        """Test loading config from YAML file."""
        yaml_content = {
            "coverage": {
                "minimum_line_coverage": 85.0,
                "minimum_branch_coverage": 75.0,
            },
            "style": {"framework": "unittest"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name

        try:
            loader = ConfigLoader(temp_file)
            config = loader.load_config()

            assert config.coverage.minimum_line_coverage == 85.0
            assert config.coverage.minimum_branch_coverage == 75.0
            assert config.style.framework == "unittest"

        finally:
            os.unlink(temp_file)

    def test_load_config_with_empty_yaml(self):
        """Test loading config from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            loader = ConfigLoader(temp_file)
            config = loader.load_config()

            # Should use defaults when file is empty
            assert config.coverage.minimum_line_coverage == 80.0

        finally:
            os.unlink(temp_file)

    def test_load_config_with_invalid_yaml(self):
        """Test loading config from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            temp_file = f.name

        try:
            loader = ConfigLoader(temp_file)
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load_config()
            assert "Invalid YAML" in str(exc_info.value)

        finally:
            os.unlink(temp_file)

    def test_load_env_config(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE": "90",
            "TESTCRAFT_STYLE__FRAMEWORK": "unittest",
            "TESTCRAFT_ENVIRONMENT__AUTO_DETECT": "false",
            "TESTCRAFT_COST_MANAGEMENT__MAX_FILE_SIZE_KB": "100",
            "TESTCRAFT_TEST_PATTERNS__EXCLUDE": "test1.py,test2.py",
        }

        with patch.dict(os.environ, env_vars):
            loader = ConfigLoader()
            config = loader.load_config()

            assert config.coverage.minimum_line_coverage == 90.0
            assert config.style.framework == "unittest"
            assert config.environment.auto_detect is False
            assert config.cost_management.max_file_size_kb == 100
            assert "test1.py" in config.test_patterns.exclude
            assert "test2.py" in config.test_patterns.exclude

    def test_parse_env_value(self):
        """Test parsing environment variable values."""
        loader = ConfigLoader()

        # Boolean values
        assert loader._parse_env_value("true") is True
        assert loader._parse_env_value("TRUE") is True
        assert loader._parse_env_value("yes") is True
        assert loader._parse_env_value("1") is True
        assert loader._parse_env_value("false") is False
        assert loader._parse_env_value("FALSE") is False
        assert loader._parse_env_value("no") is False
        assert loader._parse_env_value("0") is False

        # Numeric values
        assert loader._parse_env_value("42") == 42
        assert loader._parse_env_value("3.14") == 3.14

        # List values
        assert loader._parse_env_value("a,b,c") == ["a", "b", "c"]
        assert loader._parse_env_value("item1, item2, item3") == [
            "item1",
            "item2",
            "item3",
        ]

        # String values
        assert loader._parse_env_value("hello") == "hello"

    def test_cli_overrides(self):
        """Test CLI argument overrides."""
        cli_overrides = {
            "coverage": {"minimum_line_coverage": 95.0},
            "style": {"framework": "unittest"},
        }

        loader = ConfigLoader()
        config = loader.load_config(cli_overrides=cli_overrides)

        assert config.coverage.minimum_line_coverage == 95.0
        assert config.style.framework == "unittest"

    def test_priority_order(self):
        """Test that CLI overrides have highest priority."""
        yaml_content = {"coverage": {"minimum_line_coverage": 70.0}}

        env_vars = {"TESTCRAFT_COVERAGE__MINIMUM_LINE_COVERAGE": "80"}

        cli_overrides = {"coverage": {"minimum_line_coverage": 90.0}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, env_vars):
                loader = ConfigLoader(temp_file)
                config = loader.load_config(cli_overrides=cli_overrides)

                # CLI should have highest priority
                assert config.coverage.minimum_line_coverage == 90.0

        finally:
            os.unlink(temp_file)

    def test_config_caching(self):
        """Test that configuration is cached."""
        loader = ConfigLoader()

        config1 = loader.load_config()
        config2 = loader.load_config()

        # Should return the same cached instance
        assert config1 is config2

        # Force reload should create new instance
        config3 = loader.load_config(reload=True)
        assert config1 is not config3

    def test_validation_error_handling(self):
        """Test handling of validation errors."""
        invalid_config = {
            "coverage": {"minimum_line_coverage": "invalid"}  # Should be float
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name

        try:
            loader = ConfigLoader(temp_file)
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load_config()
            assert "Configuration validation failed" in str(exc_info.value)

        finally:
            os.unlink(temp_file)

    def test_create_sample_config(self):
        """Test creating sample configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test.yml"

            loader = ConfigLoader()
            result_path = loader.create_sample_config(config_path)

            assert result_path == config_path
            assert config_path.exists()

            # Verify content is valid YAML
            with open(config_path) as f:
                content = f.read()
                assert "TestCraft Configuration" in content
                assert "test_patterns:" in content
                assert "coverage:" in content

    def test_validate_config(self):
        """Test configuration validation."""
        loader = ConfigLoader()

        # Valid config should not raise
        valid_config = {"coverage": {"minimum_line_coverage": 80.0}}
        loader.validate_config(valid_config)  # Should not raise

        # Invalid config should raise
        invalid_config = {"coverage": {"minimum_line_coverage": "invalid"}}
        with pytest.raises(ConfigurationError):
            loader.validate_config(invalid_config)

    def test_get_config_summary(self):
        """Test getting configuration summary."""
        loader = ConfigLoader()
        summary = loader.get_config_summary()

        assert "config_file" in summary
        assert "coverage_thresholds" in summary
        assert "test_framework" in summary
        assert "environment_manager" in summary
        assert "cost_limits" in summary
        assert "quality_analysis" in summary
        assert "mutation_testing" in summary

        # Check specific values
        assert summary["test_framework"] == "pytest"
        assert summary["coverage_thresholds"]["minimum_line"] == 80.0

    def test_default_config_files_search(self):
        """Test searching for default configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create one of the default config files
            config_content = {"coverage": {"minimum_line_coverage": 85.0}}
            with open(".testcraft.yml", "w") as f:
                yaml.dump(config_content, f)

            loader = ConfigLoader()
            config = loader.load_config()

            assert config.coverage.minimum_line_coverage == 85.0

    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        loader = ConfigLoader()

        base = {"a": 1, "b": {"c": 2, "d": 3}}

        updates = {"b": {"d": 4, "e": 5}, "f": 6}

        result = loader._deep_merge(base, updates)

        expected = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}

        assert result == expected


def test_load_config_convenience_function():
    """Test the convenience load_config function."""
    from testcraft.config.loader import load_config

    config = load_config()
    assert isinstance(config, TestCraftConfig)
    assert config.coverage.minimum_line_coverage == 80.0


class TestConfigIntegration:
    """Integration tests for the configuration system."""

    def test_full_config_loading_cycle(self):
        """Test complete configuration loading with all sources."""
        # Create YAML config
        yaml_content = {
            "coverage": {"minimum_line_coverage": 75.0},
            "style": {"framework": "pytest"},
        }

        # Environment variables
        env_vars = {
            "TESTCRAFT_COVERAGE__MINIMUM_BRANCH_COVERAGE": "65",
            "TESTCRAFT_ENVIRONMENT__AUTO_DETECT": "false",
        }

        # CLI overrides
        cli_overrides = {"cost_management": {"max_file_size_kb": 75}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, env_vars):
                loader = ConfigLoader(temp_file)
                config = loader.load_config(cli_overrides=cli_overrides)

                # YAML values
                assert config.coverage.minimum_line_coverage == 75.0
                assert config.style.framework == "pytest"

                # Environment values
                assert config.coverage.minimum_branch_coverage == 65.0
                assert config.environment.auto_detect is False

                # CLI values
                assert config.cost_management.max_file_size_kb == 75

                # Default values (not overridden)
                assert config.quality.enable_quality_analysis is True

        finally:
            os.unlink(temp_file)
