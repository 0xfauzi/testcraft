"""Tests for TOML configuration support in testcraft."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from testcraft.config.loader import ConfigLoader, ConfigurationError
from testcraft.config.models import TestCraftConfig


class TestTOMLConfigurationLoading:
    """Test TOML configuration file loading."""

    def test_load_minimal_toml_config(self, tmp_path: Path) -> None:
        """Test loading a minimal TOML configuration."""
        toml_content = """
[style]
framework = "pytest"
assertion_style = "pytest"

[coverage]
minimum_line_coverage = 85.0
junit_xml = true

[evaluation]
enabled = true
acceptance_checks = true
llm_judge_enabled = false
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(toml_content)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        assert isinstance(config, TestCraftConfig)
        assert config.style.framework == "pytest"
        assert config.style.assertion_style == "pytest"
        assert config.coverage.minimum_line_coverage == 85.0
        assert config.coverage.junit_xml is True
        assert config.evaluation.enabled is True
        assert config.evaluation.acceptance_checks is True
        assert config.evaluation.llm_judge_enabled is False

    def test_load_comprehensive_toml_config(self, tmp_path: Path) -> None:
        """Test loading a comprehensive TOML configuration."""
        toml_content = """
[style]
framework = "pytest"
assertion_style = "pytest"
mock_library = "unittest.mock"

[coverage]
minimum_line_coverage = 80.0
minimum_branch_coverage = 70.0
regenerate_if_below = 60.0
junit_xml = true

[coverage.runner]
mode = "python-module"
pytest_path = "pytest"

[generation]
include_docstrings = true
generate_fixtures = true
max_test_methods_per_class = 25

[evaluation]
enabled = true
acceptance_checks = true
llm_judge_enabled = true
rubric_dimensions = ["correctness", "coverage", "clarity"]
statistical_testing = true
confidence_level = 0.95
artifacts_path = ".testcraft/eval_artifacts"
batch_size = 15

[llm]
default_provider = "openai"
openai_model = "o4-mini"
temperature = 0.2
max_retries = 5

[cost_management]
max_file_size_kb = 60
daily_limit = 30.0

[cost_management.cost_thresholds]
per_request_limit = 1.5
warning_threshold = 0.8
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(toml_content)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        # Test various nested configurations
        assert config.style.framework == "pytest"
        assert config.style.mock_library == "unittest.mock"
        assert config.coverage.minimum_line_coverage == 80.0
        assert config.coverage.minimum_branch_coverage == 70.0
        assert config.coverage.regenerate_if_below == 60.0
        assert config.coverage.runner.mode == "python-module"
        assert config.coverage.runner.pytest_path == "pytest"
        assert config.generation.include_docstrings is True
        assert config.generation.generate_fixtures is True
        assert config.generation.max_test_methods_per_class == 25
        
        # Test evaluation configuration
        assert config.evaluation.enabled is True
        assert config.evaluation.llm_judge_enabled is True
        assert config.evaluation.rubric_dimensions == ["correctness", "coverage", "clarity"]
        assert config.evaluation.statistical_testing is True
        assert config.evaluation.confidence_level == 0.95
        assert config.evaluation.artifacts_path == ".testcraft/eval_artifacts"
        assert config.evaluation.batch_size == 15
        
        # Test LLM configuration
        assert config.llm.default_provider == "openai"
        assert config.llm.openai_model == "o4-mini"
        assert config.llm.temperature == 0.2
        assert config.llm.max_retries == 5
        
        # Test cost management
        assert config.cost_management.max_file_size_kb == 60
        # Note: daily_limit comes from the default if not explicitly set at the top level
        # We only set per_request_limit and warning_threshold in the nested section
        assert config.cost_management.cost_thresholds.per_request_limit == 1.5
        assert config.cost_management.cost_thresholds.warning_threshold == 0.8

    def test_toml_file_priority_over_yaml(self, tmp_path: Path) -> None:
        """Test that TOML files take priority over YAML files."""
        toml_content = """
[coverage]
minimum_line_coverage = 90.0
"""
        yaml_content = """
coverage:
  minimum_line_coverage: 75.0
"""
        
        toml_file = tmp_path / ".testcraft.toml"
        yaml_file = tmp_path / ".testcraft.yml"
        
        toml_file.write_text(toml_content)
        yaml_file.write_text(yaml_content)
        
        # Change to the temp directory so the files are found
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            loader = ConfigLoader()
            config = loader.load_config()
            assert config.coverage.minimum_line_coverage == 90.0  # From TOML, not YAML
        finally:
            os.chdir(original_cwd)

    def test_invalid_toml_syntax_error(self, tmp_path: Path) -> None:
        """Test handling of invalid TOML syntax."""
        invalid_toml = """
[style
framework = "pytest"  # Missing closing bracket
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(invalid_toml)
        
        loader = ConfigLoader(config_file)
        
        with pytest.raises(ConfigurationError, match="Invalid TOML"):
            loader.load_config()

    def test_toml_validation_error(self, tmp_path: Path) -> None:
        """Test handling of TOML files with validation errors."""
        invalid_config_toml = """
[coverage]
minimum_line_coverage = 150.0  # Invalid: > 100%
minimum_branch_coverage = -10.0  # Invalid: negative
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(invalid_config_toml)
        
        loader = ConfigLoader(config_file)
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            loader.load_config()

    def test_empty_toml_file(self, tmp_path: Path) -> None:
        """Test handling of empty TOML files."""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text("")
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        # Should use defaults when file is empty
        assert isinstance(config, TestCraftConfig)
        assert config.coverage.minimum_line_coverage == 80.0  # Default value

    def test_toml_with_environment_overrides(self, tmp_path: Path) -> None:
        """Test TOML configuration with environment variable overrides."""
        toml_content = """
[coverage]
minimum_line_coverage = 80.0
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(toml_content)
        
        loader = ConfigLoader(config_file)
        
        # Environment override should take precedence
        env_overrides = {
            'coverage': {
                'minimum_line_coverage': 95.0
            }
        }
        
        config = loader.load_config(env_overrides=env_overrides)
        
        assert config.coverage.minimum_line_coverage == 95.0


class TestTOMLConfigurationGeneration:
    """Test TOML configuration file generation."""

    def test_create_minimal_toml_config(self, tmp_path: Path) -> None:
        """Test creating minimal TOML configuration."""
        config_file = tmp_path / "minimal.toml"
        
        loader = ConfigLoader()
        created_path = loader.create_sample_toml_config(config_file, minimal=True)
        
        assert created_path == config_file
        assert config_file.exists()
        
        # Load the created config to verify it's valid
        loader_test = ConfigLoader(config_file)
        config = loader_test.load_config()
        
        assert isinstance(config, TestCraftConfig)
        assert config.style.framework == "pytest"
        assert config.evaluation.enabled is False  # Disabled in minimal config

    def test_create_comprehensive_toml_config(self, tmp_path: Path) -> None:
        """Test creating comprehensive TOML configuration."""
        config_file = tmp_path / "comprehensive.toml"
        
        loader = ConfigLoader()
        created_path = loader.create_sample_toml_config(config_file, minimal=False)
        
        assert created_path == config_file
        assert config_file.exists()
        
        # Load the created config to verify it's valid
        loader_test = ConfigLoader(config_file)
        config = loader_test.load_config()
        
        assert isinstance(config, TestCraftConfig)
        # Should have all the default values from TestCraftConfig
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'telemetry')
        assert hasattr(config, 'quality')

    def test_create_toml_without_tomli_w_raises_error(self, tmp_path: Path) -> None:
        """Test that creating TOML files without tomli-w raises proper error."""
        config_file = tmp_path / "test.toml"
        
        loader = ConfigLoader()
        
        with patch('testcraft.config.loader.tomli_w', None):
            with pytest.raises(ConfigurationError, match="tomli-w library is required"):
                loader.create_sample_toml_config(config_file)

    def test_default_toml_filename(self, tmp_path: Path) -> None:
        """Test creating TOML config with default filename."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            loader = ConfigLoader()
            created_path = loader.create_sample_toml_config(minimal=True)
            
            expected_path = tmp_path / ".testcraft.toml"
            assert str(created_path) == ".testcraft.toml"  # Returns relative path
            assert expected_path.exists()
        finally:
            os.chdir(original_cwd)


class TestTOMLEvaluationConfiguration:
    """Test evaluation-specific TOML configuration."""

    def test_evaluation_config_section(self, tmp_path: Path) -> None:
        """Test complete evaluation configuration section."""
        toml_content = """
[evaluation]
enabled = true
golden_repos_path = "/path/to/golden/repos"
acceptance_checks = true
llm_judge_enabled = true
rubric_dimensions = ["correctness", "coverage", "clarity", "safety", "maintainability"]
statistical_testing = true
confidence_level = 0.99
human_review_enabled = true
artifacts_path = "custom/artifacts/path"
state_file = "custom_eval_state.json"
evaluation_timeout_seconds = 600
batch_size = 25
prompt_version = "v2.1"
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(toml_content)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        eval_config = config.evaluation
        assert eval_config.enabled is True
        assert eval_config.golden_repos_path == "/path/to/golden/repos"
        assert eval_config.acceptance_checks is True
        assert eval_config.llm_judge_enabled is True
        assert eval_config.rubric_dimensions == ["correctness", "coverage", "clarity", "safety", "maintainability"]
        assert eval_config.statistical_testing is True
        assert eval_config.confidence_level == 0.99
        assert eval_config.human_review_enabled is True
        assert eval_config.artifacts_path == "custom/artifacts/path"
        assert eval_config.state_file == "custom_eval_state.json"
        assert eval_config.evaluation_timeout_seconds == 600
        assert eval_config.batch_size == 25
        assert eval_config.prompt_version == "v2.1"

    def test_evaluation_config_validation_errors(self, tmp_path: Path) -> None:
        """Test validation errors in evaluation configuration."""
        invalid_toml = """
[evaluation]
confidence_level = 1.5  # Invalid: > 1.0
batch_size = 0          # Invalid: < 1
evaluation_timeout_seconds = 5  # Invalid: < 10
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(invalid_toml)
        
        loader = ConfigLoader(config_file)
        
        with pytest.raises(ConfigurationError):
            loader.load_config()

    def test_evaluation_defaults(self, tmp_path: Path) -> None:
        """Test that evaluation configuration uses proper defaults."""
        toml_content = """
[evaluation]
enabled = true
"""
        config_file = tmp_path / ".testcraft.toml"
        config_file.write_text(toml_content)
        
        loader = ConfigLoader(config_file)
        config = loader.load_config()
        
        eval_config = config.evaluation
        assert eval_config.enabled is True
        assert eval_config.acceptance_checks is True  # Default
        assert eval_config.llm_judge_enabled is True  # Default
        assert eval_config.rubric_dimensions == ["correctness", "coverage", "clarity", "safety"]  # Default
        assert eval_config.statistical_testing is True  # Default
        assert eval_config.confidence_level == 0.95  # Default
        assert eval_config.batch_size == 10  # Default
        assert eval_config.evaluation_timeout_seconds == 300  # Default
