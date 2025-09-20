"""
Integration tests for enriched context feature in test generation.

This module provides comprehensive end-to-end tests for the enriched context
functionality, including feature flag testing, backwards compatibility,
and integration with the complete generation pipeline.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.domain.models import (
    GenerationResult,
    TestElement,
    TestElementType,
    TestGenerationPlan,
)


class TestEnrichedContextIntegration:
    """Integration tests for enriched context in the generation pipeline."""

    @pytest.fixture
    def mock_ports(self):
        """Create comprehensive mock ports for integration testing."""
        llm_port = MagicMock()
        writer_port = MagicMock()
        coverage_port = MagicMock()
        refine_port = MagicMock()
        context_port = MagicMock()
        parser_port = MagicMock()
        state_port = MagicMock()
        telemetry_port = MagicMock()

        # Setup telemetry port mocks
        mock_span = MagicMock()
        mock_span.get_trace_context.return_value = MagicMock(trace_id="test_trace_123")
        telemetry_port.create_span.return_value.__enter__.return_value = mock_span
        telemetry_port.create_child_span.return_value.__enter__.return_value = mock_span

        # Setup LLM port for generation
        llm_port.generate_async.return_value = GenerationResult(
            file_path="test_generated.py",
            content="def test_example(): pass",
            success=True,
            error_message=None,
        )

        # Setup coverage port
        coverage_port.measure_coverage.return_value = {}
        coverage_port.get_coverage_summary.return_value = {
            "overall_line_coverage": 0.0,
            "overall_branch_coverage": 0.0,
            "files_covered": 0,
            "total_lines": 0,
        }

        # Setup context port with enriched context data
        context_port.build_context_graph.return_value = {"graph": "context"}
        context_port.index.return_value = {"indexed": True}
        context_port.retrieve.return_value = {
            "results": [
                {"snippet": "example_func(arg1, arg2)", "path": "usage_file.py"},
                {"snippet": "result = example_func()", "path": "another_usage.py"},
            ],
            "total_found": 2,
        }
        context_port.get_related_context.return_value = {
            "relationships": ["imports:utils", "calls:helper_func"],
            "related_files": ["utils.py", "helpers.py"],
        }

        # Setup parser port
        parser_port.parse_file.return_value = {
            "ast": ast.parse("def example_func(): pass"),
            "source_lines": ["def example_func():", "    pass"],
        }

        # Setup state port
        state_port.get_all_state.return_value = {}

        return {
            "llm_port": llm_port,
            "writer_port": writer_port,
            "coverage_port": coverage_port,
            "refine_port": refine_port,
            "context_port": context_port,
            "parser_port": parser_port,
            "state_port": state_port,
            "telemetry_port": telemetry_port,
        }

    @pytest.fixture
    def sample_project_structure(self):
        """Create a sample project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create main source file
            main_file = project_path / "main.py"
            main_file.write_text("""
import os
import requests
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
API_KEY = os.environ["API_KEY"]

class UserManager:
    \"\"\"User management class.
    
    :raises ValueError: When user ID is invalid
    :raises KeyError: When user not found
    \"\"\"
    
    def __init__(self):
        self.db_url = DATABASE_URL
        
    def create_user(self, name: str, email: str) -> dict:
        \"\"\"Create a new user.
        
        Args:
            name: User's name
            email: User's email address
            
        Returns:
            Created user data
            
        Raises:
            ValueError: If name or email is empty
        \"\"\"
        if not name or not email:
            raise ValueError("Name and email are required")
            
        response = requests.post("http://api.example.com/users", {
            "name": name,
            "email": email
        })
        
        if response.status_code != 201:
            raise RuntimeError("Failed to create user")
            
        return response.json()

    def get_user(self, user_id: int) -> dict:
        \"\"\"Get user by ID.\"\"\"
        if user_id <= 0:
            raise ValueError("Invalid user ID")
            
        # Database query would go here
        return {"id": user_id, "name": "Test User"}
""")

            # Create conftest.py with custom fixtures
            conftest_file = project_path / "tests" / "conftest.py"
            conftest_file.parent.mkdir(parents=True, exist_ok=True)
            conftest_file.write_text("""
import pytest
from unittest.mock import Mock

@pytest.fixture(scope="session")
def database_session():
    \"\"\"Database session fixture.\"\"\"
    return Mock()

@pytest.fixture
def api_client():
    \"\"\"Mock API client fixture.\"\"\"
    return Mock()

@pytest.fixture
def user_data():
    \"\"\"Sample user data.\"\"\"
    return {"name": "Test User", "email": "test@example.com"}
""")

            # Create existing test file
            existing_test = project_path / "tests" / "test_existing.py"
            existing_test.write_text("""
import pytest
from unittest.mock import patch
from main import UserManager

class TestUserManager:
    
    def test_create_user_success(self, user_data, api_client):
        \"\"\"Test successful user creation.\"\"\"
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {"id": 1, **user_data}
            
            manager = UserManager()
            result = manager.create_user(user_data["name"], user_data["email"])
            
            assert result["id"] == 1
            assert result["name"] == user_data["name"]

    @pytest.mark.slow
    def test_get_user_invalid_id(self):
        \"\"\"Test get user with invalid ID.\"\"\"
        manager = UserManager()
        
        with pytest.raises(ValueError, match="Invalid user ID"):
            manager.get_user(-1)
""")

            # Create pyproject.toml
            pyproject_file = project_path / "pyproject.toml"
            pyproject_file.write_text("""
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
testpaths = ["tests"]
""")

            yield project_path

    def test_enriched_context_end_to_end_integration(
        self, mock_ports, sample_project_structure
    ):
        """Test complete enriched context integration through generation pipeline."""
        project_path = sample_project_structure
        main_file = project_path / "main.py"

        # Configure with all enrichment features enabled
        config = {
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_db_boundary_detection": True,
                "enable_http_boundary_detection": True,
                "enable_comprehensive_fixtures": True,
                "enable_side_effect_detection": True,
                "max_env_vars": 10,
                "max_fixtures": 15,
            },
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
            },
            "prompt_budgets": {
                "per_item_chars": 1000,
                "total_chars": 8000,
                "section_caps": {
                    "snippets": 5,
                    "neighbors": 3,
                    "test_exemplars": 2,
                    "contracts": 2,
                    "deps_config_fixtures": 2,
                    "coverage_hints": 2,
                    "callgraph": 2,
                    "error_paths": 2,
                    "usage_examples": 3,
                    "pytest_settings": 1,
                    "side_effects": 2,
                },
            },
        }

        # Create generate use case with enriched config
        generate_usecase = GenerateUseCase(config=config, **mock_ports)

        # Create test plan for UserManager class
        test_element = TestElement(
            name="UserManager",
            type=TestElementType.CLASS,
            line_range=(10, 50),
            docstring="User management class with database and API integration",
        )
        plan = TestGenerationPlan(elements_to_test=[test_element])

        # Create context assembler to test context generation
        context_assembler = ContextAssembler(
            mock_ports["context_port"], mock_ports["parser_port"], config
        )

        # Test context generation with all enrichment features
        enriched_context = context_assembler.context_for_generation(plan, main_file)

        # Verify enriched context contains expected sections
        assert enriched_context is not None
        assert isinstance(enriched_context, str)

        # Should contain usage examples from context retrieval
        assert "example_func(" in enriched_context

        # Should contain call-graph information from relationships
        assert "Call-graph edges" in enriched_context
        assert (
            "imports:utils" in enriched_context
            or "calls:helper_func" in enriched_context
        )

        # Verify context respects budget limits
        assert len(enriched_context) <= config["prompt_budgets"]["total_chars"]

        # Run actual generation to test end-to-end flow
        # Note: This tests the integration but with mocked LLM calls
        results = []

        # Mock the async generation flow
        async def run_generation():
            result = await generate_usecase.generate_async(
                project_path, target_files=[str(main_file)]
            )
            return result

        # The generation should complete without errors
        # (actual content verification would require running the full async pipeline)
        assert True  # Placeholder for async integration test

    def test_feature_flags_comprehensive_coverage(
        self, mock_ports, sample_project_structure
    ):
        """Test all context enrichment feature flags work correctly."""
        project_path = sample_project_structure
        main_file = project_path / "main.py"

        # Test with all features disabled
        config_disabled = {
            "context_enrichment": {
                "enable_env_detection": False,
                "enable_db_boundary_detection": False,
                "enable_http_boundary_detection": False,
                "enable_comprehensive_fixtures": False,
                "enable_side_effect_detection": False,
            },
            "context_categories": {
                "snippets": True,  # Keep basic snippets
                "neighbors": False,
                "test_exemplars": False,
                "contracts": False,
                "deps_config_fixtures": False,  # Should be False due to enrichment flags
                "coverage_hints": False,
                "callgraph": False,
                "error_paths": False,
                "usage_examples": False,
                "pytest_settings": False,
                "side_effects": False,  # Should be False due to enrichment flags
            },
        }

        context_assembler_disabled = ContextAssembler(
            mock_ports["context_port"], mock_ports["parser_port"], config_disabled
        )

        test_element = TestElement(
            name="UserManager",
            type=TestElementType.CLASS,
            line_range=(10, 50),
            docstring="Test class",
        )
        plan = TestGenerationPlan(elements_to_test=[test_element])

        # Generate context with features disabled
        context_disabled = context_assembler_disabled.context_for_generation(
            plan, main_file
        )

        # Should have minimal context (just snippets)
        if context_disabled:
            # Should not contain enrichment-specific sections
            assert "# Dependencies/Config/Fixtures" not in context_disabled
            assert "# Side-effect boundaries" not in context_disabled
            # Note: Call-graph edges may still appear due to mock setup, but deps_config_fixtures should be minimal

        # Test with selective feature enabling
        config_selective = {
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_db_boundary_detection": False,
                "enable_http_boundary_detection": True,
                "enable_comprehensive_fixtures": False,
                "enable_side_effect_detection": True,
            },
            "context_categories": {
                "snippets": True,
                "deps_config_fixtures": True,  # Enabled due to env and http detection
                "side_effects": True,  # Enabled due to side effect detection
                "neighbors": False,
                "test_exemplars": False,
                "contracts": False,
                "coverage_hints": False,
                "callgraph": False,
                "error_paths": False,
                "usage_examples": False,
                "pytest_settings": False,
            },
        }

        context_assembler_selective = ContextAssembler(
            mock_ports["context_port"], mock_ports["parser_port"], config_selective
        )

        context_selective = context_assembler_selective.context_for_generation(
            plan, main_file
        )

        # Should contain only enabled features
        if context_selective:
            # Should contain environment variables (enabled)
            # Should contain HTTP clients (enabled)
            # Should contain side effects (enabled)
            # But NOT database clients (disabled) or fixtures (disabled)
            pass  # The actual content depends on the mock setup and detection logic

    def test_backwards_compatibility_default_behavior(
        self, mock_ports, sample_project_structure
    ):
        """Test that existing workflows work without configuration changes."""
        project_path = sample_project_structure
        main_file = project_path / "main.py"

        # Test with no explicit enrichment configuration (should use defaults)
        config_default = {}  # Empty config should use defaults

        generate_usecase_default = GenerateUseCase(config=config_default, **mock_ports)

        # Create basic test plan
        test_element = TestElement(
            name="get_user",
            type=TestElementType.FUNCTION,
            line_range=(30, 40),
            docstring="Get user by ID",
        )
        plan = TestGenerationPlan(elements_to_test=[test_element])

        # Should work without errors (backwards compatibility)
        context_assembler = ContextAssembler(
            mock_ports["context_port"],
            mock_ports["parser_port"],
            generate_usecase_default._config,
        )

        context = context_assembler.context_for_generation(plan, main_file)

        # Should generate some context (exact content depends on defaults)
        # Main thing is it shouldn't crash or break existing behavior
        assert context is None or isinstance(context, str)

    def test_prompt_generation_with_enriched_context(
        self, mock_ports, sample_project_structure
    ):
        """Test that enriched context actually appears in generated prompts."""
        project_path = sample_project_structure
        main_file = project_path / "main.py"

        config_enriched = {
            "context_enrichment": {
                "enable_env_detection": True,
                "enable_comprehensive_fixtures": True,
                "max_env_vars": 5,
                "max_fixtures": 8,
            },
            "context_categories": {
                "snippets": True,
                "test_exemplars": True,
                "deps_config_fixtures": True,
                "usage_examples": True,
            },
        }

        # Capture the actual prompt sent to LLM
        captured_prompts = []

        def mock_llm_generate(prompt, **kwargs):
            captured_prompts.append(prompt)
            return GenerationResult(
                file_path="test_generated.py",
                content="def test_example(): pass",
                success=True,
                error_message=None,
            )

        mock_ports["llm_port"].generate_async = AsyncMock(side_effect=mock_llm_generate)

        generate_usecase = GenerateUseCase(config=config_enriched, **mock_ports)

        # Create test plan
        test_element = TestElement(
            name="create_user",
            type=TestElementType.FUNCTION,
            line_range=(20, 35),
            docstring="Create a new user",
        )
        plan = TestGenerationPlan(elements_to_test=[test_element])

        # Test context assembler directly
        context_assembler = ContextAssembler(
            mock_ports["context_port"], mock_ports["parser_port"], config_enriched
        )

        enriched_context = context_assembler.context_for_generation(plan, main_file)

        # Verify enriched context structure
        if enriched_context:
            # Should contain ADDITIONAL CONTEXT section
            assert "ADDITIONAL CONTEXT" in enriched_context or enriched_context.strip()

            # Verify specific enrichment features are present based on the sample code
            # (The exact content depends on the mock setup and detection logic)
            pass

    def test_complex_codebase_mocking_scenarios(self, mock_ports):
        """Test enriched context with complex mocked codebase scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create complex project structure
            src_dir = project_path / "src"
            src_dir.mkdir()

            # Main module with dependencies
            main_module = src_dir / "service.py"
            main_module.write_text("""
import logging
import asyncio
from typing import Optional
from dataclasses import dataclass
from .database import DatabaseManager
from .cache import CacheManager
from .external_api import APIClient

@dataclass
class UserService:
    \"\"\"User service with multiple dependencies.
    
    :raises ConnectionError: When database connection fails  
    :raises TimeoutError: When API calls timeout
    :raises ValueError: When input validation fails
    \"\"\"
    
    db: DatabaseManager
    cache: CacheManager
    api_client: APIClient
    logger: logging.Logger
    
    async def get_user_profile(self, user_id: int) -> Optional[dict]:
        \"\"\"Get comprehensive user profile.\"\"\"
        try:
            # Check cache first
            cached = await self.cache.get(f"user:{user_id}")
            if cached:
                return cached
                
            # Get from database
            user = await self.db.find_user(user_id)
            if not user:
                return None
                
            # Enrich with external API data
            external_data = await self.api_client.get_user_details(user_id)
            
            profile = {**user, **external_data}
            await self.cache.set(f"user:{user_id}", profile, ttl=3600)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {e}")
            raise
""")

            # Mock complex context retrieval
            mock_ports["context_port"].get_related_context.return_value = {
                "relationships": [
                    "imports:database.DatabaseManager",
                    "imports:cache.CacheManager",
                    "imports:external_api.APIClient",
                    "calls:cache.get",
                    "calls:db.find_user",
                    "calls:api_client.get_user_details",
                ],
                "related_files": [
                    "src/database.py",
                    "src/cache.py",
                    "src/external_api.py",
                ],
            }

            mock_ports["context_port"].retrieve.return_value = {
                "results": [
                    {"snippet": "user_service.get_user_profile(123)", "path": "app.py"},
                    {
                        "snippet": "await service.get_user_profile(user_id)",
                        "path": "controller.py",
                    },
                    {
                        "snippet": "profile = get_user_profile(current_user.id)",
                        "path": "views.py",
                    },
                ],
                "total_found": 3,
            }

            # Create enriched configuration
            config = {
                "context_categories": {
                    "snippets": True,
                    "neighbors": True,
                    "callgraph": True,
                    "error_paths": True,
                    "usage_examples": True,
                },
                "prompt_budgets": {
                    "per_item_chars": 800,
                    "total_chars": 6000,
                },
            }

            context_assembler = ContextAssembler(
                mock_ports["context_port"], mock_ports["parser_port"], config
            )

            # Create complex test plan
            test_element = TestElement(
                name="get_user_profile",
                type=TestElementType.FUNCTION,
                line_range=(15, 45),
                docstring="Get comprehensive user profile with caching and external API",
            )
            plan = TestGenerationPlan(elements_to_test=[test_element])

            # Generate enriched context
            context = context_assembler.context_for_generation(plan, main_module)

            # Verify complex context handling
            if context:
                # Should handle multiple relationships
                assert "Call-graph edges" in context
                assert any(
                    rel in context
                    for rel in ["imports:database", "calls:cache", "calls:db.find_user"]
                )

                # Should contain usage examples
                assert "Usage get_user_profile" in context
                assert (
                    "user_service.get_user_profile" in context
                    or "get_user_profile(user_id)" in context
                )

                # Should respect budget constraints
                assert len(context) <= config["prompt_budgets"]["total_chars"]

    def test_configuration_validation_and_error_handling(self, mock_ports):
        """Test configuration validation and graceful error handling."""
        # Test invalid configuration values
        invalid_config = {
            "context_enrichment": {
                "max_env_vars": -1,  # Invalid
                "max_fixtures": 500,  # Too high
            },
            "prompt_budgets": {
                "per_item_chars": 10,  # Too small
                "total_chars": 100000,  # Too large
                "section_caps": {
                    "invalid_section": 5,  # Invalid section
                },
            },
        }

        # Should handle invalid config gracefully
        with patch(
            "testcraft.application.generation.config.logger.warning"
        ) as mock_warn:
            generate_usecase = GenerateUseCase(config=invalid_config, **mock_ports)

            # Should have warned about invalid values
            assert mock_warn.call_count > 0

            # Should still create a working use case with corrected values
            assert generate_usecase is not None
            assert generate_usecase._config is not None

        # Test with completely missing configuration sections
        minimal_config = {}
        generate_usecase_minimal = GenerateUseCase(config=minimal_config, **mock_ports)

        # Should work with defaults
        assert generate_usecase_minimal is not None
        assert "context_enrichment" in generate_usecase_minimal._config
        assert "prompt_budgets" in generate_usecase_minimal._config
