"""
Integration tests for generation services.

This module contains integration tests that verify the interaction between
multiple generation services working together.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testcraft.application.generation.config import GenerationConfig
from testcraft.application.generation.services.batch_executor import BatchExecutor
from testcraft.application.generation.services.content_builder import ContentBuilder
from testcraft.application.generation.services.context_assembler import ContextAssembler
from testcraft.application.generation.services.coverage_evaluator import CoverageEvaluator
from testcraft.application.generation.services.plan_builder import PlanBuilder
from testcraft.application.generation.services.state_discovery import StateSyncDiscovery
from testcraft.domain.models import TestGenerationPlan


class TestGenerationServicesIntegration:
    """Integration test cases for generation services working together."""

    @pytest.fixture
    def all_mocked_services(
        self, 
        mock_context_port,
        mock_parser_port, 
        mock_coverage_port,
        mock_state_port,
        mock_file_discovery_service,
        mock_telemetry_port,
        default_config
    ):
        """Create all generation services with mocked dependencies."""
        telemetry_port, _ = mock_telemetry_port
        
        services = {
            'state_discovery': StateSyncDiscovery(
                mock_state_port, 
                mock_file_discovery_service, 
                telemetry_port
            ),
            'plan_builder': PlanBuilder(
                mock_parser_port,
                mock_file_discovery_service,
                telemetry_port
            ),
            'content_builder': ContentBuilder(mock_parser_port),
            'context_assembler': ContextAssembler(
                mock_context_port, 
                mock_parser_port, 
                default_config
            ),
            'coverage_evaluator': CoverageEvaluator(
                mock_coverage_port, 
                telemetry_port
            ),
            'batch_executor': BatchExecutor(telemetry_port),
        }
        return services

    def test_service_instantiation_integration(self, all_mocked_services):
        """Test that all services can be instantiated together without conflicts."""
        # Verify all services are created
        assert len(all_mocked_services) == 6
        
        # Verify each service is properly instantiated
        for service_name, service in all_mocked_services.items():
            assert service is not None
            assert hasattr(service, '__class__')

    def test_state_discovery_to_plan_builder_flow(
        self, all_mocked_services, mock_file_discovery_service, mock_parser_port
    ):
        """Test the flow from state discovery to plan building."""
        state_discovery = all_mocked_services['state_discovery']
        plan_builder = all_mocked_services['plan_builder']
        
        # Setup mocks for discovery
        mock_file_discovery_service.filter_existing_files.return_value = ["test.py"]
        mock_parser_port.parse_file.return_value = {
            "ast": None,
            "source_lines": ["def test_func(): pass"],
            "elements": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # State discovery phase
            discovery_result = state_discovery.sync_and_discover(
                project_path, 
                ["test.py"]
            )
            
            # Plan building phase
            if discovery_result["files"]:
                source_files = discovery_result["files"]
                plans = plan_builder.build_plans(source_files)
                
                # Verify integration
                assert isinstance(plans, list)

    def test_content_builder_context_assembler_integration(
        self, all_mocked_services, mock_parser_port, mock_context_port
    ):
        """Test integration between content builder and context assembler."""
        content_builder = all_mocked_services['content_builder']
        context_assembler = all_mocked_services['context_assembler']
        
        # Setup mocks
        mock_parser_port.parse_file.return_value = {
            "ast": None,
            "source_lines": ["def example(): pass"]
        }
        mock_context_port.retrieve.return_value = {"results": []}
        
        # Create a test plan
        mock_plan = MagicMock(spec=TestGenerationPlan)
        mock_plan.elements_to_test = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test.py"
            source_path.write_text("def example(): pass")
            
            # Test content building
            content = content_builder.build_code_content(mock_plan, source_path)
            assert isinstance(content, str)
            
            # Test context assembly (this would normally use the content)
            context_result = context_assembler.context_for_generation(source_path, mock_plan)
            
            # Context result can be None or string depending on implementation
            assert context_result is None or isinstance(context_result, str)

    @pytest.mark.asyncio
    async def test_batch_executor_with_mock_generation(self, all_mocked_services):
        """Test batch executor with a mock generation function."""
        batch_executor = all_mocked_services['batch_executor']
        
        # Create simple test plans
        mock_plan1 = MagicMock(spec=TestGenerationPlan)
        mock_plan2 = MagicMock(spec=TestGenerationPlan)
        
        plans = [mock_plan1, mock_plan2]
        
        # Mock generation function
        async def mock_generation_fn(plan):
            from testcraft.domain.models import GenerationResult
            return GenerationResult(
                file_path="test.py",
                content="# Generated test",
                success=True,
                error_message=None,
            )
        
        # Execute in batches
        results = await batch_executor.run_in_batches(plans, 2, mock_generation_fn)
        
        # Verify results
        assert len(results) == 2
        assert all(r.success for r in results)
