"""
Tests for BatchExecutor service.

This module contains unit tests for the batch execution service.
"""

import pytest

from testcraft.application.generation.services.batch_executor import BatchExecutor
from testcraft.domain.models import GenerationResult, TestGenerationPlan


class TestBatchExecutor:
    """Test cases for BatchExecutor service."""

    @pytest.fixture
    def service(self, mock_telemetry_port):
        """Create BatchExecutor service."""
        telemetry_port, _ = mock_telemetry_port
        return BatchExecutor(telemetry_port)

    @pytest.mark.asyncio
    async def test_run_in_batches(self, service, mock_telemetry_port):
        """Test running generation in batches."""
        from testcraft.domain.models import TestElement, TestElementType

        telemetry_port, mock_span = mock_telemetry_port

        # Create test plans
        element1 = TestElement(
            name="test_func1",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="Test function 1",
        )
        element2 = TestElement(
            name="test_func2",
            type=TestElementType.FUNCTION,
            line_range=(6, 10),
            docstring="Test function 2",
        )

        plan1 = TestGenerationPlan(elements_to_test=[element1])
        plan2 = TestGenerationPlan(elements_to_test=[element2])

        plans = [plan1, plan2]

        # Mock generation function
        async def mock_generation_fn(plan):
            element_name = plan.elements_to_test[0].name
            return GenerationResult(
                file_path=f"test_{element_name}.py",
                content=f"def {element_name}(): pass",
                success=True,
                error_message=None,
            )

        results = await service.run_in_batches(plans, 2, mock_generation_fn)

        # Verify results
        assert len(results) == 2
        assert all(r.success for r in results)
        assert "test_func1" in results[0].content
        assert "test_func2" in results[1].content

        # Verify telemetry
        mock_span.set_attribute.assert_called()

    @pytest.mark.asyncio
    async def test_run_in_batches_with_failures(self, service, mock_telemetry_port):
        """Test batch execution with some failures."""
        from testcraft.domain.models import TestElement, TestElementType

        telemetry_port, mock_span = mock_telemetry_port

        element = TestElement(
            name="test_func",
            type=TestElementType.FUNCTION,
            line_range=(1, 5),
            docstring="Test function",
        )
        plan = TestGenerationPlan(elements_to_test=[element])

        # Mock generation function that raises exception
        async def failing_generation_fn(plan):
            raise Exception("Generation failed")

        results = await service.run_in_batches([plan], 1, failing_generation_fn)

        # Should handle exception gracefully
        assert len(results) == 1
        assert not results[0].success
        assert "Generation failed" in results[0].error_message
