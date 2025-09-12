"""
Tests for immediate refinement feature.

Tests the new immediate write-and-refine functionality that processes files
one at a time with generate → write → refine workflow.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from testcraft.application.generate_usecase import GenerateUseCase
from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.domain.models import GenerationResult


@pytest.fixture
def mock_ports():
    """Create mock ports for testing."""
    return {
        "llm_port": Mock(),
        "writer_port": Mock(),
        "coverage_port": Mock(),
        "refine_port": Mock(),
        "context_port": Mock(),
        "parser_port": Mock(),
        "state_port": Mock(),
        "telemetry_port": Mock(),
    }


@pytest.fixture
def immediate_config():
    """Configuration for immediate refinement mode."""
    return {
        "immediate_refinement": True,
        "max_refine_workers": 2,
        "keep_failed_writes": False,
        "refine_on_first_failure_only": True,
        "refinement_backoff_sec": 0.1,
        "enable_refinement": True,
        "max_refinement_iterations": 3,
        "batch_size": 5,
        "enable_context": False,
        "test_framework": "pytest",
        "coverage_threshold": 0.8,
        "enable_streaming": False,
    }


@pytest.fixture
def legacy_config():
    """Configuration for legacy mode."""
    return {
        "immediate_refinement": False,
        "enable_refinement": True,
        "max_refinement_iterations": 3,
        "batch_size": 5,
        "enable_context": False,
        "test_framework": "pytest",
        "coverage_threshold": 0.8,
        "enable_streaming": False,
    }


@pytest.mark.asyncio
class TestImmediateRefinement:
    """Test immediate refinement functionality."""

    async def test_immediate_mode_happy_path(self, mock_ports, immediate_config):
        """Test successful immediate pipeline: generate→write→refine on first try."""
        # Setup mocks
        mock_ports["llm_port"].generate_tests = AsyncMock(
            return_value={"tests": "def test_example(): pass"}
        )
        mock_ports["writer_port"].write_test_file.return_value = {"success": True, "bytes_written": 100}
        mock_ports["refine_port"].refine_from_failures.return_value = {
            "success": True,
            "refined_content": "def test_example(): pass",
        }
        
        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()
        
        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []
        
        # Create use case
        usecase = GenerateUseCase(
            config=immediate_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports
        )
        
        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {"files": ["src/example.py"]}
        
        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.measure_final.return_value = {"overall_line_coverage": 0.8}
        usecase._coverage_evaluator.calculate_delta.return_value = {"line_coverage_delta": 0.3}
        
        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        usecase._plan_builder.build_plans.return_value = [{"source_file": "src/example.py"}]
        usecase._plan_builder.get_source_path_for_plan.return_value = "src/example.py"
        
        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = "tests/test_example.py"
        
        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}
        
        # Mock pytest refiner to return successful refinement on first try
        with patch("testcraft.application.generate_usecase.PytestRefiner") as mock_pytest_refiner_class:
            mock_refiner = AsyncMock()
            mock_refiner.refine_until_pass = AsyncMock(
                return_value={
                    "success": True,
                    "iterations": 1,
                    "final_status": "passed",
                    "test_file": "tests/test_example.py",
                }
            )
            mock_pytest_refiner_class.return_value = mock_refiner
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the test
                results = await usecase.generate_tests(project_path=temp_dir)
                
                # Verify results
                assert results["success"] is True
                assert results["tests_generated"] > 0
                assert results["files_written"] > 0
                assert results["files_refined"] > 0
                
                # Verify refinement was called
                mock_refiner.refine_until_pass.assert_called_once()

    async def test_immediate_mode_syntax_invalid_no_write(self, mock_ports, immediate_config):
        """Test that invalid syntax prevents write when keep_failed_writes=False."""
        # Setup mock to return invalid Python syntax
        mock_ports["llm_port"].generate_tests = AsyncMock(
            return_value={"tests": "def test_invalid( pass"}  # Invalid syntax
        )
        
        # Writer should raise an exception due to syntax validation
        mock_ports["writer_port"].write_test_file.side_effect = Exception("Syntax error in generated code")
        
        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()
        
        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []
        
        # Create use case with keep_failed_writes=False
        config = immediate_config.copy()
        config["keep_failed_writes"] = False
        
        usecase = GenerateUseCase(
            config=config,
            file_discovery_service=mock_file_discovery,
            **mock_ports
        )
        
        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {"files": ["src/example.py"]}
        
        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.measure_final.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.calculate_delta.return_value = {"line_coverage_delta": 0.0}
        
        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        usecase._plan_builder.build_plans.return_value = [{"source_file": "src/example.py"}]
        usecase._plan_builder.get_source_path_for_plan.return_value = "src/example.py"
        
        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = "tests/test_example.py"
        
        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}

        with patch("testcraft.application.generate_usecase.PytestRefiner") as mock_pytest_refiner_class:
            mock_refiner = AsyncMock()
            mock_pytest_refiner_class.return_value = mock_refiner
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock rollback method
                with patch.object(usecase, "_rollback_failed_write", new_callable=AsyncMock) as mock_rollback:
                    # Run the test
                    results = await usecase.generate_tests(project_path=temp_dir)
                    
                    # Verify write was not successful
                    assert results["files_written"] == 0
                    
                    # Verify rollback was called for failed write
                    mock_rollback.assert_called()

    async def test_immediate_mode_refinement_iterations_exhausted(self, mock_ports, immediate_config):
        """Test refinement continues until max iterations are exhausted."""
        # Setup mocks
        mock_ports["llm_port"].generate_tests = AsyncMock(
            return_value={"tests": "def test_example(): assert False"}  # Test that will fail
        )
        mock_ports["writer_port"].write_test_file.return_value = {"success": True, "bytes_written": 100}
        
        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span) 
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()
        
        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []
        
        # Create use case
        usecase = GenerateUseCase(
            config=immediate_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports
        )
        
        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {"files": ["src/example.py"]}
        
        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.measure_final.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.calculate_delta.return_value = {"line_coverage_delta": 0.0}
        
        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        usecase._plan_builder.build_plans.return_value = [{"source_file": "src/example.py"}]
        usecase._plan_builder.get_source_path_for_plan.return_value = "src/example.py"
        
        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = "tests/test_example.py"
        
        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}

        # Mock pytest refiner to exhaust all iterations without success
        with patch("testcraft.application.generate_usecase.PytestRefiner") as mock_pytest_refiner_class:
            mock_refiner = AsyncMock()
            mock_refiner.refine_until_pass = AsyncMock(
                return_value={
                    "success": False,
                    "iterations": immediate_config["max_refinement_iterations"],
                    "final_status": "failed",
                    "test_file": "tests/test_example.py",
                    "error": "Maximum refinement iterations exceeded",
                }
            )
            mock_pytest_refiner_class.return_value = mock_refiner
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the test
                results = await usecase.generate_tests(project_path=temp_dir)
                
                # Verify refinement failed after max iterations
                assert results["files_refined"] == 0  # No successful refinements
                
                # Verify refinement was called with correct parameters
                mock_refiner.refine_until_pass.assert_called_once()
                call_args = mock_refiner.refine_until_pass.call_args
                assert call_args[0][1] == immediate_config["max_refinement_iterations"]  # max_iterations param

    async def test_concurrency_semaphore_respected(self):
        """Test that semaphore limits concurrent pytest operations."""
        from concurrent.futures import ThreadPoolExecutor
        
        # Create pytest refiner with limited concurrency
        mock_refine_port = Mock()
        mock_telemetry_port = Mock()
        mock_executor = ThreadPoolExecutor(max_workers=2)
        
        # Mock telemetry span
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_telemetry_port.create_child_span.return_value = telemetry_span
        
        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry_port,
            executor=mock_executor,
            max_concurrent_refines=1,  # Only allow 1 concurrent refine
            backoff_sec=0.1,
        )
        
        # Track concurrent calls
        concurrent_calls = []
        call_order = []
        
        async def mock_run_pytest(test_path):
            call_order.append(f"start_{test_path}")
            concurrent_calls.append(test_path)
            await asyncio.sleep(0.2)  # Simulate pytest execution time
            concurrent_calls.remove(test_path)
            call_order.append(f"end_{test_path}")
            return {"returncode": 0, "stdout": "", "stderr": ""}
        
        # Mock the run_pytest method
        with patch.object(refiner, "run_pytest", side_effect=mock_run_pytest):
            with patch.object(refiner, "format_pytest_failure_output", return_value=""):
                async def build_context(test_file, test_content):
                    return {}
                
                # Start multiple refinements concurrently
                tasks = []
                for i in range(3):
                    task = refiner.refine_until_pass(
                        f"test_{i}.py",
                        1,  # max_iterations
                        build_context,
                    )
                    tasks.append(task)
                
                # Run all tasks
                results = await asyncio.gather(*tasks)
                
                # Verify all succeeded
                for result in results:
                    assert result["success"] is True
                
                # Verify that concurrent calls never exceeded our limit
                # Due to semaphore, we should never have more than 1 concurrent call
                max_concurrent = 0
                current_concurrent = 0
                for call in call_order:
                    if call.startswith("start_"):
                        current_concurrent += 1
                        max_concurrent = max(max_concurrent, current_concurrent)
                    elif call.startswith("end_"):
                        current_concurrent -= 1
                
                assert max_concurrent == 1, f"Expected max 1 concurrent call, got {max_concurrent}"

    async def test_legacy_mode_still_works(self, mock_ports, legacy_config):
        """Test that legacy mode (immediate_refinement=False) still works."""
        # Setup mocks for legacy mode
        mock_ports["llm_port"].generate_tests = AsyncMock(
            return_value={"tests": "def test_example(): pass"}
        )
        mock_ports["writer_port"].write_test_file.return_value = {"success": True, "bytes_written": 100}
        mock_ports["refine_port"].refine_from_failures.return_value = {
            "success": True,
            "refined_content": "def test_example(): pass",
        }
        
        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_span.return_value = telemetry_span
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()
        
        # Setup other required mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []
        
        # Create use case with legacy config
        usecase = GenerateUseCase(
            config=legacy_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports
        )
        
        # Mock the service methods
        usecase._state_discovery = Mock()
        usecase._state_discovery.sync_and_discover.return_value = {"files": ["src/example.py"]}
        
        usecase._coverage_evaluator = Mock()
        usecase._coverage_evaluator.measure_initial.return_value = {"overall_line_coverage": 0.5}
        usecase._coverage_evaluator.measure_final.return_value = {"overall_line_coverage": 0.8}
        usecase._coverage_evaluator.calculate_delta.return_value = {"line_coverage_delta": 0.3}
        
        usecase._plan_builder = Mock()
        usecase._plan_builder.decide_files_to_process.return_value = ["src/example.py"]
        usecase._plan_builder.build_plans.return_value = [{"source_file": "src/example.py"}]
        usecase._plan_builder.get_source_path_for_plan.return_value = "src/example.py"
        
        usecase._content_builder = Mock()
        usecase._content_builder.build_code_content.return_value = "def example(): pass"
        usecase._content_builder.determine_test_path.return_value = "tests/test_example.py"
        
        usecase._context_assembler = Mock()
        usecase._context_assembler.context_for_generation.return_value = {}
        
        # Mock batch executor and pytest refiner
        usecase._batch_executor = Mock()
        usecase._batch_executor.run_in_batches = AsyncMock(
            return_value=[
                GenerationResult(
                    file_path="tests/test_example.py",
                    content="def test_example(): pass",
                    success=True,
                    error_message=None,
                )
            ]
        )
        
        with patch("testcraft.application.generate_usecase.PytestRefiner") as mock_pytest_refiner_class:
            mock_refiner = AsyncMock()
            mock_refiner.refine_until_pass = AsyncMock(
                return_value={
                    "success": True,
                    "iterations": 1,
                    "final_status": "passed",
                    "test_file": "tests/test_example.py",
                }
            )
            mock_pytest_refiner_class.return_value = mock_refiner
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the test
                results = await usecase.generate_tests(project_path=temp_dir)
                
                # Verify legacy mode works
                assert results["success"] is True
                assert results["tests_generated"] > 0
                
                # Verify batch executor was used (legacy mode)
                usecase._batch_executor.run_in_batches.assert_called_once()


@pytest.mark.integration
class TestImmediateRefinementIntegration:
    """Integration tests for immediate refinement."""

    async def test_state_incremental_logging(self, mock_ports, immediate_config):
        """Test that state.json contains incremental entries from immediate mode."""
        # This would be a more complex integration test that verifies
        # the state logging functionality works end-to-end
        # For now, we'll test the state recording method directly
        
        from testcraft.application.generate_usecase import GenerateUseCase
        from testcraft.domain.models import GenerationResult
        
        # Setup minimal mocks
        mock_file_discovery = Mock()
        mock_file_discovery.discover_test_files.return_value = []
        
        # Mock state port to track incremental updates
        incremental_logs = []
        def mock_update_state(key, value, merge_strategy=None):
            if key == "immediate_generation_log" and merge_strategy == "append":
                incremental_logs.append(value)
            return {"success": True}
        
        mock_ports["state_port"].update_state = mock_update_state
        
        # Setup telemetry mock
        telemetry_span = Mock()
        telemetry_span.__enter__ = Mock(return_value=telemetry_span)
        telemetry_span.__exit__ = Mock(return_value=None)
        mock_ports["telemetry_port"].create_child_span.return_value = telemetry_span
        mock_ports["telemetry_port"].record_metrics = Mock()
        
        usecase = GenerateUseCase(
            config=immediate_config,
            file_discovery_service=mock_file_discovery,
            **mock_ports
        )
        
        # Create test file result
        file_result = {
            "generation_result": GenerationResult(
                file_path="tests/test_example.py",
                content="def test_example(): pass",
                success=True,
                error_message=None,
            ),
            "write_result": {
                "success": True,
                "bytes_written": 100,
                "formatted": True,
                "error": None,
            },
            "refinement_result": {
                "success": True,
                "iterations": 2,
                "final_status": "passed",
                "error": None,
            },
            "success": True,
            "errors": [],
        }
        
        # Test the per-file state recording
        await usecase._record_per_file_state(file_result)
        
        # Verify incremental log was recorded
        assert len(incremental_logs) == 1
        log_entry = incremental_logs[0]
        
        assert log_entry["file_path"] == "tests/test_example.py"
        assert log_entry["success"] is True
        assert "timestamp" in log_entry
        assert "stages" in log_entry
        
        # Verify all stages were logged
        stages = log_entry["stages"]
        assert "generation" in stages
        assert "write" in stages  
        assert "refinement" in stages
        
        # Verify stage details
        assert stages["generation"]["success"] is True
        assert stages["write"]["success"] is True
        assert stages["refinement"]["success"] is True
        assert stages["refinement"]["iterations"] == 2

    async def test_per_file_progress_display(self):
        """Test that CLI shows detailed per-file progress for immediate mode."""
        # This would test the CLI display functionality
        # For now, we can test the display logic directly
        
        from testcraft.cli.main import _display_immediate_mode_results
        from testcraft.domain.models import GenerationResult
        from rich.console import Console
        from io import StringIO
        
        # Create mock results in immediate mode format
        results = {
            "metadata": {
                "config_used": {
                    "immediate_refinement": True
                }
            },
            "files_discovered": 2,
            "files_written": 2,
            "tests_generated": 2,
            "files_processed": 2,
            "final_coverage": {"overall_line_coverage": 0.85},
            "coverage_delta": {"line_coverage_delta": 0.15},
            "generation_results": [
                GenerationResult(
                    file_path="tests/test_file1.py",
                    content="def test_file1(): pass",
                    success=True,
                    error_message=None,
                ),
                GenerationResult(
                    file_path="tests/test_file2.py",
                    content="def test_file2(): pass", 
                    success=True,
                    error_message=None,
                ),
            ],
            "refinement_results": [
                {
                    "test_file": "tests/test_file1.py",
                    "success": True,
                    "iterations": 1,
                },
                {
                    "test_file": "tests/test_file2.py", 
                    "success": True,
                    "iterations": 2,
                },
            ],
        }
        
        # Capture console output
        test_console = Console(file=StringIO(), width=80)
        
        # Mock the UI components
        with patch("testcraft.cli.main.ui") as mock_ui:
            with patch("testcraft.cli.main.console", test_console):
                # Test the display function
                _display_immediate_mode_results(results)
                
                # Verify UI components were called
                mock_ui.create_project_summary_panel.assert_called_once()
                mock_ui.print_panel.assert_called_once()
                mock_ui.display_success.assert_called_once()
                
                # Verify console output contains per-file table
                output = test_console.file.getvalue()
                assert "Per-File Results (Immediate Mode)" in output
