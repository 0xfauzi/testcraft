"""
Tests for strict refinement behavior to prevent masking production bugs.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from testcraft.adapters.refine.main_adapter import RefineAdapter
from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig


class TestStrictRefinement:
    """Test strict refinement policies that prevent masking production bugs."""
    
    def test_refine_config_defaults(self):
        """Test that strict refinement flags have correct defaults."""
        config = RefineConfig()
        
        # All strict flags should default to safe values
        assert config.strict_assertion_preservation is True
        assert config.fail_on_xfail_markers is True
        assert config.allow_xfail_on_suspected_bugs is False
        assert config.report_suspected_prod_bugs is True
    
    def test_xfail_detection(self):
        """Test that XFAIL markers are correctly detected in pytest output."""
        refiner = PytestRefiner(
            refine_port=MagicMock(),
            telemetry_port=MagicMock(),
            executor=MagicMock(),
            config=RefineConfig()
        )
        
        # Test various XFAIL patterns
        assert refiner.detect_xfail_in_output("test_something XFAIL") is True
        assert refiner.detect_xfail_in_output("1 xfailed in 0.5s") is True
        assert refiner.detect_xfail_in_output("@pytest.mark.xfail") is True
        assert refiner.detect_xfail_in_output("pytest.xfail('reason')") is True
        assert refiner.detect_xfail_in_output("XPASS (unexpected pass)") is True
        assert refiner.detect_xfail_in_output("test passed normally") is False
    
    def test_production_bug_detection_fallback(self):
        """Test that basic production bug detection works as a fallback."""
        refiner = PytestRefiner(
            refine_port=MagicMock(),
            telemetry_port=MagicMock(),
            executor=MagicMock(),
            config=RefineConfig()
        )
        
        # Test that basic error patterns are detected as hints
        failure_output = """
        AssertionError: assert 42 == 50
        where 42 = calculate_answer()
        """
        result = refiner.detect_suspected_prod_bug(failure_output)
        assert result is not None
        assert result["suspected"] is True
        assert result["type"] == "potential_prod_bug"
        assert "awaiting LLM analysis" in result["description"]
        
        # Test AttributeError detection
        failure_output = """
        AttributeError: 'User' object has no attribute 'email'
        in user_service.py line 25
        """
        result = refiner.detect_suspected_prod_bug(failure_output)
        assert result is not None
        assert result["suspected"] is True
        assert result["type"] == "potential_prod_bug"
        
        # Test division by zero detection
        failure_output = """
        ZeroDivisionError: division by zero
        in calculate_ratio() line 10
        """
        result = refiner.detect_suspected_prod_bug(failure_output)
        assert result is not None
        assert result["suspected"] is True
        assert result["type"] == "potential_prod_bug"
        
        # Test that non-error output returns None
        failure_output = """
        All tests passed successfully
        """
        result = refiner.detect_suspected_prod_bug(failure_output)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_refine_until_pass_with_xfail_detection(self, tmp_path):
        """Test that refinement fails when XFAIL markers are detected."""
        test_file = tmp_path / "test_example.py"
        test_file.write_text("""
import pytest

@pytest.mark.xfail(reason="Known bug")
def test_something():
    assert False
""")
        
        config = RefineConfig(fail_on_xfail_markers=True)
        
        mock_refine_port = MagicMock()
        mock_telemetry = MagicMock()
        mock_telemetry.create_child_span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        
        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry,
            executor=MagicMock(),
            config=config
        )
        
        # Mock file status tracker
        mock_status_tracker = MagicMock()
        refiner._status_tracker = mock_status_tracker
        
        # Mock pytest to return success but with XFAIL in output
        with patch.object(refiner, 'run_pytest', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "returncode": 0,  # Tests "pass" but with xfail
                "stdout": "test_example.py::test_something XFAIL",
                "stderr": "",
                "command": "pytest test_example.py"
            }
            
            async def mock_context_fn(test_file, test_content):
                return {}
            
            result = await refiner.refine_until_pass(
                str(test_file),
                max_iterations=1,
                build_source_context_fn=mock_context_fn
            )
            
            assert result["success"] is False
            assert result["final_status"] == "xfail_detected"
            assert "xfail markers" in result["error"].lower()
            assert result.get("bug_report_created") is True
            
            # Check that the test file was marked with bug info
            marked_content = test_file.read_text()
            assert "PRODUCTION BUG DETECTED" in marked_content
            assert "pytest.mark.xfail" in marked_content
            
            # Check that bug report was created
            bug_report = test_file.parent / f"BUG_REPORT_{test_file.stem}.md"
            assert bug_report.exists()
    
    @pytest.mark.asyncio
    async def test_refine_until_pass_with_prod_bug_detection(self, tmp_path):
        """Test that refinement stops when LLM detects a production bug."""
        test_file = tmp_path / "test_math.py"
        test_file.write_text("""
def test_calculation():
    from math_module import calculate
    assert calculate(2, 3) == 5  # Expecting 5 but gets 6
""")
        
        config = RefineConfig(
            strict_assertion_preservation=True,
            report_suspected_prod_bugs=True
        )
        
        # Mock refine port to return production bug detection
        mock_refine_port = MagicMock()
        mock_refine_port.refine_from_failures.return_value = {
            "success": False,
            "final_status": "prod_bug_suspected",
            "suspected_prod_bug": "The calculate function returns 6 when given inputs (2, 3) but the test expects 5. This indicates a bug in the calculate function's implementation - it appears to be adding instead of performing the expected operation.",
            "error": "Suspected production bug detected"
        }
        
        mock_telemetry = MagicMock()
        mock_telemetry.create_child_span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        
        refiner = PytestRefiner(
            refine_port=mock_refine_port,
            telemetry_port=mock_telemetry,
            executor=MagicMock(),
            config=config
        )
        
        # Mock status tracker
        mock_status_tracker = MagicMock()
        refiner._status_tracker = mock_status_tracker
        
        # Mock pytest to return failure with assertion error
        with patch.object(refiner, 'run_pytest', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "returncode": 1,
                "stdout": """
FAILED test_math.py::test_calculation
AssertionError: assert 6 == 5
where 6 = calculate(2, 3)
""",
                "stderr": "",
                "command": "pytest test_math.py"
            }
            
            async def mock_context_fn(test_file, test_content):
                return {}
            
            result = await refiner.refine_until_pass(
                str(test_file),
                max_iterations=3,
                build_source_context_fn=mock_context_fn
            )
            
            assert result["success"] is False
            assert result["final_status"] == "prod_bug_suspected"
            assert "calculate function returns 6" in result.get("suspected_prod_bug", "")
            assert result.get("bug_report_created") is True
            
            # Check that the test file was marked with bug info
            marked_content = test_file.read_text()
            assert "PRODUCTION BUG DETECTED" in marked_content
            assert "pytest.mark.xfail" in marked_content
            assert "calculate function returns 6" in marked_content  # LLM's description
            
            # Check that bug report was created
            bug_report = test_file.parent / f"BUG_REPORT_{test_file.stem}.md"
            assert bug_report.exists()
            report_content = bug_report.read_text()
            assert "PRODUCTION BUG REPORT" in report_content
            assert "LLM Analysis" in report_content
            assert "calculate function returns 6" in report_content  # LLM's description
    
    def test_refine_adapter_with_strict_preservation(self):
        """Test that RefineAdapter includes strict preservation in prompts."""
        config = RefineConfig(strict_assertion_preservation=True)
        
        mock_llm = MagicMock()
        adapter = RefineAdapter(
            llm=mock_llm,
            config=config
        )
        
        payload = {
            "test_file_path": "test_example.py",
            "iteration": 1,
            "current_test_content": "def test_foo(): assert False",
            "pytest_failure_output": "AssertionError"
        }
        
        instructions = adapter._payload_to_instructions(payload)
        
        # Check that strict preservation rules are included
        assert "STRICT SEMANTIC PRESERVATION MODE ACTIVE" in instructions
        assert "DO NOT weaken assertions" in instructions
        assert "DO NOT change expected values to match buggy" in instructions
        assert "suspected_prod_bug" in instructions
    
    def test_refine_adapter_without_strict_preservation(self):
        """Test that RefineAdapter excludes strict rules when disabled."""
        config = RefineConfig(strict_assertion_preservation=False)
        
        mock_llm = MagicMock()
        adapter = RefineAdapter(
            llm=mock_llm,
            config=config
        )
        
        payload = {
            "test_file_path": "test_example.py",
            "iteration": 1,
            "current_test_content": "def test_foo(): assert False",
            "pytest_failure_output": "AssertionError"
        }
        
        instructions = adapter._payload_to_instructions(payload)
        
        # Check that strict preservation rules are NOT included
        assert "STRICT SEMANTIC PRESERVATION MODE ACTIVE" not in instructions
    
    def test_refine_adapter_handles_suspected_bug_response(self, tmp_path):
        """Test that RefineAdapter correctly handles suspected bug in LLM response."""
        # Create a temporary test file
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_foo(): assert False")
        
        config = RefineConfig(
            strict_assertion_preservation=True,
            report_suspected_prod_bugs=True
        )
        
        mock_llm = MagicMock()
        mock_llm.refine_content = MagicMock(return_value={
            "refined_content": "def test_foo(): assert False",
            "changes_made": "No changes - production bug suspected",
            "confidence": 0.9,
            "improvement_areas": [],
            "suspected_prod_bug": "The production code returns 6 instead of expected 5"
        })
        
        adapter = RefineAdapter(
            llm=mock_llm,
            config=config
        )
        
        result = adapter.refine_from_failures(
            test_file=str(test_file),
            failure_output="AssertionError: assert 6 == 5",
            max_iterations=1
        )
        
        assert result["success"] is False
        assert result["final_status"] == "prod_bug_suspected"
        assert "suspected_prod_bug" in result
        assert "returns 6 instead of expected 5" in result["suspected_prod_bug"]
