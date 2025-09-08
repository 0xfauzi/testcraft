"""
Main adapter for test refinement operations.

This module implements the RefinePort interface, providing functionality
for refining tests based on pytest failures and other quality issues.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ...ports.refine_port import RefinePort
from ...ports.llm_port import LLMPort
from ...domain.models import RefineOutcome
from ...adapters.io.subprocess_safe import run_subprocess_simple


class RefineAdapter:
    """
    Adapter for refining tests based on failures and quality issues.
    
    This adapter implements the RefinePort interface and uses LLM integration
    to intelligently fix pytest failures and improve test quality.
    """
    
    def __init__(self, llm: LLMPort):
        """
        Initialize the refine adapter.
        
        Args:
            llm: LLM adapter for generating refinements
        """
        self.llm = llm
    
    def refine_from_failures(
        self,
        test_file: Union[str, Path],
        failure_output: str,
        source_context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        timeout_seconds: int = 300,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Refine a test file based on pytest failure output.
        
        This method iteratively refines test files by:
        1. Analyzing pytest failure output
        2. Sending failure + current test code to LLM
        3. Getting refined test code back
        4. Safely applying changes
        5. Re-running pytest to verify fixes
        6. Repeating until success or max iterations reached
        
        Args:
            test_file: Path to the test file that failed
            failure_output: Raw pytest failure output (stdout/stderr)
            source_context: Optional source code context for fixing
            max_iterations: Maximum number of refinement attempts
            timeout_seconds: Maximum total time to spend on refinement
            **kwargs: Additional refinement parameters
            
        Returns:
            Dictionary containing:
                - 'success': Whether refinement was successful
                - 'refined_content': Updated test file content if successful
                - 'iterations_used': Number of refinement iterations performed
                - 'final_status': Final pytest run status
                - 'error': Error message if refinement failed
        """
        test_path = Path(test_file)
        if not test_path.exists():
            return {
                "success": False,
                "error": f"Test file not found: {test_path}",
                "iterations_used": 0,
                "final_status": "file_not_found"
            }
        
        start_time = time.time()
        iteration = 0
        previous_content = None
        
        for iteration in range(1, max_iterations + 1):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                return {
                    "success": False,
                    "error": f"Timeout after {timeout_seconds} seconds",
                    "iterations_used": iteration - 1,
                    "final_status": "timeout"
                }
            
            # Read current test content
            try:
                current_content = test_path.read_text(encoding="utf-8")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read test file: {e}",
                    "iterations_used": iteration - 1,
                    "final_status": "read_error"
                }
            
            # Check for no-change condition
            if previous_content is not None and current_content == previous_content:
                return {
                    "success": False,
                    "error": "No changes made in refinement iteration",
                    "iterations_used": iteration - 1,
                    "final_status": "no_change"
                }
            
            # Prepare refinement payload
            payload = self._build_refinement_payload(
                test_file=test_path,
                current_content=current_content,
                failure_output=failure_output,
                source_context=source_context,
                iteration=iteration,
                **kwargs
            )
            
            # Get refined content from LLM
            try:
                refined_content = self._get_llm_refinement(payload)
                if not refined_content or refined_content == current_content:
                    return {
                        "success": False,
                        "error": "LLM returned no changes or identical content",
                        "iterations_used": iteration,
                        "final_status": "llm_no_change"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"LLM refinement failed: {e}",
                    "iterations_used": iteration,
                    "final_status": "llm_error"
                }
            
            # Apply changes safely
            try:
                self._apply_refinement_safely(test_path, refined_content)
                previous_content = current_content
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to apply refinement: {e}",
                    "iterations_used": iteration,
                    "final_status": "apply_error"
                }
            
            # Re-run pytest to verify fixes
            pytest_result = self._run_pytest_verification(test_path)
            
            if pytest_result["success"]:
                return {
                    "success": True,
                    "refined_content": refined_content,
                    "iterations_used": iteration,
                    "final_status": "success"
                }
            else:
                # Update failure output for next iteration
                failure_output = pytest_result.get("output", failure_output)
        
        # Max iterations reached
        return {
            "success": False,
            "error": f"Max iterations ({max_iterations}) reached without success",
            "iterations_used": max_iterations,
            "final_status": "max_iterations"
        }
    
    def _build_refinement_payload(
        self,
        test_file: Path,
        current_content: str,
        failure_output: str,
        source_context: Optional[Dict[str, Any]] = None,
        iteration: int = 1,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Build the payload for LLM refinement request.
        
        Args:
            test_file: Path to the test file
            current_content: Current content of the test file
            failure_output: Pytest failure output
            source_context: Optional source code context
            iteration: Current iteration number
            **kwargs: Additional context
            
        Returns:
            Dictionary payload for LLM request
        """
        payload = {
            "task": "refine_failing_test",
            "test_file_path": str(test_file),
            "current_test_content": current_content,
            "pytest_failure_output": failure_output,
            "iteration": iteration,
            "instructions": [
                "Analyze the pytest failure output and current test content",
                "Identify the root cause of the test failure",
                "Generate fixed test code that addresses the specific failure",
                "Ensure the fix is minimal and focused on the actual issue",
                "Preserve existing test structure and style where possible",
                "Return only the corrected test file content"
            ]
        }
        
        # Add source context if available
        if source_context:
            payload["source_context"] = source_context
        
        # Add any additional context from kwargs
        payload.update({k: v for k, v in kwargs.items() if k not in payload})
        
        return payload
    
    def _get_llm_refinement(self, payload: Dict[str, Any]) -> str:
        """
        Get refined test content from LLM.
        
        Args:
            payload: Refinement request payload
            
        Returns:
            Refined test content
            
        Raises:
            Exception: If LLM request fails
        """
        # Convert payload to prompt text
        prompt = self._payload_to_prompt(payload)
        
        # Make LLM request
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.1  # Low temperature for consistent fixes
        )
        
        # Extract test content from response
        return self._extract_test_content(response)
    
    def _payload_to_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Convert refinement payload to LLM prompt.
        
        Args:
            payload: Refinement request payload
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a test refinement expert. Your task is to fix a failing test based on pytest output.

## Test File: {payload['test_file_path']}
## Iteration: {payload['iteration']}

## Current Test Content:
```python
{payload['current_test_content']}
```

## Pytest Failure Output:
```
{payload['pytest_failure_output']}
```
"""
        
        if "source_context" in payload and payload["source_context"]:
            prompt += f"""
## Source Code Context:
{payload['source_context']}
"""
        
        prompt += """
## Instructions:
1. Analyze the pytest failure output carefully
2. Identify the specific issue causing the test failure
3. Fix the test code to address the root cause
4. Keep changes minimal and focused
5. Preserve existing test structure and style
6. Return ONLY the corrected Python test code, no explanations

## Refined Test Content:
```python
"""
        
        return prompt
    
    def _extract_test_content(self, llm_response: str) -> str:
        """
        Extract test content from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Extracted test content
        """
        # Find code block in response
        if "```python" in llm_response:
            start = llm_response.find("```python") + len("```python")
            end = llm_response.find("```", start)
            if end != -1:
                return llm_response[start:end].strip()
        
        # If no code block, return the response as-is (fallback)
        return llm_response.strip()
    
    def _apply_refinement_safely(self, test_file: Path, refined_content: str) -> None:
        """
        Apply refined content to test file safely.
        
        Args:
            test_file: Path to test file to update
            refined_content: New content to write
            
        Raises:
            Exception: If write operation fails
        """
        # Create backup
        backup_path = test_file.with_suffix(test_file.suffix + ".bak")
        backup_path.write_text(test_file.read_text(encoding="utf-8"), encoding="utf-8")
        
        try:
            # Write refined content
            test_file.write_text(refined_content, encoding="utf-8")
        except Exception:
            # Restore from backup on failure
            test_file.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            raise
        finally:
            # Clean up backup
            if backup_path.exists():
                backup_path.unlink()
    
    def _run_pytest_verification(self, test_file: Path) -> Dict[str, Any]:
        """
        Run pytest on the refined test file to verify fixes.
        
        Args:
            test_file: Path to test file to run
            
        Returns:
            Dictionary with success status and output
        """
        try:
            stdout, stderr, return_code = run_subprocess_simple(
                ["python", "-m", "pytest", str(test_file), "-v"],
                timeout=60,
                raise_on_error=False
            )
            
            output = ""
            if stdout:
                output += stdout
            if stderr:
                output += stderr
            
            return {
                "success": return_code == 0,
                "output": output,
                "return_code": return_code
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Pytest execution failed: {e}",
                "return_code": -1
            }
    
    # Implement other required methods from RefinePort
    def refine(
        self,
        test_files: List[Union[str, Path]],
        source_files: Optional[List[Union[str, Path]]] = None,
        refinement_goals: Optional[List[str]] = None,
        **kwargs: Any
    ) -> RefineOutcome:
        """Basic refine method - delegates to more specific methods."""
        # This is a placeholder implementation
        # In a full implementation, this would coordinate multiple refinement operations
        return RefineOutcome(
            updated_files=[str(f) for f in test_files],
            rationale="General refinement not yet implemented",
            plan="Use refine_from_failures for pytest-based refinement"
        )
    
    def analyze_test_quality(
        self,
        test_file: Union[str, Path],
        source_file: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Analyze test quality - placeholder implementation."""
        return {
            "quality_score": 0.5,
            "coverage_score": 0.5,
            "maintainability_score": 0.5,
            "issues": ["Not implemented yet"],
            "recommendations": ["Use refine_from_failures for specific improvements"]
        }
    
    def suggest_improvements(
        self,
        test_file: Union[str, Path],
        improvement_type: str = "comprehensive",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Suggest improvements - placeholder implementation."""
        return {
            "suggestions": ["Not implemented yet"],
            "priority": ["low"],
            "estimated_effort": ["unknown"],
            "expected_benefit": ["unknown"]
        }
    
    def optimize_test_structure(
        self,
        test_file: Union[str, Path],
        optimization_goals: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Optimize test structure - placeholder implementation."""
        return {
            "optimized_structure": "Not implemented yet",
            "changes_needed": ["Not implemented yet"],
            "benefits": ["Not implemented yet"],
            "migration_plan": "Not implemented yet"
        }
    
    def enhance_test_coverage(
        self,
        test_file: Union[str, Path],
        source_file: Union[str, Path],
        coverage_gaps: Optional[List[int]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Enhance test coverage - placeholder implementation."""
        return {
            "new_tests": ["Not implemented yet"],
            "coverage_improvement": 0.0,
            "test_additions": ["Not implemented yet"],
            "coverage_analysis": "Not implemented yet"
        }
