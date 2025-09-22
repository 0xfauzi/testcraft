"""
Tests for the subprocess_safe module.

This module tests the robust subprocess execution utilities,
including timeout handling, error handling, and process cleanup.
"""

import subprocess
from unittest.mock import Mock, patch

import pytest

from testcraft.adapters.io.python_runner import run_python_module
from testcraft.adapters.io.subprocess_safe import (
    SubprocessExecutionError,
    SubprocessTimeoutError,
    run_subprocess_safe,
    run_subprocess_simple,
)


class TestSubprocessSafe:
    """Test the safe subprocess context manager."""

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_success(self, mock_popen):
        """Test that the context manager handles successful execution."""
        # Mock successful subprocess execution
        mock_proc = Mock()
        mock_proc.communicate.return_value = ("stdout", "stderr")
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0  # Process is finished
        mock_popen.return_value = mock_proc

        # Test the context manager
        cmd = ["echo", "test"]
        with run_subprocess_safe(cmd, timeout=30) as (stdout, stderr):
            assert stdout == "stdout"
            assert stderr == "stderr"

        # Verify process was created with correct parameters
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == cmd
        assert call_args[1]["start_new_session"] is True
        assert call_args[1]["stdout"] == subprocess.PIPE
        assert call_args[1]["stderr"] == subprocess.PIPE

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_timeout(self, mock_popen):
        """Test that the context manager properly handles timeout cleanup."""
        # Mock subprocess that times out
        mock_proc = Mock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(["test"], 30),  # initial call times out
            ("", ""),  # cleanup communicate succeeds
        ]
        mock_proc.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_proc

        # Test timeout handling
        cmd = ["sleep", "60"]
        with pytest.raises(SubprocessTimeoutError):
            with run_subprocess_safe(cmd, timeout=30):
                pass

        # Verify cleanup was attempted
        mock_proc.kill.assert_called_once()
        assert mock_proc.communicate.call_count == 2

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_error(self, mock_popen):
        """Test that the context manager handles subprocess errors."""
        # Mock subprocess that returns error
        mock_proc = Mock()
        mock_proc.communicate.return_value = ("stdout", "stderr")
        mock_proc.returncode = 1  # Error return code
        mock_proc.poll.return_value = 1  # Process finished with error
        mock_popen.return_value = mock_proc

        # Test error handling
        cmd = ["false"]  # Command that returns error
        with pytest.raises(SubprocessExecutionError):
            with run_subprocess_safe(cmd, timeout=30):
                pass

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_process_cleanup(self, mock_popen):
        """Test that the context manager ensures process termination in finally block."""
        # Mock subprocess that needs termination
        mock_proc = Mock()
        mock_proc.communicate.return_value = ("stdout", "stderr")
        mock_proc.returncode = 0
        mock_proc.poll.return_value = (
            None  # Process still running when finally executes
        )
        mock_popen.return_value = mock_proc

        # Test process cleanup
        cmd = ["test_command"]
        with run_subprocess_safe(cmd, timeout=30):
            pass

        # Verify termination was called since poll() returned None
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called()

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_with_input(self, mock_popen):
        """Test subprocess execution with input text."""
        # Mock successful subprocess execution
        mock_proc = Mock()
        mock_proc.communicate.return_value = ("output", "")
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        # Test with input
        cmd = ["cat"]
        input_text = "hello world"
        with run_subprocess_safe(cmd, input_text=input_text) as (stdout, stderr):
            assert stdout == "output"

        # Verify input was passed to communicate
        mock_proc.communicate.assert_called_once_with(input=input_text, timeout=30)

    @patch("subprocess.Popen")
    def test_run_subprocess_safe_with_cwd_and_env(self, mock_popen):
        """Test subprocess execution with custom working directory and environment."""
        # Mock successful subprocess execution
        mock_proc = Mock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        # Test with cwd and env
        cmd = ["pwd"]
        cwd = "/tmp"
        env = {"TEST_VAR": "test_value"}

        with run_subprocess_safe(cmd, cwd=cwd, env=env):
            pass

        # Verify Popen was called with cwd and env
        call_args = mock_popen.call_args
        assert call_args[1]["cwd"] == cwd
        assert call_args[1]["env"] == env


class TestSubprocessSimple:
    """Test the simple subprocess wrapper function."""

    @patch("testcraft.adapters.io.subprocess_safe.run_subprocess_safe")
    def test_run_subprocess_simple_success(self, mock_safe):
        """Test simple wrapper with successful execution."""
        mock_safe.return_value.__enter__.return_value = ("stdout", "stderr")

        stdout, stderr, code = run_subprocess_simple(["echo", "test"])

        assert stdout == "stdout"
        assert stderr == "stderr"
        assert code == 0

    @patch("testcraft.adapters.io.subprocess_safe.run_subprocess_safe")
    def test_run_subprocess_simple_error_raised(self, mock_safe):
        """Test simple wrapper with error and raise_on_error=True."""
        mock_safe.side_effect = SubprocessExecutionError("Command failed")

        with pytest.raises(SubprocessExecutionError):
            run_subprocess_simple(["false"], raise_on_error=True)

    @patch("testcraft.adapters.io.subprocess_safe.run_subprocess_safe")
    def test_run_subprocess_simple_error_not_raised(self, mock_safe):
        """Test simple wrapper with error and raise_on_error=False."""
        mock_safe.side_effect = SubprocessExecutionError("Command failed")

        stdout, stderr, code = run_subprocess_simple(["false"], raise_on_error=False)

        assert stdout is None
        assert "Command failed" in stderr
        assert code == 1

    @patch("testcraft.adapters.io.subprocess_safe.run_subprocess_safe")
    def test_run_subprocess_simple_timeout_not_raised(self, mock_safe):
        """Test simple wrapper with timeout and raise_on_error=False."""
        mock_safe.side_effect = SubprocessTimeoutError("Command timed out")

        stdout, stderr, code = run_subprocess_simple(
            ["sleep", "60"], raise_on_error=False
        )

        assert stdout is None
        assert "Command timed out" in stderr
        assert code == -1


class TestConvenienceFunctions:
    """Test the convenience functions."""

    @patch("testcraft.adapters.io.python_runner.run_subprocess_simple")
    def test_run_python_module(self, mock_simple):
        """Test the Python module runner."""
        import sys
        mock_simple.return_value = ("output", "", 0)

        result = run_python_module("pytest", ["--version"])

        mock_simple.assert_called_once_with(
            [sys.executable, "-m", "pytest", "--version"], timeout=30
        )
        assert result == ("output", "", 0)


class TestIntegration:
    """Integration tests using actual subprocesses (but safe ones)."""

    def test_echo_integration(self):
        """Test actual echo command execution."""
        with run_subprocess_safe(["echo", "hello"]) as (stdout, stderr):
            assert stdout.strip() == "hello"
            assert stderr == ""

    def test_timeout_integration(self):
        """Test timeout with a real subprocess."""
        # Use a command that should timeout quickly
        with pytest.raises(SubprocessTimeoutError):
            with run_subprocess_safe(["sleep", "10"], timeout=0.1):
                pass

    def test_error_integration(self):
        """Test error handling with a real subprocess."""
        # Use a command that should fail
        with pytest.raises(SubprocessExecutionError):
            with run_subprocess_safe(["false"]):  # 'false' command always returns 1
                pass

    def test_simple_wrapper_integration(self):
        """Test the simple wrapper with real commands."""
        stdout, stderr, code = run_subprocess_simple(["echo", "test"])
        assert stdout.strip() == "test"
        assert code == 0

        # Test with error suppression
        stdout, stderr, code = run_subprocess_simple(["false"], raise_on_error=False)
        assert stdout is None
        assert code != 0
