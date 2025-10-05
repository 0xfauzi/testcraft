"""
Async subprocess execution utilities.

This module provides async wrappers around the existing synchronous subprocess
utilities (python_runner and subprocess_safe), allowing async workflows to use
the established subprocess patterns without blocking the event loop.

## Usage Examples

### Async Python module execution:
```python
from testcraft.adapters.io.async_runner import run_python_module_async

async def test_execution():
    stdout, stderr, returncode = await run_python_module_async(
        "pytest",
        args=["test_file.py", "-v"],
        timeout=60
    )
    print(f"Tests {'passed' if returncode == 0 else 'failed'}")
```

### Async subprocess with existing ThreadPoolExecutor:
```python
from testcraft.adapters.io.async_runner import run_python_module_async_with_executor

async def batch_testing(executor, test_files):
    tasks = [
        run_python_module_async_with_executor(
            executor, "pytest", args=[file, "-v"]
        )
        for file in test_files
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## Design Principles

- **Reuses existing abstractions**: Built on top of python_runner.py and subprocess_safe.py
- **Thread pool management**: Flexible executor handling for different use cases
- **Consistent interface**: Same signature as sync utilities with async wrapper
- **Performance optimized**: Reuses provided executors when available
- **Robust error handling**: Comprehensive exception handling for executor and subprocess issues
- **Executor lifecycle management**: Validates executor state before use

## Error Handling

The async functions can raise several types of exceptions:
- **RuntimeError**: When ThreadPoolExecutor is shut down or invalid
- **asyncio.TimeoutError**: When the overall operation times out
- **concurrent.futures.BrokenExecutor**: When the executor is broken or cannot run tasks
- **TestCraftError**: Wrapped subprocess errors from underlying utilities
- **OSError**: When subprocess cannot be executed

Always handle these exceptions appropriately in your async code.
"""

import asyncio
import logging
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor
from pathlib import Path

from .python_runner import run_python_module
from .subprocess_safe import run_subprocess_simple

# Module-level logger
logger = logging.getLogger(__name__)


def _validate_executor(executor: ThreadPoolExecutor) -> None:
    """
    Validate that a ThreadPoolExecutor is active and can accept tasks.

    Args:
        executor: ThreadPoolExecutor to validate

    Raises:
        RuntimeError: If executor is shut down or invalid
    """
    if not isinstance(executor, ThreadPoolExecutor):
        raise RuntimeError(f"Expected ThreadPoolExecutor, got {type(executor)}")

    if hasattr(executor, "_shutdown") and executor._shutdown:
        raise RuntimeError(
            "ThreadPoolExecutor is shut down and cannot accept new tasks"
        )


def _sync_python_module_wrapper(
    module_name: str,
    args: list[str] | None = None,
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    raise_on_error: bool = False,
) -> tuple[str | None, str | None, int]:
    """Synchronous wrapper for run_python_module to avoid lambda closures."""
    return run_python_module(
        module_name=module_name,
        args=args,
        timeout=timeout,
        cwd=cwd,
        env=env,
        raise_on_error=raise_on_error,
    )


def _sync_subprocess_wrapper(
    cmd: list[str],
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    input_text: str | None = None,
    raise_on_error: bool = True,
) -> tuple[str | None, str | None, int]:
    """Synchronous wrapper for run_subprocess_simple to avoid lambda closures."""
    return run_subprocess_simple(
        cmd=cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
        input_text=input_text,
        raise_on_error=raise_on_error,
    )


async def run_python_module_async(
    module_name: str,
    args: list[str] | None = None,
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    raise_on_error: bool = False,
) -> tuple[str | None, str | None, int]:
    """
    Async wrapper around run_python_module for non-blocking Python module execution.

    Creates a temporary thread pool executor for the operation. For batch operations
    or when you have an existing executor, use run_python_module_async_with_executor.

    Args:
        module_name: Name of the Python module to run
        args: Additional arguments to pass to the module
        timeout: Maximum time to wait for completion
        cwd: Working directory for the subprocess
        env: Environment variables for the subprocess
        raise_on_error: Whether to raise exception on non-zero exit codes

    Returns:
        tuple: (stdout, stderr, return_code)

    Raises:
        asyncio.TimeoutError: If the operation times out
        RuntimeError: If the temporary executor cannot be created
        OSError: If the subprocess cannot be executed
    """
    loop = asyncio.get_event_loop()
    # Add buffer to timeout for executor cleanup
    executor_timeout = timeout + 5

    try:
        # Use temporary executor for single operations
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _sync_python_module_wrapper,
                    module_name,
                    args,
                    timeout,
                    cwd,
                    env,
                    raise_on_error,
                ),
                timeout=executor_timeout,
            )
    except TimeoutError:
        logger.warning(
            f"Python module execution timed out after {executor_timeout} seconds"
        )
        raise
    except BrokenExecutor as e:
        logger.error(f"Executor failed during Python module execution: {e}")
        raise RuntimeError(f"ThreadPoolExecutor is broken: {e}") from e


async def run_python_module_async_with_executor(
    executor: ThreadPoolExecutor,
    module_name: str,
    args: list[str] | None = None,
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    raise_on_error: bool = False,
    **kwargs,
) -> tuple[str | None, str | None, int]:
    """
    Async wrapper around run_python_module using a provided executor.

    More efficient for batch operations or when you have an existing executor.
    This is the pattern used in GenerateUseCase for consistency with the
    existing thread pool.

    Args:
        executor: ThreadPoolExecutor to use for the operation
        module_name: Name of the Python module to run
        args: Additional arguments to pass to the module
        timeout: Maximum time to wait for completion
        cwd: Working directory for the subprocess
        env: Environment variables for the subprocess
        raise_on_error: Whether to raise exception on non-zero exit codes
        **kwargs: Additional arguments passed to run_python_module

    Returns:
        tuple: (stdout, stderr, return_code)

    Raises:
        RuntimeError: If executor is shut down or invalid
        asyncio.TimeoutError: If the operation times out
        BrokenExecutor: If the executor cannot run tasks
        OSError: If the subprocess cannot be executed
    """
    # Validate executor before use
    _validate_executor(executor)

    loop = asyncio.get_event_loop()
    # Add buffer to timeout for executor cleanup
    executor_timeout = timeout + 5

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _sync_python_module_wrapper,
                module_name,
                args,
                timeout,
                cwd,
                env,
                raise_on_error,
            ),
            timeout=executor_timeout,
        )
    except TimeoutError:
        logger.warning(
            f"Python module execution timed out after {executor_timeout} seconds"
        )
        raise
    except BrokenExecutor as e:
        logger.error(f"Executor failed during Python module execution: {e}")
        raise RuntimeError(f"ThreadPoolExecutor is broken: {e}") from e


async def run_subprocess_async(
    cmd: list[str],
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    input_text: str | None = None,
    raise_on_error: bool = False,
) -> tuple[str | None, str | None, int]:
    """
    Async wrapper around run_subprocess_simple for non-blocking subprocess execution.

    Args:
        cmd: Command and arguments to execute
        timeout: Maximum time to wait for completion
        cwd: Working directory for the subprocess
        env: Environment variables for the subprocess
        input_text: Text to send to subprocess stdin
        raise_on_error: Whether to raise exception on non-zero exit codes

    Returns:
        tuple: (stdout, stderr, return_code)

    Raises:
        asyncio.TimeoutError: If the operation times out
        RuntimeError: If the temporary executor cannot be created
        OSError: If the subprocess cannot be executed
    """
    loop = asyncio.get_event_loop()
    # Add buffer to timeout for executor cleanup
    executor_timeout = timeout + 5

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _sync_subprocess_wrapper,
                    cmd,
                    timeout,
                    cwd,
                    env,
                    input_text,
                    raise_on_error,
                ),
                timeout=executor_timeout,
            )
    except TimeoutError:
        logger.warning(
            f"Subprocess execution timed out after {executor_timeout} seconds"
        )
        raise
    except BrokenExecutor as e:
        logger.error(f"Executor failed during subprocess execution: {e}")
        raise RuntimeError(f"ThreadPoolExecutor is broken: {e}") from e


async def run_subprocess_async_with_executor(
    executor: ThreadPoolExecutor,
    cmd: list[str],
    timeout: int = 30,
    cwd: str | Path | None = None,
    env: dict | None = None,
    input_text: str | None = None,
    raise_on_error: bool = True,
    **kwargs,
) -> tuple[str | None, str | None, int]:
    """
    Async wrapper around run_subprocess_simple using a provided executor.

    Args:
        executor: ThreadPoolExecutor to use for the operation
        cmd: Command and arguments to execute
        timeout: Maximum time to wait for completion
        cwd: Working directory for the subprocess
        env: Environment variables for the subprocess
        input_text: Text to send to subprocess stdin
        raise_on_error: Whether to raise exception on non-zero exit codes
        **kwargs: Additional arguments passed to run_subprocess_simple

    Returns:
        tuple: (stdout, stderr, return_code)

    Raises:
        RuntimeError: If executor is shut down or invalid
        asyncio.TimeoutError: If the operation times out
        BrokenExecutor: If the executor cannot run tasks
        OSError: If the subprocess cannot be executed
    """
    # Validate executor before use
    _validate_executor(executor)

    loop = asyncio.get_event_loop()
    # Add buffer to timeout for executor cleanup
    executor_timeout = timeout + 5

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _sync_subprocess_wrapper,
                cmd,
                timeout,
                cwd,
                env,
                input_text,
                raise_on_error,
            ),
            timeout=executor_timeout,
        )
    except TimeoutError:
        logger.warning(
            f"Subprocess execution timed out after {executor_timeout} seconds"
        )
        raise
    except BrokenExecutor as e:
        logger.error(f"Executor failed during subprocess execution: {e}")
        raise RuntimeError(f"ThreadPoolExecutor is broken: {e}") from e
