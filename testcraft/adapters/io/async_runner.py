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
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .python_runner import run_python_module
from .subprocess_safe import run_subprocess_simple


async def run_python_module_async(
    module_name: str,
    args: Optional[List[str]] = None,
    timeout: int = 30,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict] = None,
    raise_on_error: bool = False
) -> Tuple[Optional[str], Optional[str], int]:
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
    """
    loop = asyncio.get_event_loop()
    
    # Use temporary executor for single operations
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor,
            lambda: run_python_module(
                module_name=module_name,
                args=args,
                timeout=timeout,
                cwd=cwd,
                env=env,
                raise_on_error=raise_on_error
            )
        )


async def run_python_module_async_with_executor(
    executor: ThreadPoolExecutor,
    module_name: str,
    args: Optional[List[str]] = None,
    timeout: int = 30,
    **kwargs
) -> Tuple[Optional[str], Optional[str], int]:
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
        **kwargs: Additional arguments passed to run_python_module
        
    Returns:
        tuple: (stdout, stderr, return_code)
    """
    loop = asyncio.get_event_loop()
    
    return await loop.run_in_executor(
        executor,
        lambda: run_python_module(
            module_name=module_name,
            args=args,
            timeout=timeout,
            **kwargs
        )
    )


async def run_subprocess_async(
    cmd: List[str],
    timeout: int = 30,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict] = None,
    input_text: Optional[str] = None,
    raise_on_error: bool = False
) -> Tuple[Optional[str], Optional[str], int]:
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
    """
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(
            executor,
            lambda: run_subprocess_simple(
                cmd=cmd,
                timeout=timeout,
                cwd=cwd,
                env=env,
                input_text=input_text,
                raise_on_error=raise_on_error
            )
        )


async def run_subprocess_async_with_executor(
    executor: ThreadPoolExecutor,
    cmd: List[str],
    timeout: int = 30,
    **kwargs
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Async wrapper around run_subprocess_simple using a provided executor.
    
    Args:
        executor: ThreadPoolExecutor to use for the operation
        cmd: Command and arguments to execute
        timeout: Maximum time to wait for completion
        **kwargs: Additional arguments passed to run_subprocess_simple
        
    Returns:
        tuple: (stdout, stderr, return_code)
    """
    loop = asyncio.get_event_loop()
    
    return await loop.run_in_executor(
        executor,
        lambda: run_subprocess_simple(
            cmd=cmd,
            timeout=timeout,
            **kwargs
        )
    )
