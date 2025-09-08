"""
Python runner utilities.

This module provides utilities for running Python modules via
`python -m ...`, built on top of the safe subprocess execution
utilities. It intentionally focuses on Python-specific execution
patterns to keep `subprocess_safe` generic and single-purpose.
"""


from .subprocess_safe import run_subprocess_simple


def run_python_module(
    module_name: str, args: list[str] | None = None, timeout: int = 30, **kwargs
) -> tuple[str | None, str | None, int]:
    """
    Run a Python module using 'python -m module_name'.

    Args:
        module_name: Name of the Python module to run
        args: Additional arguments to pass to the module
        timeout: Maximum time to wait
        **kwargs: Additional arguments to pass to run_subprocess_simple

    Returns:
        tuple: (stdout, stderr, return_code)
    """
    cmd = ["python", "-m", module_name]
    if args:
        cmd.extend(args)

    return run_subprocess_simple(cmd, timeout=timeout, **kwargs)
