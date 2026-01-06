from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_on_path()

# Allow test suite to bypass CLl virtualenv checks without impacting production runs
os.environ.setdefault("TESTCRAFT_SKIP_ENV_CHECK", "1")
